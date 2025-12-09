import os
import sys
from pathlib import Path
import subprocess
import torch
import torchaudio
import numpy as np
import librosa

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
NUM_CHUNKS = 10
GDRIVE_REMOTE = "gdrive:diffrhythm_outputs"   # your remote

# ---------------------------------------------------------------------
# GOOGLE DRIVE UPLOAD HELPER (replaces rsync)
# ---------------------------------------------------------------------
def gdrive_upload_batch(local_dir: Path):
    """
    Uploads all files in local_dir to Google Drive using rclone.
    """
    for p in local_dir.glob("*"):
        if p.is_file():
            cmd = ["rclone", "copy", str(p.resolve()), GDRIVE_REMOTE]
            print("\n[GDRIVE] Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------
# OTHER FUNCTIONS (unchanged)
# ---------------------------------------------------------------------

# DiffRhythm imports
REPO_ROOT = Path(__file__).resolve().parent
INFER_DIR = REPO_ROOT / "infer"
sys.path.insert(0, str(INFER_DIR))

import infer
from infer_utils import (
    prepare_model,
    get_lrc_token,
    get_style_prompt,
    get_reference_latent,
    get_negative_style_prompt,
    normalize_audio,
)

ACTIVATIONS = {}

def make_hook(name):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output
        ACTIVATIONS.setdefault(name, []).append(out.detach().cpu())
    return hook

def register_dit_block_hooks(cfm):
    dit = cfm.transformer
    for name, module in dit.named_modules():
        if name.startswith("transformer_blocks.") and name.count(".") == 1:
            idx = int(name.split(".")[1])
            if idx % 4 == 0:
                module.register_forward_hook(make_hook(f"dit.{name}"))
                print("Registered hook on:", name)

KK_MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88],dtype=np.float32)
KK_MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17],dtype=np.float32)
NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def detect_key_from_waveform_torch(wave, sr):
    mono = wave.mean(dim=0) if wave.dim() == 2 else wave
    y = mono.cpu().numpy().astype(np.float32)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_energy = chroma.sum(axis=1)

    if np.all(chroma_energy == 0): 
        return "Unknown", 0

    chroma_norm = chroma_energy / (np.linalg.norm(chroma_energy) + 1e-8)

    major = KK_MAJOR / (np.linalg.norm(KK_MAJOR) + 1e-8)
    minor = KK_MINOR / (np.linalg.norm(KK_MINOR) + 1e-8)

    best_score = -999
    best_name = "Unknown"

    for i, note in enumerate(NOTE_NAMES):
        maj = np.roll(major, i)
        minr = np.roll(minor, i)

        ms = np.dot(chroma_norm, maj)
        ns = np.dot(chroma_norm, minr)

        if ms > best_score:
            best_score = ms
            best_name = f"{note} major"

        if ns > best_score:
            best_score = ns
            best_name = f"{note} minor"

    return best_name, best_score


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    audio_length = 100
    max_frames = 2100

    cfm, tokenizer, muq, vae = prepare_model(max_frames, device)
    register_dit_block_hooks(cfm)

    output_root = REPO_ROOT / "tmp_outputs"
    output_root.mkdir(exist_ok=True)

    note_names = NOTE_NAMES
    all_keys = [f"{n} major" for n in note_names] + [f"{n} minor" for n in note_names]
    moods = ["Romantic","Nostalgic","Heartfelt","Happy","Melancholic",
             "Love","Upbeat","Energetic","Uplifting","Carefree"]

    sample_idx = 0

    for mood in moods:
        for key_str in all_keys:
            global ACTIVATIONS
            ACTIVATIONS = {}

            # Generate
            lrc=""
            lrc_prompt, start_time, end_frame, song_duration = get_lrc_token(
                max_frames,lrc,tokenizer,audio_length,device)

            latent_prompt, pred_frames = get_reference_latent(
                device,max_frames,False,None,None,vae)

            style_prompt = get_style_prompt(
                muq, prompt=f"Classical, {mood}, Piano, key: {key_str}"
            )
            neg_prompt = get_negative_style_prompt(device)

            songs = infer.inference(
                cfm_model=cfm,
                vae_model=vae,
                cond=latent_prompt,
                text=lrc_prompt,
                duration=end_frame,
                style_prompt=style_prompt,
                negative_style_prompt=neg_prompt,
                start_time=start_time,
                pred_frames=pred_frames,
                batch_infer_num=1,
                song_duration=song_duration,
                chunked=True,
            )

            # Normalize + split
            song = normalize_audio(songs[0], target_dbfs=-6)
            sr = 44100
            total = song.shape[-1]
            eff = (total//NUM_CHUNKS)*NUM_CHUNKS
            song = song[:,:eff]
            chunk_size = eff//NUM_CHUNKS

            chunk_audio = []
            chunk_keys = []
            for i in range(NUM_CHUNKS):
                w = song[:, i*chunk_size:(i+1)*chunk_size]
                chunk_audio.append(w)
                k, s = detect_key_from_waveform_torch(w, sr)
                chunk_keys.append((k, s))

            # Split activations
            chunked_activations=[{} for _ in range(NUM_CHUNKS)]
            for name, tensors in ACTIVATIONS.items():
                base = tensors[0]
                if base.ndim>=3:
                    td=1
                elif base.ndim==2:
                    td=0
                else:
                    for i in range(NUM_CHUNKS):
                        chunked_activations[i][name]=base
                    continue

                full = torch.cat(tensors,dim=td) if len(tensors)>1 else base
                L = full.shape[td]
                L_trunc=(L//NUM_CHUNKS)*NUM_CHUNKS
                clen=L_trunc//NUM_CHUNKS
                cropped=full.narrow(td,0,L_trunc)

                for i in range(NUM_CHUNKS):
                    start=i*clen
                    chunked_activations[i][name]=cropped.narrow(td,start,clen)

            # SAVE + GDRIVE UPLOAD + DELETE
            for i in range(NUM_CHUNKS):
                idx = f"{sample_idx:04d}"
                wav = output_root/f"audio_{idx}.wav"
                act = output_root/f"activations_{idx}.pt"
                key = output_root/f"key_{idx}.txt"

                torchaudio.save(str(wav), chunk_audio[i], sr)
                torch.save(chunked_activations[i], act)
                with open(key,"w") as f:
                    f.write(chunk_keys[i][0]+"\n")
                    f.write("score="+str(chunk_keys[i][1])+"\n")

                print(f"[CHUNK {idx}] ready")

                # GOOGLE DRIVE UPLOAD (instead of rsync)
                gdrive_upload_batch(output_root)

                # Clean local files
                for p in output_root.glob("*"):
                    if p.is_file():
                        p.unlink()

                sample_idx += 1

    print("Done. Total samples:", sample_idx)


if __name__ == "__main__":
    main()
