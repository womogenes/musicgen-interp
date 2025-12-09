import os
import sys
import random
from pathlib import Path

import torch
import torchaudio

# ---------------------------------------------------------------------
# 1. Point Python at the DiffRhythm "infer" folder so we can import
#    infer.py and infer_utils.py, which live in infer/.
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
INFER_DIR = REPO_ROOT / "infer"
sys.path.insert(0, str(INFER_DIR))

import infer  # this is infer/infer.py, imported as a module
from infer_utils import (
    prepare_model,
    get_lrc_token,
    get_style_prompt,
    get_negative_style_prompt,
    get_reference_latent,
)


def main():
    # -----------------------------------------------------------------
    # 2. Basic config: device + duration
    # -----------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_length = 95  # seconds; 95 uses 2048 frames in the official script :contentReference[oaicite:3]{index=3}
    if audio_length == 95:
        max_frames = 2048
    elif 95 < audio_length <= 285:
        max_frames = 6144
    else:
        raise ValueError("audio_length must be 95 or between 96 and 285 (inclusive).")

    # -----------------------------------------------------------------
    # 3. Load models (same helper used by infer/infer.py)
    #    prepare_model returns: (cfm, tokenizer, muq, vae) :contentReference[oaicite:4]{index=4}
    # -----------------------------------------------------------------
    cfm, tokenizer, muq, vae = prepare_model(max_frames, device)

    # -----------------------------------------------------------------
    # 4. Lyrics (L RC) – use their example file
    #    If you want instrumental only, set lrc = "" instead.
    # -----------------------------------------------------------------
    lrc = ""
    lrc_prompt, start_time, end_frame, song_duration = get_lrc_token(
        max_frames, lrc, tokenizer, audio_length, device
    )

    # -----------------------------------------------------------------
    # 5. Style prompt
    #    Change this string to steer genre / mood / rough "key".
    # -----------------------------------------------------------------
    style_prompt = get_style_prompt(
        muq,
        prompt="heartfelt, classical, piano, key: C major"
    )
    negative_style_prompt = get_negative_style_prompt(device)

    # -----------------------------------------------------------------
    # 6. Latent prompt for editing / continuation.
    #    For simple generation (no editing), use edit=False and no ref song. :contentReference[oaicite:5]{index=5}
    # -----------------------------------------------------------------
    latent_prompt, pred_frames = get_reference_latent(
        device,
        max_frames,
        False,      # edit
        None,       # edit_segments
        None,       # ref_song
        vae,
    )

    # -----------------------------------------------------------------
    # 7. Run DiffRhythm inference (this calls cfm_model.sample + decode_audio) :contentReference[oaicite:6]{index=6}
    # -----------------------------------------------------------------
    generated_songs = infer.inference(
        cfm_model=cfm,
        vae_model=vae,
        cond=latent_prompt,
        text=lrc_prompt,
        duration=end_frame,
        style_prompt=style_prompt,
        negative_style_prompt=negative_style_prompt,
        start_time=start_time,
        pred_frames=pred_frames,
        batch_infer_num=1,
        song_duration=song_duration,
        chunked=True,  # reduce VRAM usage; recommended for 8GB cards :contentReference[oaicite:7]{index=7}
    )

    # Take one sample from the batch (here batch_infer_num=1 so it's just that one)
    song = random.choice(generated_songs)
    # -----------------------------------------------------------------
    # 8. Save to WAV
    # -----------------------------------------------------------------
    output_dir = REPO_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "diffrhythm_output.wav"

    torchaudio.save(str(output_path), song, sample_rate=44100)
    print(f"Saved DiffRhythm output to: {output_path}")


if __name__ == "__main__":
    main()
