import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]
KEY_NAMES = [f"{n}_major" for n in NOTE_NAMES] + [
    f"{n}_minor" for n in NOTE_NAMES
]
KEY_INDEX = {k: i for i, k in enumerate(KEY_NAMES)}

def load_metadata(path: Path):
    print(f"Loading metadata from: {path}")
    with open(path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries")
    return data


def compute_counts_and_agreement(metadata):
    prompted_counts = Counter()
    detected_counts = Counter()
    agreement_count = 0
    detected_total = 0

    per_key_prompted = Counter()
    per_key_agree = Counter()

    conf_values = []

    # confusion matrix: prompted (rows) vs detected (cols)
    cm = np.zeros((len(KEY_NAMES), len(KEY_NAMES)), dtype=int)

    for m in metadata:
        prompted = m.get("prompted_key")
        detected = m.get("detected_key")
        ki = m.get("key_info", {})

        if prompted is not None:
            prompted_counts[prompted] += 1
            per_key_prompted[prompted] += 1

        if detected is not None:
            detected_counts[detected] += 1
            detected_total += 1

        # confidence
        conf = ki.get("confidence", None)
        if isinstance(conf, (int, float)):
            conf_values.append(conf)

        # agreement
        if prompted is not None and detected is not None:
            if prompted == detected:
                agreement_count += 1
                per_key_agree[prompted] += 1

            # confusion matrix
            if prompted in KEY_INDEX and detected in KEY_INDEX:
                i = KEY_INDEX[prompted]
                j = KEY_INDEX[detected]
                cm[i, j] += 1

    return {
        "prompted_counts": prompted_counts,
        "detected_counts": detected_counts,
        "agreement_count": agreement_count,
        "detected_total": detected_total,
        "per_key_prompted": per_key_prompted,
        "per_key_agree": per_key_agree,
        "conf_values": conf_values,
        "confusion_matrix": cm,
    }


def print_summary(stats, out_dir: Path):
    prompted_counts = stats["prompted_counts"]
    detected_counts = stats["detected_counts"]
    agreement_count = stats["agreement_count"]
    detected_total = stats["detected_total"]
    per_key_prompted = stats["per_key_prompted"]
    per_key_agree = stats["per_key_agree"]
    conf_values = stats["conf_values"]
    cm = stats["confusion_matrix"]

    print("\n=== BASIC COUNTS ===")
    print(f"Num samples (metadata entries): {sum(prompted_counts.values())}")
    print(f"Num samples with detected_key != None: {detected_total}")

    print("\nPrompted-key counts:")
    for k in KEY_NAMES:
        if prompted_counts[k] > 0:
            print(f"  {k:8s}: {prompted_counts[k]}")

    print("\nDetected-key counts:")
    for k in KEY_NAMES:
        if detected_counts[k] > 0:
            print(f"  {k:8s}: {detected_counts[k]}")

    print("\n=== AGREEMENT ===")
    if detected_total > 0:
        overall_agreement = agreement_count / detected_total
        print(f"Overall agreement (prompted == detected | detected exists): "
              f"{agreement_count}/{detected_total} = {overall_agreement:.3f}")
    else:
        print("No detected keys found.")

    print("\nPer-key agreement (conditioned on prompted_key):")
    for k in KEY_NAMES:
        n_prompt = per_key_prompted[k]
        if n_prompt == 0:
            continue
        n_agree = per_key_agree[k]
        frac = n_agree / n_prompt
        print(f"  {k:8s}: {n_agree}/{n_prompt} = {frac:.3f}")

    print("\n=== CONFIDENCE STATS (from key_info.confidence) ===")
    if conf_values:
        conf_arr = np.array(conf_values)
        print(f"Num confidences: {len(conf_arr)}")
        print(f"  mean   : {conf_arr.mean():.4f}")
        print(f"  median : {np.median(conf_arr):.4f}")
        print(f"  min/max: {conf_arr.min():.4f} / {conf_arr.max():.4f}")
        for thr in [0.05, 0.1, 0.2, 0.3, 0.4]:
            frac = (conf_arr >= thr).mean()
            print(f"  frac >= {thr:.2f}: {frac:.3f}")
    else:
        print("No confidence values found.")

    # Save confusion matrix to CSV for nice plotting later
    cm_path = out_dir / "prompted_vs_detected_confusion.csv"
    header = ",".join([""] + KEY_NAMES)
    lines = [header]
    for i, row_key in enumerate(KEY_NAMES):
        row = [row_key] + [str(x) for x in cm[i]]
        lines.append(",".join(row))
    cm_path.write_text("\n".join(lines))
    print(f"\nSaved prompted-vs-detected confusion matrix to {cm_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    if len(sys.argv) > 1:
        meta_path = Path(sys.argv[1])
    else:
        meta_path = Path("/home/harinit9/orcd/pool/musicgen-data/dataset_metadata.json")

    if not meta_path.exists():
        print(f"ERROR: {meta_path} does not exist.")
        sys.exit(1)

    metadata = load_metadata(meta_path)

    out_dir = meta_path.parent
    stats = compute_counts_and_agreement(metadata)
    print_summary(stats, out_dir)


if __name__ == "__main__":
    main()
