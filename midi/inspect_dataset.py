from pathlib import Path
import json
from collections import Counter

BASE = Path("/home/harinit9/orcd/pool/musicgen-data")
meta_path = BASE / "dataset_metadata.json"
meta = json.loads(meta_path.read_text())

print("Num samples:", len(meta))

prompt_counts = Counter(m["prompted_key"] for m in meta)
detect_counts = Counter(m["detected_key"] for m in meta if m["detected_key"] is not None)

print("Prompted-key counts:")
print(prompt_counts)
print("Detected-key counts:")
print(detect_counts)