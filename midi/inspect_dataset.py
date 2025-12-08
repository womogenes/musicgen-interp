from pathlib import Path
import json
from collections import Counter

BASE = Path("/home/harinit9/orcd/pool/musicgen-data-nokey")
meta_path = BASE / "dataset_metadata.json"
meta = json.loads(meta_path.read_text())

print("Num samples:", len(meta))

detect_counts = Counter(m["detected_key"] for m in meta if m["detected_key"] is not None)
detect_mode = Counter(m["key_info"]["mode"] for m in meta)

print("Detected-key counts:")
print(detect_counts)
print(len(detect_counts))
print(detect_mode)
