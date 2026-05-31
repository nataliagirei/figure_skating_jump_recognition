"""
Generates data/preprocessed/metadata.json by matching preprocessed video files
to their labels in registry.json.

Run after preprocessing.py has finished.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "registry.json"
PREPROCESSED_DIR = ROOT / "data" / "preprocessed"
OUTPUT_PATH = PREPROCESSED_DIR / "metadata.json"

with open(REGISTRY_PATH, encoding="utf-8") as f:
    registry = json.load(f)

metadata: dict = {}
missing: list[str] = []

for mp4 in sorted(PREPROCESSED_DIR.rglob("*.mp4")):
    stem = mp4.stem
    if stem not in registry:
        missing.append(stem)
        continue

    entry = registry[stem].copy()
    entry["preprocessed_filepath"] = str(mp4.relative_to(ROOT)).replace("\\", "/")
    metadata[stem] = entry

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)

print(f"Metadata saved: {OUTPUT_PATH}")
print(f"Videos matched: {len(metadata)}")
if missing:
    print(f"Not found in registry ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")