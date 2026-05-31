"""
Creates stratified train/val/test splits from preprocessed metadata.
Excludes failed videos. Remaps labels to 0-based indices.

Outputs: data/splits/{train,val,test}.json
Run after video_preprocessing/metadata_generation.py.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT          = Path(__file__).resolve().parent.parent
INPUT_META    = ROOT / "data" / "preprocessed" / "metadata.json"
SPLITS_DIR    = ROOT / "data" / "splits"

TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15

# Failed excluded → 3 jump type classes
JUMP_TYPE_MAP = {"salchow": 0, "axel": 1, "toe_loop": 2}
# rotation: single→0, double→1, triple→2
ROTATION_MAP  = {1: 0, 2: 1, 3: 2}


def main() -> None:
    with open(INPUT_META, encoding="utf-8") as f:
        metadata = json.load(f)

    # Filter and remap
    videos: dict = {}
    for stem, entry in metadata.items():
        if entry["jump_type"] not in JUMP_TYPE_MAP:
            continue
        e = entry.copy()
        e["jump_type_label"] = JUMP_TYPE_MAP[entry["jump_type"]]
        e["rotation_label"]  = ROTATION_MAP[entry["rotation"]]
        videos[stem] = e

    print(f"Total after filtering failed: {len(videos)}")

    # Stratified split: each jump type contributes proportionally to each split
    groups: dict[str, list] = defaultdict(list)
    for stem, entry in videos.items():
        groups[entry["jump_type"]].append(stem)

    splits: dict[str, dict] = {"train": {}, "val": {}, "test": {}}

    for jump_type, stems in groups.items():
        random.shuffle(stems)
        n         = len(stems)
        train_end = int(n * TRAIN_SPLIT)
        val_end   = train_end + int(n * VAL_SPLIT)

        for stem in stems[:train_end]:
            splits["train"][stem] = videos[stem]
        for stem in stems[train_end:val_end]:
            splits["val"][stem] = videos[stem]
        for stem in stems[val_end:]:
            splits["test"][stem] = videos[stem]

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    for split, data in splits.items():
        out = SPLITS_DIR / f"{split}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        # Count per class
        counts = defaultdict(int)
        for v in data.values():
            counts[v["jump_type"]] += 1
        detail = "  ".join(f"{k}:{counts[k]}" for k in sorted(counts))
        print(f"{split:5s}: {len(data):3d} videos  [{detail}] → {out.name}")


if __name__ == "__main__":
    main()