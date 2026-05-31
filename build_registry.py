"""
Scans data/ directories and builds registry.json from scratch.
Run this once after adding new videos to data/.

Directory structure expected:
    data/{JumpType}/{Rotation}/{video}.mp4
    e.g. data/Axel/Double/abc123.mp4
         data/Failed/UnknownRotation/xyz.mp4
"""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

JUMP_TYPE_MAP = {
    "Axel":     ("axel",     1),
    "Salchow":  ("salchow",  0),
    "Toe Loop": ("toe_loop", 2),
    "Failed":   ("failed",   3),
}

ROTATION_MAP = {
    "Single":          1,
    "Double":          2,
    "Triple":          3,
    "UnknownRotation": 0,
}

ROTATION_NAMES = {0: "unknown", 1: "single", 2: "double", 3: "triple"}

SKIP_DIRS = {"preprocessed", "frames"}


def detect_source(stem: str) -> str:
    base = re.sub(r"_\d+$", "", stem)
    return "tiktok" if re.fullmatch(r"[0-9a-f]{32}", base) else "youtube"


def build() -> None:
    registry: dict = {}
    skipped: list[str] = []

    for mp4 in sorted(DATA_DIR.rglob("*.mp4")):
        parts = mp4.relative_to(DATA_DIR).parts

        if parts[0] in SKIP_DIRS:
            continue

        if len(parts) < 3:
            skipped.append(str(mp4))
            continue

        jump_dir, rotation_dir = parts[0], parts[1]

        if jump_dir not in JUMP_TYPE_MAP:
            skipped.append(f"Unknown jump dir: {mp4}")
            continue
        if rotation_dir not in ROTATION_MAP:
            skipped.append(f"Unknown rotation dir: {mp4}")
            continue

        jump_type, jump_type_label = JUMP_TYPE_MAP[jump_dir]
        rotation = ROTATION_MAP[rotation_dir]

        registry[mp4.stem] = {
            "filename": mp4.name,
            "filepath": str(mp4.relative_to(ROOT)).replace("\\", "/"),
            "jump_type": jump_type,
            "jump_type_label": jump_type_label,
            "rotation": rotation,
            "fall": 1 if jump_type == "failed" else 0,
            "source": detect_source(mp4.stem),
        }

    out = ROOT / "registry.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=4, ensure_ascii=False)

    print(f"Registry saved: {out}")
    print(f"Total videos: {len(registry)}")

    types = Counter(v["jump_type"] for v in registry.values())
    print("\nBy jump type:")
    for t, c in sorted(types.items()):
        print(f"  {t}: {c}")

    rotations = Counter((v["jump_type"], v["rotation"]) for v in registry.values())
    print("\nBy jump type + rotation:")
    for (t, r), c in sorted(rotations.items()):
        print(f"  {t} {ROTATION_NAMES[r]}: {c}")

    sources = Counter(v["source"] for v in registry.values())
    print("\nBy source:")
    for s, c in sorted(sources.items()):
        print(f"  {s}: {c}")

    if skipped:
        print(f"\nSkipped ({len(skipped)}):")
        for s in skipped:
            print(f"  {s}")


if __name__ == "__main__":
    build()
