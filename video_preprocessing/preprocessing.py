"""
Resizes and center-crops raw videos from data/ → data/preprocessed/.
Samples NUM_FRAMES frames per video to reduce file size.

Run before metadata_generation.py and training_preparation.py.
"""
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "data" / "preprocessed"

TARGET_SHORT_SIDE = 256
CROP_SIZE = 224
NUM_FRAMES = 32

SKIP_DIRS = {"preprocessed", "frames"}


def resize_short_side(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if h < w:
        new_h, new_w = TARGET_SHORT_SIDE, int(w * TARGET_SHORT_SIDE / h)
    else:
        new_h, new_w = int(h * TARGET_SHORT_SIDE / w), TARGET_SHORT_SIDE
    return cv2.resize(frame, (new_w, new_h))


def center_crop(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    x = w // 2 - CROP_SIZE // 2
    y = h // 2 - CROP_SIZE // 2
    return frame[y:y + CROP_SIZE, x:x + CROP_SIZE]


def sample_indices(total: int, n: int) -> list[int]:
    return list(np.linspace(0, total - 1, n, dtype=int))


def process_video(input_path: Path, output_path: Path) -> bool:
    cap = cv2.VideoCapture(str(input_path))
    all_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = center_crop(resize_short_side(frame))
        all_frames.append(frame)

    cap.release()

    if not all_frames:
        return False

    indices = sample_indices(len(all_frames), min(NUM_FRAMES, len(all_frames)))
    frames = [all_frames[i] for i in indices]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, 10, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    return True


def main() -> None:
    videos = [
        v for v in INPUT_DIR.rglob("*.mp4")
        if v.relative_to(INPUT_DIR).parts[0] not in SKIP_DIRS
    ]

    print(f"Found {len(videos)} videos to preprocess")
    failed = []

    for video in tqdm(videos):
        relative = video.relative_to(INPUT_DIR)
        output_path = OUTPUT_DIR / relative
        if output_path.exists():
            continue
        if not process_video(video, output_path):
            failed.append(str(video))

    if failed:
        print(f"\nFailed to process ({len(failed)}):")
        for f in failed:
            print(f"  {f}")
    print("Preprocessing done.")


if __name__ == "__main__":
    main()