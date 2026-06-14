"""
Extracts MediaPipe body pose keypoints from raw videos.
Saves one .npy file per video: shape (NUM_FRAMES, 99)
  → 33 landmarks × 3 values (x, y, visibility)

Uses original videos (not preprocessed) to preserve full body in frame.
Output: data/poses/{video_stem}.npy

Run after build_registry.py.
"""
import json
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm

ROOT      = Path(__file__).resolve().parent.parent
REGISTRY  = ROOT / "registry.json"
POSES_DIR = ROOT / "data" / "poses"
NUM_FRAMES = 32
NUM_LANDMARKS = 33
FEATURES_PER_LANDMARK = 3   # x, y, visibility
FEATURE_DIM = NUM_LANDMARKS * FEATURES_PER_LANDMARK  # 99

MODEL_PATH = Path(__file__).resolve().parent / "pose_landmarker_full.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)

SKIP_TYPES = {"failed"}


def _ensure_model() -> str:
    if not MODEL_PATH.exists():
        print(f"Downloading pose model -> {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")
    return str(MODEL_PATH)


def extract_poses(video_path: Path, num_frames: int) -> np.ndarray:
    """
    Returns array (num_frames, 99).
    Frames without detected pose reuse the last valid pose (or zeros).
    """
    model_path = _ensure_model()

    base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    cap   = cv2.VideoCapture(str(video_path))
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)

    indices = set(np.linspace(0, total - 1, num_frames, dtype=int).tolist())
    result  : list[np.ndarray] = []
    last_valid = np.zeros(FEATURE_DIM, dtype=np.float32)

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        for frame_idx in range(total):
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx not in indices:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            detection = landmarker.detect(mp_image)

            if detection.pose_landmarks:
                kpts = []
                for lm in detection.pose_landmarks[0]:  # first person only
                    kpts.extend([lm.x, lm.y, lm.visibility])
                last_valid = np.array(kpts, dtype=np.float32)
            # else: reuse last valid pose (keeps temporal continuity)

            result.append(last_valid.copy())

    cap.release()

    # Pad to num_frames if video was shorter or frames were missed
    while len(result) < num_frames:
        result.append(last_valid.copy())

    return np.stack(result[:num_frames])   # (num_frames, 99)


def normalize_pose(poses: np.ndarray) -> np.ndarray:
    """
    Hip-centred + shoulder-width normalisation.
    Makes poses invariant to position and body size in frame.

    poses: (T, 99) — 33 landmarks × (x, y, vis)
    """
    # Landmark indices: left_hip=23, right_hip=24, left_shoulder=11, right_shoulder=12
    LEFT_HIP, RIGHT_HIP = 23, 24
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

    out = poses.copy()

    for t in range(len(poses)):
        frame = poses[t].reshape(NUM_LANDMARKS, FEATURES_PER_LANDMARK)
        xy    = frame[:, :2]   # (33, 2)

        hip_centre   = (xy[LEFT_HIP] + xy[RIGHT_HIP]) / 2
        shoulder_dist = np.linalg.norm(xy[LEFT_SHOULDER] - xy[RIGHT_SHOULDER])
        if shoulder_dist < 1e-6:
            shoulder_dist = 1.0

        norm_xy = (xy - hip_centre) / shoulder_dist

        norm_frame = frame.copy()
        norm_frame[:, :2] = norm_xy
        out[t] = norm_frame.reshape(-1)

    return out


def main() -> None:
    with open(REGISTRY, encoding="utf-8") as f:
        registry = json.load(f)

    POSES_DIR.mkdir(parents=True, exist_ok=True)

    videos = [
        (stem, entry) for stem, entry in registry.items()
        if entry["jump_type"] not in SKIP_TYPES
    ]

    print(f"Videos to process: {len(videos)}")

    failed    : list[str] = []
    low_detect: list[str] = []

    for stem, entry in tqdm(videos):
        out_path = POSES_DIR / f"{stem}.npy"
        if out_path.exists():
            continue

        video_path = ROOT / entry["filepath"]
        if not video_path.exists():
            failed.append(stem)
            continue

        poses = extract_poses(video_path, NUM_FRAMES)
        poses = normalize_pose(poses)

        # Warn if many frames had no detection (all-zero keypoints)
        zero_frames = np.all(poses.reshape(NUM_FRAMES, NUM_LANDMARKS, -1)[:, :, :2] == 0, axis=(1, 2))
        if zero_frames.mean() > 0.5:
            low_detect.append(stem)

        np.save(out_path, poses)

    print(f"\nDone. Saved to {POSES_DIR}")
    if failed:
        print(f"Missing video files ({len(failed)}): {failed[:5]}")
    if low_detect:
        print(f"Low detection rate ({len(low_detect)} videos) - pose may be unreliable:")
        for s in low_detect[:10]:
            print(f"  {s}")


if __name__ == "__main__":
    main()