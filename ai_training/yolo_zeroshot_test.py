"""
YOLO zero-shot diagnostic.

Runs YOLOv8n on the takeoff frame of N_SAMPLES videos per class.
Saves a 4-column comparison grid and a text summary.

Output: reports/figures/yolo_zeroshot/
  {class}_grid.png   — Original | YOLO detections | YOLO blade crop | Heuristic crop
  summary.txt        — detection rates, confidence, bbox coverage stats

Columns compared:
  1. Original takeoff frame
  2. YOLO detections (person bbox=green, blade zone=cyan)
  3. YOLO blade crop  — bottom FOOT_FRAC of person bbox, resized to 224×224
  4. Heuristic crop   — current bottom-50% of frame (extract_local_features.py)

Usage:
    pip install ultralytics
    python ai_training/yolo_zeroshot_test.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT      = Path(__file__).resolve().parent.parent
FLOW_DIR  = ROOT / "data" / "features_flow"
METAS     = [
    ROOT / "data" / "frames" / "dataset_metadata_train.json",
    ROOT / "data" / "frames" / "dataset_metadata_val.json",
    ROOT / "data" / "frames" / "dataset_metadata_test.json",
]
OUT_DIR = ROOT / "reports" / "figures" / "yolo_zeroshot"

N_SAMPLES   = 5     # videos per class
FOOT_FRAC   = 0.30  # bottom 30% of person bbox = blade region
CONF_THRESH = 0.25  # YOLO confidence threshold
COCO_PERSON = 0     # COCO class id for "person"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_NAMES   = {0: "salchow", 1: "axel", 2: "toe_loop"}


# ── helpers ────────────────────────────────────────────────────────────────────

def denorm(frame_npy: np.ndarray) -> np.ndarray:
    """(3,H,W) float32 ImageNet-normalised  ->  (H,W,3) uint8 RGB."""
    img = frame_npy.transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def resize224(img: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)


def heuristic_crop(img: np.ndarray) -> np.ndarray:
    """Bottom 50% of frame, resized to 224x224 (current extract_local_features logic)."""
    H = img.shape[0]
    return resize224(img[H // 2:, :, :])


def find_takeoff_idx(stem: str, n_frames: int) -> int:
    """Replicates extract_local_features.py takeoff detection."""
    flow_path = FLOW_DIR / f"{stem}.npy"
    if not flow_path.exists():
        return 2
    flow    = np.load(flow_path)     # (45,)
    mean_v  = flow[1::3]             # (15,) vertical flow per pair
    pair_idx = int(np.argmin(mean_v))
    return min(pair_idx + 1, n_frames - 1)


def load_all_metadata() -> dict:
    entries = {}
    for path in METAS:
        data = json.loads(path.read_text(encoding="utf-8"))
        entries.update(data)
    return entries


# ── YOLO helpers ───────────────────────────────────────────────────────────────

def extract_blade_crop(
    img: np.ndarray, result
) -> tuple[np.ndarray | None, dict | None]:
    """
    Find highest-confidence person bbox in result.
    Crop bottom FOOT_FRAC of that bbox as the blade region.
    Returns (crop_224, stats) or (None, None).
    """
    H, W = img.shape[:2]
    best_conf = -1.0
    best_box  = None

    for box in result.boxes:
        if int(box.cls.item()) != COCO_PERSON:
            continue
        conf = float(box.conf.item())
        if conf > CONF_THRESH and conf > best_conf:
            best_conf = conf
            best_box  = box.xyxy[0].cpu().numpy()

    if best_box is None:
        return None, None

    x1, y1, x2, y2 = (
        max(0, int(best_box[0])),
        max(0, int(best_box[1])),
        min(W, int(best_box[2])),
        min(H, int(best_box[3])),
    )

    bbox_h  = y2 - y1
    foot_y1 = max(y1, int(y2 - bbox_h * FOOT_FRAC))

    blade = img[foot_y1:y2, x1:x2]
    if blade.size == 0:
        return None, None

    stats = {
        "conf":       best_conf,
        "bbox_area":  (x2 - x1) * (y2 - y1) / (H * W),
        "blade_area": (x2 - x1) * (y2 - foot_y1) / (H * W),
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "foot_y1": foot_y1,
    }
    return resize224(blade), stats


def annotate_frame(img: np.ndarray, result, stats: dict | None) -> np.ndarray:
    """Draw person bbox (green) and blade zone (cyan) on a copy of img."""
    import cv2
    out = img.copy()

    for box in result.boxes:
        conf = float(box.conf.item())
        if conf < CONF_THRESH:
            continue
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].cpu().numpy())
        if int(box.cls.item()) == COCO_PERSON:
            color = (0, 200, 0)
            label = f"person {conf:.2f}"
        else:
            color = (150, 150, 150)
            cls_name = result.names.get(int(box.cls.item()), str(int(box.cls.item())))
            label = f"{cls_name} {conf:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    if stats is not None:
        cv2.rectangle(
            out,
            (stats["x1"], stats["foot_y1"]),
            (stats["x2"], stats["y2"]),
            (0, 220, 220), 2,
        )
        cv2.putText(out, "blade zone", (stats["x1"], max(stats["foot_y1"] - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 220), 1)

    return out


# ── visualisation ──────────────────────────────────────────────────────────────

def save_class_grid(
    class_name: str,
    rows: list[dict],
    out_path: Path,
) -> None:
    n    = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = [axes]

    col_titles = [
        "Original (takeoff frame)",
        "YOLO detections  (person=green, blade=cyan)",
        f"YOLO blade crop  (bottom {int(FOOT_FRAC * 100)}% of person bbox)",
        "Heuristic crop  (bottom 50% of frame)",
    ]

    for row_i, row in enumerate(rows):
        for col_i, key in enumerate(["orig", "annotated", "blade", "heuristic"]):
            ax  = axes[row_i][col_i]
            img = row.get(key)
            if img is not None:
                ax.imshow(img)
            else:
                ax.set_facecolor("#1a1a1a")
                ax.text(0.5, 0.5, "no person\ndetected",
                        ha="center", va="center", color="#aaa",
                        fontsize=9, transform=ax.transAxes)
            if row_i == 0:
                ax.set_title(col_titles[col_i], fontsize=8, pad=4)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_i == 0:
                det_label = (f"conf {row['conf']:.2f}" if row["detected"]
                             else "no person")
                ax.set_ylabel(f"v{row_i + 1} | {det_label}", fontsize=8)

    fig.suptitle(
        f"YOLO zero-shot — {class_name}  ({n} videos sampled)",
        fontsize=11, y=1.005,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── main ───────────────────────────────────────────────────────────────────────

def run_yolo_test() -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not found.  Install with:  pip install ultralytics")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading YOLO model (yolo11n.pt) ...")
    model = YOLO("yolo11n.pt")   # auto-downloads on first run

    all_meta = load_all_metadata()

    by_class: dict[int, list[tuple[str, list[str]]]] = {0: [], 1: [], 2: []}
    for stem, entry in all_meta.items():
        lbl = entry.get("jump_type_label")
        if lbl in by_class:
            frames = entry["frames"]
            if all(Path(p).exists() for p in frames):
                by_class[lbl].append((stem, frames))

    summary_rows = []

    for label_id in [0, 1, 2]:
        class_name = CLASS_NAMES[label_id]
        pool       = by_class[label_id][:N_SAMPLES]

        if not pool:
            print(f"\n[WARN] No valid frames for class {class_name}, skipping.")
            continue

        print(f"\nClass: {class_name}  ({len(pool)} videos)")

        rows       = []
        det_confs  = []
        det_bboxes = []
        det_blades = []

        for stem, frame_paths in pool:
            t_idx     = find_takeoff_idx(stem, len(frame_paths))
            frame_npy = np.load(frame_paths[t_idx])   # (3,224,224)
            orig      = denorm(frame_npy)              # (224,224,3) uint8 RGB

            results   = model(orig[:, :, ::-1], verbose=False)  # BGR for YOLO
            result    = results[0]

            blade_img, stats = extract_blade_crop(orig, result)
            annotated        = annotate_frame(orig, result, stats)

            if stats:
                det_confs.append(stats["conf"])
                det_bboxes.append(stats["bbox_area"])
                det_blades.append(stats["blade_area"])

            rows.append({
                "orig":      orig,
                "annotated": annotated,
                "blade":     blade_img,
                "heuristic": heuristic_crop(orig),
                "detected":  stats is not None,
                "conf":      stats["conf"] if stats else 0.0,
            })

        n_det     = sum(r["detected"] for r in rows)
        avg_conf  = float(np.mean(det_confs))  if det_confs  else 0.0
        avg_bbox  = float(np.mean(det_bboxes)) if det_bboxes else 0.0
        avg_blade = float(np.mean(det_blades)) if det_blades else 0.0

        summary_rows.append({
            "class":      class_name,
            "n":          len(pool),
            "detected":   n_det,
            "det_rate":   n_det / max(len(pool), 1),
            "avg_conf":   avg_conf,
            "avg_bbox":   avg_bbox,
            "avg_blade":  avg_blade,
        })

        save_class_grid(class_name, rows, OUT_DIR / f"{class_name}_grid.png")

    # ── summary ────────────────────────────────────────────────────────────────
    lines = [
        "YOLO zero-shot diagnostic summary",
        f"Model: yolo11n.pt  |  conf_thresh={CONF_THRESH}  |  foot_frac={FOOT_FRAC}",
        f"N_SAMPLES={N_SAMPLES} per class",
        "",
        f"{'Class':<12} {'n':<5} {'detected':<10} {'det_rate':>9}  {'avg_conf':>9}  "
        f"{'avg_person_bbox':>15}  {'avg_blade_crop':>14}",
        "-" * 78,
    ]
    for r in summary_rows:
        lines.append(
            f"{r['class']:<12} {r['n']:<5} {r['detected']:<10} "
            f"{r['det_rate']:>8.1%}  {r['avg_conf']:>9.3f}  "
            f"{r['avg_bbox']:>14.3f}   {r['avg_blade']:>13.3f}"
        )
    lines += [
        "",
        "Interpretation guide:",
        "  det_rate   — % of sampled frames where a person was found (aim >80%)",
        "  avg_conf   — YOLO confidence for the person detection",
        "  avg_person_bbox — fraction of frame area covered by person bbox",
        "  avg_blade_crop  — fraction of frame area in the YOLO blade crop",
        "                    (vs heuristic: bottom-50% = 0.500 of the frame)",
        "",
        "If det_rate < 60%: YOLO struggles with this footage; fine-tuning needed.",
        "If avg_blade_crop << 0.10: crop is tight (good — less background noise).",
        "If avg_blade_crop > 0.20: person bbox is small; crop may still include ice.",
    ]
    summary_text = "\n".join(lines) + "\n"
    (OUT_DIR / "summary.txt").write_text(summary_text, encoding="utf-8")

    print(f"\n{'='*60}")
    print(summary_text)
    print(f"All outputs -> {OUT_DIR}")


if __name__ == "__main__":
    run_yolo_test()