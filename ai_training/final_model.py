"""
Final hybrid inference model.

Jump type   — LogReg (sklearn) on Video (R3D-18 512-d) + Pose (297-d aggregated)
              Best combo from Exp 6: 46.2% CV accuracy, most reliable type estimate.
              Checkpoint: checkpoints/final_logreg_type.pkl

Rotation    — Pose BiLSTM (best rotation epoch from Exp 3)
              Peak: 67.4% val accuracy, 34pp above chance.
              Checkpoint: checkpoints/best_pose_model_rotation.pth
              Fallback:   checkpoints/best_pose_model.pth  (best type epoch, 45.6% rotation)

Usage as module:
    from final_model import FinalModel
    model = FinalModel()
    result = model.predict_from_stem("axel_clip_001")
    # → {"type": "axel", "rotation": "double", "type_proba": {...}, "rotation_proba": {...}}

Usage as script (quick sanity check on val set):
    python final_model.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose_model import PoseModel

ROOT     = Path(__file__).resolve().parent.parent
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
DATA_DIR = ROOT / "data"

TYPE_NAMES = ["salchow", "axel", "toe_loop"]
ROT_NAMES  = ["single", "double", "triple"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _aggregate_pose(pose_seq: np.ndarray) -> np.ndarray:
    """(T, 99) → (297,) mean + std + mean|velocity| for LogReg."""
    mean_pos = pose_seq.mean(axis=0)
    std_pos  = pose_seq.std(axis=0)
    velocity = np.abs(np.diff(pose_seq, axis=0)).mean(axis=0)
    return np.concatenate([mean_pos, std_pos, velocity])


class FinalModel:
    def __init__(
        self,
        logreg_path: Path | None = None,
        bilstm_path: Path | None = None,
    ) -> None:
        logreg_path = logreg_path or CKPT_DIR / "final_logreg_type.pkl"
        if not logreg_path.exists():
            raise FileNotFoundError(
                f"LogReg checkpoint not found: {logreg_path}\n"
                "Run: python train_sklearn.py"
            )

        rot_primary  = bilstm_path or CKPT_DIR / "best_pose_model_rotation.pth"
        rot_fallback = CKPT_DIR / "best_pose_model.pth"
        if rot_primary.exists():
            actual_bilstm = rot_primary
        elif rot_fallback.exists():
            print(f"[FinalModel] best_pose_model_rotation.pth not found, "
                  f"using best_pose_model.pth (rotation accuracy ~45.6% vs ~67.4%)")
            actual_bilstm = rot_fallback
        else:
            raise FileNotFoundError(
                f"Pose BiLSTM checkpoint not found.\n"
                "Run: python train_pose.py"
            )

        self.type_clf  = joblib.load(logreg_path)

        self.rot_model = PoseModel().to(DEVICE)
        self.rot_model.load_state_dict(
            torch.load(actual_bilstm, map_location=DEVICE, weights_only=True)
        )
        self.rot_model.eval()

        self._logreg_path = logreg_path
        self._bilstm_path = actual_bilstm

    def predict(
        self,
        video_feat: np.ndarray,
        pose_seq: np.ndarray,
    ) -> dict:
        """
        Args:
            video_feat: (512,) R3D-18 avgpool vector (from data/features/{stem}.npy)
            pose_seq:   (T, 99) MediaPipe pose sequence (from data/poses/{stem}.npy)

        Returns:
            {
              "type": "axel",
              "type_idx": 1,
              "type_proba": {"salchow": 0.12, "axel": 0.76, "toe_loop": 0.12},
              "rotation": "double",
              "rotation_idx": 1,
              "rotation_proba": {"single": 0.05, "double": 0.82, "triple": 0.13},
            }
        """
        pose_agg = _aggregate_pose(pose_seq)
        x = np.concatenate([video_feat, pose_agg])[np.newaxis, :]  # (1, 809)
        type_idx   = int(self.type_clf.predict(x)[0])
        type_proba = self.type_clf.predict_proba(x)[0].tolist()

        pose_tensor = torch.from_numpy(pose_seq).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _, rot_logits = self.rot_model(pose_tensor)
            rot_proba = F.softmax(rot_logits, dim=-1).squeeze(0).cpu().numpy()
        rot_idx = int(rot_proba.argmax())

        return {
            "type":           TYPE_NAMES[type_idx],
            "type_idx":       type_idx,
            "type_proba":     {n: float(p) for n, p in zip(TYPE_NAMES, type_proba)},
            "rotation":       ROT_NAMES[rot_idx],
            "rotation_idx":   rot_idx,
            "rotation_proba": {n: float(p) for n, p in zip(ROT_NAMES, rot_proba)},
        }

    def predict_from_stem(self, stem: str) -> dict:
        """Load precomputed features by stem name and predict."""
        video_feat = np.load(DATA_DIR / "features" / f"{stem}.npy")
        pose_seq   = np.load(DATA_DIR / "poses"    / f"{stem}.npy")
        return self.predict(video_feat, pose_seq)

    @staticmethod
    def checkpoints_ready() -> tuple[bool, bool]:
        """Returns (logreg_ready, bilstm_ready)."""
        logreg_ok = (CKPT_DIR / "final_logreg_type.pkl").exists()
        bilstm_ok = (CKPT_DIR / "best_pose_model_rotation.pth").exists() or \
                    (CKPT_DIR / "best_pose_model.pth").exists()
        return logreg_ok, bilstm_ok


def _quick_eval() -> None:
    """Sanity check on val set. Type acc is inflated (LogReg trained on all data).
    Rotation acc is meaningful (BiLSTM held out val during training)."""
    print(f"Loading FinalModel (device: {DEVICE})")
    model = FinalModel()
    print(f"  Type:     {model._logreg_path.name}")
    print(f"  Rotation: {model._bilstm_path.name}")

    meta_path = DATA_DIR / "frames" / "dataset_metadata_val.json"
    if not meta_path.exists():
        print("Val metadata not found — run training_preparation.py first")
        return

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    FAILED = 3

    type_correct = type_total = 0
    rot_correct  = rot_total  = 0

    print(f"\n{'stem':<35} {'pred_type':<12} {'gt_type':<12} "
          f"{'pred_rot':<10} {'gt_rot':<10} {'T/R'}")
    print("-" * 95)

    for stem, entry in meta.items():
        if entry["jump_type_label"] == FAILED:
            continue
        vid_path  = DATA_DIR / "features" / f"{stem}.npy"
        pose_path = DATA_DIR / "poses"    / f"{stem}.npy"
        if not vid_path.exists() or not pose_path.exists():
            continue

        result = model.predict_from_stem(stem)
        gt_type    = TYPE_NAMES[entry["jump_type_label"]]
        gt_rot_lbl = entry.get("rotation", 0)  # field name in metadata is "rotation"
        gt_rot     = ROT_NAMES[gt_rot_lbl - 1] if gt_rot_lbl > 0 else "?"

        t_ok = result["type"] == gt_type
        type_correct += int(t_ok)
        type_total   += 1

        r_ok = False
        if gt_rot != "?":
            r_ok = result["rotation"] == gt_rot
            rot_correct += int(r_ok)
            rot_total   += 1

        flag = ("+" if t_ok else "-") + ("+" if r_ok else "-")
        print(f"{stem:<35} {result['type']:<12} {gt_type:<12} "
              f"{result['rotation']:<10} {gt_rot:<10} {flag}")

    print("-" * 95)
    print(f"Val type acc     : {type_correct}/{type_total} = {type_correct/type_total:.1%}"
          f"  (NOTE: LogReg trained on all 305 videos; reliable estimate = 46.2% CV)")
    if rot_total > 0:
        print(f"Val rotation acc : {rot_correct}/{rot_total} = {rot_correct/rot_total:.1%}"
              f"  (BiLSTM val held out during training - this is a real estimate)")


if __name__ == "__main__":
    _quick_eval()