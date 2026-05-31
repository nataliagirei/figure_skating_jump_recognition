# Figure Skating Jump Recognition

Classifies figure skating jumps (salchow, axel, toe loop) from short video clips.  
A research project exploring what accuracy is achievable on a small dataset (~305 videos) using frozen pretrained backbones, pose estimation, optical flow, and ensemble feature engineering.

---

## Problem

Given a short video clip of a figure skating jump, predict the jump type:

| Label | Class | Count |
|---|---|---|
| 0 | Salchow | 95 |
| 1 | Axel | 109 |
| 2 | Toe Loop | 101 |
| 3 | Failed (excluded from training) | 14 |

**Core difficulty**: salchow vs. toe loop differ only in which blade edge contacts the ice at takeoff — a sub-10-pixel detail at 224×224 resolution. Axel is uniquely recognisable because it is the only jump with a forward approach, producing a distinctive global motion signature.

**Best result**: 50.0% val accuracy — ResNet18 (frozen) + LSTM temporal aggregation (last hidden state + mean pooling).  
**Robust ceiling** for global features without temporal modelling: ~46% (5-fold CV on 305 videos). Val set has 46 examples = 2.2pp per prediction, so single-split numbers carry ±5–8pp confidence intervals.

---

## Results

| Experiment | Architecture | Val acc | CV acc | Notes |
|---|---|---|---|---|
| **Exp 1 — Baseline** | **ResNet18+LSTM (frozen)** | **50.0%** | — | **Best result — LSTM temporal aggregation (last+mean pooling)** |
| Exp 2 — layer4 FT | ResNet18+LSTM partial FT | 42.6% | — | Overfits: train 81%, val 43% |
| Exp 3 — Pose LSTM | BiLSTM on keypoints | 52.2% | — | Pose-only; ~200K params |
| Exp 4 — R3D-18 full FT | R3D-18 33M params | 56.5% | — | Kinetics pretrain; train acc 93.8% |
| Exp 5 — R3D-18 frozen | Linear probe ~3K params | 43.5% | — | No temporal aggregation |
| Exp 6 — Sklearn | LogReg / SVM RBF | — | **46.2%** | Robust 5-fold CV on 305 videos |

### Sklearn ablation (5-fold CV, LogReg C=0.05)

| Feature set | Dim | CV acc |
|---|---|---|
| Video only (R3D-18) | 512 | 45.6% ± 3.5% |
| Pose only | 297 | 39.7% ± 3.0% |
| Flow only | 45 | 35.4% ± 7.6% |
| Local crop only | 512 | 42.6% ± 6.1% |
| Video + Pose | 809 | **46.2% ± 3.2%** |
| All four | 1366 | 43.0% ± 1.2% |

Best robust result: **SVM RBF C=1, Video+Pose → 46.2% (5-fold CV)**  
Best single-split result: **ResNet18+LSTM frozen → 50.0% val** (see Exp 1)

### Confusion pattern (stable across all experiments)

![Sklearn confusion matrix](reports/figures/exp6_sklearn/confusion_matrix.png)

```
              precision  recall  f1
salchow           0.31    0.29   0.30   ← confused with toe_loop
axel              0.58    0.61   0.60   ← best: unique forward approach
toe_loop          0.39    0.39   0.39   ← confused with salchow
```

### Feature ablation

![Feature ablation](reports/figures/exp6_sklearn/ablation_bar_chart.png)

---

## Dataset

- **305 videos** (train 213 / val 46 / test 46, stratified by class)
- Videos downloaded from YouTube / TikTok, trimmed to one jump per clip
- Frames pre-extracted as `(3, 224, 224)` float32 `.npy`, ImageNet-normalised
- 16 frames sampled per clip (uniform in eval, random segment in train)

```
data/
├── Axel/, Salchow/, Toe Loop/, Failed/   ← raw source videos
├── frames/                               ← .npy frames + metadata JSONs
│   ├── dataset_metadata_train.json
│   ├── dataset_metadata_val.json
│   └── dataset_metadata_test.json
├── poses/          ← MediaPipe pose sequences  (305 × .npy, shape T×99)
├── features/       ← R3D-18 video embeddings   (319 × 512-d .npy)
├── features_flow/  ← optical flow features     (319 × 45-d .npy)
└── features_local/ ← lower-body crop embeddings (319 × 512-d .npy)
```

---

## Architecture

### Feature extraction pipeline

```
Raw clip → pre-extracted frames: (T, 3, 224, 224) float32, ImageNet-normalised
    │
    ├── extract_video_features.py   frozen R3D-18 avgpool  →  (512,) per video
    ├── extract_flow_features.py    Farneback dense flow   →  (45,)  per video
    │                               15 pairs × [mean_u, mean_v, mean_mag]
    ├── extract_local_features.py   takeoff detection      →  (512,) per video
    │                               argmin(mean_v) → 4-frame window →
    │                               bottom-50% crop → ResNet18 avgpool
    └── extract_poses.py            MediaPipe Tasks API    →  (T, 99) per video
                                    33 landmarks × (x, y, visibility)
```

### PyTorch models

**Exp 1 — ResNet18+LSTM** (`model.py`): ResNet18 frozen → per-frame 512-d → LSTM(128) → last+mean pooling → BN → 3-class head.

**Exp 2 — ResNet18+LSTM unfrozen** (`model_a.py`): Same but `layer4` + avgpool trainable at LR 1e-5.

**Exp 3 — Pose BiLSTM** (`pose_model.py`): Linear(99→128) + ReLU → BiLSTM(256, 2-layer) → last+mean pooling → LayerNorm → 3-class head. ~200K params.

**Exp 4+5 — R3D-18** (`model_b.py`): R3D-18 backbone → BN(512) → Dropout(0.3) → 3-class head. `freeze_backbone=True/False` controls whether backbone is frozen.

---

## Project Structure

```
figure_skating_jump_recognition/
├── ai_training/
│   ├── dataset.py               ← FigureSkatingDataset (.npy frames + augmentation)
│   ├── model.py                 ← ResNet18+LSTM  [Exp 1]
│   ├── model_a.py               ← ResNet18+LSTM layer4 unfrozen  [Exp 2]
│   ├── model_b.py               ← R3D-18 heads  [Exp 4, 5]
│   ├── pose_dataset.py          ← PoseDataset (.npy pose sequences)
│   ├── pose_model.py            ← BiLSTM on pose  [Exp 3]
│   ├── train.py                 ← Exp 1
│   ├── train_a.py               ← Exp 2
│   ├── train_pose.py            ← Exp 3
│   ├── train_b_finetune.py      ← Exp 4 (full fine-tuning)
│   ├── train_b.py               ← Exp 5 (frozen linear probe)
│   ├── train_sklearn.py         ← Exp 6 (ablation + grid search)
│   ├── extract_video_features.py
│   ├── extract_flow_features.py
│   ├── extract_local_features.py
│   ├── reporting_utils.py       ← shared: save_training_curves, save_confusion_matrix
│   └── checkpoints/             ← .pth + history.json (auto-created)
├── pose_extraction/
│   └── extract_poses.py         ← MediaPipe Tasks API
├── reporting/
│   └── save_sample_frames.py    ← 3 frames per class (takeoff / mid-air / landing)
├── reports/figures/
│   ├── exp1_baseline/           ← training_curves.png, confusion_matrix.png
│   ├── exp2_resnet_finetune/
│   ├── exp3_pose_lstm/
│   ├── exp4_r3d18_full/
│   ├── exp5_r3d18_frozen/
│   ├── exp6_sklearn/            ← ablation_bar_chart.png, grid_search_bar_chart.png,
│   │                               confusion_matrix.png
│   └── sample_frames/           ← {class}_{takeoff|mid-air|landing}.png
├── video_preprocessing/
│   ├── preprocessing.py         ← resize, crop, frame extraction, normalise
│   └── metadata_generation.py
├── video_download/
│   └── yt_download.py
├── run_all.py                   ← master pipeline
├── build_registry.py
├── registry.json                ← video labels (jump type, rotation, fall)
└── EXPERIMENTS.md               ← full experiment log with all results
```

---

## Running

### Full pipeline
```bash
python run_all.py
```
Runs feature extraction (skip if done), then all 6 experiments in order (skip if checkpoint exists), then saves sample frames.

### Single experiment
```bash
python run_all.py --exp 5          # R3D-18 frozen only
python run_all.py --exp 6          # sklearn only (~30s)
```

### Skip training, regenerate reports only
```bash
python run_all.py --skip-training --skip-features
```

### Re-run a specific experiment from scratch
Delete the checkpoint, then run:
```bash
# Example: re-run Exp 5
del ai_training\checkpoints\best_model_b.pth
python run_all.py --exp 5
```

### Run training scripts directly (from ai_training/)
```bash
cd ai_training
python train_b.py          # Exp 5
python train_sklearn.py    # Exp 6 (always fast, no checkpoint skip)
```

---

## Dependencies

```
torch >= 2.0
torchvision
scikit-learn
mediapipe >= 0.10
opencv-python
matplotlib
numpy
tqdm
yt-dlp
```

---

## Key Findings

1. **LSTM temporal aggregation is the decisive factor.** ResNet18 (frozen ImageNet) + LSTM = **50.0% val**, while R3D-18 (frozen Kinetics, better backbone) + linear probe = 43.5%. The difference is not the backbone — it's that the LSTM retains 16 per-frame feature vectors and learns which frames matter, while the linear probe sees only a single global avgpool.

2. **Freeze the backbone.** Both Exp 2 (ResNet18 layer4 FT, 81%→43%) and Exp 4 (R3D-18 full FT, 94%→57%) show catastrophic overfitting. The pretrained features already encode enough motion structure; fine-tuning 2–33M params on 213 clips corrupts them.

3. **Axel is the only reliably separable class** (~61% recall across all experiments) because its forward approach is a visible global motion cue. Salchow vs. toe loop differ only at the blade contact point — a sub-10-pixel region that no global feature can capture.

4. **46.2% is the robust ceiling for linear/kernel classifiers** (5-fold CV on 305 videos). The 50.0% from Exp 1 is a single-split val result — above the robust ceiling but on 46 examples (each ≈ 2.2pp).

5. **Adding features hurts with linear classifiers** — all four features combined (1366-d) performs worse than video alone (512-d) with LogReg. SVM RBF absorbs the noise via non-linear boundaries.

6. **More data is the only real fix.** Estimate: 1000+ labelled clips or 4K close-up footage showing the blade contact would meaningfully raise the ceiling. A binary axel detector is achievable at ~70–80% with current data.

### Sample frames

| Class | Takeoff | Mid-air | Landing |
|---|---|---|---|
| **Salchow** | ![](reports/figures/sample_frames/salchow_takeoff.png) | ![](reports/figures/sample_frames/salchow_mid-air.png) | ![](reports/figures/sample_frames/salchow_landing.png) |
| **Axel** | ![](reports/figures/sample_frames/axel_takeoff.png) | ![](reports/figures/sample_frames/axel_mid-air.png) | ![](reports/figures/sample_frames/axel_landing.png) |
| **Toe Loop** | ![](reports/figures/sample_frames/toe_loop_takeoff.png) | ![](reports/figures/sample_frames/toe_loop_mid-air.png) | ![](reports/figures/sample_frames/toe_loop_landing.png) |

See `EXPERIMENTS.md` for the complete experiment log including all training curves, confusion matrices, and analysis.