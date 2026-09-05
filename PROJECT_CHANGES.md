# PROJECT_CHANGES.md
> Append-only. Newest first. Format: `[DATE] [TYPE] Description`
> Types: TRAIN | CODE | CONFIG | DATA | CLEANUP | FIX | PLAN

---

## [2026-08-21] BENCHMARK — Phase 3: 2026 Full Benchmark & Robustness Matrix Complete
- Scripts: `scripts/evaluate_v10_full_battery.py` and `scripts/evaluate_robustness.py`.
- Kaggle In-Domain ($N=20,000$): **0.9999 ROC-AUC**, **99.69% Calibrated Accuracy**, **0.0038 Brier Score**.
- FaceForensics++ 100% Actor-Disjoint ($N=14,200$ across 5 cohorts): **0.6070 Macro AUC** (Deepfakes: **0.6736 AUC**, FaceShifter: **0.6200 AUC**, FaceSwap: **0.5836 AUC**, NeuralTextures: **0.5801 AUC**, Face2Face: **0.5775 AUC**).
- 14-Distortion Robustness Battery:
  - Pristine: `0.9074` AUC
  - JPEG Q=90: `0.9067` AUC (99.9% retention)
  - JPEG Q=70: `0.9070` AUC (100.0% retention)
  - JPEG Q=50: `0.9051` AUC (99.7% retention)
  - JPEG Q=30: `0.9010` AUC (99.3% retention)
  - Gaussian Blur ($k=7$): `0.9056` AUC (99.8% retention)
  - Downscaling ($0.5\times$): `0.9075` AUC (100.0% retention)
- Output files: `results/benchmark_eval_v10/v10_final_evaluation.json`, `results/benchmark_eval_v10/robustness_results.json`.

---

## [2026-08-21] TRAIN — Phase 2: DeepTrace V10 Robust Multi-Source Training Complete
- Script: `scripts/train_v10_robust.py` (10 epochs, batch size 32, accumulation 2, AdamW cosine annealing).
- Dataset: 33,100 training samples (FF++ 600 actor-disjoint frames + Kaggle 5k real/5k fake + On-The-Fly SBI blending).
- Best Validation ROC-AUC: **`0.8602`** (Epoch 7), Best Accuracy: **`76.82%`**.
- Checkpoint: `checkpoints/v10_robust/best_model.pth` (592 MB).

---

## [2026-08-21] FIX — Phase 1: Robust JPEG Augmentations & FF++ Split Deprecation
- `datasets/transforms.py`: Raised JPEG compression probability to `p=0.7` standalone (moved out of `OneOf` block) to force model to learn compression-invariant boundary features.
- `datasets/faceforensics.py`: Replaced corrupted sequential-slice `_apply_split()` method with explicit `RuntimeError` deprecation guard pointing to `utils/actor_splits.py`.
- Automated tests passed: tensor shape verification (3, 160, 160) and deprecation guard check.

---

## [2026-08-21] PLAN — V10 Robust Multi-Source Training Implementation Plan
- Full forensic audit confirmed shortcut learning (StyleGAN lattice memorization) as root cause of 59.31% FF++ cross-dataset AUC.
- 5-phase plan created: (1) Fix JPEG augmentation p=0.7, (2) Multi-source training, (3) Full benchmark battery, (4) Diffusion data expansion, (5) Final eval + governance.
- Architecture freeze decision: no new modules or parameters — fix data pipeline first.
- Governance files updated: PROJECT_MEMORY.yaml, PROJECT_DECISIONS.md, AGENT_HANDOVER.md.
- Implementation plan: stored in Antigravity artifact directory as implementation_plan.md.

---

## [2026-08-18] ENSEMBLE & VISUAL — V6 Dual-Stream Multimodal Ensemble Complete
- Script: `scripts/evaluate_v6_ensemble.py` & `scripts/generate_v6_gradcam_artifacts.py`
- Fusion: Convex combination of Macro Stream (V3-E2E: Spatial + 2D-DCT + CLIP) and Microscopic Residual Stream (V5-SRM: Spatial + 3-channel SRM High-Pass Residuals + CLIP).
- In-Domain Kaggle Retention (N=20,000): Accuracy = **99.69%**, ROC-AUC = **0.99991**, F1 = **0.9969** (Dominating MesoNet 84.16% and Xception 98.34%).
- FaceForensics++ Breakouts:
  - DeepFakeDetection: Accuracy = **81.35%**, ROC-AUC = **0.9999**, F1 = **0.8428**.
  - Deepfakes: Accuracy = **58.03%**, ROC-AUC = **0.6254**, F1 = **0.5597**.
  - FaceShifter: Accuracy = **55.65%**, ROC-AUC = **0.5827**, F1 = **0.5229**.
  - FaceSwap: Accuracy = **53.12%**, ROC-AUC = **0.5446**, F1 = **0.4816**.
  - Face2Face: Accuracy = **51.75%**, ROC-AUC = **0.5383**, F1 = **0.4582**.
  - Overall FF++ Benchmark (N=14,000): ROC-AUC = **0.6321** (Up from 0.5275 baseline), F1 = **0.6730**.
- GradCAM Overlays (results/gradcam_v6/): Authentic (P=0.0017), StyleGAN (P=0.9997), FaceSwap (P=0.8239), DeepFakeDetection (P=1.0000).
- Checkpoints: `checkpoints/v5_srm_residual/best_model.pth` + `checkpoints/v3_e2e_multisource/best_model.pth`.
- JSON Metric Report: `results/benchmark_eval_v6/v6_ensemble_evaluation.json`.

## [2026-08-17] TRAIN — Xception Baseline (Rössler et al., 2019) Complete
- Script: `scripts/train_xception_baseline.py`
- Training Data: Kaggle 100k split (Adam lr=0.0002, CosineAnnealing, 2 Epochs, Seed 42)
- Test Results: Accuracy: **98.34%**, ROC-AUC: **0.9998**, F1: **0.9837**, Brier: **0.0123**
- Checkpoint: `checkpoints/xception_baseline_seed42/best_model.pth`
- Metrics: `results/benchmark_eval_v2/xception_baseline_seed42.json`

## [2026-08-16] TRAIN — MesoNet-4 Baseline Reproduction Complete
- Script: `scripts/train_mesonet_baseline.py`
- Result: 15 Epochs trained on Kaggle 100k split (Seed 42)
- Test Accuracy: 84.16%, ROC-AUC: 0.9204, Brier Score: 0.1147
- Saved: `results/benchmark_eval_v2/mesonet_baseline_seed42.json`

## [2026-08-16] AUDIT — Pre-Test Audit: 7 P0 Blockers Identified
- Source: `audit_phase2_pretest/README.md` (read-only code/config/data audit)
- P0-1: External-folder evaluation double-counts images on Windows (case-insensitive)
- P0-2: GradCAM passes zeros for DCT branch (wrong inference path)
- P0-3: GradCAM narrative contradicts saved JSON (98.2% claimed vs 0.4677 actual)
- P0-4: Calibration reported as good but metrics are WORSE post-calibration
- P0-5: V2 evaluation uses V1 calibration file (wrong model)
- P0-6: Report mixes 0.5-threshold accuracy with 0.1341-threshold confusion matrix
- P0-7: FF++/CelebDF/DFDC loaders have invalid ordered splits (not stratified)
- Decision: ALL P0 items must be fixed before any new experiments
- Checkpoint manifest created: CHECKPOINT_MANIFEST.yaml
- Phase-2 checkpoint backed up to D:\deepfake_checkpoints_backup\

## [2026-08-16] PLAN — v3_multisource_finetune Roadmap
- Phase 0: Fix 7 audit blockers + unified preprocessing + manifests
- Phase 1: Zero-shot FF++ baseline (no fine-tuning, per-manipulation AUC)
- Phase 2: v3_multisource_finetune from Phase-2 weights (Kaggle 50% + FF++ 50%)
- Phase 3: Final eval on Kaggle + FF++ + Celeb-DF (unseen)
- Phase 4: Fair MesoNet (≥3 seeds) + GradCAM 30/class + honest report
- Architecture: DeepfakeDetector (V1 trained). NOT detector_v2.py.
- Starting weights: checkpoints/v2_clip_finetune/best_model.pth (SHA256: 1FC7332B...)
- New checkpoint: checkpoints/v3_multisource_finetune/

## [2026-08-16] AUDIT — Diagnostic Full Audit Completed
- Script: scripts/diagnostic_full_audit.py
- Key finding: Model is a StyleGAN-specific artifact detector, NOT a general deepfake detector
- Compression does NOT cause _new_dataset failure (AUC stays 1.0 at 10KB)
- _new_dataset labels are unreliable (no provenance, visual spot-check inconclusive)
- test_data (TPDNE) is same generator family as training — NOT cross-dataset
- Calibration paradox explained: T=4.39 softens already-correct predictions, increasing error

## [2026-07-03] CODE — Phase 1 GAN Finetune Throughput Optimization
- Added optional CLIP alignment loss flag in `models/clip_alignment.py` and `models/detector.py`
- `scripts/train_gan_finetune.py` now skips the unused frozen CLIP visual forward during GAN finetune
- Enabled CUDA benchmark/TF32, non-blocking transfers, input channels-last tensors, and DataLoader workers
- Kept fused AdamW opt-in only via `--fused_optimizer` after local AMP crash in `scaler.step(optimizer)`
- Defaults: `batch_size=16`, `accumulation=4`, `num_workers=2`; use `--num_workers 0` if Windows worker/pagefile errors return

## [2026-07-03] FIX — Phase 1 GAN Finetune Save + Threshold Consistency
- Fixed `train_gan_finetune.py` validation AUC lookup: `roc_auc` is now used instead of missing `auc`
- Initialized `best_auc=-1.0` so the first valid epoch can save `best_model.pth`
- Added calibrated threshold `0.1341` to `checkpoints/kaggle_realfake/calibration.json`
- Set `configs/config.yaml` fallback threshold to `0.1341`
- Updated handover/memory docs to use `.venv\Scripts\python.exe` and reflect current Phase 1 checkpoint state

## [2026-06-25] TRAIN — Phase 1 GAN Finetune STARTED on RTX 4050
- Command: `.venv\Scripts\python.exe scripts/train_gan_finetune.py --epochs 1 --batch_size 4`
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU | 6.4 GB VRAM
- Train set: 110,012 images (100k kaggle + 10k stylegan)
- Smoke test (1 epoch) running — full 15-epoch run next

## [2026-06-25] FIX — Wrong Python Binary (Critical)
- System `python` = `2.10.0+cpu` — NO GPU support
- `.venv\Scripts\python.exe` = `torch 2.6.0+cu124` + RTX 4050 ✅
- `venv\Scripts\python.exe`  = `torch 2.11.0+cu128` + RTX 4050 ✅
- Fix: always use `.venv\Scripts\python.exe` for training

## [2026-06-25] FIX — train_gan_finetune.py: 9 bugs fixed
- `get_device(use_cuda=)` → `prefer_cuda=`
- `load_checkpoint(strict=)` → direct `torch.load` + `load_state_dict(strict=False)`
- `get_grad_scaler(use_amp=)` → `enabled=`
- `autocast(device_type="cuda")` → `device_type=device.type` (CPU-safe)
- `compute_metrics(list, list)` → numpy arrays + correct positional args
- `None` in batch → custom `collate_fn` to skip None cache fields
- `num_workers>0` on Windows → `nw=0` (avoids spawn crash)
- `pin_memory=True` on CPU → `pin=device.type=="cuda"`
- `dct_input=` kwarg (×2) → `dct=` (actual model forward signature)

## [2026-06-25] DATA — 10k GAN images ready on D: drive
- Source: kaggle_realfake test/fake split (10k, never seen by V1 model)
- Copied with stylegan_ prefix → auto-labelled generator_type=0
- Path: D:\deepfake_data\kaggle_realfake\real_vs_fake\real-vs-fake\train\fake\
- Zero overlap with training set confirmed (filename check)

## [2026-06-25] CODE — AGENT_HANDOVER.md + PROJECT_MISTAKES.md created
- Full handover doc for any future agent/IDE
- Covers: arch changes, freeze/unfreeze log, method changes, mistakes log
- Rule: append every bug+fix to AGENT_HANDOVER.md



## [2026-06-25] DATA — StyleGAN Download Started (Phase 1)
- Running: download_stylegan_train.py --count 5000 → D:\deepfake_data\...
- Junction: data/stylegan_images → D: (loader sees it transparently)
- ETA: ~92 min | 12 already on C:

## [2026-06-25] CODE — train_gan_finetune.py Created (Phase 1)
- Loads V1 best_model.pth (strict=False)
- Freezes: temporal_model, physiology_encoder, clip_visual backbone
- Trains: spatial_encoder, frequency_encoder, fusion, detection_head + GeneratorHead
- Loss: binary BCE + generator attribution CE (weight=0.4), ignore_index=-1 for reals
- LR: 1e-5, epochs: 15, batch: 16, accumulation: 4
- Saves: checkpoints/v1_gan_finetune/best_model.pth

## [2026-06-25] CODE — datasets/kaggle_realfake.py Patched (Phase 1)
- Added GENERATOR_TYPES map: real=-1, stylegan=0, diffusion=1, other_gan=2, unknown=3
- generator_type inferred from filename stem at load time
- __getitem__ now returns generator_type tensor



## [2026-06-21] PLAN — New 4-Phase Implementation Plan Created
- Phase 1: GAN fine-tune (download 5k StyleGAN → train_gan_finetune.py)
- Phase 2: CLIP partial unfreeze (last 2 blocks, LR=5e-6)
- Phase 3: V2 stages 1,2,4,5 (stage 3 deferred — no video data)
- Phase 4: Recalibrate + RAG DB + evaluate
- Artifact: `implementation_plan.md` in brain artifacts

## [2026-06-21] CODE — `scripts/pretrain_mae.py` Created
- MAE self-supervised pretraining for Video Swin-T temporal encoder
- Masks 75% patches, 2-layer transformer decoder
- Handles images (pseudo-clips via jitter) and real videos
- Completes the missing Stage 0 component from original plan

## [2026-06-21] FIX — `context_summary.md` Calibration Contradiction Fixed
- Changed: "Calibration pending: status=done" → "None — calibration is complete"
- Updated Next Actions to reflect current state

## [2026-06-21] FIX — `project_state.json` Checkpoint List Updated
- Removed stale `epoch_5.pth` from `available` list
- Now accurately reflects only `best_model.pth`

## [2026-06-21] CLEANUP — Major Project Cleanup
- Deleted 21 root-level junk/one-off files
- Deleted 6 old V1 stage training scripts (superseded by train_v2.py)
- Deleted 82 old training log files
- Deleted `results_test/` (14 attention visualisation PNGs, 2.3MB)
- Deleted `torchvision-0.21.0+cu124-cp313-cp313-win_amd64.whl` (6MB)
- Deleted `inference/calibration.py` (duplicate shim)
- Deleted all `__pycache__/` dirs
- Kept: `calibration.py` (imported by pipeline), `evaluation/`, `explainability/`

## [2026-06-21] CLEANUP — Deleted Intermediate Epoch Checkpoints
- Deleted: epoch_5.pth, epoch_10.pth, epoch_15.pth, epoch_20.pth
- Reclaimed: ~2.5 GB
- Remaining: best_model.pth (630MB) + calibration.json

## [2026-06-21] TRAIN — V1 Training Complete (kaggle_realfake)
- Accuracy: 99.39%, AUC: 0.9998, best_epoch: 12
- Temperature calibrated: T=4.396, threshold=0.1341
- Checkpoint: checkpoints/kaggle_realfake/best_model.pth

## [2026-06-21] DATA — StyleGAN Download Incomplete
- Only 12/10000 images downloaded before interruption (2026-06-18 04:39)
- Script: scripts/download_stylegan_train.py (resumable)
- Target: data/kaggle_realfake/.../train/fake/stylegan_*.jpg

## [2026-06-21] CODE — V2 Architecture Fully Coded (16 new files)
- spectral_branches.py, identity_encoder.py, generator_head.py
- rag_retrieval.py, knowledge_base.py, uncertainty.py
- fusion_transformer.py, detection_head_v2.py, detector_v2.py
- model_config_v2.yaml, losses_v2.py, trainer_v2.py, train_v2.py
- build_rag_db.py, pretrain_dino.py, adversarial_transforms.py
- Status: coded, import smoke tests pass, no trained checkpoint
