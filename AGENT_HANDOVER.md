# AGENT_HANDOVER.md
> Complete context for any AI agent / IDE picking up this project.
> Read this + PROJECT_MEMORY.yaml before doing anything.
> Updated every session. Newest entries at top of each section.

---

## HOW TO START (any new agent)
1. Read `PROJECT_MEMORY.yaml` — full state, architecture, roadmap
2. Read `AGENT_HANDOVER.md` (this file) — decisions, mistakes, rules
3. Do NOT scan the full repo — waste of tokens
4. Check `project.status` in MEMORY — tells you exactly where we are
5. Continue from `roadmap` section — find first `pending` item

> [!IMPORTANT]
> **ALWAYS use `.venv\Scripts\python.exe` for training — NOT system `python`**
> System python = `2.10.0+cpu` (no GPU). `.venv` = `torch 2.6.0+cu124` + RTX 4050.
> `venv` also works: `torch 2.11.0+cu128` + RTX 4050.
> Run all scripts as: `.venv\Scripts\python.exe scripts/train_gan_finetune.py ...`


---

## AGENT RULES
```
on_mistake    → append to AGENT_HANDOVER.md § Mistakes & Fixes
on_arch_change→ append to AGENT_HANDOVER.md § Architecture Changes
on_freeze     → append to AGENT_HANDOVER.md § Freeze/Unfreeze Log
on_method     → append to AGENT_HANDOVER.md § Method Changes
on_session_end→ update PROJECT_MEMORY.yaml status field
language      → short, clear, low-token (caveman-style ok)
```

---

## ⚠️ AUDIT 2026-10-10 — READ FIRST
- Full report: `audit_v10_claude/AUDIT_REPORT.md`. V10 numbers below reproduce exactly, but:
- Kaggle test fakes (all 10k) were in V1 training via `D:\deepfake_data\...\train\fake` (MD5 copies). Folder moved to `D:\deepfake_data\_QUARANTINE_kaggle_test_fake_copies\`.
- V10 lineage (V3/V4/V5) trained on FF++ "test" actors → "100% actor-disjoint" is false for V9/V10 weights.
- V10 val set = 3,000 test-actor frames; "calibrated acc" tuned on test labels; robustness 0.907 is pooled Kaggle+FF++ (FF++ alone 0.644).
- FF++ frames were never face-cropped. V10 memorises training scenes (mixed-pair probe AUC 0.10).
- **Do not warm-start from any V1–V10 checkpoint.** Use `scripts/train_v11_clean.py` (guards enforce this).

### V11 status — TRAINED + TESTED (2026-10-11). SUPERSEDED by V12 (`checkpoints/v12_dense/best_model.pth`, see below)
- Best = epoch 9 (val-selected), early stop epoch 13. clean_lineage=True. Frozen thresholds: FF++ 0.62, Kaggle 0.78
- TEST (run once, `results/benchmark_eval_v11/v11_honest_evaluation.json`): FF++ 400 unseen actors macro frame AUC **0.802** (video 0.838; V10 0.607) · DF 0.907 · FSh 0.828 · FS 0.808 · F2F 0.765 · NT 0.700 · bal acc 72.0% @ frozen τ
- Kaggle test AUC 0.9645 / bal acc 90.0% (V10's 0.9999 was contaminated). Scene probe 0.63 (V10 0.10). FF++ JPEG q30 0.778. OOD _new_dataset 0.45 (labels unverified)
- Weak: NT/F2F cohorts, residual scene memorisation, Kaggle below V10. Do NOT re-tune on test; any new experiment → select on val, then test once
- Logs: `logs/v11_clean/` (train.log epochs 1-5, train_resume.log 6-13, eval_test.log)

### Model/output work 2026-10-11 (user: do NOT touch app.py / ui/ — app files restored to git HEAD)
- `scripts/val_output_study.py` (VAL only): hflip TTA helps all val cohorts → ON; temperature T=1.525; thresholds FF++ 0.66 / Kaggle 0.60 → `checkpoints/v11_clean/output_calibration.json`
- V11 + those settings on TEST (`results/benchmark_eval_v11_clean/`): FF++ macro AUC 0.802→**0.807**, Brier 0.204→0.184, ECE 0.078; Kaggle AUC 0.9645→0.968
- `inference/v11_predictor.py` (model-side inference: MTCNN crop → V11), `scripts/fit_v11_app_threshold.py` → `checkpoints/v11_clean/app_threshold.json`
- FINDING: through MTCNN crop path FF++ test AUC = 0.8016 (identical) but Kaggle test 0.905 (vs 0.9645 raw) — Kaggle was trained un-cropped. Fix in V12: crop ALL sources with the same MTCNN pipeline (`scripts/crop_kaggle_faces.py`)
- **V12 DONE — NEW BEST / ACTIVE MODEL: `checkpoints/v12_dense/best_model.pth`** (epoch 13 of 18, early stop; clean_lineage=True)
  - Changes vs V11: 32 frames/video for 540 train actors (66,555 crops, `manifests/ffpp_c23_faces256_dense_train_manifest.csv`) · Kaggle MTCNN-cropped like FF++ (25k/class train; `manifests/kaggle_faces256_manifest.csv`) · input 224 · same val images as V11 (`--val_from`)
  - Val-chosen output settings (`checkpoints/v12_dense/output_calibration.json`): hflip TTA ON, T=1.384, thresholds FF++ 0.63 / Kaggle 0.74
  - TEST (once, `results/benchmark_eval_v12_dense/`, robustness not yet run): FF++ unseen-actor macro AUC **0.873** (video 0.905; V11 0.807, V10 0.607) · DF 0.951 · FS 0.883 · FSh 0.887 · F2F 0.844 · NT 0.799 · bal acc 78.5% · recall 68.9% / specificity 88.0%
  - Kaggle test (MTCNN crops = app path) AUC **0.989**, recall 94.5%, specificity 95.3% (V11 via same path 0.905)
  - Scene probe 0.78 (V11 0.63, V10 0.10). OOD `_new_dataset` still fails (AUC 0.47, recall 2.7%); `test_data` AUC 0.92
  - REPORT 2026-10-12: `reports/v12/V12_REPORT.md` (+ `reports/v12/extras.json`, figures). Cross-dataset Kaggle-only → FF++: DeepTrace 0.590 (1 epoch), MesoNet 0.500 (3 epochs) — both stopped early for time; MesoNet same-split Kaggle AUC 0.806 vs V12 0.989
  - NEXT: robustness run (`evaluate_v11_honest.py --checkpoint checkpoints/v12_dense/best_model.pth` without --skip_robustness); NT/F2F still weakest; Celeb-DF cross-dataset test
- Windows RAM: DataLoaders passing full-res frames hit MemoryError when 2+ GPU jobs run → run data jobs ONE at a time (2 workers, prefetch 1); extract/crop scripts are resumable

## CURRENT STATUS (2026-08-21) — superseded by audit above
- **Active Production Model**: DeepTrace V10 Robust Multi-Source (`checkpoints/v10_robust/best_model.pth`).
- **Core Problem Solved**: Persistent JPEG augmentation (`p=0.7`) + balanced multi-source training successfully broke StyleGAN shortcut memorization and cured the web compression collapse.
- **In-Domain Kaggle Retention (N=20,000)**: **99.69% Calibrated Accuracy · 0.9999 ROC-AUC · 0.0038 Brier Score** (9,966 TN / 9,971 TP).
- **FF++ 100% Actor-Disjoint (5 Cohorts, N=14,200)**: **0.6070 Macro AUC** (Deepfakes: **0.6736 AUC**, FaceShifter: **0.6200 AUC**, FaceSwap: **0.5836 AUC**, NeuralTextures: **0.5801 AUC**, Face2Face: **0.5775 AUC**).
- **Robustness Battery (14 Distortion Conditions)**:
  - Pristine: `0.9074` AUC
  - JPEG Q=90: `0.9067` AUC (99.9% retention)
  - JPEG Q=70: `0.9070` AUC (100.0% retention)
  - JPEG Q=50: `0.9051` AUC (99.7% retention)
  - JPEG Q=30: `0.9010` AUC (99.3% retention)
  - Gaussian Blur ($k=7$): `0.9056` AUC (99.8% retention)
  - Downscale (scale=0.5): `0.9075` AUC (100.0% retention)
- **All Checkpoints Preserved**: v1, v2, v3, v4, v5, v7, v9, v10_robust.
- **Next Step (Phase 4)**: Ingest Diffusion datasets (CIFAKE / GenImage) via [`scripts/download_diffusion_data.py`](file:///c:/Users/Udit/Desktop/deepfake1/scripts/download_diffusion_data.py) to train V10-D.

---

## ARCHITECTURE CHANGES (newest first)

### [2026-07-03] models/clip_alignment.py + detector.py — optional CLIP alignment loss
- Added `compute_alignment_loss` / `compute_clip_alignment_loss` flag
- Default remains `True` for normal training/inference
- `train_gan_finetune.py` passes `False` because GAN finetune loss does not use `clip_alignment_loss`
- Saves the frozen CLIP ViT forward pass during Phase 1 without changing binary/gen losses

### [2026-06-25] datasets/kaggle_realfake.py — generator_type added
- Added `GENERATOR_TYPES` map: real=-1, stylegan=0, diffusion=1, other_gan=2, unknown=3
- Auto-infers type from filename stem at load time
- Added `extra_fake_dirs` param → scans D: drive GAN folder transparently
- `__getitem__` now returns `generator_type` tensor

### [2026-06-25] scripts/train_gan_finetune.py — CREATED
- Phase 1 GAN finetune script
- Loads V1 `best_model.pth` with `strict=False`
- Adds inline `GeneratorHead` (Linear 512→128→4) on top of `fused_features`
- Loss: `binary_BCE + 0.4 * generator_CE (ignore_index=-1 for reals)`
- Config: LR=1e-5, epochs=15, batch=16, accumulation=4

### [2026-06-25] models/clip_alignment.py — partial_unfreeze() PLANNED
- Not yet implemented — Phase 2
- Will unfreeze last 2 ViT transformer blocks + ln_post + proj
- LR for CLIP blocks = 5e-6 (separate param group)

### [2026-06-25] scripts/pretrain_mae.py — CREATED
- MAE self-supervised pretraining for Video Swin-T
- 75% patch masking, 2-layer transformer decoder
- Status: created, never run

### [2026-06-21] V2 architecture — CODED (16 files, untrained)
- New modules: spectral_combiner (FFT+Wavelet+Noise), identity_encoder (ArcFace-R18),
  rag_retrieval (FAISS), fusion_transformer (4L×8H), detection_head_v2
- Entry: `scripts/train_v2.py` stages 1→5
- Status: import smoke tests pass, no checkpoint

---

## FREEZE / UNFREEZE LOG (newest first)

### [2026-06-25] Phase 1 GAN finetune freeze config
```
FROZEN:   temporal_model, physiology_encoder, clip_visual (backbone only)
TRAINED:  spatial_encoder, frequency_encoder, fusion, detection_head, GeneratorHead (new)
REASON:   GAN fingerprints live in spatial+frequency domain; temporal unused in image mode
VRAM est: ~4.5 GB
```

### [2026-06-25] Phase 2 CLIP partial unfreeze (PLANNED)
```
FROZEN:   temporal_model, physiology_encoder, CLIP blocks 0..9
TRAINED:  spatial+freq encoders, fusion, detection_head, CLIP blocks 10-11 + ln_post + proj
REASON:   Last 2 blocks contain highest-level semantic features; full unfreeze = VRAM OOM
LR:       1e-5 main | 5e-6 CLIP blocks (separate param group)
VRAM est: ~5.5 GB (tight — reduce batch to 8 if OOM)
```

### [2026-06-21] V1 training freeze config (completed)
```
FROZEN:   clip_vit_b32 (88M params), temporal_swin_t, physiology_bilstm
TRAINED:  spatial_encoder, frequency_encoder, clip_projection, fusion, detection_head
RESULT:   99.39% acc, 0.9998 AUC, T=4.396, threshold=0.1341
```

---

## METHOD CHANGES (newest first)

### [2026-10-10] V11 clean protocol (`scripts/train_v11_clean.py`, `scripts/evaluate_v11_honest.py`)
- Init ImageNet EffNet-B0 + OpenAI CLIP; CLIP ViT frozen (never runs at inference anyway — fused "CLIP token" = spatial_projection)
- Unfrozen: EffNet blocks 4-6 + head, spectral combiner, fusion, head, CLIP projections
- Sampler 50/50 real/fake, equal FF++ cohorts, SBI 10%; plain BCE + 0.15 manip CE
- Select on val actors (0.75·FF++ val macro AUC + 0.25·Kaggle val AUC); eval with frozen val thresholds, per-domain, video-level + bootstrap CI

### [2026-07-03] Phase 1 training speed optimization
- Skips unused frozen CLIP visual pass during GAN finetune
- Enables CUDA `cudnn.benchmark`, TF32, and non-blocking H2D transfers
- Defaults changed to `batch_size=16`, `accumulation=4`, `num_workers=2`
- Fused AdamW is opt-in only with `--fused_optimizer`; default disabled because it can crash with AMP on this PyTorch/CUDA build
- If Windows pagefile/spawn errors occur, rerun with `--num_workers 0`

### [2026-07-03] Fused AdamW + AMP crash
- Error: `AssertionError` in `torch.optim.adamw._multi_tensor_adamw` during `scaler.step(optimizer)`
- Cause: unsafe fused/foreach AdamW path with AMP on local PyTorch/CUDA build
- Fix: default optimizer is standard AdamW; fused optimizer is now opt-in via `--fused_optimizer`

### [2026-06-25] GAN data source — 3 attempts before correct solution
1. `thispersondoesnotexist.com` → wrong URL (returns HTML not image)
2. `this-person-does-not-exist.com/new` API → only ~500 unique images, 83% dup rate
3. HuggingFace datasets → all target datasets missing from Hub
4. ✅ **FINAL:** copied `test/fake` split (10k, held-out, zero overlap) with `stylegan_` prefix

### [2026-06-25] Training script deletion — old vs new
- Deleted: `train_stage1/2/3/4.py`, `train_kaggle.py` (V1 era, superseded)
- Current: `train_v2.py` (all stages), `train_gan_finetune.py` (Phase 1), `train_clip_finetune.py` (Phase 2, TBD)

### [2026-06-21] Calibration method
- Post-hoc temperature scaling (not trained into model)
- Calibrated on val set after training: T=4.396576
- Threshold tuned separately: 0.1341 (not 0.5 — see DECISIONS)

---

## MISTAKES & FIXES (newest first)

### [2026-10-11] V11 resume hung at epoch-6 validation — Windows commit-memory MemoryError
- Val DataLoader spawned 4 extra workers (each imports torch+CUDA, reserves GBs of commit) while game running → worker died at import, main process hung; epoch 6 lost
- Fix: `--val_workers` default 0 (validate in-process), train `--num_workers` 3. Also earlier: `pin_memory` off by default (CUDA "resource already mapped")

### [2026-10-10] CRITICAL — Kaggle test fakes used as training data (V1 → all later models)
- `test/fake` copied as `stylegan_*` into extra GAN dir; "zero overlap" checked by FILENAME only
- Fix: folder quarantined; `utils/leakage_guard.py` checks MD5 content; V11 retrains from ImageNet/CLIP

### [2026-10-10] CRITICAL — V9/V10 warm-started from FF++-test-actor-contaminated V3/V4/V5
- Fix: V11 clean init; `assert_clean_lineage()` refuses warm starts without `clean_lineage=True`

### [2026-10-10] V10 val = test actors; eval threshold tuned on test labels
- Fix: `get_actor_disjoint_split_3way` (540/60/400, same test actors); thresholds frozen from val in ckpt

### [2026-10-10] V10 manipulation loss silently zero
- `V10RobustDetector` read `manipulation_type_logits`; head returns `manipulation_logits` → `loss_manip` never computed
- Fix: `V11Detector` captures `manipulation_logits` via hook

### [2026-10-10] FF++ fed as full video frames (no face crop); SBI applied to fakes
- Fix: `scripts/crop_ffpp_faces.py` (MTCNN, identical pipeline all classes) → `manifests/ffpp_c23_faces256_manifest.csv`; V11 SBI on reals → label fake

### [2026-07-03] CRITICAL — train_gan_finetune.py read wrong AUC metric key
- `compute_metrics()` returns `roc_auc`, but `train_gan_finetune.py` read `metrics.get("auc", 0.0)`
- Result: validation AUC stayed 0.0 and `best_model.pth` might never be saved
- Fix: read `roc_auc` with `auc` fallback and initialize `best_auc=-1.0` so first epoch can save

### [2026-07-03] Calibration threshold sidecar missing
- `calibration.json` had only `temperature`, so inference fell back to `configs/config.yaml`
- `configs/config.yaml` had stale threshold 0.162 while docs/project state use 0.1341
- Fix: added `threshold: 0.1341` to `calibration.json` and set config fallback to 0.1341

### [2026-06-25] CRITICAL — Wrong Python binary used for training
- `python` in PATH = `2.10.0+cpu` — NO CUDA, runs on CPU only
- Fix: use `.venv\Scripts\python.exe` (cu124) or `venv\Scripts\python.exe` (cu128)

### [2026-06-25] train_gan_finetune.py — 9 API bugs (util signatures not checked)
- `get_device(use_cuda=)` → `prefer_cuda=`
- `load_checkpoint()` has no `strict` param → replaced with direct `torch.load`
- `get_grad_scaler(use_amp=)` → `enabled=`
- `torch.autocast(device_type="cuda")` → use `device.type` for CPU safety
- `compute_metrics(list, list)` → needs `(np.array, np.array, np.array)`
- DataLoader collate crashes on `None` (cache miss) → custom `collate_fn`
- `num_workers=4` on Windows → causes multiprocessing spawn crash → set to `0`
- `pin_memory=True` on CPU → warning + waste → `pin = device.type=="cuda"`
- `model(dct_input=)` → actual kwarg is `dct=`

### [2026-06-25] Unicode `->` in Python print → cp1252 crash on Windows
- `UnicodeEncodeError` in `download_gan_fast.py`
- Fix: replaced all `→` with ASCII `->`

### [2026-06-25] `trust_remote_code=True` deprecated in newer `datasets`
- Fix: removed param — but datasets didn't exist on Hub anyway

### [2026-06-25] Wrong download URL for StyleGAN site
- `thispersondoesnotexist.com` returns HTML page (4KB), not image
- Fix: found real endpoint at `this-person-does-not-exist.com/new` (JSON API)
- Then found site pool too small (500 uniq / 10k needed) — abandoned

### [2026-06-25] Duplicate checkpoint confusion
- `project_state.json` listed deleted epoch checkpoints as available
- Fix: updated to only list `best_model.pth`

### [2026-06-25] Calibration status contradiction
- `context_summary.md` said pending, `project_state.json` said done
- Fix: both updated to reflect calibration complete

### [2026-06-25] `inference/calibration.py` was dead shim
- Just re-exported from root `calibration.py` — no value
- Fix: deleted, `pipeline.py` imports root directly

---

## NEXT ACTIONS (in order — V10 Implementation Plan)
1. **Phase 1**: Fix `datasets/transforms.py` — JPEG compression `p=0.7` standalone. Deprecate `faceforensics.py._apply_split()` with RuntimeError.
2. **Phase 2**: Create `scripts/train_v10_robust.py` (copy V9, add Kaggle 10k+10k, use updated transforms). Train 20 epochs from V9/V7 warm start.
3. **Phase 3**: Evaluate V10 — Kaggle in-domain, FF++ actor-disjoint 5-cohort, JPEG/blur/noise robustness. Create `scripts/evaluate_robustness.py`.
4. **Phase 4**: Download CIFAKE + GenImage diffusion data. Create `scripts/train_v10d_diffusion.py`. Retrain with diffusion fakes.
5. **Phase 5**: Final evaluation battery + update all governance files (MEMORY, HANDOVER, DECISIONS, CHANGES).

---

## KEY NUMBERS (don't recompute)
| Metric | Value |
|--------|-------|
| V1 accuracy | 99.39% |
| V1 AUC | 0.9998 |
| Calibration T | 4.396576 |
| Optimal threshold | 0.1341 |
| Total params | 51.7M |
| Trained params | 19.8M |
| CLIP params (frozen) | 88M |
| image_size | 160×160 (fixed — VRAM constraint) |
| GPU | RTX 4050 6GB VRAM |
| Max safe VRAM | ~5.8 GB |
