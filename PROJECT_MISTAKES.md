# PROJECT_MISTAKES.md
> Every bug, wrong assumption, failed attempt — logged with fix.
> Use this for project writeup / explanation later.
> Format: [DATE] MISTAKE → FIX

---

## [2026-10-11] Kaggle preprocessing mismatch between training and real use
- **Mistake:** V11 trained on raw Kaggle 256x256 images, but real uploads are MTCNN-cropped first; Kaggle test AUC fell 0.9645 → 0.905 through that path
- **Fix:** V12 crops every source with the same MTCNN pipeline (`scripts/crop_kaggle_faces.py`); V12 Kaggle test via crops: AUC 0.989

## [2026-10-11] Windows MemoryError / hangs in DataLoader workers
- **Mistake:** Several GPU jobs at once, each with workers passing full-res frames (or 4 extra val workers) exhausted commit memory → worker died, main process hung (lost V11 epoch 6, half of dense extraction, Kaggle crop run)
- **Fix:** One data/GPU job at a time; val runs in-process (`--val_workers 0`); 2 workers + `prefetch_factor=1`; extraction/crop scripts resumable; `pin_memory` off (CUDA "resource already mapped")

## [2026-10-10] CRITICAL — Kaggle test fakes used as training data
- **Mistake:** `test/fake` (10k) copied as `stylegan_*` into an extra GAN folder used by V1 training; overlap checked by FILENAME, so renaming hid it. All later models inherited it
- **Fix:** Folder quarantined (`D:\deepfake_data\_QUARANTINE_kaggle_test_fake_copies`); `utils/leakage_guard.py` checks MD5 content before every training/eval run

## [2026-10-10] CRITICAL — "Actor-disjoint" V9/V10 warm-started from contaminated V3/V4/V5
- **Mistake:** Split definition was clean, but weights descended from runs that trained on all 400 FF++ test actors
- **Fix:** V11/V12 trained from ImageNet/CLIP; `assert_clean_lineage()` refuses warm starts; `verify_no_leakage.py` CHECK 8

## [2026-10-10] V10 validation = test actors; thresholds tuned on test labels; pooled robustness metric
- **Fix:** 3-way actor split 540/60/400; thresholds/temperature/TTA chosen on val only (`scripts/val_output_study.py`); per-domain reporting with recall/specificity (`scripts/evaluate_v11_honest.py`)

## [2026-10-10] FF++ fed as full video frames; SBI on fakes; dead manipulation loss; CLIP not in inference path
- **Mistake:** No face crop (face ≈ few % of 160px input); SBI blended fakes instead of reals; `manipulation_type_logits` key never existed → loss always 0; CLIP ViT skipped at inference
- **Fix:** MTCNN face crops (`scripts/crop_ffpp_faces.py`); SBI on reals → label fake; hook captures `manipulation_logits`; CLIP kept frozen (documented)

## [2026-07-03] GAN fine-tune AUC key mismatch
- **Mistake:** `train_gan_finetune.py` used `metrics.get("auc", 0.0)`, but `compute_metrics()` returns `roc_auc`
- **Fix:** Read `roc_auc` with `auc` fallback and initialize `best_auc=-1.0` so the first valid epoch can save `best_model.pth`

## [2026-07-03] Calibration threshold missing from sidecar
- **Mistake:** `checkpoints/kaggle_realfake/calibration.json` only stored temperature, so inference used the config fallback threshold
- **Fix:** Added `threshold: 0.1341` to the sidecar and changed the config fallback threshold to `0.1341`

## [2026-06-25] StyleGAN download — wrong URL
- **Mistake:** Used `thispersondoesnotexist.com` directly → returns 4KB HTML not image
- **Fix:** Found real API at `this-person-does-not-exist.com/new` returning JSON with image path

## [2026-06-25] StyleGAN download — tiny image pool
- **Mistake:** Site only has ~500 unique images, 83% of requests were duplicates → would take 1000+ min for 10k
- **Fix:** Switched to copying 10k images from the dataset's own `test/fake` split (held-out, never trained on, zero overlap)

## [2026-06-25] HuggingFace datasets — deprecated param
- **Mistake:** Used `trust_remote_code=True` → error: not supported in newer `datasets` version
- **Fix:** Removed param — but all target HF datasets also didn't exist on Hub

## [2026-06-25] Unicode arrow in print → Windows cp1252 crash
- **Mistake:** Used `→` character in f-string → `UnicodeEncodeError` on Windows terminal
- **Fix:** Replaced with ASCII `->` 

## [2026-06-25] Old stage scripts left after cleanup
- **Mistake:** `train_stage1/2/3/4.py` + `train_kaggle.py` still existed alongside `train_v2.py` → confusion which to use
- **Fix:** Deleted all old stage scripts, `train_v2.py` is single entry point

## [2026-06-25] `inference/calibration.py` duplicate shim
- **Mistake:** `inference/calibration.py` just did `from calibration import *` — redundant, confusing
- **Fix:** Deleted it. `inference/pipeline.py` imports from root `calibration.py` directly

## [2026-06-25] `calibration.py` status contradiction in context_summary.md
- **Mistake:** `context_summary.md` said "Calibration: pending" but `project_state.json` said "done"
- **Fix:** Updated both files to reflect calibration is complete (T=4.396, threshold=0.1341)

## [2026-06-25] Epoch checkpoint bloat
- **Mistake:** epoch_5/10/15/20.pth kept on disk after training — wasted 2.5GB
- **Fix:** Deleted all intermediate checkpoints, kept only `best_model.pth`

## [2026-06-25] `project_state.json` listed deleted checkpoints
- **Mistake:** `available_checkpoints` still listed epoch_5.pth after deletion
- **Fix:** Updated list to only contain `best_model.pth`
