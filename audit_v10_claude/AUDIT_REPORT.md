# DeepTrace V10 — Independent Leakage & Accuracy Audit

Date: 2026-10-10 · Checkpoint: `checkpoints/v10_robust/best_model.pth` (epoch 7) · No project code was modified.
All numbers below come from the scripts in this folder; raw outputs are in `results/`.

## 1. Verdict

| Reported claim | Verdict |
|---|---|
| Kaggle in-domain 99.69% / AUC 0.9999 | **Procedurally contaminated, but not inflated in practice.** All 10,000 Kaggle *test* fakes were trained on by V1, and V10 inherits those weights. Even so, V10 scores the same on never-trained Kaggle *valid* images (AUC 0.9999, 99.65% @0.5), with no test-vs-valid gap (p=0.15). It is a real in-domain StyleGAN-vs-FFHQ number. It says nothing about general deepfake detection. |
| FF++ "100% actor-disjoint" macro AUC 0.607 | **Numbers reproduce exactly; "actor-disjoint" is false at the weight level.** V10 ← V9 ← V7 ← V5 ← V4 ← V3, and V3 trained on frames from **all 400** "test" actors (V4/V5: 235 of them). 49% of each cohort's benchmark frames were also V10's validation set. The true signal is weak: frame-AUC 95% CIs (video-clustered) are about ±0.06, e.g. FaceSwap 0.52–0.64. |
| "Calibrated accuracy" 57.9% FF++ | Threshold is tuned on **test labels**. Honest acc @0.5: 54.7–59.3%. The handover's "≈50.05% at τ=0.5" is **wrong**. |
| Compression hardening ">90% AUC retained" | **Misleading.** The 0.907 is a pooled Kaggle+FF++ AUC. Kaggle alone is 0.9999 and FF++ alone is **0.644**. The pooled value is mostly the model telling the two datasets apart. On FF++, JPEG q30 gives 0.637 against 0.644 pristine: there was little to retain. |
| Grokking | **No evidence.** V10 ran 10 epochs (not 20; `last.pth` = epoch 10). Val AUC is flat at 0.846–0.860 from epoch 1. The pattern is ordinary overfitting / memorisation: train-actor AUC 0.85 vs test-actor 0.62. |

**Main real problem (new):** V10 has learned to recognise *scenes it saw labelled "real"*, not manipulation artefacts. Fakes built on a training actor's background video, with a test actor's face swapped in, get P(fake)=0.10 and AUC **0.10** against real frames. It is confidently wrong. See §3.2.

## 2. Handover (Gemini) claims checked against the code

| ID | Claim | Status | Evidence |
|---|---|---|---|
| L2 | Grad-CAM called without `dct` | TRUE | `inference/pipeline.py:328,342` |
| L3 | DFD domain shortcut | TRUE | DFD fakes vs FF++ test reals: AUC 0.946 (studio 1920×1080 vs YouTube), while real FF++ cohorts are ~0.6 |
| L4 | CelebDF/DFDC sequential slice | TRUE (code) | `datasets/celebdf.py:61`, `datasets/dfdc.py:67`; not on the V10 path |
| L5 | V3 frame-level random split | TRUE, **and V3 is an ancestor of V10** | `scripts/train_v3_e2e.py:147-185` |
| L6 | V7/V9 val ⊂ train | TRUE | 60.3% (V7) / 50.0% (V9) of val frames are training frames, i.e. 100% of the FF++ part |
| L7 | V10 val = test actors | TRUE | 3,000 benchmark frames were the val set used to pick the epoch |
| L8 | SBI applied to fakes, not reals | TRUE | `scripts/train_v10_robust.py:157` (also V9 `:139`) |
| L9 | Threshold tuned on test | TRUE, and it is the script actually used: `scripts/evaluate_v10_full_battery.py:117-122`. Effect is small: Kaggle +0.05 pt, FF++ +1 to +3.7 pt |
| — | last.pth = epoch 20 | FALSE | checkpoint metadata: epoch 10; best = epoch 7 |
| — | Noise σ = 0.005–0.025 | FALSE | code uses σ = 0.05–0.25 (×255), `evaluate_robustness.py:197` |
| — | `np.load` pickle RCE | FALSE | numpy 2.4.3 defaults to `allow_pickle=False` |
| — | Flask: no `MAX_CONTENT_LENGTH`, extension-only check, binds 0.0.0.0 | TRUE (static) | `ui/flask_app.py:76-82,177`. Path traversal is **not** possible: the save name is `upload{ext}`. |
| — | Gradio both bind 0.0.0.0 | PARTLY | `app.py:198` yes; `ui/app.py` uses the default localhost |
| — | retain_graph memory leak | MOSTLY WRONG | the graph is freed on return. The actual bug is hooks not being removed if an exception fires (no `try/finally`, `app.py:77-86`) |
| — | `.venv` torch 2.6 | stale | only `venv/` exists, torch 2.11+cu128 |

Live endpoint fuzzing (500 MB upload, concurrency VRAM test) was **not run**. The security items above are static findings.

## 3. New findings (not in the handover)

### 3.1 Kaggle test fakes were copied into training (T2)
`D:\deepfake_data\kaggle_realfake\real_vs_fake\real-vs-fake\train\fake` holds **10,000 byte-identical (MD5) copies of all 10,000 Kaggle test fakes**, renamed `stylegan_*`. `train_gan_finetune.py` passes it as `extra_fake_dirs` ("Train set: 110,012 images", PROJECT_CHANGES.md 2026-06-25). The "zero overlap" check compared **filenames**, and the renaming defeated it. Lineage: V1-GAN → V2 → V3 → V4 → V5 → V7 → V9 → V10.
Measured effect on V10: none detectable. Test-fake vs valid-fake logits give AUC 0.49 (p=0.15). The test set is still invalid by construction.

### 3.2 V10 recognises scenes, not manipulations (T5)
FF++ fake ids are `<target>_<source>`: the target supplies the scene, the source supplies the face. "Mixed" pairs were discarded from training:

| Mixed-pair fake | mean P(fake) | AUC vs real test frames |
|---|---|---|
| scene = **train** actor, face = test actor | **0.10** | **0.10** |
| scene = test actor, face = **train** actor | 0.52 | 0.57 |
| (normal test fakes / test reals) | 0.56 / 0.48 | — |

The model saw the training actors' scenes labelled "real". When those scenes reappear with a swapped face, it says "real" with high confidence. Supporting evidence:
- Train-actor AUC 0.85 vs test-actor AUC 0.62 for every cohort (e.g. Deepfakes 0.90 vs 0.68).
- V9→V10 raised train-actor AUC from 0.74 to 0.85 but test-actor AUC only from 0.58 to 0.62.

### 3.3 FF++ input is not face-cropped
FF++ frames are full 640×480 to 1920×1080 video frames (`results/sample_ffpp_real.jpg`), squashed to 160×160. Kaggle images are tight 256×256 face crops. This causes three problems:
- The face is a small fraction of the input, which destroys blending artefacts.
- The source dataset is trivially identifiable: image size alone separates Kaggle from FF++ with AUC 1.0, and the model separates Kaggle reals from FF++ reals with AUC 0.97.
- V10 calls **59% of real FF++ frames "fake"**, against 0.5% of Kaggle reals.

Face2Face and NeuralTextures frames are also saved at different resolutions from their source reals (576/832 px vs 656/854 px wide). A metadata-only classifier reaches AUC 0.73 on those cohorts, higher than V10's 0.58.

### 3.4 The ancestor contamination (T1) is real but has no measurable frame-level effect (T5)
- 4,817 of 8,520 unique benchmark frames, and all 284 benchmark videos, were in an ancestor's training set.
- A within-video paired test of seen vs unseen frames shows no memorisation (fakes p=0.44, reals p=0.28).
- An earlier between-video comparison (p=4e-11) was confounded by which videos were sampled. Disregard it.

### 3.5 Other
- V9 "clean baseline" = **epoch 1**. It was selected on a leaked val (AUC 0.954), so it is essentially V7 plus one epoch.
- V10 trains on 16,000 reals vs 27,100 fakes, with fakes weighted 2.0–3.5×. That pushes FF++ scores toward "fake" (optimal τ 0.62–0.67).
- Horizontal flip changes **14.8%** of FF++ decisions (Kaggle 0.4%), so FF++ decisions sit near noise.
- OOD `_new_dataset` Test (N=10,905): AUC **0.31** (below chance, inverted); acc 41.6%; ECE 0.41.
- `scripts/verify_no_leakage.py` prints "ALL CHECKS PASSED" right after its own CHECK 7 flags V4/V5 as contaminated. It checks only the split definition, not weights, val overlap or file hashes.
- **V10's manipulation-type loss was always zero.** `V10RobustDetector` reads `manipulation_type_logits`, but `DetectionHead` returns `manipulation_logits` (`models/detection_head.py:95`). The 0.15·L_manip term was never computed.
- **CLIP is not in the inference path.** With `compute_alignment_loss=False` (all eval scripts), `CLIPAlignment` skips the ViT, and the fused "CLIP token" is `spatial_projection(EfficientNet features)`. CLIP only supplies a training-time alignment target.
- The near-duplicate (dHash) test was inconclusive: the 64-bit hash collides on unrelated aligned faces. Only the MD5 result is conclusive.

## 4. Numbers that are defensible today

| Metric | Value |
|---|---|
| Kaggle valid (never trained), AUC / acc@0.5 | 0.9999 / 99.65% (StyleGAN2 vs FFHQ only) |
| FF++ test actors, frame AUC by cohort, all 4,000 reals | DF 0.68, FS 0.59, FSh 0.63, F2F 0.59, NT 0.59 (±~0.06, 95% CI) |
| FF++ video-level AUC (benchmark set) | 0.58–0.68; all above chance (video-permutation p ≤ 0.01) |
| FF++ acc@0.5 (balanced cohorts) | 54.7–59.3% |
| OOD `_new_dataset` | AUC 0.31 |

Caveat for all FF++ rows: the weight lineage saw these actors.

## 5. Recommended fixes (priority order)
1. **Quarantine** `D:\deepfake_data\...\train\fake`. Add an MD5-based train/test overlap check to every builder.
2. **Retrain from clean pretrained backbones** (ImageNet EfficientNet + OpenCLIP), not from V9/V7. Any warm start inherits V1's test fakes and V3–V5's test actors.
3. **Face-crop FF++** (RetinaFace/MTCNN, fixed margin, square) and resize the same way as Kaggle. This is the single largest accuracy lever.
4. **Three-way actor split** (e.g. 540 train / 60 val / 400 test). Do model selection and threshold freezing on val actors only; report acc@frozen-τ.
5. Report video-level AUC with video-cluster bootstrap CIs, per domain. Never pool Kaggle+FF++.
6. Fix SBI: blend on **reals** and relabel them as fake, as V7 did. Balance classes or weights.
7. Extend `verify_no_leakage.py` to cover checkpoint lineage, val∩test, and file hashes.

## 6. Fix progress (V11) — paused 2026-10-11 00:08 after epoch 5/15

| Fix | Status |
|---|---|
| 1. Quarantine duplicate test fakes | ✅ moved to `D:\deepfake_data\_QUARANTINE_kaggle_test_fake_copies\` (with README) |
| 2. Clean init, no V1–V10 warm start | ✅ `scripts/train_v11_clean.py`; `assert_clean_lineage()` refuses dirty checkpoints |
| 3. Face-crop FF++ | ✅ `scripts/crop_ffpp_faces.py` → 59,980/60,000 crops (max drop rate 0.15%, Deepfakes) |
| 4. Three-way actor split | ✅ `get_actor_disjoint_split_3way`: 540/60/400; test actors identical to V10; no video discarded |
| 5. Honest reporting | ✅ `scripts/evaluate_v11_honest.py` (frozen val thresholds, per domain, video AUC + CI, scene probe); **not yet run on test** |
| 6. SBI on reals, balanced sampling | ✅ in V11 |
| 7. Guards in verifier | ✅ `utils/leakage_guard.py` (MD5 + actors); `verify_no_leakage.py` CHECK 8 (lineage) |
| Bonus | ✅ fixed the dead manipulation loss (V10 key bug) |

Full-run guard: 40,798 training files, 0 MD5 duplicates vs val / Kaggle test / FF++ test, 0 shared actors.

V11 validation (60 unseen actors, face crops). Noisy: 34 fake videos per cohort.

| epoch | FF++ val macro AUC | Kaggle val AUC | FF++ val bal. acc @ frozen τ |
|---|---|---|---|
| 1 | 0.714 | 0.751 | 0.672 |
| 2 | 0.766 | 0.868 | 0.714 |
| 3 | 0.802 | 0.914 | 0.743 |
| 4 | 0.810 | 0.932 | 0.750 |
| 5 | **0.812** | **0.942** | 0.750 |

Resume: `venv\Scripts\python.exe scripts\train_v11_clean.py --resume checkpoints/v11_clean/last.pth`
Then (once): `venv\Scripts\python.exe scripts\evaluate_v11_honest.py --checkpoint checkpoints/v11_clean/best_model.pth`

## 7. Re-running
```
venv\Scripts\python.exe audit_v10_claude\t1_split_lineage_audit.py      # ~1 min, CPU
venv\Scripts\python.exe audit_v10_claude\t2_duplicates_and_shortcuts.py # ~10 min, CPU
venv\Scripts\python.exe audit_v10_claude\t3_infer_cache.py --tag v10 --robustness   # ~15 min, GPU
venv\Scripts\python.exe audit_v10_claude\t4_analysis.py v10
venv\Scripts\python.exe audit_v10_claude\t5_memorisation_probes.py
```
