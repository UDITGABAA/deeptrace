# DeepTrace V12 — Evaluation Report (Updated: published-detector comparison + full-pipeline cross-dataset check)

*2026-10-12 · checkpoint `checkpoints/v12_dense/best_model.pth` (epoch 13) · all numbers reproducible with the commands in §11*

DeepTrace V12 is a leak-free retrain of the DeepTrace architecture. It replaces V10, whose reported numbers were affected by train/test leakage (see `audit_v10_claude/AUDIT_REPORT.md`).

- Every decision (checkpoint, decision thresholds, temperature, test-time augmentation) was made on **validation data only**.
- The test sets were evaluated **once**.
- Before every training and evaluation run, an MD5 guard verifies that no test file is byte-identical to a training file. No FF++ actor appears in both training and test.

---

## 1. Summary

| | Value |
|---|---|
| FF++ c23, 400 actors never seen in training: frame AUC / video AUC | **0.873 / 0.905** |
| FF++ balanced accuracy, recall, specificity (frozen val threshold) | **78.5% · 68.9% · 88.0%** |
| Kaggle real-vs-StyleGAN test (MTCNN crops): AUC / balanced accuracy | **0.989 / 94.9%** |
| Cross-dataset: DeepTrace trained on **Kaggle only**, tested on FF++ (no fine-tuning) | macro AUC **0.590** (MesoNet 0.500) — §8 |
| Reproduced baseline (MesoNet-4) on the same Kaggle split | AUC 0.806 vs V12 0.989 — §7 |

---

## 2. Experimental setup

### Data and preprocessing

All sources go through one identical pipeline:
1. MTCNN face detection.
2. A square crop at 1.3× the face box.
3. Resize to 256×256 and JPEG-75 encoding.
4. Model input at 224×224 with ImageNet normalisation.

The same pipeline is used for training, evaluation and inference.

| Split | Kaggle real-vs-fake (FFHQ vs StyleGAN) | FaceForensics++ c23 |
|---|---|---|
| Train | 49,981 face crops (25k per class from Kaggle *train*) | 540 actors: 5,400 real + 15,400 fake frames (10/video) + 66,555 extra frames (32/video) |
| Validation | 1,999 images from Kaggle *valid* | 60 actors: 600 real + 1,700 fake frames |
| Test | 19,991 images from Kaggle *test* (9 had no detectable face) | 400 actors: 4,000 real + 7,100 fake frames (1,420 per manipulation) |

### Actor split

- FF++ actors are split 540 / 60 / 400 (train / val / test). Seed 42 (test) and 4242 (val).
- Fake videos `A_B` and `B_A` share both actors, so whole actor pairs are assigned to a single split.
- No fake video is discarded, and the test actors are identical to those used for V9/V10.

### Leakage guards (`utils/leakage_guard.py`, run automatically)

- **Content:** 0 of 19,991 Kaggle test, 11,100 FF++ test and 25,782 FF++ mixed-pair files are byte-identical to any of the 137,334 training files.
- **Identity:** 0 shared actors between train and val, or between train and test.
- **Lineage:** weights are initialised from ImageNet EfficientNet-B0 and OpenAI CLIP only. Warm starts from any V1–V10 checkpoint are refused, because those descend from runs that trained on test data.

### Model and training

- **Architecture:** V10/V11 architecture, 119.7M parameters, 31.2M trainable.
  - Branches: EfficientNet-B0 spatial, EfficientNet-B0 frequency, and a multi-spectral combiner (FFT phase, wavelet packet, SRM/Gabor).
  - Fusion: cross-attention, followed by a binary head and a 5-class manipulation head.
- **Sampling:** a balanced sampler (50% real / 50% fake, equal mass per FF++ manipulation).
- **Self-blended images:** SBI is applied to real faces and relabelled fake (Shiohara & Yamasaki, 2022).
- **Loss and optimiser:** BCE plus 0.15 × manipulation cross-entropy; AdamW with 1-epoch warm-up and cosine decay.
- **Augmentation:** JPEG, blur, noise, colour jitter and cutout.
- **Schedule:** 30k samples per epoch; early stopping on validation (patience 5). Training stopped at epoch 18; the best epoch is 13.

---

## 3. Table I — Main results on held-out TEST

The threshold is frozen from validation (FF++ 0.63, Kaggle 0.74). Inference uses horizontal-flip TTA and temperature T = 1.384.

| Test set | Frame AUC (95% CI)¹ | Video AUC | Recall (fake caught) | Specificity (real passed) | Precision | Balanced acc. |
|---|---|---|---|---|---|---|
| FF++ Deepfakes | 0.951 (0.934–0.967) | 0.970 | 88.2% | 88.0% | 72.3% | 88.1% |
| FF++ FaceSwap | 0.883 (0.857–0.907) | 0.921 | 70.7% | 88.0% | 67.7% | 79.4% |
| FF++ FaceShifter | 0.887 (0.861–0.909) | 0.926 | 68.9% | 88.0% | 67.1% | 78.5% |
| FF++ Face2Face | 0.844 (0.806–0.876) | 0.877 | 63.8% | 88.0% | 65.4% | 75.9% |
| FF++ NeuralTextures | 0.799 (0.761–0.835) | 0.832 | 53.1% | 88.0% | 61.2% | 70.6% |
| **FF++ all (macro)** | **0.873** (all frames pooled: 0.848–0.895) | **0.905** | **68.9%** | **88.0%** | 91.1% | **78.5%** |
| **Kaggle (StyleGAN vs FFHQ)** | **0.989** | — | **94.5%** | **95.3%** | 95.2% | **94.9%** |
| `test_data` (99 images) | 0.922 | — | 72.0% | 98.0% | 97.3% | 85.0% |
| `_new_dataset` (labels unverified)² | 0.470 | — | 2.7% | 97.6% | 53.8% | 50.2% |

¹ 95% CI from a video-cluster bootstrap (frames of one video are resampled together).
² This folder has no source or documentation for its labels. It is reported for transparency, not as a benchmark.

**Progress on the same FF++ test actors:**

| Model | FF++ macro AUC |
|---|---|
| V10 (leaky) | 0.607 |
| V11 (leak-free) | 0.802 |
| V11 + TTA/calibration | 0.807 |
| **V12** | **0.873** |

---

## 4. Table II — Calibration and decision settings (chosen on VALIDATION only)

| Setting | Value | Chosen how |
|---|---|---|
| Horizontal-flip TTA | on | Improved validation AUC on every FF++ cohort and on Kaggle |
| Temperature T | **1.384** | Minimises validation negative log-likelihood |
| Threshold FF++ / Kaggle | **0.63 / 0.74** | Maximises validation balanced accuracy |

| Brier score / ECE (15 bins) | Before T | After T |
|---|---|---|
| Validation, all (n = 4,299) | 0.0773 / 0.0345 | **0.0759 / 0.0082** |
| Validation, FF++ (n = 2,300) | 0.1062 / 0.0573 | **0.1024 / 0.0235** |
| Validation, Kaggle (n = 1,999) | **0.0441 / 0.0152** | 0.0455 / 0.0277 |
| Test, FF++ (n = 11,100) | — | 0.1570 / 0.0787 |
| Test, Kaggle (n = 19,991) | — | 0.0478 / 0.0207 |

A single temperature fit on the pooled validation set improves FF++ calibration but slightly worsens Kaggle calibration. Per-domain temperatures would fix this; they were not used, because the source domain is unknown at inference time.

---

## 5. Confusion matrices and class-wise breakdown (TEST)

| | Pred. real | Pred. fake |
|---|---|---|
| **Kaggle: true real** | 9,520 (TN) | 472 (FP) |
| **Kaggle: true fake** | 545 (FN) | 9,454 (TP) |
| **FF++: true real** | 3,521 (TN) | 479 (FP) |
| **FF++: true fake** | 2,205 (FN) | 4,895 (TP) |

**FF++ test, per class:** share correctly classified at threshold 0.63.

| Class | Frames (videos) | Correct | Mean P(fake) |
|---|---|---|---|
| Real (passed as real) | 4,000 (400) | 88.0% | 0.228 |
| Deepfakes | 1,420 (142) | 88.2% | 0.886 |
| FaceSwap | 1,420 (142) | 70.7% | 0.746 |
| FaceShifter | 1,420 (142) | 68.9% | 0.736 |
| Face2Face | 1,420 (142) | 63.8% | 0.675 |
| NeuralTextures | 1,420 (142) | 53.1% | 0.586 |

Figures:
- `figures/confusion_kaggle_test.png`
- `figures/confusion_ffpp_test_all.png`
- `figures/ffpp_classwise.png`

The two reenactment methods (Face2Face, NeuralTextures) are the hardest. They change only expression and mouth texture and leave identity and blending boundaries largely intact.

---

## 6. Grad-CAM on test images

Test images were chosen by a **fixed rule**: the first file in sorted order for each category, plus the first missed FF++ fake. They were not hand-picked. Grad-CAM is computed on the spatial EfficientNet `conv_head` with respect to the fake logit.

| Figure | Truth | Calibrated P(fake) | Verdict |
|---|---|---|---|
| `figures/gradcam_ffpp_fake_Deepfakes.png` | fake | 0.994 | FAKE ✔ |
| `figures/gradcam_ffpp_real.png` | real | 0.154 | REAL ✔ |
| `figures/gradcam_kaggle_real.png` | real | 0.122 | REAL ✔ |
| `figures/gradcam_kaggle_fake.png` | fake | 0.667 | REAL ✘ (just under 0.74) |
| `figures/gradcam_ffpp_fake_MISSED_Deepfakes.png` | fake | 0.110 | REAL ✘ |

**What to read from these figures:**
- **Attention is often outside the inner face.** In the correctly detected Deepfakes example, Grad-CAM peaks at the face border and background, not on the facial features. This matches the scene-memorisation probe (§9): part of the evidence still comes from context rather than the manipulation itself.
- **Grad-CAM shows only one branch.** It covers the spatial branch, not the frequency or spectral branches, so it is a partial explanation of the decision.

---

## 7. Baseline reproduced on the identical split

Both models were trained on **exactly the same Kaggle training images** (49,981 MTCNN face crops), selected on the **same 1,999 validation images**, and tested on the **same 19,991 test images**. The decision threshold was fit on Kaggle validation only.

> ⚠️ Due to time limits, both runs were **stopped early**: MesoNet after 3 of 10 planned epochs, and the Kaggle-only DeepTrace after 1 epoch. The numbers are **directional, not converged**. V12 itself trained to early stopping (epoch 13 of 18).

| Model (Kaggle test, n = 19,991) | Training data | Epochs | AUC | Recall | Specificity | Bal. acc. | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| MesoNet-4 (Afchar et al., 2018), re-trained | Kaggle only | 3 (of 10) | 0.806 | 74.9% | 70.8% | 72.9% | 0.187 | 0.073 |
| DeepTrace, Kaggle only | Kaggle only | 1 (of 8) | 0.873 | 75.6% | 82.5% | 79.0% | 0.156 | 0.097 |
| **DeepTrace V12** | Kaggle + FF++ | 13 | **0.989** | **94.5%** | **95.3%** | **94.9%** | **0.048** | **0.021** |

Confusion matrices (Kaggle test):

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| MesoNet | 7,074 | 2,918 | 2,508 | 7,491 |
| DeepTrace, Kaggle only | 8,240 | 1,752 | 2,445 | 7,554 |
| V12 | 9,520 | 472 | 545 | 9,454 |

Even after only 1 epoch, the DeepTrace architecture beats the 3-epoch MesoNet on the identical split (AUC 0.873 vs 0.806). The full V12 reaches 0.989. A converged MesoNet/XceptionNet comparison is still to do (`scripts/train_baseline_same_split.py --arch mesonet|xception`).

---

## 7b. Comparison with published detectors (DeepfakeBench, NeurIPS 2023)

The published values below are frame-level AUC from the DeepfakeBench README (github.com/SCLBD/DeepfakeBench), with every detector trained and tested on FF++ c23.

> ⚠️ **Not a like-for-like comparison.**
> - DeepfakeBench uses the official FF++ split (720/140/140 videos), 256 px, and 4 manipulations.
> - DeepTrace V12 uses a stricter actor-disjoint split (540/60/400 actors) at 224 px, includes FaceShifter, and trains jointly with Kaggle StyleGAN data.
> - Read the gaps as approximate.

| Detector (published) | FF++ c23 AUC | DF | F2F | FS | NT |
|---|---|---|---|---|---|
| Meso4 | 0.608 | 0.677 | 0.617 | 0.595 | 0.570 |
| MesoInception | 0.758 | 0.854 | 0.809 | 0.742 | 0.652 |
| Capsule | 0.842 | 0.867 | 0.863 | 0.873 | 0.780 |
| CNN-Aug | 0.849 | 0.905 | 0.879 | 0.903 | 0.731 |
| FWA | 0.877 | 0.921 | 0.900 | 0.884 | 0.812 |
| **DeepTrace V12 (ours)** | **0.873** | **0.951** | **0.844** | **0.883** | **0.799** |
| EfficientNet-B4 | 0.957 | 0.976 | 0.976 | 0.980 | 0.931 |
| Face X-ray | 0.959 | 0.979 | 0.987 | 0.987 | 0.929 |
| Xception | 0.964 | 0.980 | 0.979 | 0.983 | 0.939 |
| UCF (best in benchmark) | 0.971 | 0.988 | 0.984 | 0.990 | 0.944 |

**Reading:**
- **Position.** V12 sits in the middle tier, level with FWA, Capsule and CNN-Aug, and about **0.09 AUC below** the Xception, EfficientNet-B4 and UCF tier.
- **Where the gap is.** It is smallest on Deepfakes (0.951 vs 0.980) and largest on Face2Face and NeuralTextures (−0.13 to −0.14).
- **Likely causes.**
  - Partial fine-tuning: only the top EfficientNet blocks are unfrozen.
  - 224 px input.
  - Far fewer FF++ training frames (about 1,700 videos at 10–32 frames each).
  - Joint training with StyleGAN data.
- **Cross-dataset context.** In DeepfakeBench, the FF++→Celeb-DF v2 AUC of the top detectors is only 0.73–0.77. Cross-dataset weakness is the norm for this field, not specific to DeepTrace. Our own cross-dataset test (Kaggle→FF++, §8) uses a different direction and is not directly comparable.

---

## 8. Cross-dataset generalisation: train on Kaggle, test on FaceForensics++ (no fine-tuning)

**Protocol.**
- Models were trained on Kaggle (StyleGAN vs FFHQ faces) only, and FF++ was **never seen** in training or validation.
- They were then tested **without fine-tuning** on all 400 FF++ c23 test actors: 4,000 real + 7,100 fake face crops.
- The threshold comes from Kaggle validation and was not adjusted for FF++.

| FF++ test (cross-dataset) | DeepTrace, Kaggle only (1 epoch) | MesoNet, Kaggle only (3 epochs) | *V12 (trained on FF++ train actors; within-dataset reference)* |
|---|---|---|---|
| Deepfakes AUC | **0.714** | 0.493 | *0.951* |
| FaceSwap AUC | 0.445 | 0.499 | *0.883* |
| FaceShifter AUC | 0.580 | 0.515 | *0.887* |
| Face2Face AUC | 0.597 | 0.499 | *0.844* |
| NeuralTextures AUC | 0.613 | 0.492 | *0.799* |
| **Macro AUC** | **0.590** | 0.500 | *0.873* |
| Recall / specificity (all) | 71.8% / 40.1% | 52.7% / 48.1% | *68.9% / 88.0%* |
| Balanced accuracy (all) | 56.0% | 50.4% | *78.5%* |

**What this shows:**
- **Kaggle-only training does not transfer to video face forgeries.**
  - MesoNet falls to chance (AUC 0.50).
  - Kaggle-only DeepTrace keeps a weak signal (macro 0.59, best on Deepfakes at 0.71), but it is worse than chance on FaceSwap (0.45).
  - With the Kaggle threshold it calls 60% of real FF++ faces fake (specificity 40%). The score scale itself shifts across datasets.
- **V12's FF++ performance comes from training on FF++ manipulations.** V12 scores 0.873 on the same test actors, which is consistent with the cross-dataset literature (Celeb-DF, DFDC, DeepfakeBench): detectors generalise poorly to manipulation families they never saw.
- **Caveat:** both cross-dataset models are under-trained (1 and 3 epochs). A converged Kaggle-only model could score differently, in either direction.

---

### 8b. Full-pipeline cross-dataset check (300 raw FF++ frames)

**Pipeline.** Raw frame → MTCNN face detection → 1.3× crop → JPEG-75 → model (all branches + fusion) → flip TTA → temperature → threshold.

**Settings for the Kaggle-only model.** TTA, temperature and threshold were chosen on **Kaggle validation only** (2,000 raw images through the same pipeline). No FF++ information was used.

**Test set.** 300 raw FF++ c23 frames from the held-out test actors: 150 real and 30 per manipulation, one frame per video, fixed seed. Script: `scripts/cross_dataset_full_pipeline_check.py`; output: `results/cross_dataset_full_pipeline_300.json`.

| Same 300 raw FF++ frames | AUC | Recall | Specificity | Bal. acc. | Brier | ECE |
|---|---|---|---|---|---|---|
| DeepTrace, Kaggle only (cross-dataset; T = 0.524, thr 0.41 from Kaggle val) | **0.640** | 82.7% | 33.3% | 58.0% | 0.274 | 0.168 |
| Same model without TTA or temperature | 0.640 | 84.0% | 32.7% | 58.3% | 0.246 | 0.080 |
| *V12, within-dataset reference (T = 1.384, thr 0.63)* | *0.873* | *69.3%* | *87.3%* | *78.3%* | *0.151* | *0.086* |

Per-manipulation AUC, Kaggle-only model (V12 in brackets):

| Manipulation | Kaggle-only | V12 |
|---|---|---|
| Deepfakes | 0.721 | 0.945 |
| FaceSwap | 0.540 | 0.931 |
| FaceShifter | 0.630 | 0.897 |
| Face2Face | 0.600 | 0.812 |
| NeuralTextures | 0.707 | 0.781 |

**Findings:**
1. **The full raw-frame pipeline reproduces the benchmark.**
   - V12 scores AUC 0.873, recall 69.3% and specificity 87.3% on these 300 raw frames.
   - Its full 11,100-frame test gave 0.873, 68.9% and 88.0%.
2. **Cross-dataset transfer stays weak through the full pipeline (AUC 0.64).** The model labels most FF++ faces "fake" (specificity 33%).
3. **Calibration does not transfer across datasets.**
   - The temperature fit on Kaggle (T = 0.52) sharpens probabilities that are already wrong on FF++, so ECE rises from 0.080 to 0.168.
   - TTA did not change Kaggle-validation AUC (0.872 both ways).
4. **Small sample.** With 300 frames, AUC is uncertain by roughly ±0.06, and each per-manipulation AUC rests on only 30 fakes.

---

## 9. Robustness and shortcut checks

- **Scene-memorisation probe.** This uses FF++ fakes whose background video belongs to a *training* actor while the swapped-in face belongs to a test actor. Such fakes are scored against the test reals.
  - Results: V10 AUC **0.10** (memorised scenes) → V11 **0.63** → V12 **0.78**.
  - Normal test fakes score 0.87, so a residual scene cue remains.
- **Compression and noise (measured on V11).** On FF++, the AUC moves from 0.807 pristine to:

  | Condition | FF++ AUC |
  |---|---|
  | JPEG quality 50 | 0.788 |
  | JPEG quality 30 | 0.783 |
  | Blur k = 7 | 0.752 |
  | Gaussian noise σ = 0.05 | 0.737 |

  The V12 robustness battery has not been run yet.

---

## 10. Related work

**Benchmarks and baselines.**
- **FaceForensics++** (Rössler et al., ICCV 2019) introduced the four manipulation methods used here (Deepfakes, Face2Face, FaceSwap, NeuralTextures; FaceShifter was added later). It established XceptionNet as the standard frame-level baseline.
- **MesoNet** (Afchar et al., WIFS 2018) is a deliberately shallow network that targets mesoscopic image properties. It remains a common lightweight baseline, and we reproduce it in §7.
- **Celeb-DF** (Li et al., CVPR 2020) and **DFDC** (Dolhansky et al., 2020) showed that detectors trained on FF++ drop sharply on newer, higher-quality forgeries. That observation motivates cross-dataset testing.
- **DeepfakeBench** (Yan et al., NeurIPS 2023 D&B) standardised training and evaluation protocols. It showed that many reported gains shrink under consistent preprocessing and splits, which is the same lesson as our V10 audit.

**Blending- and artefact-based detection.**
- **Face X-ray** (Li et al., CVPR 2020) detects the blending boundary that most face-swap pipelines leave, and trains on synthetic blends of real faces.
- **Self-Blended Images (SBI)** (Shiohara & Yamasaki, CVPR 2022) blend a transformed copy of a real face into itself. This produces hard, generic fakes that improve cross-dataset generalisation. DeepTrace uses SBI on real faces; V10's original implementation wrongly blended fakes.

**Frequency-domain detection.**
- GAN up-sampling leaves spectral fingerprints: Frank et al. (ICML 2020) and Durall et al. (CVPR 2020).
- **F³-Net** (Qian et al., ECCV 2020) mines frequency-aware clues and is particularly helpful on compressed video.
- Steganalysis rich models (**SRM**, Fridrich & Kodovsky, 2012) supply fixed high-pass residual filters.
- DeepTrace's spectral combiner (FFT phase, wavelet packet, SRM/Gabor) follows this line.

**Attention, temporal and semantic approaches.**
- Multi-attentional detection (Zhao et al., CVPR 2021) localises subtle artefacts with multiple attention maps.
- **LipForensics** (Haliassos et al., CVPR 2021) uses mouth-motion semantics and generalises better across manipulations.
- For fully generated images, Wang et al. (CVPR 2020) showed that a ResNet trained on one GAN with heavy augmentation transfers to others.
- **UnivFD** (Ojha et al., CVPR 2023) uses frozen CLIP features with nearest-neighbour or linear probing for universal fake-image detection. Our audit found that CLIP in DeepTrace only acts as a training-time alignment target and is not used at inference.

**Explainability and calibration.**
- Grad-CAM (Selvaraju et al., ICCV 2017) gives class-discriminative localisation.
- Temperature scaling (Guo et al., ICML 2017) is the standard post-hoc calibration method we use.
- MTCNN (Zhang et al., 2016) provides the face detection.

**Positioning.** DeepTrace combines spatial, spectral and blending-based (SBI) cues in one network. Its main methodological contribution in V11/V12 is the evaluation protocol rather than a new layer:
- an actor-disjoint three-way split,
- content-hash leakage guards,
- weight-lineage checks,
- identical preprocessing for all sources,
- validation-only calibration.

---

## 11. Limitations

1. **Within-dataset ≠ general.**
   - The headline FF++ numbers are on *unseen actors of the same dataset*: same capture pipeline, compression and manipulation methods as in training.
   - Cross-dataset (§8): trained on Kaggle only, DeepTrace drops to **0.590 macro AUC** on FF++ (MesoNet 0.500). Specificity falls to 40% with the Kaggle threshold, against 0.873 / 88% when FF++ manipulations are seen in training.
   - Performance therefore depends on having seen the manipulation family. These cross-dataset models were stopped early (1 and 3 epochs) and are directional only.
2. **Reenactment is weak.** Recall is 53% on NeuralTextures and 64% on Face2Face at 88% specificity.
3. **Residual shortcut.** The scene probe (0.78 vs 0.87) and the Grad-CAM maps (§6) show that part of the signal is context outside the face.
4. **Calibration is domain-dependent.**
   - One temperature fit on pooled validation improves FF++ and slightly worsens Kaggle calibration (§4).
   - FF++ test ECE (0.079) is higher than validation ECE (0.024).
5. **Untested sources.** No diffusion-model images (Stable Diffusion, Midjourney), no Celeb-DF / DFDC, and no phone-captured or social-media-recompressed video. `_new_dataset` (AUC 0.47) shows failure on an unknown source, though its labels are unverified.
6. **Single run.** One seed per model. The confidence intervals capture test-sample variance but not training-run variance.
7. **Small validation set.** 34 fake videos per manipulation, so model selection is noisy. Val→test drop: V12 FF++ 0.923 (validation, with TTA) → 0.873 (test).
8. **Frame-level model.** Video verdicts average frame scores; there is no temporal modelling.

---

## 12. Reproducing

```
venv\Scripts\python.exe scripts\crop_ffpp_faces.py                    # FF++ face crops
venv\Scripts\python.exe scripts\extract_ffpp_dense_train_faces.py     # 32 frames/video, train actors only
venv\Scripts\python.exe scripts\crop_kaggle_faces.py                  # Kaggle MTCNN crops
venv\Scripts\python.exe scripts\train_v11_clean.py --dense_manifest manifests/ffpp_c23_faces256_dense_train_manifest.csv --kaggle_faces_manifest manifests/kaggle_faces256_manifest.csv --val_from checkpoints/v11_clean/data_manifest.json --max_kaggle 25000 --image_size 224 --epochs 20 --patience 5 --out_dir checkpoints/v12_dense
venv\Scripts\python.exe scripts\val_output_study.py --checkpoint checkpoints/v12_dense/best_model.pth      # TTA / T / thresholds on VAL
venv\Scripts\python.exe scripts\evaluate_v11_honest.py --checkpoint checkpoints/v12_dense/best_model.pth   # TEST, once
venv\Scripts\python.exe scripts\v12_report_extras.py                                                       # Grad-CAM, Brier, confusion
venv\Scripts\python.exe scripts\train_v11_clean.py --kaggle_only --kaggle_faces_manifest manifests/kaggle_faces256_manifest.csv --val_from checkpoints/v11_clean/data_manifest.json --max_kaggle 25000 --image_size 224 --epochs 8 --patience 3 --out_dir checkpoints/v12_kaggle_only
venv\Scripts\python.exe scripts\evaluate_v11_honest.py --checkpoint checkpoints/v12_kaggle_only/best_model.pth --no_calibration --skip_robustness
venv\Scripts\python.exe scripts\train_baseline_same_split.py --arch mesonet
venv\Scripts\python.exe scripts\eval_cross_and_baseline.py      # evaluates the Kaggle-only DeepTrace + MesoNet checkpoints
```

Result files:
- `results/benchmark_eval_v12_dense/`
- `results/cross_dataset_and_baseline.json` (Kaggle-only DeepTrace + MesoNet)
- `reports/v12/extras.json`
