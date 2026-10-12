# DeepTrace — Numbers for the Presentation (V12, verified 2026-10-11)

Model: `checkpoints/v12_dense/best_model.pth` (clean lineage, trained from ImageNet/CLIP weights).
Test set touched once. Thresholds, temperature and TTA were chosen on validation only.
Sources: `results/benchmark_eval_v12_dense/v11_honest_evaluation.json` and `audit_v10_claude/AUDIT_REPORT.md` §6b.

## Headline
- **FaceForensics++ (400 actors never seen in training):** AUC **0.873** frame-level, **0.905** video-level, **78.5%** balanced accuracy.
- **Kaggle StyleGAN vs real faces:** AUC **0.989**, **94.9%** balanced accuracy.
- **Leakage check:** 0 of the test files are byte-identical to any of the 137,334 training files, and no actor appears in both training and test.

## Recall (fakes caught) per dataset, at the frozen threshold

| Dataset | AUC | Recall | Specificity (reals passed) | Balanced acc. |
|---|---|---|---|---|
| FF++ Deepfakes | 0.951 | 88.2% | 88.0% | 88.1% |
| FF++ FaceSwap | 0.883 | 70.7% | 88.0% | 79.4% |
| FF++ FaceShifter | 0.887 | 68.9% | 88.0% | 78.5% |
| FF++ Face2Face | 0.844 | 63.8% | 88.0% | 75.9% |
| FF++ NeuralTextures | 0.799 | 53.1% | 88.0% | 70.6% |
| **FF++ all cohorts** | **0.873** | **68.9%** | **88.0%** | **78.5%** |
| Kaggle (StyleGAN) | 0.989 | 94.5% | 95.3% | 94.9% |
| `test_data` (99 images, small) | 0.922 | 72.0% | 98.0% | 85.0% |
| `_new_dataset` (labels unverified) | 0.470 | 2.7% | 97.6% | 50.2% |

## Progress story (same FF++ test actors)

| Version | FF++ macro AUC | What changed |
|---|---|---|
| V10 (old) | 0.607 | Leaky lineage; full frames; validation = test actors |
| V11 | 0.802 | Leak-free retrain, face crops, 3-way actor split, MD5 guards |
| V11 + TTA + calibration | 0.807 | Output settings chosen on validation |
| **V12** | **0.873** | 3× more training frames, Kaggle face-cropped like FF++, 224 px input |

## Calibration (Table II, validation, chosen on validation only)
- Temperature **T = 1.384**; thresholds FF++ **0.63**, Kaggle **0.74**; flip TTA on.
- Validation Brier **0.0773 → 0.0759**, ECE **0.0345 → 0.0082** after temperature scaling.
- Test Brier: FF++ 0.157, Kaggle 0.048.

## Baseline on the same Kaggle split, and cross-dataset (train Kaggle only → test FF++, no fine-tuning)

| Model | Kaggle test AUC | FF++ cross-dataset macro AUC |
|---|---|---|
| MesoNet-4 (3 of 10 epochs) | 0.806 | 0.500 |
| DeepTrace, Kaggle only (1 epoch) | 0.873 | 0.590 |
| **DeepTrace V12 (Kaggle + FF++)** | **0.989** | *0.873 (within-dataset, unseen actors)* |

Both extra models were stopped early for time, so treat their results as directional only.

Figures (confusion matrices, class-wise bar chart, Grad-CAM): `reports/v12/figures/`. Full report: `reports/v12/V12_REPORT.md`.

## Say honestly if asked
- **Hardest cohorts:** NeuralTextures and Face2Face (recall 53–64%).
- **Not a general detector:** `_new_dataset` fails (AUC 0.47). Its labels have no known source. Diffusion images and phone video are untested.
- **Robustness for V12 is not measured yet.** V11 lost only about 0.02 AUC on FF++ at JPEG quality 30.
- **Scene memorisation:** reduced (probe 0.10 in V10 → 0.78 in V12) but not zero.
- **Old README/SUBMISSION figures** (99.69%, 0.9999, 0.607, "100% actor-disjoint", robustness 0.90) come from V10. They are superseded; see `audit_v10_claude/AUDIT_REPORT.md`.
