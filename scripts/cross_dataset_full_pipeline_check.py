"""
Cross-dataset check through the FULL inference pipeline (evaluation only, no training):

  raw image -> MTCNN face detection -> 1.3x square crop -> JPEG-75 -> model (all branches + fusion)
  -> hflip TTA -> temperature -> threshold

Model: DeepTrace trained on Kaggle ONLY (checkpoints/v12_kaggle_only). TTA on/off, temperature and threshold
are chosen on KAGGLE validation only (raw Kaggle valid images through the same pipeline) -> no FF++ information.
Test: 300 raw FF++ c23 frames from the 400 held-out test actors (150 real + 30 per manipulation, one frame per
video, fixed seed). V12 (which saw FF++ train actors) is scored on the same 300 frames as a within-dataset reference.

Writes results/cross_dataset_full_pipeline_300.json
"""
import sys
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.v11_predictor import V11Predictor, NoFaceError
from utils.actor_splits import get_actor_disjoint_split, _extract_video_id
from scripts.train_baseline_same_split import ece

COHORTS = ["Deepfakes", "FaceSwap", "FaceShifter", "Face2Face", "NeuralTextures"]
sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -20, 20)))


def run(pred, paths, batch=32):
    """Full pipeline on raw files. Returns (plain_logits, tta_logits); NaN where no face found."""
    z0, zt = np.full(len(paths), np.nan), np.full(len(paths), np.nan)
    faces, idx = [], []

    def flush():
        if faces:
            a = pred.logits_for_faces(faces)[0]
            b = pred.logits_for_faces([np.ascontiguousarray(f[:, ::-1]) for f in faces])[0]
            z0[idx], zt[idx] = a, 0.5 * (a + b)
            faces.clear(); idx.clear()

    for i, p in enumerate(paths):
        try:
            faces.append(pred.crop_face(np.array(Image.open(p).convert("RGB")))[0]); idx.append(i)
        except NoFaceError:
            pass
        if len(faces) == batch:
            flush()
    flush()
    return z0, zt


def metrics(y, z, T, thr):
    ok = ~np.isnan(z); y, z = np.asarray(y)[ok], z[ok]
    p = sig(z / T); pr = p >= thr
    tp, fn = int(np.sum(pr & (y == 1))), int(np.sum(~pr & (y == 1)))
    tn, fp = int(np.sum(~pr & (y == 0))), int(np.sum(pr & (y == 0)))
    return {"n_scored": int(ok.sum()), "no_face": int((~ok).sum()), "auc": round(float(roc_auc_score(y, z)), 4),
            "recall_fake": round(tp / max(1, tp + fn), 4), "specificity_real": round(tn / max(1, tn + fp), 4),
            "bal_acc": round(0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp)), 4),
            "brier": round(float(brier_score_loss(y, p)), 4), "ece": ece(y, p),
            "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp}}


def bal_thr(y, p):
    grid = np.linspace(0.02, 0.98, 97)
    return float(grid[int(np.argmax([0.5 * (np.mean(p[y == 1] >= t) + np.mean(p[y == 0] < t)) for t in grid]))])


def fit_T(y, z):
    def nll(T):
        p = np.clip(sig(z / T), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


def main():
    rng = random.Random(7)
    # ---- 300 raw FF++ test frames: one frame per video, unseen actors
    raw = pd.read_csv("manifests/ffpp_c23_manifest.csv")
    sp = get_actor_disjoint_split(raw, "filepath")
    test = []
    reals = sp["test_reals"].assign(vid=lambda d: d.filepath.map(_extract_video_id))
    for vid in rng.sample(sorted(reals.vid.unique()), 150):
        test.append((rng.choice(reals[reals.vid == vid].filepath.tolist()), 0, "real"))
    for m in COHORTS:
        f = sp["test_fakes"][m].assign(vid=lambda d: d.filepath.map(_extract_video_id))
        for vid in rng.sample(sorted(f.vid.unique()), 30):
            test.append((rng.choice(f[f.vid == vid].filepath.tolist()), 1, m))
    tp_ = [t[0] for t in test]; ty = np.array([t[1] for t in test]); tm = np.array([t[2] for t in test])

    out = {"pipeline": "raw image -> MTCNN -> 1.3x crop -> JPEG75 -> model (all branches+fusion) -> hflip TTA -> T -> threshold",
           "test_set": "300 raw FF++ c23 frames, 400 held-out test actors: 150 real + 30 per manipulation, 1 frame/video, seed 7"}

    # ---- A. Kaggle-only model: settings from KAGGLE validation only
    ko = V11Predictor("checkpoints/v12_kaggle_only/best_model.pth", threshold=0.5)
    val = [(p, y) for p, y, g in json.load(open("checkpoints/v11_clean/data_manifest.json"))["val"] if g.startswith("kaggle")]
    vy = np.array([y for _, y in val])
    v0, vt = run(ko, [p for p, _ in val])
    m = ~np.isnan(v0)
    auc0, auct = roc_auc_score(vy[m], v0[m]), roc_auc_score(vy[m], vt[m])
    use_tta = auct >= auc0
    vz = vt if use_tta else v0
    T = fit_T(vy[m], vz[m]); thr = bal_thr(vy[m], sig(vz[m] / T))
    out["kaggle_only_model"] = {
        "trained_on": "Kaggle only (1 epoch, stopped early for time)",
        "settings_from_kaggle_val": {"val_auc_plain": round(auc0, 4), "val_auc_tta": round(auct, 4), "use_tta": bool(use_tta),
                                     "temperature": round(T, 4), "threshold": round(thr, 4)},
        "kaggle_val_full_pipeline": metrics(vy, vz, T, thr),
    }
    t0, tt = run(ko, tp_)
    tz = tt if use_tta else t0
    r = metrics(ty, tz, T, thr)
    r["per_cohort_auc_vs_150_reals"] = {c: round(float(roc_auc_score(ty[np.isin(tm, ["real", c])][~np.isnan(tz[np.isin(tm, ["real", c])])],
                                                                    tz[np.isin(tm, ["real", c])][~np.isnan(tz[np.isin(tm, ["real", c])])])), 4) for c in COHORTS}
    out["kaggle_only_model"]["ffpp_cross_dataset_300"] = r
    out["kaggle_only_model"]["ffpp_cross_dataset_300_plain_no_tta_no_T"] = metrics(ty, t0, 1.0, bal_thr(vy[m], sig(v0[m])))
    del ko

    # ---- B. Reference: V12 (trained on FF++ train actors) on the same 300 frames, its own val-chosen settings
    cal = json.loads(Path("checkpoints/v12_dense/output_calibration.json").read_text())
    v12 = V11Predictor("checkpoints/v12_dense/best_model.pth", threshold=0.5)
    a0, at = run(v12, tp_)
    az = at if cal["use_tta"] else a0
    r = metrics(ty, az, cal["temperature"], cal["thresholds"]["ffpp"])
    r["per_cohort_auc_vs_150_reals"] = {c: round(float(roc_auc_score(ty[np.isin(tm, ["real", c])][~np.isnan(az[np.isin(tm, ["real", c])])],
                                                                    az[np.isin(tm, ["real", c])][~np.isnan(az[np.isin(tm, ["real", c])])])), 4) for c in COHORTS}
    out["v12_reference_same_300"] = {"trained_on": "Kaggle + FF++ train actors (within-dataset reference)",
                                     "settings": {"use_tta": cal["use_tta"], "temperature": cal["temperature"], "threshold": cal["thresholds"]["ffpp"]},
                                     "ffpp_300": r}
    Path("results/cross_dataset_full_pipeline_300.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
