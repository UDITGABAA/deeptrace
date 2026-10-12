"""
Fit the web-app decision threshold for V11 on VALIDATION data, through the exact app pipeline
(raw image -> MTCNN crop -> model), and optionally verify that the app pipeline reproduces the
benchmark on TEST data (verification only — nothing is tuned on test).

  venv\\Scripts\\python.exe scripts\\fit_v11_app_threshold.py [--check_test]

Writes checkpoints/v11_clean/app_threshold.json
"""
import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.v11_predictor import V11Predictor, NoFaceError, DEFAULT_CKPT
from utils.actor_splits import get_actor_disjoint_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fit_v11_app_threshold")
COHORTS = ["Deepfakes", "FaceSwap", "FaceShifter", "Face2Face", "NeuralTextures"]


def run(pred, paths, batch=64):
    """Return logits (NaN where no face found)."""
    out = np.full(len(paths), np.nan)
    faces, idx = [], []
    for i, p in enumerate(paths):
        try:
            faces.append(pred.crop_face(np.array(Image.open(p).convert("RGB")))[0]); idx.append(i)
        except NoFaceError:
            pass
        if len(faces) == batch or (i == len(paths) - 1 and faces):
            out[idx] = pred.logits_for_faces(faces)[0]
            faces, idx = [], []
        if i % 2000 == 0:
            logger.info(f"{i}/{len(paths)}")
    return out


def bal_acc(y, p, t):
    return 0.5 * (np.mean(p[y == 1] >= t) + np.mean(p[y == 0] < t))


def summarize(y, z, thr):
    ok = ~np.isnan(z)
    y, z = np.asarray(y)[ok], z[ok]
    p = 1 / (1 + np.exp(-z))
    return {"n": int(len(y)), "no_face": int((~ok).sum()), "auc": round(float(roc_auc_score(y, z)), 4),
            "bal_acc@thr": round(float(bal_acc(y, p, thr)), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--check_test", action="store_true")
    args = ap.parse_args()
    pred = V11Predictor(args.checkpoint, threshold=0.5)
    ckdir = Path(args.checkpoint).parent

    # ---- validation pools (same items as V11 training val), as RAW inputs
    val = json.load(open(ckdir / "data_manifest.json"))["val"]
    faces_df = pd.read_csv("manifests/ffpp_c23_faces256_manifest.csv")
    crop2src = dict(zip(faces_df.filepath, faces_df.src_filepath))
    vk = [(p, y) for p, y, g in val if g.startswith("kaggle")]
    vf = [(crop2src[p], y) for p, y, g in val if g.startswith("ffpp")]
    zk = run(pred, [p for p, _ in vk]); yk = np.array([y for _, y in vk])
    zf = run(pred, [p for p, _ in vf]); yf = np.array([y for _, y in vf])

    # threshold: maximise the MEAN of per-domain balanced accuracy (each domain weighted equally)
    pk, pf = 1 / (1 + np.exp(-zk)), 1 / (1 + np.exp(-zf))
    mk, mf = ~np.isnan(zk), ~np.isnan(zf)
    grid = np.linspace(0.02, 0.98, 97)
    scores = [0.5 * (bal_acc(yk[mk], pk[mk], t) + bal_acc(yf[mf], pf[mf], t)) for t in grid]
    thr = float(round(grid[int(np.argmax(scores))], 4))
    report = {"threshold": thr, "fit_on": "V11 validation (Kaggle valid 2,000 + FF++ val actors), app pipeline",
              "checkpoint": args.checkpoint,
              "val_kaggle": summarize(yk, zk, thr), "val_ffpp": summarize(yf, zf, thr)}

    if args.check_test:
        K = Path("data/kaggle_realfake/real_vs_fake/real-vs-fake/test")
        rng = np.random.RandomState(0)
        kt = []
        for c, y in [("real", 0), ("fake", 1)]:
            ps = sorted((K / c).glob("*.jpg"))
            kt += [(str(ps[i]), y) for i in rng.choice(len(ps), 1000, replace=False)]
        zkt = run(pred, [p for p, _ in kt])
        raw = pd.read_csv("manifests/ffpp_c23_manifest.csv")
        sp = get_actor_disjoint_split(raw, "filepath")
        ft = [(p, 0, "real") for p in sp["test_reals"]["filepath"]] + \
             [(p, 1, m) for m in COHORTS for p in sp["test_fakes"][m]["filepath"]]
        zft = run(pred, [p for p, _, _ in ft])
        yft, mft = np.array([y for _, y, _ in ft]), np.array([m for _, _, m in ft])
        report["test_check_app_pipeline"] = {
            "kaggle_test_2000": summarize([y for _, y in kt], zkt, thr),
            "ffpp_test_raw_frames_all": summarize(yft, zft, thr),
            "ffpp_per_cohort_auc": {m: summarize(yft[np.isin(mft, ["real", m])], zft[np.isin(mft, ["real", m])], thr)["auc"]
                                    for m in COHORTS},
            "note": "verification only; compare with results/benchmark_eval_v11/v11_honest_evaluation.json",
        }

    (ckdir / "app_threshold.json").write_text(json.dumps(report, indent=2))
    logger.info(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
