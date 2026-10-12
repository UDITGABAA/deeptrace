"""
Honest evaluation for V11 (rules from audit_v10_claude/AUDIT_REPORT.md):

  * thresholds are FROZEN from the checkpoint (fit on val actors) — never tuned on test labels
    (a test-tuned "oracle" accuracy is reported only as a labelled upper bound)
  * MD5 check: no evaluated file may be byte-identical to a training file
  * domains are reported SEPARATELY (Kaggle vs FF++); never pooled
  * FF++: frame AUC with ALL 4,000 test reals, video-level AUC, video-cluster bootstrap 95% CI
  * robustness per domain; scene-memorisation probe on mixed-pair fakes; OOD set

Usage:
  venv\\Scripts\\python.exe scripts\\evaluate_v11_honest.py --checkpoint checkpoints/v11_clean/best_model.pth
"""
import sys
import json
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detector import DeepfakeDetector
from scripts.evaluate_v10_full_battery import V10EvalDetector
from datasets.transforms import get_val_transforms
from utils.actor_splits import get_actor_disjoint_split, get_actor_split, _extract_video_id
from utils.device import AMPContext
from utils.leakage_guard import assert_no_content_overlap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("v11_honest_eval")

COHORTS = ["Deepfakes", "FaceSwap", "FaceShifter", "Face2Face", "NeuralTextures"]
KAGGLE = Path("data/kaggle_realfake/real_vs_fake/real-vs-fake")
IMG_SIZE = 160  # overwritten from checkpoint args in main()
TTA = False
TEMP = 1.0     # temperature from <ckpt_dir>/output_calibration.json (fit on val)


def _jpeg(im, q):
    _, enc = cv2.imencode(".jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


PERTURB = {
    "jpeg_q50": lambda im, rs: _jpeg(im, 50),
    "jpeg_q30": lambda im, rs: _jpeg(im, 30),
    "blur_k7": lambda im, rs: cv2.GaussianBlur(im, (7, 7), 0),
    "noise_s05": lambda im, rs: np.clip(im + rs.normal(0, 0.05 * 255, im.shape), 0, 255).astype(np.uint8),
}


class DS(Dataset):
    def __init__(self, paths, perturb=None):
        self.paths, self.perturb, self.tf = paths, perturb, get_val_transforms(IMG_SIZE)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = np.array(Image.open(self.paths[i]).convert("RGB"))
        if self.perturb:
            im = PERTURB[self.perturb](im, np.random.RandomState(i))
        return self.tf(image=im)["image"]


@torch.no_grad()
def infer(model, paths, device, perturb=None):
    dl = DataLoader(DS(paths, perturb), batch_size=64, num_workers=4, pin_memory=True)
    out = []
    for x in dl:
        x = x.to(device, non_blocking=True)
        with AMPContext(device=device, enabled=True):
            z = model(x)
            if TTA:  # average logits with the horizontally flipped view
                z = 0.5 * (z + model(torch.flip(x, dims=[3])))
        out.append(z.float().cpu().numpy())
    return np.nan_to_num(np.concatenate(out), nan=0.0) / TEMP  # temperature-scaled logits


sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -20, 20)))
r4 = lambda v: round(float(v), 4)


def bal_acc(y, p, t):
    pred = p >= t
    return r4(0.5 * (np.mean(pred[y == 1]) + np.mean(~pred[y == 0])))


def oracle_bal_acc(y, p):
    return max(bal_acc(y, p, t) for t in np.linspace(0.02, 0.98, 97))


def video_boot_ci(y, s, vids, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    uv = np.unique(vids)
    idx = {v: np.where(vids == v)[0] for v in uv}
    out = []
    for _ in range(B):
        sel = np.concatenate([idx[v] for v in rng.choice(uv, len(uv))])
        if len(np.unique(y[sel])) == 2:
            out.append(roc_auc_score(y[sel], s[sel]))
    return [r4(np.percentile(out, 2.5)), r4(np.percentile(out, 97.5))]


def ece(y, p, bins=15):
    conf = np.where(p >= 0.5, p, 1 - p); pred = (p >= 0.5).astype(int)
    e = 0.0
    edges = np.linspace(0.5, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf <= hi) if lo == 0.5 else (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return r4(e)


def binary_report(y, logit, thr, vids=None):
    y = np.asarray(y); p = sig(np.asarray(logit))
    pred = p >= thr
    tp, fn = int(np.sum(pred & (y == 1))), int(np.sum(~pred & (y == 1)))
    tn, fp = int(np.sum(~pred & (y == 0))), int(np.sum(pred & (y == 0)))
    rep = {"n": int(len(y)), "auc": r4(roc_auc_score(y, logit)), "ece": ece(y, p),
           "recall_fake@frozen_thr": r4(tp / max(1, tp + fn)),        # share of fakes caught
           "specificity_real@frozen_thr": r4(tn / max(1, tn + fp)),   # share of reals passed
           "precision_fake@frozen_thr": r4(tp / max(1, tp + fp)),
           "confusion@frozen_thr": {"tp": tp, "fn": fn, "tn": tn, "fp": fp},
           "bal_acc@frozen_thr": bal_acc(y, p, thr), "frozen_thr": thr, "bal_acc@0.5": bal_acc(y, p, 0.5),
           "oracle_bal_acc(test-tuned, upper bound only)": oracle_bal_acc(y, p),
           "brier": r4(brier_score_loss(y, p))}
    if vids is not None:
        vids = np.asarray(vids)
        v = pd.DataFrame({"v": vids, "y": y, "s": logit}).groupby("v").agg(y=("y", "first"), s=("s", "mean"))
        rep["video_auc"] = r4(roc_auc_score(v.y, v.s))
        rep["n_videos"] = int(len(v))
        rep["frame_auc_95ci_video_bootstrap"] = video_boot_ci(y, np.asarray(logit), vids)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/v11_clean/best_model.pth")
    ap.add_argument("--faces_manifest", default="manifests/ffpp_c23_faces256_manifest.csv")
    ap.add_argument("--out_dir", default=None, help="default: results/benchmark_eval_<checkpoint dir name>")
    ap.add_argument("--skip_robustness", action="store_true")
    ap.add_argument("--tta", action="store_true", help="hflip TTA (only if chosen on VAL first)")
    ap.add_argument("--no_calibration", action="store_true",
                    help="ignore output_calibration.json (TTA/temperature/thresholds chosen on val)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not ckpt.get("clean_lineage"):
        raise RuntimeError("Checkpoint lacks clean_lineage flag — not a guarded V11 run.")
    thr = ckpt["thresholds"]
    global IMG_SIZE, TTA, TEMP
    IMG_SIZE = int(ckpt.get("args", {}).get("image_size", 160))
    TTA = args.tta
    calib_file = Path(args.checkpoint).parent / "output_calibration.json"
    calib = None
    if calib_file.exists() and not args.no_calibration:
        calib = json.loads(calib_file.read_text())
        TTA, TEMP, thr = calib["use_tta"], calib["temperature"], calib["thresholds"]
        logger.info(f"Using val-chosen output settings: TTA={TTA}, T={TEMP}, thresholds={thr}")
    model = V10EvalDetector(DeepfakeDetector())
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    out_dir = Path(args.out_dir or f"results/benchmark_eval_{Path(args.checkpoint).parent.name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- evaluation pools
    k_paths, k_y = [], []
    kfm = ckpt.get("args", {}).get("kaggle_faces_manifest")
    if kfm:  # model trained on MTCNN-cropped Kaggle -> evaluate on the cropped test images (= app pipeline)
        km = pd.read_csv(kfm)
        kt = km[km.split == "test"].sort_values("filepath")
        k_paths, k_y = kt.filepath.tolist(), kt.label.tolist()
        report_kaggle_note = f"MTCNN crops of Kaggle test ({len(kt)} of 20000 had a detectable face)"
    else:
        for c, y in [("real", 0), ("fake", 1)]:
            ps = sorted(str(p) for p in (KAGGLE / "test" / c).glob("*.jpg"))
            k_paths += ps; k_y += [y] * len(ps)
        report_kaggle_note = "raw Kaggle test images (no MTCNN crop)"
    fdf = pd.read_csv(args.faces_manifest)
    sp = get_actor_disjoint_split(fdf, filepath_col="filepath")
    ff = pd.concat([sp["test_reals"].assign(manip="real", y=0)] +
                   [sp["test_fakes"][m].assign(manip=m, y=1) for m in COHORTS], ignore_index=True)
    mixed = pd.concat([sp["discarded_fakes"][m].assign(manip=m) for m in COHORTS], ignore_index=True)

    # ---------- leakage guard against the checkpoint's own training list
    manifest = Path(args.checkpoint).parent / "data_manifest.json"
    train_paths = sorted({t[0] for t in json.load(open(manifest))["train"]})
    guard = assert_no_content_overlap(train_paths, {"kaggle_test": k_paths, "ffpp_test": ff.filepath.tolist(),
                                                    "ffpp_mixed_pairs": mixed.filepath.tolist()})

    report = {"checkpoint": args.checkpoint, "epoch": ckpt.get("epoch"), "image_size": IMG_SIZE, "tta": TTA, "temperature": TEMP, "output_calibration": calib, "lineage": ckpt.get("lineage"),
              "frozen_thresholds_from_val": thr, "val_metrics_at_selection": ckpt.get("metrics"),
              "leakage_guard": guard}

    # ---------- Kaggle (in-domain StyleGAN2 vs FFHQ)
    k_logit = infer(model, k_paths, device)
    pd.DataFrame({"filepath": k_paths, "y": k_y, "logit": k_logit}).to_csv(out_dir / "kaggle_test_logits.csv", index=False)
    report["kaggle_test"] = binary_report(k_y, k_logit, thr["kaggle"])
    report["kaggle_test"]["input"] = report_kaggle_note

    # ---------- FF++ test actors (face crops)
    ff["logit"] = infer(model, ff.filepath.tolist(), device)
    reals = ff[ff.y == 0]
    report["ffpp_test"] = {}
    for m in COHORTS:
        d = pd.concat([reals, ff[ff.manip == m]])
        report["ffpp_test"][m] = binary_report(d.y.values, d.logit.values, thr["ffpp"], d.video_id.values)
    report["ffpp_test"]["macro_frame_auc"] = r4(np.mean([report["ffpp_test"][m]["auc"] for m in COHORTS]))
    report["ffpp_test"]["macro_video_auc"] = r4(np.mean([report["ffpp_test"][m]["video_auc"] for m in COHORTS]))
    report["ffpp_test"]["all_cohorts"] = binary_report(ff.y.values, ff.logit.values, thr["ffpp"], ff.video_id.values)

    # ---------- scene-memorisation probe (fake id = <target scene>_<source face>)
    train_actors, _ = get_actor_split()
    mixed["logit"] = infer(model, mixed.filepath.tolist(), device)
    mixed["scene_train"] = mixed.video_id.str.split("_").str[0].isin(train_actors)
    probe = {}
    for flag, name in [(True, "scene=train_actor,face=test_actor"), (False, "scene=test_actor,face=train_actor")]:
        g = mixed[mixed.scene_train == flag]
        y = np.r_[np.ones(len(g)), np.zeros(len(reals))]
        probe[name] = {"n": int(len(g)), "mean_p_fake": r4(sig(g.logit).mean()),
                       "auc_vs_test_reals": r4(roc_auc_score(y, np.r_[g.logit, reals.logit]))}
    probe["note"] = "V10: scene=train gave AUC 0.10 (memorised scenes). ~= normal test AUC means no scene memorisation."
    report["scene_memorisation_probe"] = probe

    # ---------- OOD
    ood = {}
    for name, root, classes in [("newds_test", Path("data/_new_dataset_extracted/Test"), [("Real", 0), ("Fake", 1)]),
                                ("test_data", Path("test_data"), [("real", 0), ("fake", 1)])]:
        ps, ys = [], []
        for c, y in classes:
            fs = sorted(str(p) for p in (root / c).glob("*") if p.is_file())
            ps += fs; ys += [y] * len(fs)
        if ps:
            zs = infer(model, ps, device)
            pd.DataFrame({"filepath": ps, "y": ys, "logit": zs}).to_csv(out_dir / f"{name}_logits.csv", index=False)
            ood[name] = binary_report(ys, zs, thr["kaggle"])
    report["ood"] = ood

    # ---------- robustness per domain
    if not args.skip_robustness:
        rob = {"kaggle_test_2000": {}, "ffpp_test_all": {}}
        rng = np.random.RandomState(0)
        ky_all = np.asarray(k_y)
        ksel = np.r_[rng.choice(np.where(ky_all == 0)[0], 1000, replace=False),
                     rng.choice(np.where(ky_all == 1)[0], 1000, replace=False)]
        kp, ky = [k_paths[i] for i in ksel], np.array(k_y)[ksel]
        rob["kaggle_test_2000"]["pristine"] = r4(roc_auc_score(ky, k_logit[ksel]))
        rob["ffpp_test_all"]["pristine"] = report["ffpp_test"]["all_cohorts"]["auc"]
        for name in PERTURB:
            rob["kaggle_test_2000"][name] = r4(roc_auc_score(ky, infer(model, kp, device, name)))
            rob["ffpp_test_all"][name] = r4(roc_auc_score(ff.y, infer(model, ff.filepath.tolist(), device, name)))
            logger.info(f"robustness {name}: {rob['kaggle_test_2000'][name]} / {rob['ffpp_test_all'][name]}")
        report["robustness_auc_per_domain"] = rob

    with open(out_dir / "v11_honest_evaluation.json", "w") as f:
        json.dump(report, f, indent=2)
    ff[["filepath", "video_id", "manip", "y", "logit"]].to_csv(out_dir / "ffpp_test_logits.csv", index=False)
    logger.info(json.dumps({k: report[k] for k in ("kaggle_test", "scene_memorisation_probe")}, indent=2))
    logger.info("FF++ per cohort: " + json.dumps({m: (report["ffpp_test"][m]["auc"], report["ffpp_test"][m]["video_auc"],
                                                     report["ffpp_test"][m]["bal_acc@frozen_thr"]) for m in COHORTS}))
    logger.info(f"Saved -> {out_dir / 'v11_honest_evaluation.json'}")


if __name__ == "__main__":
    main()
