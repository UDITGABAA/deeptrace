"""
Extra evidence for the V12 report (no training, no tuning):

  1. Grad-CAM overlays on real TEST images, picked by a fixed rule (first file in sorted order of each
     category: FF++ fake / FF++ real / Kaggle fake / Kaggle real) + the first misclassified FF++ fake.
  2. Brier score + ECE on the VALIDATION predictions, before and after temperature scaling
     (the values that go next to the temperature / threshold in Table II).
  3. Confusion matrices and a class-wise breakdown on TEST, from the saved per-image logits of
     scripts/evaluate_v11_honest.py (results/benchmark_eval_v12_dense/*.csv; already temperature-scaled).

Outputs: reports/v12/extras.json and reports/v12/figures/*.png
"""
import sys
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import brier_score_loss, roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.transforms import get_val_transforms
from models.detector import DeepfakeDetector
from scripts.evaluate_v10_full_battery import V10EvalDetector

CKPT = Path("checkpoints/v12_dense/best_model.pth")
RES = Path("results/benchmark_eval_v12_dense")
OUT = Path("reports/v12"); FIG = OUT / "figures"
COHORTS = ["Deepfakes", "FaceSwap", "FaceShifter", "Face2Face", "NeuralTextures"]
sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -20, 20)))


def ece(y, p, bins=15):
    conf = np.where(p >= 0.5, p, 1 - p); pred = (p >= 0.5).astype(int)
    e, edges = 0.0, np.linspace(0.5, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf <= hi) if lo == 0.5 else (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return round(float(e), 4)


def load_model(device):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    m = V10EvalDetector(DeepfakeDetector())
    m.load_state_dict(ck["model_state_dict"], strict=True)
    return m.to(device).eval(), int(ck["args"]["image_size"])


def gradcam(model, tf, face, device):
    layer = model.spatial_encoder.backbone.conv_head
    st = {}
    h1 = layer.register_forward_hook(lambda m, i, o: st.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: st.__setitem__("g", go[0]))
    try:
        x = tf(image=face)["image"].unsqueeze(0).to(device).requires_grad_(True)
        model.zero_grad(set_to_none=True)
        z = model(x)
        z.sum().backward()
        a, g = st["a"].detach()[0], st["g"].detach()[0]
        cam = torch.relu((g.mean(dim=(1, 2))[:, None, None] * a).sum(0)).float().cpu().numpy()
    finally:
        h1.remove(); h2.remove()
    cam = cam / cam.max() if cam.max() > 0 else cam
    cam = cv2.resize(cam, (face.shape[1], face.shape[0]))
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return np.uint8(0.45 * heat + 0.55 * face), float(z.item())


def confusion_fig(cm, title, path):
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred real", "pred fake"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true real", "true fake"])
    ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calib = json.loads((CKPT.parent / "output_calibration.json").read_text())
    T, thr = calib["temperature"], calib["thresholds"]
    model, size = load_model(device)
    tf = get_val_transforms(size)
    out = {"checkpoint": str(CKPT), "temperature": T, "thresholds": thr, "tta": calib["use_tta"]}

    # ── 2. Brier / ECE on VALIDATION (same TTA + temperature as deployment) ───────────────────
    val = json.load(open(CKPT.parent / "data_manifest.json"))["val"]
    y = np.array([v[1] for v in val]); g = np.array([v[2] for v in val])
    zs = []
    with torch.no_grad():
        for i in range(0, len(val), 64):
            x = torch.stack([tf(image=np.array(Image.open(v[0]).convert("RGB")))["image"] for v in val[i:i + 64]]).to(device)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                z = 0.5 * (model(x) + model(torch.flip(x, dims=[3]))) if calib["use_tta"] else model(x)
            zs.append(z.float().cpu().numpy())
    z = np.concatenate(zs)
    b = {}
    for name, sel in [("all_val", np.ones(len(y), bool)), ("kaggle_val", np.char.startswith(g, "kaggle")),
                      ("ffpp_val", np.char.startswith(g, "ffpp"))]:
        p0, pT = sig(z[sel]), sig(z[sel] / T)
        b[name] = {"n": int(sel.sum()), "auc": round(float(roc_auc_score(y[sel], z[sel])), 4),
                   "brier_before_T": round(float(brier_score_loss(y[sel], p0)), 4),
                   "brier_after_T": round(float(brier_score_loss(y[sel], pT)), 4),
                   "ece_before_T": ece(y[sel], p0), "ece_after_T": ece(y[sel], pT)}
    out["validation_calibration"] = b

    # ── 3. Confusion matrices + class-wise breakdown on TEST ─────────────────────────────────────
    ff = pd.read_csv(RES / "ffpp_test_logits.csv")      # logits already divided by T in the evaluator
    kg = pd.read_csv(RES / "kaggle_test_logits.csv")
    ff["p"], kg["p"] = sig(ff.logit), sig(kg.logit)
    cms = {}
    for name, d, t in [("kaggle_test", kg, thr["kaggle"]), ("ffpp_test_all", ff, thr["ffpp"])]:
        pr = d.p >= t
        cm = np.array([[int(((d.y == 0) & ~pr).sum()), int(((d.y == 0) & pr).sum())],
                       [int(((d.y == 1) & ~pr).sum()), int(((d.y == 1) & pr).sum())]])
        cms[name] = {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
                     "threshold": t, "brier_test": round(float(brier_score_loss(d.y, d.p)), 4), "ece_test": ece(d.y.values, d.p.values)}
        confusion_fig(cm, f"V12 {name} (thr {t})", FIG / f"confusion_{name}.png")
    out["test_confusion"] = cms
    cw = {}
    for cls in ["real"] + COHORTS:
        d = ff[ff.manip == cls]
        rate = float((d.p >= thr["ffpp"]).mean())
        cw[cls] = {"n_frames": int(len(d)), "n_videos": int(d.video_id.nunique()),
                   ("detected_as_fake" if cls != "real" else "passed_as_real"): round(rate if cls != "real" else 1 - rate, 4),
                   "mean_p_fake": round(float(d.p.mean()), 4)}
    out["ffpp_classwise_test"] = cw
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    names = ["real"] + COHORTS
    vals = [cw["real"]["passed_as_real"]] + [cw[c]["detected_as_fake"] for c in COHORTS]
    ax.bar(names, [v * 100 for v in vals], color=["#4c9a6a"] + ["#c0504d"] * 5)
    for i, v in enumerate(vals):
        ax.text(i, v * 100 + 1, f"{v * 100:.1f}%", ha="center", fontsize=9)
    ax.set_ylim(0, 105); ax.set_ylabel("% correct"); ax.set_title("V12 on FF++ test (unseen actors): per-class correct rate", fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=20)
    fig.tight_layout(); fig.savefig(FIG / "ffpp_classwise.png", dpi=160); plt.close(fig)

    # ── 1. Grad-CAM on TEST images chosen by a fixed rule ───────────────────────────────────────
    ff_sorted, kg_sorted = ff.sort_values("filepath"), kg.sort_values("filepath")
    picks = [("ffpp_fake_Deepfakes", ff_sorted[ff_sorted.manip == "Deepfakes"].iloc[0]),
             ("ffpp_real", ff_sorted[ff_sorted.manip == "real"].iloc[0]),
             ("kaggle_fake", kg_sorted[kg_sorted.y == 1].iloc[0]),
             ("kaggle_real", kg_sorted[kg_sorted.y == 0].iloc[0])]
    missed = ff_sorted[(ff_sorted.y == 1) & (ff_sorted.p < thr["ffpp"])]
    if len(missed):
        picks.append(("ffpp_fake_MISSED_" + missed.iloc[0].manip, missed.iloc[0]))
    gc = []
    for tag, row in picks:
        face = np.array(Image.open(row.filepath).convert("RGB"))
        overlay, _ = gradcam(model, tf, face, device)
        is_ffpp = "ffpp" in tag
        t = thr["ffpp"] if is_ffpp else thr["kaggle"]
        p = float(row.p)
        verdict = "FAKE" if p >= t else "REAL"
        truth = "fake" if int(row.y) == 1 else "real"
        fig, axs = plt.subplots(1, 2, figsize=(5.6, 3.0))
        axs[0].imshow(face); axs[0].set_title(f"input ({truth})", fontsize=9)
        axs[1].imshow(overlay); axs[1].set_title(f"Grad-CAM  p_fake={p:.3f} -> {verdict}", fontsize=9)
        for a in axs:
            a.axis("off")
        fig.tight_layout(); fig.savefig(FIG / f"gradcam_{tag}.png", dpi=160); plt.close(fig)
        gc.append({"figure": f"figures/gradcam_{tag}.png", "source": str(row.filepath), "truth": truth,
                   "p_fake_calibrated_tta": round(p, 4), "threshold": t, "verdict": verdict})
    out["gradcam_examples"] = gc
    out["gradcam_note"] = ("Grad-CAM of the spatial EfficientNet conv_head w.r.t. the binary logit (single, un-flipped view); "
                           "p_fake shown is the evaluator's calibrated TTA score. Images chosen by a fixed rule, not by hand.")

    (OUT / "extras.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
