"""
T2 — Duplicate / near-duplicate leakage + trivial-metadata shortcut test (no model).

For Kaggle real-vs-fake (train/valid/test) and FF++ c23 frames:
  * exact duplicates (MD5) across splits and across labels
  * near duplicates (64-bit dHash, Hamming <= 4) between test and train/valid
  * file metadata (bytes, W, H, JPEG quantisation tables, subsampling, pixel stats)
  * "dumb classifier": gradient boosting on metadata only, fit on train split,
    scored on the test split. A high AUC means the labels can be predicted
    without looking at faces at all -> the benchmark has a shortcut.
"""
import sys, os, json, hashlib
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)


def featurize(path):
    try:
        b = open(path, "rb").read()
        md5 = hashlib.md5(b).hexdigest()
        im = Image.open(path)
        w, h = im.size
        q = getattr(im, "quantization", None) or {}
        q0 = float(np.sum(q.get(0, [0]))); q1 = float(np.sum(q.get(1, [0]))) if 1 in q else 0.0
        try:
            from PIL import JpegImagePlugin
            sub = JpegImagePlugin.get_sampling(im)
        except Exception:
            sub = -2
        im.draft("RGB", (64, 64))
        rgb = np.asarray(im.convert("RGB").resize((64, 64)), dtype=np.float32)
        g = np.asarray(Image.fromarray(rgb.astype(np.uint8)).convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
        bits = (g[:, 1:] > g[:, :-1]).flatten()
        dh = int("".join("1" if x else "0" for x in bits), 2)
        lap = np.abs(np.diff(rgb.mean(2), axis=0)).mean()
        return dict(path=path, md5=md5, bytes=len(b), w=w, h=h, q0=q0, q1=q1, sub=sub, dhash=str(dh),
                    mean_r=rgb[..., 0].mean(), mean_g=rgb[..., 1].mean(), mean_b=rgb[..., 2].mean(),
                    std=rgb.std(), grad=lap, ok=1)
    except Exception as e:
        return dict(path=path, ok=0, err=repr(e))


def near_dups(test_df, ref_df, max_ham=4):
    """Pigeonhole multi-index: Hamming<=3 guaranteed found via 4x16-bit exact chunk match; we verify <=max_ham."""
    from collections import defaultdict
    idx = [defaultdict(list) for _ in range(4)]
    ref_h = ref_df["dhash"].astype(object).tolist()
    for i, hsh in enumerate(ref_h):
        for c in range(4):
            idx[c][(hsh >> (16 * c)) & 0xFFFF].append(i)
    hits = []
    for tp, th in zip(test_df["path"], test_df["dhash"].astype(object)):
        cands = set()
        for c in range(4):
            cands.update(idx[c].get((th >> (16 * c)) & 0xFFFF, []))
        best = None
        for i in cands:
            d = bin(th ^ ref_h[i]).count("1")
            if d <= max_ham and (best is None or d < best[0]):
                best = (d, i)
        if best:
            hits.append((tp, ref_df["path"].iloc[best[1]], best[0]))
    return hits


def dumb_classifier(train_df, test_df, feats):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    clf = HistGradientBoostingClassifier(max_iter=200)
    clf.fit(train_df[feats].values, train_df["label"].values)
    p = clf.predict_proba(test_df[feats].values)[:, 1]
    out = {"auc_all_meta": round(float(roc_auc_score(test_df["label"], p)), 4)}
    for f in feats:  # single-feature AUCs (direction-agnostic)
        a = roc_auc_score(test_df["label"], test_df[f])
        out[f"auc_{f}"] = round(float(max(a, 1 - a)), 4)
    return out


if __name__ == "__main__":
    report = {}
    K = Path("data/kaggle_realfake/real_vs_fake/real-vs-fake")
    rows = []
    for s in ["train", "valid", "test"]:
        for c, y in [("real", 0), ("fake", 1)]:
            for p in sorted((K / s / c).glob("*.jpg")):
                rows.append((str(p), s, y, "kaggle"))
    ext = Path(r"D:\deepfake_data\kaggle_realfake\real_vs_fake\real-vs-fake\train\fake")
    if ext.exists():
        for p in sorted(ext.glob("*")):
            rows.append((str(p), "ext_stylegan_dir", 1, "kaggle_ext"))
    ff = pd.read_csv("manifests/ffpp_c23_manifest.csv")
    from utils.actor_splits import get_actor_disjoint_split
    sp = get_actor_disjoint_split(ff, "filepath")
    tr = set(sp["train_reals"]["filepath"]); te = set(sp["test_reals"]["filepath"])
    for m in sp["train_fakes"]:
        tr |= set(sp["train_fakes"][m]["filepath"]); te |= set(sp["test_fakes"][m]["filepath"])
    for p, y, m in zip(ff["filepath"], ff["label"], ff["manipulation_type"]):
        s = "ff_train" if p in tr else ("ff_test" if p in te else "ff_other")
        rows.append((p, s, int(y), "ffpp_" + m))

    meta = pd.DataFrame(rows, columns=["path", "split", "label", "source"])
    cache = OUT / "t2_features.pkl"
    if cache.exists():
        feats = pd.read_pickle(cache)
    else:
        with Pool(8) as pool:
            res = pool.map(featurize, meta["path"].tolist(), chunksize=256)
        feats = pd.DataFrame(res)
        feats["dhash"] = feats["dhash"].fillna("0").astype(str)
        feats.drop(columns=[c for c in ["err"] if c in feats]).to_pickle(cache)
        feats = pd.read_pickle(cache)
    feats["dhash"] = feats["dhash"].map(lambda x: int(x))
    df = meta.merge(feats, on="path", how="left")
    report["unreadable_files"] = int((df["ok"] != 1).sum())
    df = df[df["ok"] == 1].copy()

    # ---------- exact duplicates
    dup = df.groupby("md5").agg(n=("path", "size"), splits=("split", lambda s: tuple(sorted(set(s)))),
                                labels=("label", lambda s: tuple(sorted(set(s)))), src=("source", lambda s: tuple(sorted(set(s)))))
    dup = dup[dup.n > 1]
    report["exact_dup_groups"] = int(len(dup))
    report["exact_dup_cross_split_groups"] = int((dup.splits.map(len) > 1).sum())
    report["exact_dup_conflicting_label_groups"] = int((dup.labels.map(len) > 1).sum())
    report["exact_dup_split_combos"] = {str(k): int(v) for k, v in dup.splits.value_counts().head(20).items()}
    dup.reset_index().head(200).to_csv(OUT / "t2_exact_dups_sample.csv", index=False)

    # ---------- near duplicates (Kaggle test vs train+valid; FF++ test vs train)
    kt = df[(df.source == "kaggle") & (df.split == "test")]
    kr = df[(df.source.str.startswith("kaggle")) & (df.split != "test")]
    nd = near_dups(kt, kr)
    report["kaggle_test_near_dups_in_train_or_valid(ham<=4)"] = len(nd)
    pd.DataFrame(nd, columns=["test", "ref", "ham"]).to_csv(OUT / "t2_kaggle_near_dups.csv", index=False)
    ft = df[df.split == "ff_test"]; fr = df[df.split == "ff_train"]
    nd2 = near_dups(ft, fr)
    report["ffpp_test_near_dups_in_train(ham<=4)"] = len(nd2)
    pd.DataFrame(nd2, columns=["test", "ref", "ham"]).to_csv(OUT / "t2_ffpp_near_dups.csv", index=False)

    # ---------- metadata distributions
    F = ["bytes", "w", "h", "q0", "q1", "sub", "mean_r", "mean_g", "mean_b", "std", "grad"]
    desc = {}
    for (src, y), g in df.groupby(["source", "label"]):
        desc[f"{src}|label={y}"] = {f: {"mean": round(float(g[f].mean()), 3), "std": round(float(g[f].std()), 3),
                                        "uniq": int(g[f].nunique())} for f in ["bytes", "w", "h", "q0", "q1", "sub", "grad"]}
    report["metadata_by_source_label"] = desc

    # ---------- dumb classifiers
    kd = df[df.source == "kaggle"]
    report["dumb_clf_kaggle(train->test)"] = dumb_classifier(kd[kd.split == "train"], kd[kd.split == "test"], F)
    fd = df[df.source.str.startswith("ffpp") & (df.source != "ffpp_DeepFakeDetection")]
    report["dumb_clf_ffpp_all(train_actors->test_actors)"] = dumb_classifier(fd[fd.split == "ff_train"], fd[fd.split == "ff_test"], F)
    for m in ["Deepfakes", "FaceSwap", "FaceShifter", "Face2Face", "NeuralTextures"]:
        sub = fd[fd.source.isin(["ffpp_real", "ffpp_" + m])]
        report[f"dumb_clf_ffpp_{m}"] = dumb_classifier(sub[sub.split == "ff_train"], sub[sub.split == "ff_test"], F)
    # cross-source: can metadata tell Kaggle-real from FF++-real? (dataset identity)
    xs = df[df.source.isin(["kaggle", "ffpp_real"]) & (df.label == 0)].copy()
    xs["label"] = (xs.source == "ffpp_real").astype(int)
    xs_tr = xs[xs.split.isin(["train", "ff_train"])]; xs_te = xs[xs.split.isin(["test", "ff_test"])]
    report["dumb_clf_dataset_identity_real_only"] = dumb_classifier(xs_tr, xs_te, F)

    json.dump(report, open(OUT / "t2_duplicates_and_shortcuts.json", "w"), indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))
