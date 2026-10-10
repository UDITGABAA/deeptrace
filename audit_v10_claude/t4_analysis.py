"""
T4 — Statistical tests on cached logits from t3 (no GPU).

  A  reproduce the reported numbers (sanity: are we running the same thing?)
  B  threshold honesty: test-tuned vs frozen thresholds
  C  Kaggle test-fake memorisation (test fakes were in V1 training via D:\\deepfake_data)
  D  FF++: val-overlap, ancestor-seen frames, video-level, clustered CIs, permutation p
  E  dataset-identity confound (Kaggle vs FF++ source)
  F  robustness-pool decomposition
  G  horizontal-flip stability
  H  OOD sets
  I  calibration (ECE / Brier)
"""
import sys, os, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
R = Path(__file__).parent / "results"
TAG = sys.argv[1] if len(sys.argv) > 1 else "v10"
COH = ["Deepfakes", "FaceSwap", "FaceShifter", "Face2Face", "NeuralTextures"]
rng = np.random.default_rng(0)

sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -20, 20)))
r4 = lambda x: None if x is None else round(float(x), 4)


def auc(y, s):
    y = np.asarray(y)
    return r4(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None


def acc(y, p, t):
    return r4(np.mean((np.asarray(p) >= t) == np.asarray(y)))


def tuned_acc(y, p):  # exactly what evaluate_v10_full_battery.py does (uses TEST labels)
    best_t, best = 0.5, np.mean((p >= 0.5) == y)
    for t in np.linspace(0.1, 0.9, 81):
        a = np.mean((p >= t) == y)
        if a > best:
            best, best_t = a, t
    return r4(best), r4(best_t)


def fit_thresh(y, p):
    return tuned_acc(np.asarray(y), np.asarray(p))[1]


def cluster_boot_auc(df, B=2000):
    vids = df.video.unique()
    g = {v: d for v, d in df.groupby("video")}
    out = []
    for _ in range(B):
        s = pd.concat([g[v] for v in rng.choice(vids, len(vids))])
        if s.label.nunique() == 2:
            out.append(roc_auc_score(s.label, s.logit))
    return [r4(np.percentile(out, 2.5)), r4(np.percentile(out, 97.5))]


def cluster_perm_p(df, B=2000):
    """Permute labels at the VIDEO level (frames of a video share fate)."""
    obs = roc_auc_score(df.label, df.logit)
    vl = df.groupby("video").label.first()
    cnt = 0
    for _ in range(B):
        perm = dict(zip(vl.index, rng.permutation(vl.values)))
        if roc_auc_score(df.video.map(perm), df.logit) >= obs:
            cnt += 1
    return r4((cnt + 1) / (B + 1))


def ece(y, p, bins=15):
    y, p = np.asarray(y), np.asarray(p)
    conf = np.where(p >= 0.5, p, 1 - p); pred = (p >= 0.5).astype(int)
    e, edges = 0.0, np.linspace(0.5, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return r4(e)


df = pd.read_pickle(R / f"t3_logits_{TAG}.pkl")
df["p"] = sig(df.logit)
df["pathn"] = df.path.str.lower().map(lambda s: str(Path(s)).lower())
rep = {}

# ------------------------------------------------------------------ A reproduce
from utils.actor_splits import get_actor_disjoint_split
sp = get_actor_disjoint_split(pd.read_csv("manifests/ffpp_c23_manifest.csv"), "filepath")
ff = df[df.pool == "ffpp_test"].set_index("path")
test_reals = sp["test_reals"]["filepath"].tolist()
bench = {}
for m in COH:
    tf = sp["test_fakes"][m]["filepath"].tolist(); n = min(len(test_reals), len(tf))
    b = ff.loc[test_reals[:n] + tf[:n]].reset_index(); b["cohort"] = m
    bench[m] = b
kt = df[df.pool == "kaggle_test"]
A = {"kaggle_test": {"n": len(kt), "auc": auc(kt.label, kt.p), "acc@0.5": acc(kt.label, kt.p, 0.5),
                     "test_tuned_acc,thr": tuned_acc(kt.label.values, kt.p.values),
                     "brier": r4(brier_score_loss(kt.label, kt.p)), "ece": ece(kt.label, kt.p),
                     "reported": {"auc": 0.9999, "cal_acc": 0.9969, "acc_05": 0.9964, "brier": 0.0038, "ece": 0.0215}}}
for m in COH:
    b = bench[m]
    A[m] = {"n": len(b), "auc": auc(b.label, b.p), "acc@0.5": acc(b.label, b.p, 0.5),
            "test_tuned_acc,thr": tuned_acc(b.label.values, b.p.values)}
A["ffpp_macro_auc"] = r4(np.mean([A[m]["auc"] for m in COH]))
A["unreadable_images"] = int((df.ok == 0).sum())
rep["A_reproduction"] = A

# ------------------------------------------------------------------ B thresholds
kv = df[df.pool == "kaggle_valid_v10val"]
ffv = json.load(open(R / "t1_v10_val_samples.json"))["V10"]
ffv_paths = {str(Path(p)).lower() for p, y in ffv if "ffpp" in p.lower()}
ftr = df[df.pool == "ffpp_train"]
t_kval = fit_thresh(kv.label, kv.p)
t_fftrain = fit_thresh(ftr.label, ftr.p)
B = {"thr_fit_on_kaggle_valid(v10 val)": t_kval, "thr_fit_on_ffpp_train_actors": t_fftrain,
     "kaggle_test": {"acc@0.5": acc(kt.label, kt.p, 0.5), "acc@kaggle_valid_thr": acc(kt.label, kt.p, t_kval),
                     "acc@test_tuned(reported)": tuned_acc(kt.label.values, kt.p.values)[0]}}
for m in COH:
    b = bench[m]
    B[m] = {"acc@0.5": acc(b.label, b.p, 0.5), "acc@ffpp_train_thr": acc(b.label, b.p, t_fftrain),
            "acc@kaggle_valid_thr": acc(b.label, b.p, t_kval),
            "acc@test_tuned(reported)": tuned_acc(b.label.values, b.p.values)[0],
            "pct_real_called_fake@0.5": r4(np.mean(b[b.label == 0].p >= 0.5)),
            "pct_fake_called_fake@0.5": r4(np.mean(b[b.label == 1].p >= 0.5))}
rep["B_threshold_honesty"] = B

# ------------------------------------------------------------------ C Kaggle test-fake memorisation
kvall = df[df.pool.isin(["kaggle_valid_v10val", "kaggle_valid_unused"])]
ktr = df[df.pool.isin(["kaggle_train_v10used", "kaggle_train_v10unused"])]
C = {}
for nm, d in [("test", kt), ("valid", kvall), ("train", ktr)]:
    for y in (0, 1):
        s = d[d.label == y].logit
        C[f"{nm}_{'fake' if y else 'real'}"] = {"n": len(s), "mean_logit": r4(s.mean()), "median_logit": r4(s.median()),
                                                 "err_rate@0.5": r4(np.mean((sig(s) >= 0.5) != y))}
tf_, vf_ = kt[kt.label == 1].logit, kvall[kvall.label == 1].logit
tr_, vr_ = kt[kt.label == 0].logit, kvall[kvall.label == 0].logit
C["fake_logits_test_vs_valid_AUC(0.5=no memorisation)"] = auc(np.r_[np.ones(len(tf_)), np.zeros(len(vf_))], np.r_[tf_, vf_])
C["fake_logits_test_vs_valid_MWU_p"] = float(mannwhitneyu(tf_, vf_).pvalue)
C["real_logits_test_vs_valid_AUC(control)"] = auc(np.r_[np.ones(len(tr_)), np.zeros(len(vr_))], np.r_[tr_, vr_])
C["real_logits_test_vs_valid_MWU_p"] = float(mannwhitneyu(tr_, vr_).pvalue)
C["auc_valid_real_vs_valid_fake"] = auc(kvall.label, kvall.logit)
C["auc_test_real_vs_test_fake"] = auc(kt.label, kt.logit)
# bootstrap: fake error-rate test vs valid
fe_t = (sig(tf_) < 0.5).values; fe_v = (sig(vf_) < 0.5).values
C["fake_miss_rate_test"] = r4(fe_t.mean()); C["fake_miss_rate_valid"] = r4(fe_v.mean())
rep["C_kaggle_testfake_memorisation"] = C

# ------------------------------------------------------------------ D FF++ deep-dive
flags = json.load(open(R / "t1_benchmark_exposure_flags.json"))
D = {}
allb = pd.concat(bench.values())
allb["pathn"] = allb.path.map(lambda s: str(Path(s)).lower())
allb["in_val"] = allb.pathn.isin(ffv_paths)
allb["anc_seen"] = allb.pathn.map(lambda k: flags.get(k, {}).get("in_ancestor_train", False))
for m in COH:
    b = allb[allb.cohort == m]
    clean = b[~b.in_val]
    vid = b.groupby("video").agg(label=("label", "first"), logit=("logit", "mean")).reset_index()
    D[m] = {
        "frame_auc_reported_set": auc(b.label, b.logit),
        "frame_auc_95CI_video_cluster_boot": cluster_boot_auc(b[["video", "label", "logit"]], 1000),
        "video_level_auc": auc(vid.label, vid.logit), "n_videos": len(vid),
        "perm_p_video_level_labels": cluster_perm_p(b[["video", "label", "logit"]], 1000),
        "auc_excluding_v10_val_frames": auc(clean.label, clean.logit), "n_clean": len(clean),
        "auc_only_v10_val_frames": auc(b[b.in_val].label, b[b.in_val].logit),
        "auc_ancestor_seen_frames": auc(b[b.anc_seen].label, b[b.anc_seen].logit),
        "auc_ancestor_unseen_frames": auc(b[~b.anc_seen].label, b[~b.anc_seen].logit),
        "n_ancestor_seen": int(b.anc_seen.sum()),
    }
# memorisation probe independent of class balance: for FAKES, seen vs unseen mean logit
fk = allb[allb.label == 1]; rl = allb[allb.label == 0].drop_duplicates("path")
D["fake_mean_logit_seen_vs_unseen"] = [r4(fk[fk.anc_seen].logit.mean()), r4(fk[~fk.anc_seen].logit.mean())]
D["real_mean_logit_seen_vs_unseen"] = [r4(rl[rl.anc_seen].logit.mean()), r4(rl[~rl.anc_seen].logit.mean())]
D["fake_seen_vs_unseen_MWU_p"] = float(mannwhitneyu(fk[fk.anc_seen].logit, fk[~fk.anc_seen].logit).pvalue)
D["real_seen_vs_unseen_MWU_p"] = float(mannwhitneyu(rl[rl.anc_seen].logit, rl[~rl.anc_seen].logit).pvalue)
# full test set (all 4000 reals, all fakes), train-actor gap, mixed pairs, DFD
fft = df[df.pool == "ffpp_test"]
D["full_test_all_cohorts_auc"] = auc(fft.label, fft.logit)
D["train_actor_sample_auc(seen in V10 training)"] = auc(ftr.label, ftr.logit)
for m in COH:
    D.setdefault("per_cohort_train_vs_test_auc", {})[m] = [
        auc(ftr[ftr.manip.isin(["real", m])].label, ftr[ftr.manip.isin(["real", m])].logit),
        auc(fft[fft.manip.isin(["real", m])].label, fft[fft.manip.isin(["real", m])].logit)]
mp = pd.concat([df[df.pool == "ffpp_mixedpair"], fft[fft.label == 0]])
D["mixed_pair_fakes_vs_test_reals_auc"] = auc(mp.label, mp.logit)
dfd = pd.concat([df[df.pool == "ffpp_dfd"], fft[fft.label == 0]])
D["DFD_vs_ffpp_test_reals_auc(domain-confounded)"] = auc(dfd.label, dfd.logit)
rep["D_ffpp"] = D

# ------------------------------------------------------------------ E dataset identity
E = {}
kr = kt[kt.label == 0]; fr = fft[fft.label == 0]
E["mean_p_fake"] = {"kaggle_real": r4(kr.p.mean()), "kaggle_fake": r4(kt[kt.label == 1].p.mean()),
                    "ffpp_real": r4(fr.p.mean()), "ffpp_fake": r4(fft[fft.label == 1].p.mean()),
                    "newds_real": r4(df[(df.pool == "newds_test") & (df.label == 0)].p.mean()),
                    "newds_fake": r4(df[(df.pool == "newds_test") & (df.label == 1)].p.mean())}
E["pct_ffpp_REAL_called_fake@0.5"] = r4((fr.p >= 0.5).mean())
E["pct_kaggle_REAL_called_fake@0.5"] = r4((kr.p >= 0.5).mean())
E["auc_kaggle_real_vs_ffpp_real (source, both REAL)"] = auc(np.r_[np.zeros(len(kr)), np.ones(len(fr))], np.r_[kr.logit, fr.logit])
rep["E_dataset_identity"] = E

# ------------------------------------------------------------------ F robustness decomposition
F = {}
rp_f = R / f"t3_robustness_{TAG}.pkl"
if rp_f.exists():
    rp = pd.read_pickle(rp_f)
    for c in ["pristine", "jpeg_q50", "jpeg_q30", "noise_s025", "blur_k7"]:
        k, f = rp[rp.src == "kaggle"], rp[rp.src == "ffpp"]
        cross = pd.concat([k[k.label == 0], f[f.label == 1]])
        F[c] = {"pooled_auc(reported style)": auc(rp.label, rp[c]), "kaggle_only_auc": auc(k.label, k[c]),
                "ffpp_only_auc": auc(f.label, f[c]),
                "kaggle_real_vs_ffpp_fake_auc(cross-source pairs)": auc(cross.label, cross[c]),
                "ffpp_only_acc@0.5": acc(f.label, sig(f[c]), 0.5)}
    fp = pd.read_pickle(R / f"t3_ffpp_test_perturbed_{TAG}.pkl")
    base = fft.set_index("path").logit
    fp["pristine"] = fp.path.map(base)
    F["ffpp_full_test"] = {c: {"auc": auc(fp.label, fp[c]),
                               "per_cohort": {m: auc(fp[fp.manip.isin(["real", m])].label, fp[fp.manip.isin(["real", m])][c]) for m in COH}}
                           for c in ["pristine", "jpeg_q50", "jpeg_q30"]}
rep["F_robustness_decomposition"] = F

# ------------------------------------------------------------------ G flip stability
G = {}
for pool in ["kaggle_test", "ffpp_test", "test_data"]:
    d = df[(df.pool == pool) & df.logit_hflip.notna()]
    if len(d):
        G[pool] = {"pearson_r(logit,flip_logit)": r4(np.corrcoef(d.logit, d.logit_hflip)[0, 1]),
                   "decision_flip_rate@0.5": r4(np.mean((d.logit >= 0) != (d.logit_hflip >= 0))),
                   "mean_abs_dlogit": r4(np.mean(np.abs(d.logit - d.logit_hflip))),
                   "auc_orig": auc(d.label, d.logit), "auc_flipped": auc(d.label, d.logit_hflip)}
rep["G_flip_stability"] = G

# ------------------------------------------------------------------ H OOD
H = {}
for pool in ["newds_test", "test_data", "kaggle_valid_unused"]:
    d = df[df.pool == pool]
    if len(d):
        H[pool] = {"n": len(d), "auc": auc(d.label, d.p), "acc@0.5": acc(d.label, d.p, 0.5),
                   "acc@kaggle_valid_thr": acc(d.label, d.p, t_kval), "test_tuned_acc": tuned_acc(d.label.values, d.p.values)}
rep["H_ood"] = H

# ------------------------------------------------------------------ I calibration
I = {}
for nm, d in [("kaggle_test", kt), ("ffpp_test_full", fft), ("newds_test", df[df.pool == "newds_test"])]:
    if len(d):
        I[nm] = {"ece@15bins": ece(d.label, d.p), "brier": r4(brier_score_loss(d.label, d.p))}
rep["I_calibration"] = I

json.dump(rep, open(R / f"t4_analysis_{TAG}.json", "w"), indent=2, default=str)
print(json.dumps(rep, indent=2, default=str))
