"""
T1 — Static split / lineage leakage audit (no GPU, no model).

Reconstructs the exact sample lists used by every training script in the
V10 weight lineage (V3 -> V4 -> V5 -> V7 -> V9 -> V10) by importing their own
builder functions, then measures how much of the V10 FF++ "actor-disjoint"
benchmark was (a) used for V10 checkpoint selection and (b) trained on by an
ancestor checkpoint that V10 was warm-started from.

Run from repo root:  venv\\Scripts\\python.exe audit_v10_claude\\t1_split_lineage_audit.py
"""
import sys, json, random, logging
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)
logging.disable(logging.INFO)

import pandas as pd
from utils.actor_splits import get_actor_disjoint_split, get_actor_split, _extract_video_id, _extract_actor_ids

OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)
COHORTS = ["Deepfakes", "FaceSwap", "FaceShifter", "Face2Face", "NeuralTextures"]


def norm(p):
    return str(Path(p)).lower()


def actors_of(path):
    vid = _extract_video_id(path)
    a, b = _extract_actor_ids(vid)
    return {a, b}


df = pd.read_csv("manifests/ffpp_c23_manifest.csv")
split = get_actor_disjoint_split(df, filepath_col="filepath")
train_actors, test_actors = split["train_actors"], split["test_actors"]
report = {}

# ---------------------------------------------------------------- 1. split sanity
r = {}
r["actor_overlap"] = len(train_actors & test_actors)
r["n_train_actors"], r["n_test_actors"] = len(train_actors), len(test_actors)
r["test_reals_frames"] = len(split["test_reals"])
r["train_reals_frames"] = len(split["train_reals"])
for m in COHORTS:
    tf, ef = split["train_fakes"][m], split["test_fakes"][m]
    leak = 0
    for p in ef["filepath"]:
        if actors_of(p) & train_actors:
            leak += 1
    r[m] = {"train_frames": len(tf), "test_frames": len(ef), "discarded": len(split["discarded_fakes"][m]),
            "test_frames_with_train_actor": leak,
            "test_videos": ef["filepath"].map(_extract_video_id).nunique()}
# Frames per video / manifest sanity
df["vid"] = df["filepath"].map(_extract_video_id)
r["frames_per_video"] = df.groupby(["manipulation_type", "vid"]).size().describe().to_dict()
report["1_actor_split_sanity"] = r

# ---------------------------------------------------------------- 2. reconstruct V10 benchmark sets exactly
test_reals = split["test_reals"]["filepath"].tolist()
bench = {}  # cohort -> list[(path,label)]
for m in COHORTS:
    tf = split["test_fakes"][m]["filepath"].tolist()
    n = min(len(test_reals), len(tf))
    bench[m] = [(p, 0) for p in test_reals[:n]] + [(p, 1) for p in tf[:n]]
bench_paths = {norm(p) for v in bench.values() for p, _ in v}

robust_ffpp = [(p, 0) for p in test_reals[:1000]]
for m in COHORTS:
    robust_ffpp += [(p, 1) for p in split["test_fakes"][m]["filepath"].tolist()[:200]]

# ---------------------------------------------------------------- 3. reconstruct each lineage training set
lineage = {}
from scripts.train_v10_robust import build_v10_dataset
tr10, va10 = build_v10_dataset(max_kaggle=10000)
lineage["V10"] = (tr10, va10)
from scripts.train_v9_actor_disjoint import build_v9_dataset
lineage["V9"] = build_v9_dataset()
from scripts.train_v7_sota_spectral import build_v7_dataset
lineage["V7"] = build_v7_dataset()
from scripts.train_v5_srm_residual import build_srm_dataset
lineage["V5"] = build_srm_dataset()
from scripts.train_v4_seam_hardmining import build_hard_mined_dataset
lineage["V4"] = build_hard_mined_dataset()
import scripts.train_v3_e2e as v3
_orig = v3.build_datasets
# build_datasets also builds DataLoaders; we only need the sample lists -> patch DataLoader-creating parts
try:
    res = _orig()
    # function returns loaders/datasets; dig out sample lists
    def dig(o):
        if hasattr(o, "dataset"): o = o.dataset
        return getattr(o, "samples", None)
    lineage["V3"] = (dig(res[0]), dig(res[1]))
except Exception as e:
    report["V3_reconstruct_error"] = repr(e)

r = {}
for name, (tr, va) in lineage.items():
    if tr is None:
        continue
    tr_set = {norm(s[0]) for s in tr}
    va_set = {norm(s[0]) for s in va}
    tr_ff = [s for s in tr if "ffpp" in norm(s[0])]
    va_ff = [s for s in va if "ffpp" in norm(s[0])]
    tr_actors = set().union(*[actors_of(s[0]) for s in tr_ff]) if tr_ff else set()
    va_actors = set().union(*[actors_of(s[0]) for s in va_ff]) if va_ff else set()
    entry = {
        "n_train": len(tr), "n_val": len(va),
        "ffpp_train_frames": len(tr_ff), "ffpp_val_frames": len(va_ff),
        "train_val_exact_overlap": len(tr_set & va_set),
        "val_frames_also_in_train_pct": round(100 * len(tr_set & va_set) / max(1, len(va_set)), 2),
        "test_actors_seen_in_train": len(tr_actors & test_actors),
        "test_actors_seen_in_val": len(va_actors & test_actors),
        "benchmark_frames_in_train": len(bench_paths & tr_set),
        "benchmark_frames_in_val": len(bench_paths & va_set),
        "train_labels": {"real": sum(1 for s in tr if s[1] == 0), "fake": sum(1 for s in tr if s[1] == 1)},
        "kaggle_test_in_train": sum(1 for s in tr if "real-vs-fake\\test" in norm(s[0]) or "real-vs-fake/test" in norm(s[0])),
        "kaggle_test_in_val": sum(1 for s in va if "real-vs-fake\\test" in norm(s[0]) or "real-vs-fake/test" in norm(s[0])),
    }
    # per cohort benchmark exposure
    entry["benchmark_exposure_by_cohort"] = {
        m: {"in_train": sum(1 for p, _ in bench[m] if norm(p) in tr_set),
            "in_val": sum(1 for p, _ in bench[m] if norm(p) in va_set),
            "n": len(bench[m])} for m in COHORTS}
    r[name] = entry
report["2_lineage_exposure"] = r

# union of ancestor training sets (weights V10 inherits)
anc = set()
for name in ["V3", "V4", "V5", "V7", "V9"]:
    if name in lineage and lineage[name][0] is not None:
        anc |= {norm(s[0]) for s in lineage[name][0]}
anc_videos = {_extract_video_id(p) for p in anc if "ffpp" in p}
bench_vids = {_extract_video_id(p) for p in bench_paths}
report["3_ancestor_union"] = {
    "benchmark_frames_seen_by_any_ancestor_training": len(bench_paths & anc),
    "benchmark_frames_total_unique": len(bench_paths),
    "benchmark_videos_with_any_frame_in_ancestor_training": len(bench_vids & anc_videos),
    "benchmark_videos_total": len(bench_vids),
    "v10_val_frames_in_benchmark": len(bench_paths & {norm(s[0]) for s in va10}),
}

# save per-frame exposure flags for T3 (model eval) to stratify on
flags = {}
v10_val = {norm(s[0]) for s in va10}
for m in COHORTS:
    for p, y in bench[m]:
        k = norm(p)
        flags[k] = {"in_v10_val": k in v10_val, "in_ancestor_train": k in anc,
                    "video_in_ancestor_train": _extract_video_id(p) in anc_videos}
json.dump(flags, open(OUT / "t1_benchmark_exposure_flags.json", "w"))
json.dump({k: [(s[0], s[1]) for s in v[1]] for k, v in {"V10": lineage["V10"]}.items()},
          open(OUT / "t1_v10_val_samples.json", "w"))
json.dump([(s[0], s[1]) for s in tr10], open(OUT / "t1_v10_train_samples.json", "w"))

# ---------------------------------------------------------------- 4. V10 SBI / label sanity
r = {}
r["sbi_applied_to_label"] = "fake (binary_label == 1) — see train_v10_robust.py:157; reals never get SBI, label never flipped"
r["v10_train_ffpp_fake_weight_vs_real"] = "fakes weighted 2.0-3.5, reals 1.2 (focal BCE)"
report["4_v10_sbi"] = r

json.dump(report, open(OUT / "t1_split_lineage_audit.json", "w"), indent=2, default=str)
print(json.dumps(report, indent=2, default=str))
