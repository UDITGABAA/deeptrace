"""
T5 — Memorisation / identity-shortcut probes (no GPU; uses t3 cache).

1. Within-video paired test: for FF++ test videos that have BOTH frames an
   ancestor checkpoint trained on AND frames it never saw, compare the same
   video's scores.  Removes "those videos are just easier" confounding.
2. Mixed-pair fakes: FF++ fake id is <target>_<source> (target video supplies
   the scene/body, source supplies the face).  If the model recognises scenes
   it saw as REAL in training, fakes whose TARGET is a train actor should be
   scored as real.
"""
import sys, os, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
R = Path(__file__).parent / "results"
from utils.actor_splits import get_actor_split, _extract_video_id

train_actors, test_actors = get_actor_split()
df = pd.read_pickle(R / "t3_logits_v10.pkl")
flags = json.load(open(R / "t1_benchmark_exposure_flags.json"))

# rebuild ancestor-train union for ALL ffpp_test frames (flags only cover the benchmark subset)
import logging; logging.disable(logging.INFO)
from scripts.train_v4_seam_hardmining import build_hard_mined_dataset
from scripts.train_v5_srm_residual import build_srm_dataset
import scripts.train_v3_e2e as v3
anc = set()
for tr, _ in [build_hard_mined_dataset(), build_srm_dataset()]:
    anc |= {str(Path(s[0])).lower() for s in tr}
anc |= {str(Path(s[0])).lower() for s in v3.build_datasets()[0].samples}

out = {}
ft = df[df.pool == "ffpp_test"].copy()
ft["seen"] = ft.path.map(lambda p: str(Path(p)).lower() in anc)
for lab, name in [(1, "fakes"), (0, "reals")]:
    d = ft[ft.label == lab]
    g = d.groupby(["video", "seen"]).logit.mean().unstack()
    g = g.dropna()
    diff = (g[True] - g[False])
    out[f"within_video_{name}"] = {
        "n_videos_with_both": int(len(g)),
        "mean_logit_seen": round(float(g[True].mean()), 4), "mean_logit_unseen": round(float(g[False].mean()), 4),
        "mean_diff(seen-unseen)": round(float(diff.mean()), 4),
        "frac_videos_seen_higher": round(float((diff > 0).mean()), 4),
        "wilcoxon_p": float(wilcoxon(diff).pvalue) if len(diff) > 5 else None,
        "expected_sign_if_memorised": "+ (more fake)" if lab == 1 else "- (more real)",
    }

mp = df[df.pool == "ffpp_mixedpair"].copy()
mp["tgt"] = mp.video.str.split("_").str[0]; mp["src"] = mp.video.str.split("_").str[1]
mp["case"] = np.where(mp.tgt.isin(train_actors), "target(scene)=TRAIN actor, face=test actor",
                      "target(scene)=test actor, face=TRAIN actor")
reals = ft[ft.label == 0]
res = {}
for c, g in mp.groupby("case"):
    s = pd.concat([g, reals])
    res[c] = {"n_fake_frames": len(g), "mean_p_fake": round(float((1 / (1 + np.exp(-g.logit))).mean()), 4),
              "auc_vs_test_reals": round(float(roc_auc_score(s.label, s.logit)), 4)}
res["reference_test_fakes_mean_p_fake"] = round(float((1 / (1 + np.exp(-ft[ft.label == 1].logit))).mean()), 4)
res["reference_test_reals_mean_p_fake"] = round(float((1 / (1 + np.exp(-reals.logit))).mean()), 4)
tr = df[df.pool == "ffpp_train"]
res["reference_train_reals_mean_p_fake"] = round(float((1 / (1 + np.exp(-tr[tr.label == 0].logit))).mean()), 4)
res["reference_train_fakes_mean_p_fake"] = round(float((1 / (1 + np.exp(-tr[tr.label == 1].logit))).mean()), 4)
out["mixed_pair_identity_probe"] = res

json.dump(out, open(R / "t5_memorisation_probes.json", "w"), indent=2)
print(json.dumps(out, indent=2))
