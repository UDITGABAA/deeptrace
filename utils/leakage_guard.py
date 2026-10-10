"""
Leakage guard: hard checks that training data never overlaps evaluation data.

Added after the 2026-10 audit (audit_v10_claude/AUDIT_REPORT.md), which found that
10,000 Kaggle test fakes had been copied into a training folder under new filenames.
A filename check could not catch that; these checks compare FILE CONTENT (MD5).

    assert_no_content_overlap(train_paths, {"val": val_paths, "test": test_paths})
    assert_actor_disjoint(train_ffpp_paths, {"val": ..., "test": ...})
    assert_clean_lineage(checkpoint_path)   # refuses warm-starts from contaminated checkpoints
"""
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable

import torch

from utils.actor_splits import _extract_actor_ids, _extract_video_id

logger = logging.getLogger(__name__)

CACHE_FILE = Path(__file__).resolve().parent.parent / "data" / ".md5_cache.json"
CLEAN_LINEAGE_KEY = "clean_lineage"


def _load_cache() -> dict:
    try:
        return json.loads(CACHE_FILE.read_text())
    except Exception:
        return {}


def md5_many(paths: Iterable[str]) -> Dict[str, str]:
    """MD5 per path, cached on (path, size, mtime) so repeated runs are fast."""
    cache = _load_cache()
    out, dirty = {}, False
    for p in paths:
        st = os.stat(p)
        key = f"{os.path.normcase(os.path.abspath(p))}|{st.st_size}|{int(st.st_mtime)}"
        h = cache.get(key)
        if h is None:
            with open(p, "rb") as f:
                h = hashlib.md5(f.read()).hexdigest()
            cache[key] = h
            dirty = True
        out[p] = h
    if dirty:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(cache))
    return out


def assert_no_content_overlap(train_paths: Iterable[str], eval_sets: Dict[str, Iterable[str]]) -> dict:
    """Raise if any training file is byte-identical to any evaluation file."""
    train_paths = list(train_paths)
    train_h = md5_many(train_paths)
    train_set = set(train_h.values())
    report = {"n_train": len(train_paths)}
    for name, paths in eval_sets.items():
        paths = list(paths)
        h = md5_many(paths)
        dup = [p for p, v in h.items() if v in train_set]
        report[name] = {"n": len(paths), "content_duplicates_in_train": len(dup)}
        if dup:
            raise RuntimeError(
                f"LEAKAGE: {len(dup)} {name} files are byte-identical to training files, e.g. {dup[:3]}"
            )
    logger.info(f"Leakage guard (MD5): OK {report}")
    return report


def _actors(paths: Iterable[str]) -> set:
    out = set()
    for p in paths:
        a, b = _extract_actor_ids(_extract_video_id(p))
        out.update((a, b))
    return out


def assert_actor_disjoint(train_ffpp_paths: Iterable[str], eval_sets: Dict[str, Iterable[str]]) -> dict:
    """Raise if any FF++ actor appearing in training also appears in an evaluation set."""
    tr = _actors(train_ffpp_paths)
    report = {"n_train_actors": len(tr)}
    for name, paths in eval_sets.items():
        ov = tr & _actors(paths)
        report[name] = {"shared_actors": len(ov)}
        if ov:
            raise RuntimeError(f"LEAKAGE: {len(ov)} FF++ actors shared between train and {name}, e.g. {sorted(ov)[:5]}")
    logger.info(f"Leakage guard (actors): OK {report}")
    return report


def assert_clean_lineage(checkpoint_path: str):
    """
    Refuse to warm-start from a checkpoint unless it was produced by a guarded
    training run (which stores clean_lineage=True). V1-V10 checkpoints all descend
    from runs that trained on Kaggle test fakes and/or FF++ test actors.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not ckpt.get(CLEAN_LINEAGE_KEY, False):
        raise RuntimeError(
            f"Refusing warm start from {checkpoint_path}: no '{CLEAN_LINEAGE_KEY}' flag. "
            "Pre-V11 checkpoints are contaminated (see audit_v10_claude/AUDIT_REPORT.md)."
        )
