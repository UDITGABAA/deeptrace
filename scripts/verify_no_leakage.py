"""
Post-hoc Verification: Confirms Zero Actor Leakage Between Train and Test.

Validates BOTH:
(a) The split utility's logic (code-correctness check)
(b) The actual manifest indices that would go into training — matching
    the exact [:N] slicing logic from train_v7_sota_spectral.py

Run BEFORE and AFTER training to confirm no contamination path exists.
"""

import sys
import logging
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.actor_splits import get_actor_split, get_actor_disjoint_split, _extract_video_id, _extract_actor_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("leakage_verification")


def main():
    manifest_path = Path("manifests/ffpp_c23_manifest.csv")
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return

    df = pd.read_csv(manifest_path)
    col = "filepath" if "filepath" in df.columns else "image_path"

    train_actors, test_actors = get_actor_split()
    split = get_actor_disjoint_split(df, filepath_col=col)

    all_passed = True

    # ═══ CHECK 1: Actor sets are disjoint ═══
    overlap = train_actors & test_actors
    if len(overlap) == 0:
        logger.info("✅ CHECK 1 PASS: train_actors ∩ test_actors = ∅")
    else:
        logger.error(f"❌ CHECK 1 FAIL: {len(overlap)} actors in both sets: {sorted(overlap)[:10]}")
        all_passed = False

    # ═══ CHECK 2: Every test fake has both src AND tgt in test_actors ═══
    for manip, test_df in split["test_fakes"].items():
        test_df = test_df.copy()
        test_df["video_id"] = test_df[col].apply(_extract_video_id)
        violations = 0
        for vid in test_df["video_id"].unique():
            src, tgt = _extract_actor_ids(vid)
            if src not in test_actors or tgt not in test_actors:
                violations += 1
        if violations == 0:
            logger.info(f"✅ CHECK 2 PASS [{manip}]: All {test_df['video_id'].nunique()} test fake videos have both actors in test set")
        else:
            logger.error(f"❌ CHECK 2 FAIL [{manip}]: {violations} test fake videos have actors in training set")
            all_passed = False

    # ═══ CHECK 3: Every train fake has both src AND tgt in train_actors ═══
    for manip, train_df in split["train_fakes"].items():
        train_df = train_df.copy()
        train_df["video_id"] = train_df[col].apply(_extract_video_id)
        violations = 0
        for vid in train_df["video_id"].unique():
            src, tgt = _extract_actor_ids(vid)
            if src not in train_actors or tgt not in train_actors:
                violations += 1
        if violations == 0:
            logger.info(f"✅ CHECK 3 PASS [{manip}]: All {train_df['video_id'].nunique()} train fake videos have both actors in train set")
        else:
            logger.error(f"❌ CHECK 3 FAIL [{manip}]: {violations} train fake videos have actors in test set")
            all_passed = False

    # ═══ CHECK 4: Train reals contain ONLY train actors ═══
    train_real_actors = set(split["train_reals"][col].apply(_extract_video_id).unique())
    leak = train_real_actors & test_actors
    if len(leak) == 0:
        logger.info(f"✅ CHECK 4 PASS: All {len(train_real_actors)} train real actors are in train set")
    else:
        logger.error(f"❌ CHECK 4 FAIL: {len(leak)} train real actors are in test set: {sorted(leak)[:10]}")
        all_passed = False

    # ═══ CHECK 5: Test reals contain ONLY test actors ═══
    test_real_actors = set(split["test_reals"][col].apply(_extract_video_id).unique())
    leak = test_real_actors & train_actors
    if len(leak) == 0:
        logger.info(f"✅ CHECK 5 PASS: All {len(test_real_actors)} test real actors are in test set")
    else:
        logger.error(f"❌ CHECK 5 FAIL: {len(leak)} test real actors are in train set: {sorted(leak)[:10]}")
        all_passed = False

    # ═══ CHECK 6: DFD is excluded from all cohort evaluations ═══
    dfd_count = len(split["dfd_excluded"])
    if dfd_count > 0:
        logger.info(f"✅ CHECK 6 PASS: DFD excluded ({dfd_count} frames set aside, not in any train/test split)")
    else:
        logger.warning("⚠️ CHECK 6 WARN: No DFD frames found in manifest")

    # ═══ CHECK 7: Validate the OLD training script's contamination ═══
    logger.info("\n═══ HISTORICAL CONTAMINATION CHECK (old [:5000]/[:2000] slicing) ═══")
    df["video_id"] = df[col].apply(_extract_video_id)

    old_train_reals = df[df["manipulation_type"] == "real"].head(5000)
    old_train_real_actors = set(old_train_reals["video_id"].unique())
    old_leak = old_train_real_actors & test_actors
    logger.info(
        f"Old [:5000] real slice: {len(old_train_real_actors)} actors, "
        f"{len(old_leak)} in current test set → "
        f"{'❌ CONTAMINATED' if old_leak else '✅ CLEAN'}"
    )

    for manip in ["Deepfakes", "FaceSwap", "FaceShifter", "Face2Face", "NeuralTextures"]:
        old_fakes = df[df["manipulation_type"] == manip].head(2000)
        old_fake_vids = old_fakes["video_id"].unique()
        contaminated = 0
        for vid in old_fake_vids:
            src, tgt = _extract_actor_ids(vid)
            if src in test_actors or tgt in test_actors:
                contaminated += 1
        logger.info(
            f"Old [:2000] {manip}: {len(old_fake_vids)} vids, "
            f"{contaminated} contaminated → "
            f"{'❌ CONTAMINATED' if contaminated else '✅ CLEAN'}"
        )

    # ═══ CHECK 8: Checkpoint lineage (a clean split is useless if weights descend from a dirty run) ═══
    from utils.leakage_guard import assert_clean_lineage
    for ckpt in ["checkpoints/v11_clean/best_model.pth", "checkpoints/v10_robust/best_model.pth",
                 "checkpoints/v9_actor_disjoint/best_model.pth"]:
        if Path(ckpt).exists():
            try:
                assert_clean_lineage(ckpt)
                logger.info(f"✅ CHECK 8 PASS: {ckpt} has clean_lineage=True")
            except RuntimeError:
                logger.error(f"❌ CHECK 8: {ckpt} has NO clean lineage (descends from runs trained on test "
                             f"actors / Kaggle test fakes) — do not report it as actor-disjoint")

    # ═══ FINAL VERDICT ═══
    if all_passed:
        logger.info("\n" + "=" * 60)
        logger.info("🟢 SPLIT CHECKS PASSED — zero actor overlap in the split DEFINITION")
        logger.info("   (this does not cover weights: see CHECK 8, and audit_v10_claude/AUDIT_REPORT.md)")
        logger.info("=" * 60)
    else:
        logger.error("\n" + "=" * 60)
        logger.error("🔴 CHECKS FAILED — Actor leakage detected")
        logger.error("=" * 60)


if __name__ == "__main__":
    main()
