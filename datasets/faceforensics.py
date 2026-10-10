"""
FaceForensics++ dataset loader.
Expected structure:
    data/FaceForensics++/
        original_sequences/youtube/c23/videos/*.mp4
        manipulated_sequences/Deepfakes/c23/videos/*.mp4
        manipulated_sequences/Face2Face/c23/videos/*.mp4
        manipulated_sequences/FaceSwap/c23/videos/*.mp4
        manipulated_sequences/NeuralTextures/c23/videos/*.mp4
"""

from pathlib import Path
import logging
from datasets.base_dataset import BaseDeepfakeDataset

logger = logging.getLogger(__name__)

MANIPULATION_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


class FaceForensicsDataset(BaseDeepfakeDataset):
    """FaceForensics++ dataset."""

    def __init__(self, root_dir: str, compression: str = "c23", **kwargs):
        self.compression = compression
        super().__init__(root_dir, **kwargs)

    def _load_samples(self) -> list:
        samples = []
        ff_root = self.root_dir / "FaceForensics++"

        # Real videos
        real_dir = ff_root / "original_sequences" / "youtube" / self.compression / "videos"
        if real_dir.exists():
            for vid in sorted(real_dir.glob("*.mp4")):
                samples.append({
                    "path": str(vid),
                    "label": 0,
                    "manipulation_type": "real",
                })

        # Manipulated videos
        for method in MANIPULATION_METHODS:
            manip_dir = ff_root / "manipulated_sequences" / method / self.compression / "videos"
            if manip_dir.exists():
                for vid in sorted(manip_dir.glob("*.mp4")):
                    samples.append({
                        "path": str(vid),
                        "label": 1,
                        "manipulation_type": method,
                    })

        # Split
        samples = self._apply_split(samples)
        return samples

    def _apply_split(self, samples: list) -> list:
        """DEPRECATED: This split is broken (sequential slice on concat list).
        Use utils/actor_splits.py and datasets/ffpp_zeroshot.py instead."""
        raise RuntimeError(
            "FaceForensicsDataset._apply_split() is DEPRECATED due to a critical "
            "sequential-slice bug. Use utils/actor_splits.py for FF++ partitioning. "
            "See PROJECT_DECISIONS.md [2026-08-21]."
        )
