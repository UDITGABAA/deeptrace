"""
CelebDF v2 dataset loader.
Expected structure:
    data/CelebDF/
        Celeb-real/*.mp4
        Celeb-synthesis/*.mp4
        YouTube-real/*.mp4     (optional)
        List_of_testing_videos.txt
"""

from pathlib import Path
import logging
from datasets.base_dataset import BaseDeepfakeDataset

logger = logging.getLogger(__name__)


class CelebDFDataset(BaseDeepfakeDataset):
    """CelebDF v2 dataset."""

    def _load_samples(self) -> list:
        samples = []
        cdf_root = self.root_dir / "CelebDF"

        # Real videos
        for real_dir_name in ["Celeb-real", "YouTube-real"]:
            real_dir = cdf_root / real_dir_name
            if real_dir.exists():
                for vid in sorted(real_dir.glob("*.mp4")):
                    samples.append({
                        "path": str(vid),
                        "label": 0,
                        "manipulation_type": "real",
                    })

        # Fake videos
        synth_dir = cdf_root / "Celeb-synthesis"
        if synth_dir.exists():
            for vid in sorted(synth_dir.glob("*.mp4")):
                samples.append({
                    "path": str(vid),
                    "label": 1,
                    "manipulation_type": "Deepfakes",
                })

        # Use official test split if available
        test_list_path = cdf_root / "List_of_testing_videos.txt"
        if test_list_path.exists() and self.split == "test":
            test_videos = set()
            with open(test_list_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        test_videos.add(parts[1])
            samples = [s for s in samples if Path(s["path"]).name in test_videos]
        else:
            samples = self._apply_split(samples)

        return samples

    def _apply_split(self, samples: list) -> list:
        """Apply train/val/test split (80/10/10)."""
        n = len(samples)
        if self.split == "train":
            return samples[: int(n * 0.8)]
        elif self.split == "val":
            return samples[int(n * 0.8) : int(n * 0.9)]
        else:
            return samples[int(n * 0.9) :]
