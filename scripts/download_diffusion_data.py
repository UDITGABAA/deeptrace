"""
Diffusion Dataset Downloader and Ingestion Engine.
Supports fetching synthetic diffusion fakes (e.g. CIFAKE, GenImage subsets)
for Phase 4 multi-generator generalization hardening.

Target directory: data/diffusion_fakes/
Structure:
  data/diffusion_fakes/
    ├── train/
    │   ├── real/
    │   └── fake/
    └── test/
        ├── real/
        └── fake/
"""

import sys
import shutil
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_diffusion")


def setup_diffusion_folders(dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "test"]:
        for cls in ["real", "fake"]:
            (dest_dir / split / cls).mkdir(parents=True, exist_ok=True)
    logger.info(f"Initialized diffusion folder structure at {dest_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download & Setup Diffusion Dataset")
    parser.add_argument("--dest", type=str, default="data/diffusion_fakes")
    args = parser.parse_args()

    dest_dir = Path(args.dest)
    setup_diffusion_folders(dest_dir)
    logger.info("Ready for diffusion fake ingestion (CIFAKE / Stable Diffusion / Midjourney / Flux).")


if __name__ == "__main__":
    main()
