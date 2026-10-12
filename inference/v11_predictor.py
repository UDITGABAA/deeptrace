"""
V11 predictor compatibility module — delegates to the canonical V12-Dense engine.
Ensures zero breakage for existing scripts while upgrading inference to the audited V12 model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.v12_predictor import V12Predictor, NoFaceError, MANIP_FAMILIES, DEFAULT_CKPT

# Canonical alias
class V11Predictor(V12Predictor):
    """Backwards-compatible alias for V12Predictor."""
    def __init__(self, checkpoint: str = DEFAULT_CKPT, device: str = None, threshold: float = None):
        super().__init__(checkpoint=checkpoint, device=device, threshold=threshold)

    def gradcam_overlay(self, face):
        """Backwards-compatible single-face GradCAM overlay."""
        cam_crop, _ = self.gradcam_overlays(face, face, [0, 0, face.shape[1], face.shape[0]])
        return cam_crop
