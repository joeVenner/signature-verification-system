"""Single source of truth for every determinism control (see docs/DETERMINISM.md).

The verification path itself uses no randomness. The controls here cover:
* OpenCV worker threads: pinned to 1 so no result depends on scheduling.
* The only stochastic code (robustness-benchmark noise) draws from
  `seeded_rng`, a PCG64 generator derived from GLOBAL_SEED + a stable offset.
"""

from __future__ import annotations

import cv2
import numpy as np

GLOBAL_SEED = 20261003
OPENCV_THREADS = 1

_configured = False


def configure_determinism() -> None:
    """Apply process-wide determinism settings (idempotent). Called at import
    of the feature extractor so it runs before any verification work."""
    global _configured
    if _configured:
        return
    cv2.setNumThreads(OPENCV_THREADS)
    _configured = True


def seeded_rng(offset: int) -> np.random.Generator:
    """Reproducible generator for a named stream: GLOBAL_SEED + offset.

    Use a stable offset (e.g. derived from an image SHA-256 prefix) so the
    stream is independent of evaluation order.
    """
    return np.random.Generator(np.random.PCG64(GLOBAL_SEED + int(offset)))
