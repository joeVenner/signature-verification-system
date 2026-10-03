"""Input conditions applied identically (label-agnostic) to every benchmark image.

``raw``         the files as stored.
``harmonized``  flat-field photometric normalisation: paper -> pure white,
                ink darkness rescaled to a fixed level. Removes scan-session
                appearance (background tint, exposure, ink darkness) while
                preserving stroke geometry.

Why this exists: in the CEDAR subset the forgeries were scanned in a different
session from the genuine signatures. Comparing only the *paper brightness* of
two images separates genuine from skilled-forgery pairs with ROC-AUC = 1.000
(see benchmark/experiments.md, EXP-000). A verifier evaluated on raw images can
therefore look accurate while partly detecting the scanner, not the signer.
Reported decisions use the harmonized condition; the raw-vs-harmonized gap is
reported as "shortcut reliance".
"""

from __future__ import annotations

from typing import Callable, Dict

import cv2
import numpy as np


def harmonize(image: np.ndarray) -> np.ndarray:
    """Deterministic flat-field normalisation. Returns a 3-channel uint8 image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    paper = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    paper = cv2.GaussianBlur(paper, (0, 0), 5).astype(np.float64)
    ratio = np.clip(gray.astype(np.float64) / np.maximum(paper, 1.0), 0.0, 1.0)
    darkness = cv2.GaussianBlur(1.0 - ratio, (0, 0), 0.8)
    darkness[darkness < 0.06] = 0.0                      # flatten paper texture / scanner noise
    ink = darkness > 0.15
    scale = float(np.quantile(darkness[ink], 0.9)) if int(ink.sum()) > 50 else max(float(darkness.max()), 1e-3)
    darkness = np.clip(darkness / scale, 0.0, 1.0) * 0.8
    out = np.round(255.0 * (1.0 - darkness)).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


CONDITIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "raw": lambda img: img,
    "harmonized": harmonize,
}
