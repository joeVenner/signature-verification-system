"""Remove printed form structure from a signature crop (EXP-009).

Cheque crops contain printed elements the verifier was never trained to see:
the signatory rule under the signature and dashed signing-box borders. These
are removed before verification; signature strokes that cross a rule are
preserved (descender-safe baseline removal).
"""

from __future__ import annotations

import cv2
import numpy as np

from signature_verification_system.src.preprocessing.ink_extractor import remove_horizontal_baseline
from signature_verification_system.src.preprocessing.normalization import ensure_dark_ink, to_gray

MIN_RULE_FRACTION = 0.25    # horizontal run >= 25% of crop width is a printed rule
MAX_DASH_HEIGHT = 5         # flat components (dashes, rule fragments) are at most this tall
MIN_DASH_ELONGATION = 3.0   # ... and at least this many times wider than tall


def clean_printed_rules(crop: np.ndarray) -> np.ndarray:
    """Return a copy of `crop` with printed rules and dashes painted as paper."""
    gray, inverted = ensure_dark_ink(to_gray(crop))
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    width = ink.shape[1]
    kernel_w = max(15, int(MIN_RULE_FRACTION * width))
    without_rules = remove_horizontal_baseline(ink, preserve_descenders=True, kernel_width=kernel_w)
    remove = cv2.bitwise_and(ink, cv2.bitwise_not(without_rules))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(without_rules, connectivity=8)
    for i in range(1, n):
        w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        if h <= MAX_DASH_HEIGHT and w >= MIN_DASH_ELONGATION * h:
            remove[labels == i] = 255

    if not remove.any():
        return crop.copy()
    paper = int(np.median(gray[ink == 0])) if (ink == 0).any() else 255
    cleaned = gray.copy()
    cleaned[cv2.dilate(remove, np.ones((3, 3), np.uint8)) > 0] = paper
    if inverted:
        cleaned = 255 - cleaned
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR) if crop.ndim == 3 else cleaned
