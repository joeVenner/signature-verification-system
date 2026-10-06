"""Photometric and geometric normalisation of a signature crop.

Removes scan-session appearance (paper tint, exposure, ink darkness, scanner
noise) so verification depends on stroke geometry only. This matters because
a specimen card and a presented cheque are never scanned under identical
conditions; see benchmark/experiments.md EXP-000 for the measured impact.

All operations are deterministic (fixed kernels, no randomness, no
data-dependent iteration order).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from signature_verification_system.src.preprocessing.background import PreparedSignature, prepare_signature
from signature_verification_system.src.preprocessing.isolation import isolate_signature_ink

PAPER_CLOSE_KERNEL = 15     # px; larger than any stroke width in 256-900 px crops
PAPER_BLUR_SIGMA = 5.0
INK_SMOOTH_SIGMA = 0.8
PAPER_NOISE_FLOOR = 0.06    # darkness below this is treated as paper texture
INK_LEVEL_QUANTILE = 0.9    # ink darkness quantile mapped to INK_TARGET
INK_TARGET = 0.8
INK_MASK_LEVEL = 0.25       # normalised darkness above this counts as ink
CROP_TRIM_QUANTILE = 0.002  # ignore extreme 0.2% ink pixels when cropping (stray specks)
# Flat-field passes. A second pass re-estimates paper on an already-flattened
# image and removes residual scan texture that otherwise produces spurious SIFT
# keypoints (raw inputs gave ~2x the keypoints of pre-normalised ones).
# Measured in EXP-002: 1 pass skilled EER 19.3%, 2 passes 13.5%, 3-4 passes
# 12.3-13.5% (plateau) under writer-disjoint CV. 2 = smallest value on the plateau.
NORMALIZATION_PASSES = 2
# Non-local-means denoising of the raw input, once, before flat-fielding.
# EXP-006: additive scanner noise (sigma 10) pushed 33% of genuine queries into
# REJECT; with NLM h=12 that drops to 10% with CV skilled AUC 0.910 vs 0.915.
# h chosen from {7, 10, 12, 15}; differences in CV AUC between them (~0.03)
# are at the noise level of a 12-writer benchmark.
DENOISE_STRENGTH = 12.0


@dataclass(frozen=True)
class NormalizedSignature:
    """Photometrically normalised, tightly cropped ink-darkness map."""

    ink: np.ndarray                 # float64 in [0, 1], 1 = darkest ink, tight crop
    gray: np.ndarray                # uint8 harmonised full image (paper = 255)
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) of the crop in the input image
    ink_pixel_count: int
    polarity_inverted: bool = False  # True if input was light ink on dark background


def to_gray(image: np.ndarray) -> np.ndarray:
    """uint8 single-channel view of a 1/3/4-channel uint8 or uint16 image.

    Raises:
        ValueError: unsupported dtype or channel count.
    """
    if image.dtype == np.uint16:
        image = (image >> 8).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise ValueError(f"Unsupported image dtype {image.dtype}")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    raise ValueError(f"Unsupported image shape {image.shape}")


def is_inverted_polarity(gray: np.ndarray) -> bool:
    """Signatures cover a small fraction of the crop, so the median pixel is
    background and the ink is the minority tail. Normal polarity: the tail
    extends *darker* than the background; inverted (negative scans,
    white-on-black renders): it extends *lighter*. Comparing tail extents is
    robust to under-exposed grey paper, unlike a fixed median < 128 rule."""
    median = float(np.median(gray))
    dark_tail = median - float(np.quantile(gray, 0.01))
    light_tail = float(np.quantile(gray, 0.99)) - median
    return light_tail > dark_tail


def ensure_dark_ink(gray: np.ndarray) -> Tuple[np.ndarray, bool]:
    if is_inverted_polarity(gray):
        return 255 - gray, True
    return gray, False


def harmonize_photometric(image: np.ndarray, passes: int = NORMALIZATION_PASSES) -> np.ndarray:
    """Flat-field normalisation: paper -> 255, ink p90 -> fixed darkness.

    Inverted-polarity inputs are flipped first. Returns a uint8 grayscale image
    of the same size.
    """
    gray, _ = ensure_dark_ink(to_gray(image))
    if DENOISE_STRENGTH > 0:
        gray = cv2.fastNlMeansDenoising(gray, None, h=DENOISE_STRENGTH, templateWindowSize=7, searchWindowSize=21)
    for _ in range(max(1, passes)):
        gray = _flat_field_pass(gray)
    return gray


def _flat_field_pass(gray: np.ndarray) -> np.ndarray:
    """One deterministic flat-field + ink-level normalisation pass (uint8 in/out)."""
    paper = cv2.morphologyEx(
        gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (PAPER_CLOSE_KERNEL, PAPER_CLOSE_KERNEL))
    )
    paper = cv2.GaussianBlur(paper, (0, 0), PAPER_BLUR_SIGMA).astype(np.float64)
    ratio = np.clip(gray.astype(np.float64) / np.maximum(paper, 1.0), 0.0, 1.0)
    darkness = cv2.GaussianBlur(1.0 - ratio, (0, 0), INK_SMOOTH_SIGMA)
    darkness[darkness < PAPER_NOISE_FLOOR] = 0.0
    ink = darkness > 0.15
    if int(ink.sum()) > 50:
        scale = float(np.quantile(darkness[ink], INK_LEVEL_QUANTILE))
    else:
        scale = max(float(darkness.max()), 1e-3)
    darkness = np.clip(darkness / max(scale, 1e-3), 0.0, 1.0) * INK_TARGET
    return np.round(255.0 * (1.0 - darkness)).astype(np.uint8)


def prepare_image(image: np.ndarray) -> PreparedSignature:
    """Background / resolution routing of an input image (see background.py).

    Compute it once per image and pass it to `normalize_signature` and
    `quality.assess_signature_quality` to avoid repeating the work.
    """
    return prepare_signature(image, ensure_dark_ink(to_gray(image))[0])


def normalize_signature(image: np.ndarray, prepared: Optional[PreparedSignature] = None) -> NormalizedSignature:
    """Harmonise and tight-crop a signature image.

    Textured (cheque security) backgrounds are removed and small crops are
    upscaled first (see background.py). `bbox`, `gray` and `ink_pixel_count`
    are therefore in the coordinates of the prepared (possibly upscaled)
    image, whereas the quality gate reports sizes at input resolution.

    Args:
        image: input signature image.
        prepared: `prepare_image(image)` if already computed (must come from
            this exact image).

    Raises:
        ValueError: if the image is empty, unsupported, fails preprocessing or
            contains no detectable ink.
    """
    if image is None or image.size == 0:
        raise ValueError("Empty signature image")
    gray, inverted = ensure_dark_ink(to_gray(image))
    try:
        if prepared is None:
            prepared = prepare_signature(image, gray)
        if not prepared.dimensions_supported:
            # Extreme aspect ratios make every later stage pathologically slow.
            raise ValueError("Unsupported image dimensions")
        harmonized = harmonize_photometric(prepared.work_image)
    except cv2.error as exc:
        # ValueError is the verifier's INCONCLUSIVE path; the OpenCV message stays out of responses.
        raise ValueError("Image preprocessing failed") from exc
    darkness = (255.0 - harmonized.astype(np.float64)) / 255.0
    # Printed rules and caption text would otherwise set the crop (EXP-016).
    darkness = isolate_signature_ink(darkness, INK_MASK_LEVEL)
    mask = darkness > INK_MASK_LEVEL
    ys, xs = np.nonzero(mask)
    if len(ys) < 20:
        raise ValueError("No signature ink detected")
    y0, y1 = np.quantile(ys, [CROP_TRIM_QUANTILE, 1 - CROP_TRIM_QUANTILE], method="lower").astype(int)
    x0, x1 = np.quantile(xs, [CROP_TRIM_QUANTILE, 1 - CROP_TRIM_QUANTILE], method="lower").astype(int)
    y0, x0 = max(0, int(y0) - 2), max(0, int(x0) - 2)
    y1 = min(darkness.shape[0], int(y1) + 3)
    x1 = min(darkness.shape[1], int(x1) + 3)
    return NormalizedSignature(
        ink=darkness[y0:y1, x0:x1],
        gray=harmonized,
        bbox=(x0, y0, x1 - x0, y1 - y0),
        ink_pixel_count=int(len(ys)),
        polarity_inverted=inverted,
    )


def canonicalize(ink: np.ndarray, width: int, height: int, keep_aspect: bool) -> np.ndarray:
    """Resize an ink map to a fixed canvas (optionally letter-boxed)."""
    if not keep_aspect:
        return cv2.resize(ink, (width, height), interpolation=cv2.INTER_AREA)
    h, w = ink.shape
    scale = min(width / w, height / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(ink, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width), dtype=ink.dtype)
    y0, x0 = (height - nh) // 2, (width - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas
