"""Capture-resolution matching by pen width (EXP-026, EXP-027).

The stroke-width and curvature signals read the pen width on the letter-boxed
canvas. That measurement is not proportional to capture resolution: pen blur and
the fixed-pixel smoothing in normalisation add a near-constant floor, so a
signature captured at half resolution looks ~30% thicker relative to its size and
one captured at double resolution ~17% thinner (dev, canvas stroke width). On dev
those two signals lose most of their skilled-forgery power under a resolution
change (stroke width AUC 0.70 -> 0.51-0.60, curvature 0.75 -> 0.59-0.63).

The cue used here is the pen width in *source* pixels. It moves with the capture,
but varies little between writers scanned at one resolution. The signature's
overall size is NOT used: it is identity evidence, and L-002 lost clean accuracy
by harmonising on it. Two measurements of it are used:

* per image, the distance-transform pen width (clean CEDAR dev 3.9-6.6 px): above
  PEN_WIDTH_MAX_PX (finer capture than any clean scan) the image is downsampled to
  PEN_WIDTH_TARGET_PX before features are extracted;
* per pair, the ink width (ink area / skeleton length). It counts the whole
  blurred stroke, so at half resolution it drops below every clean dev scan for
  86% of images (distance-transform width: 37%). When one image is below
  INK_WIDTH_MIN_PX, both images' capture scales are estimated with a blur-floor
  model (`model_capture_scale`) and the finer image is downsampled to the coarser
  one's scale. Upsampling the coarse image instead was measured and does not help
  (it cannot remove the capture blur), so images are only ever made coarser.

Inside the band nothing happens, so clean scans are bit-identical to the previous
path. Everything is deterministic: fixed thresholds, a skeleton, a distance
transform and INTER_AREA resampling.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize

PEN_WIDTH_MAX_PX = 7.0       # above clean dev max 6.58 (and phone photo 6.88) with margin
PEN_WIDTH_TARGET_PX = 5.2    # clean dev median
PEN_INK_THRESHOLD = 0.35     # same stroke threshold as StrokeParams.ink_threshold
MAX_DOWNSAMPLE_PASSES = 2    # the blur floor makes one correction undershoot slightly
MIN_RESAMPLE_FACTOR = 0.25
# Pair factors are snapped to this grid so sub-pixel width noise does not produce a
# different resampling for every pair (and repeated pairs reuse a result).
FACTOR_STEP = 0.05
# Two images whose capture scales are this close are already matched.
MATCHED_RATIO = 1.0 - FACTOR_STEP / 2.0
INK_MASK_LEVEL = 0.25        # normalisation.INK_MASK_LEVEL
INK_WIDTH_MIN_PX = 4.9       # clean dev min 5.15 x 0.95
# Blur-floor model of the ink width in working px: w = A * s + B (dev medians, clean / 0.5x).
FLOOR_MODEL_A = 3.484
FLOOR_MODEL_B = 2.903


def capture_widths(ink: np.ndarray, work_scale: float) -> Tuple[float, float]:
    """(distance-transform pen width, ink width) of a normalised ink map, in input pixels.

    Args:
        ink: float ink-darkness crop (1 = ink) at the working resolution.
        work_scale: working / input linear scale (the low-resolution route upscales).

    Returns:
        Mean of twice the distance-to-paper along the skeleton, and ink area over
        skeleton length; both 0.0 without strokes. One skeleton serves both.
    """
    binary = ink > PEN_INK_THRESHOLD
    skeleton = skeletonize(binary)
    length = float(skeleton.sum())
    if length == 0 or work_scale <= 0:
        return 0.0, 0.0
    distance = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
    pen = float(np.mean(2.0 * distance[skeleton])) / work_scale
    return pen, float((ink > INK_MASK_LEVEL).sum()) / length / work_scale


def high_resolution_factor(pen_px: float) -> Optional[float]:
    """Downsampling factor for an image captured finer than the clean band, else None."""
    if pen_px <= PEN_WIDTH_MAX_PX:
        return None
    return max(MIN_RESAMPLE_FACTOR, PEN_WIDTH_TARGET_PX / pen_px)


def snap_factor(ratio: float) -> Optional[float]:
    """Snap a downsampling factor to the grid; None when the images are already matched."""
    if ratio >= MATCHED_RATIO:
        return None
    snapped = round(round(ratio / FACTOR_STEP) * FACTOR_STEP, 2)
    return min(max(snapped, MIN_RESAMPLE_FACTOR), 1.0 - FACTOR_STEP) if snapped < 1.0 else None


def model_capture_scale(ink_width: float, work_scale: float) -> float:
    """Capture scale of an image relative to a clean scan, from its ink width alone (blur-floor model)."""
    return max((ink_width * work_scale - FLOOR_MODEL_B) / FLOOR_MODEL_A, 0.0) / work_scale


def pair_factor(fine_ink_px: float, fine_work_scale: float,
                coarse_ink_px: float, coarse_work_scale: float) -> Optional[float]:
    """Factor to downsample the finer image of a pair by, or None when no matching is needed.

    Only triggered when the coarser image's ink width lies below the clean band;
    pairs of in-band images (every clean scan) are never touched, so pen-width
    differences between writers keep their evidential value. An in-band finer image
    counts as a clean capture (scale 1).
    """
    if coarse_ink_px <= 0 or fine_ink_px <= 0 or coarse_ink_px >= INK_WIDTH_MIN_PX:
        return None
    fine_scale = 1.0 if fine_ink_px >= INK_WIDTH_MIN_PX else model_capture_scale(fine_ink_px, fine_work_scale)
    if fine_scale <= 0:
        return None
    return snap_factor(model_capture_scale(coarse_ink_px, coarse_work_scale) / fine_scale)


def downsample(image: np.ndarray, factor: float) -> np.ndarray:
    """Area-downsample by `factor` in (0, 1) (never upsamples; at least 8 px per side)."""
    if not 0.0 < factor < 1.0:
        raise ValueError(f"downsample factor must be in (0, 1), got {factor}")
    h, w = image.shape[:2]
    size = (max(8, int(round(w * factor))), max(8, int(round(h * factor))))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
