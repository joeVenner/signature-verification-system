"""Signature ink extraction from textured cheque backgrounds and low-resolution crops.

Cheque crops differ from specimen scans in two ways the clean path was never
built for (benchmark/experiments.md EXP-025):

* **Security texture.** Guilloche (dense wavy line families), tinted paper and
  pantograph patterns cover 20-60% of the background. On BCSD cheques the
  texture is 3-10x lighter than signature ink relative to the paper (median
  darkness 17 vs 166 grey levels at native resolution) and has no single
  separating stroke width (texture lines p50 5.5 px vs ink 7.4 px), so
  darkness, not width or orientation, is the separating cue.
* **Low resolution.** Field crops are ~200 px wide. Upscaling them before
  feature extraction alone cut skilled EER from 25% to 16% on low-res
  composites (cheque_composite.py, ``clean_lowres``).

`prepare_signature` routes an image:

* textured background -> background flattening (morphological closing), then
  hysteresis thresholding seeded from the darkest strokes and grown down to
  just above the measured texture level; ink keeps its own darkness (the
  pressure signals read it) on white paper;
* width below LOWRES_MAX_WIDTH -> bicubic upscale (textured or not);
* anything else -> the input object itself, so clean full-size scans are
  bit-identical to the pre-existing path.

Everything is deterministic: fixed kernels, quantiles and OpenCV
raster-order labels; no randomness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# Paper estimate: closing kernel as a fraction of the shorter side, clamped so
# it stays wider than a pen stroke on small crops and cheap on large ones.
PAPER_KERNEL_FRAC = 0.12
PAPER_KERNEL_MIN_PX = 7
PAPER_KERNEL_MAX_PX = 31
INK_QUANTILE = 0.99            # darkness quantile taken as the ink level
STRONG_INK_FRAC = 0.5          # darkness above this share of the ink level is ink core
NEAR_INK_PX = 3                # anti-aliased halo around ink excluded from the background
MIN_BACKGROUND_PX = 50

# Texture detector. A background pixel is "textured" when it is darker than
# both an absolute floor and a share of the ink level. EXP-025 (fraction of
# textured background pixels): clean CEDAR dev and all 11 capture conditions
# max 0.009 except ruled_lines (2 px rule every 36 px) max 0.061; guilloche
# composites p10 0.27-0.31, the field screenshot 0.47-0.53.
TEXTURE_MIN_DARKNESS = 0.08
TEXTURE_REL_DARKNESS = 0.15
TEXTURED_MIN_FRACTION = 0.15   # ~2.5x the ruled-lines maximum

# Extraction (hysteresis on the darkness map).
TEXTURE_LEVEL_QUANTILE = 0.98  # texture darkness level, measured away from ink
SEED_FRAC = 0.6                # seeds: level + 0.6 x (ink - level)
GROW_MARGIN = 0.05             # grow: just above the texture level

# Separability check: a signature yields a few large connected strokes; a pure
# texture yields many fragments. EXP-025: share of the mask in the 5 largest
# components >= 0.41 on every routed composite / BCSD crop, <= 0.05 median on
# pure guilloche; pure-texture masks also cover more of the image.
TOP_COMPONENTS = 5
MIN_TOP_COMPONENT_SHARE = 0.35
MAX_INK_COVERAGE = 0.35

# Low-resolution route. Clean CEDAR scans are >= 270 px wide, so they never
# take it; cheque field crops (~200 px) always do.
LOWRES_MAX_WIDTH = 256
LOWRES_TARGET_WIDTH = 512
# Upscaled images never exceed this many pixels (a 16 x 3,000,000 input would
# otherwise become ~49 Gpx); larger results keep scale 1.0.
MAX_WORK_PIXELS = 4_000_000
MIN_SIDE_PX = 16               # smaller inputs are left to the quality gate
# Longest / shortest side. Real crops: BCSD max 6.3 (signature boxes 7.2),
# CEDAR max 4.1. Beyond this the image is not processed (gate blocks it).
MAX_ASPECT_RATIO = 20.0


@dataclass(frozen=True)
class TextureReport:
    """Measured background texture of one image (dark-ink grayscale)."""

    texture_fraction: float        # share of background pixels darker than the texture threshold
    ink_level: float               # darkness quantile INK_QUANTILE, 0 = paper, 1 = black
    is_textured: bool


@dataclass(frozen=True)
class PreparedSignature:
    """Images to feed the verifier after background / resolution routing."""

    work_image: np.ndarray         # for normalisation + features (may be upscaled)
    gate_image: np.ndarray         # for the quality gate, at input resolution
    texture: TextureReport
    upscale: float                 # work / input linear scale (1.0 = unchanged)
    background_removed: bool       # textured route applied
    separable: bool                # False: textured, but no signature could be separated
    dimensions_supported: bool = True  # False: aspect ratio above MAX_ASPECT_RATIO, nothing done


def _odd(x: float) -> int:
    n = int(round(x))
    return n if n % 2 else n + 1


def paper_kernel(shape: Tuple[int, ...]) -> int:
    """Closing kernel (odd px) for the paper estimate of an image of this shape."""
    return _odd(np.clip(PAPER_KERNEL_FRAC * min(shape[:2]), PAPER_KERNEL_MIN_PX, PAPER_KERNEL_MAX_PX))


def darkness_map(gray: np.ndarray, kernel: int) -> np.ndarray:
    """Ink darkness relative to the local paper level, float64 in [0, 1].

    A grey closing larger than any stroke and texture period erases both and
    leaves the paper (and its tint / shading) level.
    """
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    paper = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se)
    paper = cv2.GaussianBlur(paper, (0, 0), kernel / 3.0).astype(np.float64)
    return np.clip(1.0 - gray.astype(np.float64) / np.maximum(paper, 1.0), 0.0, 1.0)


def _background_mask(darkness: np.ndarray, ink_level: float) -> np.ndarray:
    """Pixels farther than NEAR_INK_PX from ink core."""
    strong = (darkness > STRONG_INK_FRAC * ink_level).astype(np.uint8)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * NEAR_INK_PX + 1, 2 * NEAR_INK_PX + 1))
    return cv2.dilate(strong, se) == 0


@dataclass(frozen=True)
class _DarknessAnalysis:
    """Darkness map plus the ink level and background mask derived from it."""

    darkness: np.ndarray
    ink_level: float
    background: np.ndarray


def _analyse(gray: np.ndarray) -> _DarknessAnalysis:
    darkness = darkness_map(gray, paper_kernel(gray.shape))
    ink_level = float(np.quantile(darkness, INK_QUANTILE))
    return _DarknessAnalysis(darkness, ink_level, _background_mask(darkness, ink_level))


def _texture_report(analysis: _DarknessAnalysis) -> TextureReport:
    background = analysis.background
    if int(background.sum()) < MIN_BACKGROUND_PX:
        return TextureReport(0.0, analysis.ink_level, False)
    threshold = max(TEXTURE_MIN_DARKNESS, TEXTURE_REL_DARKNESS * analysis.ink_level)
    fraction = float(np.mean(analysis.darkness[background] > threshold))
    # The routing decision uses the exact fraction; only the reported value is rounded.
    return TextureReport(round(fraction, 6), round(analysis.ink_level, 6), fraction >= TEXTURED_MIN_FRACTION)


def measure_background_texture(gray: np.ndarray) -> TextureReport:
    """Detect a textured (guilloche / pattern) background.

    Args:
        gray: uint8 grayscale image with dark ink on light paper.
    """
    return _texture_report(_analyse(gray))


def _is_signature_like(mask: np.ndarray) -> bool:
    if not mask.any() or float(mask.mean()) > MAX_INK_COVERAGE:
        return False
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = np.sort(stats[1:, cv2.CC_STAT_AREA])[::-1]
    return float(areas[:TOP_COMPONENTS].sum()) >= MIN_TOP_COMPONENT_SHARE * float(areas.sum())


def extract_ink_layer(gray: np.ndarray, analysis: Optional[_DarknessAnalysis] = None) -> Optional[np.ndarray]:
    """Separate signature ink from a textured background.

    Args:
        gray: uint8 grayscale image with dark ink on light paper.
        analysis: `_analyse(gray)` if the caller already has it.

    Returns:
        uint8 image of the same shape: paper = 255, ink pixels keep their
        darkness relative to the paper. None when no signature-like ink is
        clearly darker than the texture (the caller must not trust the image).
    """
    a = analysis if analysis is not None else _analyse(gray)
    darkness, ink_level, background = a.darkness, a.ink_level, a.background
    if int(background.sum()) < MIN_BACKGROUND_PX:
        return None
    level = float(np.quantile(darkness[background], TEXTURE_LEVEL_QUANTILE))
    if ink_level <= level:
        return None
    seeds = darkness > level + SEED_FRAC * (ink_level - level)
    grow = (darkness > level + GROW_MARGIN).astype(np.uint8)
    n, labels = cv2.connectedComponents(grow, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[np.unique(labels[seeds & (grow > 0)])] = True
    keep[0] = False
    mask = keep[labels]
    if not _is_signature_like(mask):
        return None
    return np.round(255.0 * (1.0 - np.where(mask, darkness, 0.0))).astype(np.uint8)


def upscale_factor(h: int, w: int) -> float:
    """Linear upscale for a low-resolution crop; 1.0 if wide enough or the result would be too large."""
    if w >= LOWRES_MAX_WIDTH:
        return 1.0
    scale = LOWRES_TARGET_WIDTH / float(w)
    if h * w * scale * scale > MAX_WORK_PIXELS:
        return 1.0
    return scale


def _upscale(image: np.ndarray, scale: float) -> np.ndarray:
    h, w = image.shape[:2]
    return cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_CUBIC)


def prepare_signature(image: np.ndarray, gray: np.ndarray) -> PreparedSignature:
    """Route a signature image through texture removal and/or low-res upscaling.

    Args:
        image: the input image (any channel layout accepted by the verifier).
        gray: its uint8 grayscale view with dark ink (see normalization.ensure_dark_ink).

    Returns:
        PreparedSignature. For a clean image at least LOWRES_MAX_WIDTH wide both
        images are `image` itself (same object).
    """
    h, w = gray.shape[:2]
    if min(h, w) < MIN_SIDE_PX:
        return PreparedSignature(image, image, TextureReport(0.0, 0.0, False), 1.0, False, True)
    if max(h, w) > MAX_ASPECT_RATIO * min(h, w):
        return PreparedSignature(image, image, TextureReport(0.0, 0.0, False), 1.0, False, True,
                                 dimensions_supported=False)
    analysis = _analyse(gray)
    texture = _texture_report(analysis)
    scale = upscale_factor(h, w)
    if not texture.is_textured:
        work = image if scale == 1.0 else _upscale(image, scale)
        return PreparedSignature(work, image, texture, scale, False, True)
    if scale == 1.0:
        work_gray, extracted = gray, extract_ink_layer(gray, analysis)
    else:
        work_gray = _upscale(gray, scale)
        extracted = extract_ink_layer(work_gray)
    if extracted is None:
        return PreparedSignature(work_gray, gray, texture, scale, False, False)
    gate = extracted if scale == 1.0 else cv2.resize(extracted, (w, h), interpolation=cv2.INTER_AREA)
    return PreparedSignature(extracted, gate, texture, scale, True, True)
