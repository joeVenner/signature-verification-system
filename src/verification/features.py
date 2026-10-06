"""Per-image signature descriptors (extracted once, compared many times).

Two complementary, writer-independent representations:

* **Global shape** — magnitude-weighted gradient-direction histograms over a
  coarse spatial grid of the (aspect-normalised) ink map. Captures overall
  layout and dominant stroke directions; strong against random forgeries.
* **Local keypoints** — SIFT keypoints/descriptors on the letter-boxed ink
  map. Compared with geometric (RANSAC) verification, they capture fine
  stroke detail that skilled forgers fail to reproduce in the right places.
* **Stroke geometry** — skeleton, local stroke directions, ink profiles, slant
  and pen width (see stroke_geometry.py): what a human examiner looks at.

Capture resolution is matched by pen width (capture_scale.py, EXP-026/027): images
captured finer than any clean scan are downsampled here, and `match_capture_scale`
makes the finer image of a pair as coarse as a low-resolution partner.

Determinism: SIFT runs single-threaded on a fixed-size uint8 canvas; keypoint
order is made canonical by sorting.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from signature_verification_system.src.core.config import DEFAULT_CONFIG, RepresentationParams
from signature_verification_system.src.core.determinism import configure_determinism
from signature_verification_system.src.preprocessing.background import PreparedSignature
from signature_verification_system.src.preprocessing.normalization import (
    NormalizedSignature,
    canonicalize,
    normalize_signature,
)
from signature_verification_system.src.verification.capture_scale import (
    MAX_DOWNSAMPLE_PASSES,
    MIN_RESAMPLE_FACTOR,
    capture_widths,
    downsample,
    high_resolution_factor,
    pair_factor,
)
from signature_verification_system.src.verification.stroke_geometry import (
    StrokeGeometry,
    extract_stroke_geometry,
)


@dataclass(frozen=True)
class SignatureFeatures:
    """Immutable descriptor bundle for one signature image."""

    normalized: NormalizedSignature
    shape_descriptor: np.ndarray          # L2-normalised gradient-grid histogram
    keypoints: np.ndarray                 # (N, 2) float32 keypoint coordinates
    descriptors: Optional[np.ndarray]     # (N, 128) float32 SIFT descriptors or None
    aspect_ratio: float                   # crop width / height
    ink_density: float                    # fraction of crop pixels that are ink
    stroke: StrokeGeometry                # stroke-level descriptors (direction, slant, profiles, width)
    # Kept so a comparison can re-extract this image at a coarser capture (EXP-026).
    source_image: Optional[np.ndarray] = None   # the image the descriptors were extracted from
    pen_width_px: float = 0.0                   # mean pen width in source_image pixels
    ink_width_px: float = 0.0                   # ink area / skeleton length, source_image pixels
    work_scale: float = 1.0                     # normalised (work) / source_image linear scale


def gradient_grid_descriptor(canvas: np.ndarray, rows: int, cols: int, bins: int, blur_sigma: float) -> np.ndarray:
    """Spatial grid of magnitude-weighted gradient-direction histograms (sqrt + L2)."""
    img = cv2.GaussianBlur(canvas, (0, 0), blur_sigma)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    ang = np.mod(np.arctan2(gy, gx), 2 * np.pi)
    idx = (ang / (2 * np.pi) * bins).astype(np.int64) % bins
    h, w = img.shape
    cells = []
    for i in range(rows):
        for j in range(cols):
            sl = (slice(i * h // rows, (i + 1) * h // rows), slice(j * w // cols, (j + 1) * w // cols))
            cells.append(np.bincount(idx[sl].ravel(), weights=mag[sl].ravel(), minlength=bins))
    vec = np.sqrt(np.concatenate(cells))
    return vec / (np.linalg.norm(vec) + 1e-9)


# One OpenCV worker thread, applied once at import (not lazily on first use)
# so behaviour never depends on call order. See src/core/determinism.py.
configure_determinism()


def _sift() -> "cv2.SIFT":
    """Fresh detector per call: no shared mutable state across threads."""
    return cv2.SIFT_create()


def keypoint_descriptors(ink: np.ndarray, width: int, height: int) -> tuple[np.ndarray, Optional[np.ndarray]]:
    canvas = canonicalize(ink, width, height, keep_aspect=True)
    u8 = np.round(255.0 * (1.0 - np.clip(canvas, 0.0, 1.0))).astype(np.uint8)
    kps, des = _sift().detectAndCompute(u8, None)
    if des is None or len(kps) == 0:
        return np.zeros((0, 2), np.float32), None
    pts = np.float32([k.pt for k in kps])
    order = np.lexsort((des.sum(axis=1), pts[:, 0], pts[:, 1]))
    return pts[order], des[order]


def _measure(norm: NormalizedSignature, source: np.ndarray) -> Tuple[float, float, float]:
    """(pen width, ink width, work scale) of a normalised signature, widths in `source` pixels."""
    work_scale = norm.gray.shape[1] / source.shape[1]
    pen, ink_width = capture_widths(norm.ink, work_scale)
    return pen, ink_width, work_scale


def _normalize_at_capture_scale(
    image: np.ndarray, prepared: Optional[PreparedSignature] = None
) -> Tuple[NormalizedSignature, np.ndarray, Tuple[float, float, float]]:
    """Normalise `image`, first downsampling it if it was captured finer than the clean band.

    `prepared` belongs to `image` and is therefore only used for the first normalisation.

    Returns:
        (normalised signature, the image it was taken from, `_measure` of it).
    """
    source = image
    norm = normalize_signature(source, prepared)
    widths = _measure(norm, source)
    total = 1.0
    for _ in range(MAX_DOWNSAMPLE_PASSES):
        factor = high_resolution_factor(widths[0])
        if factor is None:
            break
        # Resample the original once by the accumulated factor (no compounding of resampling blur).
        total = max(total * factor, MIN_RESAMPLE_FACTOR)
        source = downsample(image, total)
        norm = normalize_signature(source)
        widths = _measure(norm, source)
    return norm, source, widths


def _describe(norm: NormalizedSignature, source: np.ndarray, widths: Tuple[float, float, float],
              p: RepresentationParams) -> SignatureFeatures:
    shape_canvas = canonicalize(norm.ink, p.shape_canvas_width, p.shape_canvas_height, keep_aspect=False)
    shape = gradient_grid_descriptor(shape_canvas, p.shape_grid_rows, p.shape_grid_cols, p.shape_bins, p.shape_blur_sigma)
    pts, des = keypoint_descriptors(norm.ink, p.keypoint_canvas_width, p.keypoint_canvas_height)
    h, w = norm.ink.shape
    return SignatureFeatures(
        normalized=norm,
        shape_descriptor=shape,
        keypoints=pts,
        descriptors=des,
        aspect_ratio=float(w) / float(h),
        ink_density=float(np.mean(norm.ink > 0.25)),
        stroke=extract_stroke_geometry(norm.ink, p.stroke, p.keypoint_canvas_width, p.keypoint_canvas_height),
        source_image=source,
        pen_width_px=widths[0],
        ink_width_px=widths[1],
        work_scale=widths[2],
    )


def extract_features(
    image: np.ndarray,
    params: Optional[RepresentationParams] = None,
    prepared: Optional[PreparedSignature] = None,
) -> SignatureFeatures:
    """Extract all descriptors for one signature image (raises ValueError if no ink).

    `prepared` is `normalization.prepare_image(image)` if already computed.
    """
    p = params or DEFAULT_CONFIG.representation
    norm, source, widths = _normalize_at_capture_scale(image, prepared)
    return _describe(norm, source, widths, p)


# Benchmarks compare one reference with many low-resolution queries, which asks for the
# same (image, factor) re-extraction repeatedly. Bounded (LRU), keyed by image content,
# factor and parameters, so a result never depends on whether an entry was cached.
_COARSE_CACHE_SIZE = 32
_coarse_cache: "OrderedDict[tuple, SignatureFeatures]" = OrderedDict()
_coarse_lock = threading.Lock()


def _coarser(features: SignatureFeatures, factor: float, p: RepresentationParams) -> SignatureFeatures:
    """`features` re-extracted from its source image downsampled by `factor` (raises ValueError if no ink).

    The downsampled image goes through the normal preparation path
    (`normalize_signature` -> `prepare_signature`), so every routing rule and the
    pixel cap apply; it is never larger than an image that already passed them.
    """
    image = features.source_image
    key = (hashlib.sha256(image.tobytes()).hexdigest(), image.shape, str(image.dtype), factor, p.model_dump_json())
    with _coarse_lock:
        if key in _coarse_cache:
            _coarse_cache.move_to_end(key)
            return _coarse_cache[key]
    source = downsample(image, factor)
    norm = normalize_signature(source)
    result = _describe(norm, source, _measure(norm, source), p)
    with _coarse_lock:
        _coarse_cache[key] = result
        while len(_coarse_cache) > _COARSE_CACHE_SIZE:
            _coarse_cache.popitem(last=False)
    return result


def match_capture_scale(
    a: SignatureFeatures, b: SignatureFeatures, params: Optional[RepresentationParams] = None
) -> Tuple[SignatureFeatures, SignatureFeatures]:
    """Make the finer image of a pair as coarse as a low-resolution partner (EXP-027).

    Returns `(a, b)` unchanged unless the image with the smaller ink width is below
    the clean band (`capture_scale.pair_factor`); then the other image is
    re-extracted after downsampling it to the same modelled capture scale. If that
    fails (no ink left), the original features are kept: matching is an accuracy
    aid, never a reason to fail.
    """
    if a.source_image is None or b.source_image is None:
        return a, b
    p = params or DEFAULT_CONFIG.representation
    a_is_finer = a.ink_width_px > b.ink_width_px
    fine, coarse = (a, b) if a_is_finer else (b, a)
    factor = pair_factor(fine.ink_width_px, fine.work_scale, coarse.ink_width_px, coarse.work_scale)
    if factor is None:
        return a, b
    try:
        matched = _coarser(fine, factor, p)
    except ValueError:
        return a, b
    return (matched, b) if a_is_finer else (a, matched)
