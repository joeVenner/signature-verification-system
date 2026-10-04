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

Determinism: SIFT runs single-threaded on a fixed-size uint8 canvas; keypoint
order is made canonical by sorting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from signature_verification_system.src.core.config import DEFAULT_CONFIG, RepresentationParams
from signature_verification_system.src.core.determinism import configure_determinism
from signature_verification_system.src.preprocessing.normalization import (
    INK_MASK_LEVEL,
    NormalizedSignature,
    canonicalize,
    normalize_signature,
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
    # Kept so a comparison can re-extract this image at a lower resolution (EXP-021).
    source_image: Optional[np.ndarray] = None   # the image exactly as passed to extract_features
    ink_radius: float = 0.0                     # ink radius of gyration in source pixels


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


def ink_radius_of_gyration(ink: np.ndarray) -> float:
    """Darkness-weighted RMS distance of ink pixels from their centroid (pixels).

    Rotation invariant and exactly proportional to image scale, unlike the crop
    box (which grows when the signature is rotated).
    """
    ys, xs = np.nonzero(ink > INK_MASK_LEVEL)
    if len(ys) == 0:
        return 0.0
    weights = ink[ys, xs]
    var_x = np.average((xs - np.average(xs, weights=weights)) ** 2, weights=weights)
    var_y = np.average((ys - np.average(ys, weights=weights)) ** 2, weights=weights)
    return float(np.sqrt(var_x + var_y))


def extract_features(image: np.ndarray, params: Optional[RepresentationParams] = None) -> SignatureFeatures:
    """Extract all descriptors for one signature image (raises ValueError if no ink)."""
    p = params or DEFAULT_CONFIG.representation
    norm = normalize_signature(image)
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
        source_image=image,
        ink_radius=ink_radius_of_gyration(norm.ink),
    )


def downsampled_features(
    features: SignatureFeatures, factor: float, params: Optional[RepresentationParams] = None
) -> SignatureFeatures:
    """Re-extract `features` from its source image shrunk by `factor` (< 1).

    Returns `features` unchanged when there is no source image, `factor` >= 1 (the
    pair-level scale harmonisation only ever downsamples, EXP-021), or the shrunk
    image no longer has detectable ink (extreme size mismatch): comparing at native
    resolution is then the only option, and it is what the system did before.
    """
    if features.source_image is None or factor >= 1.0:
        return features
    h, w = features.source_image.shape[:2]
    size = (max(8, round(w * factor)), max(8, round(h * factor)))
    try:
        return extract_features(cv2.resize(features.source_image, size, interpolation=cv2.INTER_AREA), params)
    except ValueError:
        return features
