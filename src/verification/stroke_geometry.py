"""Stroke-level signals for comparing one specimen with one questioned signature.

The v3 verifier originally decided on two signals only (coarse layout and SIFT
inliers). A human examiner also compares *how* the strokes run: slant, stroke
direction along matching strokes, proportions of the horizontal and vertical ink
distribution, and pen width. Skilled forgers copy the outline but rarely get
these right, so they add independent evidence (EXP-015).

All steps are deterministic: skeletonisation, KD-tree nearest neighbours, SVD and
banded dynamic programming contain no randomness and no data-dependent ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from signature_verification_system.src.core.config import StrokeParams
from signature_verification_system.src.preprocessing.normalization import canonicalize

SLANT_BINS = 18
# Largest possible circular EMD between two slant histograms (verified numerically);
# dividing by it maps the distance onto [0, 1].
SLANT_MAX_EMD = SLANT_BINS / 2.0
MIN_SKELETON_POINTS = 10
MIN_ORIENTATION_POINTS = 5     # fewer points cannot define a local stroke direction
MIN_DIRECTION_MATCHES = 5      # fewer corresponding strokes make the mean cosine meaningless
# Real signatures skeletonise to ~1-2k points. Noise or mesh images can fill the whole
# canvas (>80k points); an even, deterministic stride bounds the per-point loops.
MAX_SKELETON_POINTS = 20_000


@dataclass(frozen=True)
class StrokeGeometry:
    """Per-signature stroke descriptors (extracted once, compared many times)."""

    skeleton_points: np.ndarray    # (N, 2) float64 (x, y) on the keypoint canvas
    orientations: np.ndarray       # (N,) local stroke direction in radians, [0, pi)
    column_profile: np.ndarray     # ink mass per column, aspect-normalised, sums to 1
    row_profile: np.ndarray        # ink mass per row, aspect-normalised, sums to 1
    slant_histogram: np.ndarray    # (SLANT_BINS,) gradient-orientation mass mod pi, sums to 1
    stroke_width: float            # mean stroke width in canvas pixels (0.0 when unmeasurable)


@dataclass(frozen=True)
class StrokeSignals:
    """Similarity signals between two `StrokeGeometry` objects, all in higher-is-more-similar form."""

    direction_agreement: float     # local stroke-direction agreement x coverage, after alignment
    slant: float                   # 1 - normalised circular earth-mover distance of slant histograms, in [0, 1]
    column_profile: float          # banded-DTW similarity of horizontal ink profiles
    row_profile: float             # banded-DTW similarity of vertical ink profiles
    stroke_width: float            # exp(-|log width ratio|)


def _local_orientations(points: np.ndarray, radius: float) -> np.ndarray:
    """Stroke direction at each skeleton point from the principal axis of its neighbourhood."""
    if len(points) < MIN_ORIENTATION_POINTS:
        return np.zeros(len(points))
    neighbours = cKDTree(points).query_ball_point(points, radius)
    angles = np.zeros(len(points))
    for i, idx in enumerate(neighbours):
        centred = points[idx] - points[idx].mean(axis=0)
        cov = centred.T @ centred
        angles[i] = 0.5 * np.arctan2(2.0 * cov[0, 1], cov[0, 0] - cov[1, 1])
    return np.mod(angles, np.pi)


def _resample_profile(profile: np.ndarray, length: int) -> np.ndarray:
    """Area-resample a 1-D profile to `length` samples, preserving total mass."""
    resized = cv2.resize(profile.reshape(1, -1), (length, 1), interpolation=cv2.INTER_AREA).ravel()
    total = resized.sum()
    return resized / total if total > 0 else resized


def extract_stroke_geometry(ink: np.ndarray, params: StrokeParams, width: int, height: int) -> StrokeGeometry:
    """Describe stroke layout of a normalised ink map (1 = ink, tight crop).

    Args:
        ink: float ink-darkness map from `normalize_signature`.
        params: stroke parameters.
        width, height: letter-boxed canvas shared with the keypoint stage so that a
            keypoint transform can be reused as the alignment starting point.
    """
    canvas = canonicalize(ink, width, height, keep_aspect=True)
    binary = canvas > params.ink_threshold
    skeleton = skeletonize(binary)
    points = np.argwhere(skeleton)[:, ::-1].astype(np.float64)
    if len(points) > MAX_SKELETON_POINTS:
        points = points[:: -(-len(points) // MAX_SKELETON_POINTS)]

    stroke_width = 0.0
    if skeleton.any():
        distance = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
        stroke_width = float(np.mean(2.0 * distance[skeleton]))

    flat = canonicalize(ink, params.profile_columns, params.profile_rows, keep_aspect=False)
    column = cv2.GaussianBlur(flat.sum(axis=0).reshape(1, -1), (0, 0), params.profile_blur_sigma).ravel()
    row = cv2.GaussianBlur(flat.sum(axis=1).reshape(1, -1), (0, 0), params.profile_blur_sigma).ravel()
    column = column / column.sum() if column.sum() > 0 else column
    row = row / row.sum() if row.sum() > 0 else row

    smooth = cv2.GaussianBlur(canvas, (0, 0), params.slant_blur_sigma)
    gx = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    bins = (angle / np.pi * SLANT_BINS).astype(np.int64) % SLANT_BINS
    slant = np.bincount(bins.ravel(), weights=np.hypot(gx, gy).ravel(), minlength=SLANT_BINS)
    slant = slant / slant.sum() if slant.sum() > 0 else slant

    return StrokeGeometry(
        skeleton_points=points,
        orientations=_local_orientations(points, params.orientation_radius),
        column_profile=column,
        row_profile=row,
        slant_histogram=slant,
        stroke_width=stroke_width,
    )


# --------------------------------------------------------------------------- alignment

_IDENTITY = np.hstack([np.eye(2), np.zeros((2, 1))])


def _similarity_fit(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Least-squares rotation + uniform scale + translation (Umeyama), no reflection."""
    mean_src, mean_dst = src.mean(axis=0), dst.mean(axis=0)
    centred_src, centred_dst = src - mean_src, dst - mean_dst
    covariance = centred_dst.T @ centred_src / len(src)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[1, 1] = -1.0
    rotation = u @ sign @ vt
    variance = float((centred_src ** 2).sum() / len(src))
    scale = float(np.trace(np.diag(singular) @ sign) / max(variance, 1e-9))
    return scale, rotation, mean_dst - scale * rotation @ mean_src


def _transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:, :2].T + matrix[:, 2]


def _compose(newer: np.ndarray, older: np.ndarray) -> np.ndarray:
    linear = newer[:, :2] @ older[:, :2]
    shift = newer[:, :2] @ older[:, 2] + newer[:, 2]
    return np.hstack([linear, shift[:, None]])


def _icp(points: np.ndarray, target_tree: cKDTree, target: np.ndarray, start: np.ndarray,
         params: StrokeParams) -> np.ndarray:
    """Trimmed iterative-closest-point refinement of a similarity transform."""
    matrix = start.copy()
    for _ in range(params.icp_iterations):
        moved = _transform_points(matrix, points)
        distance, nearest = target_tree.query(moved)
        keep = distance <= np.quantile(distance, params.icp_trim_quantile)
        if int(keep.sum()) < MIN_SKELETON_POINTS:
            break   # too few correspondences to refine: keep the current transform (start seed on step 1)
        scale, rotation, shift = _similarity_fit(moved[keep], target[nearest[keep]])
        scale = float(np.clip(scale, params.icp_min_step_scale, params.icp_max_step_scale))
        matrix = _compose(np.hstack([scale * rotation, shift[:, None]]), matrix)
    return matrix


def _chamfer_quality(moved: np.ndarray, target: np.ndarray, target_tree: cKDTree, tolerance: float) -> float:
    """1 - mean truncated symmetric nearest-stroke distance (in units of `tolerance`)."""
    forward = target_tree.query(moved)[0]
    backward = cKDTree(moved).query(target)[0]
    return 1.0 - 0.5 * (np.minimum(forward, tolerance).mean() + np.minimum(backward, tolerance).mean()) / tolerance


def _direction_agreement(a: StrokeGeometry, b: StrokeGeometry, target_tree: cKDTree, matrix: np.ndarray,
                         params: StrokeParams) -> float:
    """Mean cos(2 * delta-direction) over strokes that land within tolerance, times coverage.

    Both directions are evaluated and averaged, so the signal is symmetric in the
    two signatures once the alignment is fixed. The product with coverage punishes
    signatures whose strokes do not overlap at all.
    """
    rotation = np.arctan2(matrix[1, 0], matrix[0, 0])
    moved = _transform_points(matrix, a.skeleton_points)
    rotated = a.orientations + rotation

    forward_distance, forward_index = target_tree.query(moved)
    forward_ok = forward_distance < params.direction_tolerance_px
    backward_distance, backward_index = cKDTree(moved).query(b.skeleton_points)
    backward_ok = backward_distance < params.direction_tolerance_px

    def mean_cosine(ok: np.ndarray, own: np.ndarray, other: np.ndarray) -> float:
        if int(ok.sum()) <= MIN_DIRECTION_MATCHES:
            return 0.0
        return float(np.mean(np.cos(2.0 * (own[ok] - other[ok]))))

    forward = mean_cosine(forward_ok, rotated, b.orientations[forward_index])
    backward = mean_cosine(backward_ok, b.orientations, rotated[backward_index])
    coverage = 0.5 * (float(forward_ok.mean()) + float(backward_ok.mean()))
    return 0.5 * (forward + backward) * coverage


def direction_signal(a: StrokeGeometry, b: StrokeGeometry, keypoint_transform: Optional[np.ndarray],
                     params: StrokeParams) -> float:
    """Align specimen `a` onto `b`, then score local stroke-direction agreement.

    Two starting points are tried (identity and, when available, the RANSAC keypoint
    transform); the one with the better chamfer quality wins. Without the second
    start, signatures written at a slant align poorly (EXP-015).

    Directional by design, like the whole comparison (specimen `a` onto questioned
    `b`): the ICP samples `a`'s points and the keypoint seed maps `a` to `b`.
    """
    if len(a.skeleton_points) < MIN_SKELETON_POINTS or len(b.skeleton_points) < MIN_SKELETON_POINTS:
        return 0.0
    target_tree = cKDTree(b.skeleton_points)
    sample = a.skeleton_points[:: params.icp_point_stride]
    starts = [_IDENTITY]
    if keypoint_transform is not None:
        starts.append(np.asarray(keypoint_transform, dtype=np.float64))
    best_quality, best_matrix = -np.inf, _IDENTITY
    for start in starts:
        matrix = _icp(sample, target_tree, b.skeleton_points, start, params)
        quality = _chamfer_quality(_transform_points(matrix, a.skeleton_points), b.skeleton_points,
                                   target_tree, params.chamfer_tolerance_px)
        if quality > best_quality:
            best_quality, best_matrix = quality, matrix
    return _direction_agreement(a, b, target_tree, best_matrix, params)


# --------------------------------------------------------------------------- 1-D signals

def _circular_emd(p: np.ndarray, q: np.ndarray) -> float:
    """Circular earth-mover distance between histograms; in [0, SLANT_MAX_EMD]."""
    cumulative = np.cumsum(p - q)
    return float(np.abs(cumulative - np.median(cumulative)).sum())


def _banded_dtw(a: np.ndarray, b: np.ndarray, band: int) -> float:
    """Length-normalised DTW cost inside a Sakoe-Chiba band (equal-length inputs)."""
    n, m = len(a), len(b)
    previous = np.full(m + 1, np.inf)
    previous[0] = 0.0
    for i in range(1, n + 1):
        current = np.full(m + 1, np.inf)
        for j in range(max(1, i - band), min(m, i + band) + 1):
            current[j] = abs(a[i - 1] - b[j - 1]) + min(previous[j], previous[j - 1], current[j - 1])
        previous = current
    return float(previous[m] / (n + m))


def _profile_similarity(a: np.ndarray, b: np.ndarray, length: int, band_fraction: float) -> float:
    """1 - DTW cost between two mass profiles resampled to `length` samples (scaled so cost is O(1))."""
    ra, rb = _resample_profile(a, length), _resample_profile(b, length)
    band = max(1, int(round(length * band_fraction)))
    return 1.0 - min(1.0, _banded_dtw(ra * length, rb * length, band))


def stroke_signals(a: StrokeGeometry, b: StrokeGeometry, keypoint_transform: Optional[np.ndarray],
                   params: StrokeParams) -> StrokeSignals:
    """All stroke-level similarity signals for specimen `a` against questioned `b`."""
    width_ratio = 0.0
    if a.stroke_width > 0 and b.stroke_width > 0:
        width_ratio = float(np.exp(-abs(np.log(a.stroke_width / b.stroke_width))))
    return StrokeSignals(
        direction_agreement=direction_signal(a, b, keypoint_transform, params),
        slant=float(np.clip(1.0 - _circular_emd(a.slant_histogram, b.slant_histogram) / SLANT_MAX_EMD, 0.0, 1.0)),
        column_profile=_profile_similarity(a.column_profile, b.column_profile,
                                           params.column_dtw_length, params.dtw_band_fraction),
        row_profile=_profile_similarity(a.row_profile, b.row_profile,
                                        params.row_dtw_length, params.dtw_band_fraction),
        stroke_width=width_ratio,
    )
