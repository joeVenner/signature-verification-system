"""Stroke-level signals for comparing one specimen with one questioned signature.

The v3 verifier originally decided on two signals only (coarse layout and SIFT
inliers). A human examiner also compares *how* the strokes run: slant, stroke
direction along matching strokes, proportions of the horizontal and vertical ink
distribution, and pen width. Skilled forgers copy the outline but rarely get
these right, so they add independent evidence (EXP-015).

EXP-017 adds two stroke-quality signals aimed at skilled forgeries that copy the
shape but are drawn slowly: where along the strokes the ink is darker or lighter
(pressure and speed; compared by rank, so scan tone cannot move it) and the
distribution of contour curvature (smooth fluent arcs vs. angular, hesitant lines).

All steps are deterministic: skeletonisation, KD-tree nearest neighbours, SVD and
banded dynamic programming contain no randomness and no data-dependent ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from scipy.stats import rankdata
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
# Contour curvature x stroke width is dimensionless; real signatures put ~all of their
# contour mass below 1.0 (tighter turns than the pen width are clipped into the last bin).
CURVATURE_BINS = 20
CURVATURE_MAX = 1.0
# Largest possible (linear) EMD between two CURVATURE_BINS histograms: all mass moved
# from the first bin to the last. Dividing by it maps the distance onto [0, 1].
CURVATURE_MAX_EMD = float(CURVATURE_BINS - 1)
# A rank correlation over fewer corresponding stroke points is mostly noise.
MIN_PRESSURE_MATCHES = 20


@dataclass(frozen=True)
class StrokeGeometry:
    """Per-signature stroke descriptors (extracted once, compared many times)."""

    skeleton_points: np.ndarray    # (N, 2) float64 (x, y) on the keypoint canvas
    orientations: np.ndarray       # (N,) local stroke direction in radians, [0, pi)
    column_profile: np.ndarray     # ink mass per column, aspect-normalised, sums to 1
    row_profile: np.ndarray        # ink mass per row, aspect-normalised, sums to 1
    slant_histogram: np.ndarray    # (SLANT_BINS,) gradient-orientation mass mod pi, sums to 1
    stroke_width: float            # mean stroke width in canvas pixels (0.0 when unmeasurable)
    point_darkness: np.ndarray     # (N,) ink darkness at each skeleton point, averaged along the stroke
    curvature_histogram: np.ndarray  # (CURVATURE_BINS,) contour curvature x stroke width, sums to 1 (or all 0)


@dataclass(frozen=True)
class StrokeSignals:
    """Similarity signals between two `StrokeGeometry` objects, all in higher-is-more-similar form."""

    direction_agreement: float     # local stroke-direction agreement x coverage, after alignment
    slant: float                   # 1 - normalised circular earth-mover distance of slant histograms, in [0, 1]
    column_profile: float          # banded-DTW similarity of horizontal ink profiles
    row_profile: float             # banded-DTW similarity of vertical ink profiles
    stroke_width: float            # exp(-|log width ratio|)
    pressure_pattern: float        # rank correlation of darkness at corresponding stroke points x coverage, [-1, 1]
    curvature: float               # 1 - normalised EMD of contour-curvature histograms, in [0, 1]


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


def _along_stroke_mean(points: np.ndarray, values: np.ndarray, radius: float) -> np.ndarray:
    """Mean of `values` over the skeleton points within `radius` of each point (itself included).

    Single-pixel darkness is dominated by scan noise and anti-aliasing; averaging along
    the stroke keeps the slower pressure/speed variation a writer produces.
    """
    sums = values.astype(np.float64).copy()
    if len(points) < 2:
        return sums
    pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
    counts = np.ones(len(points))
    np.add.at(sums, pairs[:, 0], values[pairs[:, 1]])
    np.add.at(sums, pairs[:, 1], values[pairs[:, 0]])
    np.add.at(counts, pairs[:, 0], 1.0)
    np.add.at(counts, pairs[:, 1], 1.0)
    return sums / counts


def _curvature_histogram(binary: np.ndarray, stroke_width: float, min_points: int) -> np.ndarray:
    """Distribution of contour curvature, made dimensionless by the stroke width (EXP-017).

    Fluent writing has long smooth arcs; slow, drawn forgeries add angular corners,
    wobble and blunt stroke ends, which move contour mass to higher curvature. Each
    contour is smoothed at a scale of one stroke width so that pixel staircases do not
    count as curvature, and curvature is multiplied by the width so the descriptor does
    not depend on image resolution. All zeros when there is no measurable stroke.
    """
    histogram = np.zeros(CURVATURE_BINS)
    if stroke_width <= 0:
        return histogram
    sigma = max(1.0, stroke_width)
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    edges = np.linspace(0.0, CURVATURE_MAX, CURVATURE_BINS + 1)
    top = np.nextafter(CURVATURE_MAX, 0.0)   # sharper turns are counted in the last bin
    for contour in contours:
        points = contour[:, 0, :].astype(np.float64)
        if len(points) < min_points:
            continue
        xs = gaussian_filter1d(points[:, 0], sigma, mode="wrap")   # contours are closed curves
        ys = gaussian_filter1d(points[:, 1], sigma, mode="wrap")
        dx, dy = np.gradient(xs), np.gradient(ys)
        ddx, ddy = np.gradient(dx), np.gradient(dy)
        curvature = np.abs(dx * ddy - dy * ddx) / np.maximum((dx * dx + dy * dy) ** 1.5, 1e-6)
        histogram += np.histogram(np.clip(curvature * stroke_width, 0.0, top), edges)[0]
    total = histogram.sum()
    return histogram / total if total > 0 else histogram


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

    # Relative darkness pattern along the strokes (pen pressure / speed). Only its rank
    # order is ever compared (stroke_signals), so a scanner tone curve cannot move it.
    blurred = cv2.GaussianBlur(canvas, (0, 0), params.pressure_blur_sigma)
    pixel = points.astype(np.int64)
    darkness = _along_stroke_mean(points, blurred[pixel[:, 1], pixel[:, 0]], params.pressure_radius)

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
        point_darkness=darkness,
        curvature_histogram=_curvature_histogram(binary, stroke_width, params.curvature_min_contour_points),
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


@dataclass(frozen=True)
class _Correspondence:
    """Nearest-stroke pairing of aligned specimen points with questioned points, both ways."""

    rotation: float                # rotation of the alignment (radians)
    forward_index: np.ndarray      # for each specimen point: nearest questioned point
    forward_ok: np.ndarray         # ... and whether it lies within the tolerance
    backward_index: np.ndarray     # for each questioned point: nearest aligned specimen point
    backward_ok: np.ndarray

    @property
    def coverage(self) -> float:
        return 0.5 * (float(self.forward_ok.mean()) + float(self.backward_ok.mean()))


def _correspond(a: StrokeGeometry, b: StrokeGeometry, target_tree: cKDTree, matrix: np.ndarray,
                params: StrokeParams) -> _Correspondence:
    moved = _transform_points(matrix, a.skeleton_points)
    forward_distance, forward_index = target_tree.query(moved)
    backward_distance, backward_index = cKDTree(moved).query(b.skeleton_points)
    return _Correspondence(
        rotation=float(np.arctan2(matrix[1, 0], matrix[0, 0])),
        forward_index=forward_index,
        forward_ok=forward_distance < params.direction_tolerance_px,
        backward_index=backward_index,
        backward_ok=backward_distance < params.direction_tolerance_px,
    )


def _direction_agreement(a: StrokeGeometry, b: StrokeGeometry, match: _Correspondence) -> float:
    """Mean cos(2 * delta-direction) over strokes that land within tolerance, times coverage.

    Both directions are evaluated and averaged, so the signal is symmetric in the
    two signatures once the alignment is fixed. The product with coverage punishes
    signatures whose strokes do not overlap at all.
    """
    rotated = a.orientations + match.rotation

    def mean_cosine(ok: np.ndarray, own: np.ndarray, other: np.ndarray) -> float:
        if int(ok.sum()) <= MIN_DIRECTION_MATCHES:
            return 0.0
        return float(np.mean(np.cos(2.0 * (own[ok] - other[ok]))))

    forward = mean_cosine(match.forward_ok, rotated, b.orientations[match.forward_index])
    backward = mean_cosine(match.backward_ok, b.orientations, rotated[match.backward_index])
    return 0.5 * (forward + backward) * match.coverage


def align_strokes(a: StrokeGeometry, b: StrokeGeometry, keypoint_transform: Optional[np.ndarray],
                  params: StrokeParams) -> Optional[Tuple[np.ndarray, cKDTree]]:
    """Similarity transform mapping specimen `a`'s skeleton onto `b`'s, plus `b`'s KD-tree.

    Two starting points are tried (identity and, when available, the RANSAC keypoint
    transform); the one with the better chamfer quality wins. Without the second
    start, signatures written at a slant align poorly (EXP-015). Returns None when
    either skeleton is too small to align.

    Directional by design, like the whole comparison (specimen `a` onto questioned
    `b`): the ICP samples `a`'s points and the keypoint seed maps `a` to `b`.
    """
    if len(a.skeleton_points) < MIN_SKELETON_POINTS or len(b.skeleton_points) < MIN_SKELETON_POINTS:
        return None
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
    return best_matrix, target_tree


def direction_signal(a: StrokeGeometry, b: StrokeGeometry, keypoint_transform: Optional[np.ndarray],
                     params: StrokeParams) -> float:
    """Align specimen `a` onto `b`, then score local stroke-direction agreement (0.0 if unalignable)."""
    aligned = align_strokes(a, b, keypoint_transform, params)
    if aligned is None:
        return 0.0
    matrix, target_tree = aligned
    return _direction_agreement(a, b, _correspond(a, b, target_tree, matrix, params))


def _rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation (average ranks for ties); 0.0 when undefined or too few samples."""
    if len(x) < MIN_PRESSURE_MATCHES:
        return 0.0
    rx, ry = rankdata(x), rankdata(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return 0.0
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))


def _pressure_agreement(a: StrokeGeometry, b: StrokeGeometry, match: _Correspondence) -> float:
    """Do the two signatures write darker / lighter at the same places along the strokes? (EXP-017)

    Pairs every stroke point that lands within tolerance of the other signature (both
    directions, like the direction agreement), then takes the rank correlation of the
    along-stroke darkness at the paired points, times coverage. A fluent writer
    repeats where strokes thin out and darken (speed and pressure follow the motor
    program); a forger tracing the outline slowly does not. Rank correlation is
    invariant to any monotone tone mapping, so scanner exposure and the CEDAR
    scan-session difference cannot drive it (same-vs-cross session AUC 0.508 on
    different-writer dev pairs, vs 0.609 for a darkness-histogram comparison).
    """
    fwd, bwd = match.forward_ok, match.backward_ok
    specimen = np.concatenate([a.point_darkness[fwd], a.point_darkness[match.backward_index[bwd]]])
    questioned = np.concatenate([b.point_darkness[match.forward_index[fwd]], b.point_darkness[bwd]])
    return _rank_correlation(specimen, questioned) * match.coverage


# --------------------------------------------------------------------------- 1-D signals

def _circular_emd(p: np.ndarray, q: np.ndarray) -> float:
    """Circular earth-mover distance between histograms; in [0, SLANT_MAX_EMD]."""
    cumulative = np.cumsum(p - q)
    return float(np.abs(cumulative - np.median(cumulative)).sum())


def _linear_emd(p: np.ndarray, q: np.ndarray) -> float:
    """Earth-mover distance between two histograms on an ordered (non-circular) axis."""
    return float(np.abs(np.cumsum(p - q)).sum())


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
    direction, pressure = 0.0, 0.0
    aligned = align_strokes(a, b, keypoint_transform, params)   # one alignment serves both signals
    if aligned is not None:
        match = _correspond(a, b, aligned[1], aligned[0], params)
        direction = _direction_agreement(a, b, match)
        pressure = _pressure_agreement(a, b, match)
    curvature = 0.0
    if a.curvature_histogram.sum() > 0 and b.curvature_histogram.sum() > 0:
        distance = _linear_emd(a.curvature_histogram, b.curvature_histogram)
        curvature = float(np.clip(1.0 - distance / CURVATURE_MAX_EMD, 0.0, 1.0))
    return StrokeSignals(
        direction_agreement=direction,
        slant=float(np.clip(1.0 - _circular_emd(a.slant_histogram, b.slant_histogram) / SLANT_MAX_EMD, 0.0, 1.0)),
        column_profile=_profile_similarity(a.column_profile, b.column_profile,
                                           params.column_dtw_length, params.dtw_band_fraction),
        row_profile=_profile_similarity(a.row_profile, b.row_profile,
                                        params.row_dtw_length, params.dtw_band_fraction),
        stroke_width=width_ratio,
        pressure_pattern=pressure,
        curvature=curvature,
    )
