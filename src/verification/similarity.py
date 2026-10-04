"""Deterministic comparison functions over `SignatureFeatures`."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from signature_verification_system.src.core.config import (
    DEFAULT_CONFIG,
    FUSION_SIGNALS,
    FusionModel,
    RepresentationParams,
)
from signature_verification_system.src.preprocessing.normalization import (
    canonicalize,
    letterbox_transform,
    rotate_ink,
)
from signature_verification_system.src.verification.features import (
    SignatureFeatures,
    downsampled_features,
    gradient_grid_descriptor,
)
from signature_verification_system.src.verification.stroke_geometry import (
    StrokeGeometry,
    StrokeSignals,
    extract_stroke_geometry,
    stroke_signals,
)


@dataclass(frozen=True)
class KeypointMatch:
    """Geometrically verified SIFT correspondence between two signatures."""
    similarity: float     # inliers / geometric-mean keypoint count
    inliers: int
    tentative_matches: int
    ref_keypoints: int
    query_keypoints: int
    transform: Optional[np.ndarray] = None   # 2x3 similarity transform ref -> query (keypoint canvas)


@dataclass(frozen=True)
class PairSimilarity:
    """Fused comparison of one specimen with one questioned signature."""
    shape: float          # max(unaligned, aligned) layout similarity
    keypoint: KeypointMatch
    fused_logit: float
    probability: float    # match probability under equal priors (over-confident; display only)
    signals: Dict[str, float]   # every fused signal (config.FUSION_SIGNALS) plus the unfused "layout"
    alignment: Optional[Alignment] = None
    alignment_used: bool = False   # True only if the aligned layout score strictly beat the unaligned one


def shape_similarity(a: SignatureFeatures, b: SignatureFeatures) -> float:
    """Cosine similarity of global gradient-grid descriptors (in [0, 1])."""
    return float(np.clip(np.dot(a.shape_descriptor, b.shape_descriptor), 0.0, 1.0))


def keypoint_similarity(
    a: SignatureFeatures, b: SignatureFeatures, params: Optional[RepresentationParams] = None
) -> KeypointMatch:
    """Geometrically verified SIFT correspondence ratio.

    Lowe ratio test, then a RANSAC similarity transform (rotation + uniform
    scale + translation). OpenCV's RANSAC seeds its RNG with a fixed constant on
    every call, so the inlier count is reproducible.
    """
    p = params or DEFAULT_CONFIG.representation
    na, nb = len(a.keypoints), len(b.keypoints)
    if a.descriptors is None or b.descriptors is None or na < 2 or nb < 2:
        return KeypointMatch(0.0, 0, 0, na, nb)
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(a.descriptors, b.descriptors, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < p.keypoint_ratio_test * m[1].distance]
    norm = float(np.sqrt(na * nb))
    if len(good) < 4:
        # No geometric verification possible: unverified correspondences earn no
        # credit (previously len(good)/min(na, nb) on a different scale, which let
        # 3 chance matches score up to 0.75 on sparse signatures; EXP-011).
        return KeypointMatch(0.0, 0, len(good), na, nb)
    src = a.keypoints[[g.queryIdx for g in good]]
    dst = b.keypoints[[g.trainIdx for g in good]]
    transform, mask = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=p.keypoint_ransac_px,
        maxIters=2000, confidence=0.99, refineIters=10,
    )
    inliers = int(mask.sum()) if mask is not None else 0
    return KeypointMatch(inliers / norm, inliers, len(good), na, nb, transform)


@dataclass(frozen=True)
class Alignment:
    """Plausible specimen→query similarity transform recovered by RANSAC."""
    rotation_deg: float
    scale: float
    shape_similarity: float   # gradient-grid cosine after warping the specimen onto the query


def aligned_shape(
    a: SignatureFeatures, b: SignatureFeatures, k: KeypointMatch, params: Optional[RepresentationParams] = None
) -> Optional[Alignment]:
    """Shape similarity after undoing rotation/scale/shift (EXP-014).

    The coarse gradient grid is not rotation invariant, so a genuine signature
    written or scanned at a slant scores poorly. When the keypoint match yields a
    plausible transform (enough inliers, bounded rotation and scale), the
    specimen's letter-boxed ink map is warped onto the query's before the
    descriptor is computed. Returns None when no trustworthy alignment exists.
    The layout score is asymmetric by design (specimen `a` → query `b`). Rotation is in
    image coordinates (y down): a positive angle turns the specimen clockwise.
    """
    p = params or DEFAULT_CONFIG.representation
    if k.transform is None or k.inliers < p.align_min_inliers:
        return None
    m = k.transform
    scale = float(np.hypot(m[0, 0], m[1, 0]))
    rotation = float(np.degrees(np.arctan2(m[1, 0], m[0, 0])))
    if abs(rotation) > p.align_max_rotation_deg or not (p.align_min_scale < scale < p.align_max_scale):
        return None
    w, h = p.keypoint_canvas_width, p.keypoint_canvas_height
    ca = canonicalize(a.normalized.ink, w, h, keep_aspect=True)
    cb = canonicalize(b.normalized.ink, w, h, keep_aspect=True)
    warped = cv2.warpAffine(ca, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    da = gradient_grid_descriptor(warped, p.shape_grid_rows, p.shape_grid_cols, p.shape_bins, p.align_blur_sigma)
    db = gradient_grid_descriptor(cb, p.shape_grid_rows, p.shape_grid_cols, p.shape_bins, p.align_blur_sigma)
    return Alignment(rotation, scale, float(np.clip(np.dot(da, db), 0.0, 1.0)))


def derotation_angle(k: KeypointMatch, params: Optional[RepresentationParams] = None) -> Optional[float]:
    """Rotation (degrees, OpenCV convention) that undoes the query's rotation, or None.

    Slant, ink profiles and stroke width are measured in the image frame, so a
    questioned signature captured at an angle loses on all of them although the
    strokes are the same. When the RANSAC keypoint transform is trustworthy and
    reports a rotation in [derotate_min_deg, derotate_max_deg], the query is
    turned back into the specimen's frame before its stroke signals are measured.
    Small rotations are left alone: they are within normal writer variation and
    their estimates are noisy (see config.RepresentationParams.derotate_min_deg).
    """
    p = params or DEFAULT_CONFIG.representation
    if k.transform is None or k.inliers < p.align_min_inliers:
        return None
    m = k.transform
    scale = float(np.hypot(m[0, 0], m[1, 0]))
    if not (p.align_min_scale < scale < p.align_max_scale):
        return None
    # The transform maps specimen -> query in image coordinates (y down), so the
    # query is turned by -rotation in OpenCV's counter-clockwise convention.
    rotation = float(np.degrees(np.arctan2(m[1, 0], m[0, 0])))
    if not (p.derotate_min_deg <= abs(rotation) <= p.derotate_max_deg):
        return None
    return rotation


def _query_stroke_geometry(
    a: SignatureFeatures, b: SignatureFeatures, k: KeypointMatch, p: RepresentationParams
) -> Tuple[StrokeGeometry, Optional[np.ndarray]]:
    """Stroke geometry of the query in the specimen's frame, plus the ICP seed transform.

    The keypoint transform seeds the stroke alignment when it is trustworthy;
    after derotation it is re-expressed in the derotated query's canvas
    (specimen canvas -> query canvas -> query crop -> derotated crop -> its canvas).
    """
    seed = _trusted_seed(k, p)
    angle = derotation_angle(k, p)
    if angle is None:
        return b.stroke, seed
    w, h = p.keypoint_canvas_width, p.keypoint_canvas_height
    upright, crop_rotation = rotate_ink(b.normalized.ink, angle)
    to_canvas = _homogeneous(letterbox_transform(upright.shape, w, h))
    from_canvas = np.linalg.inv(_homogeneous(letterbox_transform(b.normalized.ink.shape, w, h)))
    seed = (to_canvas @ _homogeneous(crop_rotation) @ from_canvas @ _homogeneous(k.transform))[:2]
    return extract_stroke_geometry(upright, p.stroke, w, h), seed


def _homogeneous(affine: np.ndarray) -> np.ndarray:
    """3x3 homogeneous form of a 2x3 affine matrix."""
    return np.vstack([np.asarray(affine, dtype=np.float64), [0.0, 0.0, 1.0]])


def proportion_agreement(a: SignatureFeatures, b: SignatureFeatures) -> float:
    """exp(-|log aspect-ratio ratio|): 1.0 = identical width:height proportions."""
    return float(np.exp(-abs(np.log(a.aspect_ratio / b.aspect_ratio))))


def ink_density_agreement(a: SignatureFeatures, b: SignatureFeatures) -> float:
    return float(min(a.ink_density, b.ink_density) / max(a.ink_density, b.ink_density, 1e-9))


# Fused signal -> name reported by the explanation layer (and its reference file).
EVIDENCE_SIGNAL_NAMES: Mapping[str, str] = MappingProxyType({
    "stroke_direction": "stroke_direction_agreement",
    "slant": "slant_agreement",
    "column_profile": "horizontal_profile_agreement",
    "row_profile": "vertical_profile_agreement",
    "stroke_width": "stroke_width_agreement",
    "pressure_pattern": "pressure_pattern_agreement",
    "curvature": "stroke_curvature_agreement",
})


def pair_evidence_signals(pair: PairSimilarity) -> Dict[str, float]:
    """The stroke-level signals of `pair`, keyed by their explanation names."""
    return {EVIDENCE_SIGNAL_NAMES[name]: pair.signals[name] for name in EVIDENCE_SIGNAL_NAMES}


def explanation_signals(
    a: SignatureFeatures, b: SignatureFeatures, pair: Optional[PairSimilarity] = None
) -> Dict[str, float]:
    """Measured signals reported by the explanation layer (diagnostic only)."""
    p = pair or compare(a, b)
    return {
        "shape_similarity": p.shape,
        "keypoint_similarity": p.keypoint.similarity,
        "keypoint_inliers": float(p.keypoint.inliers),
        "proportion_agreement": proportion_agreement(a, b),
        "ink_density_agreement": ink_density_agreement(a, b),
        **pair_evidence_signals(p),
    }


def fuse_signals(signals: Dict[str, float], fusion: FusionModel) -> float:
    """Logistic fusion: bias + sum(weight * signal) over `FUSION_SIGNALS` (log-odds)."""
    return float(fusion.bias + sum(fusion.signal_weights[name] * signals[name] for name in FUSION_SIGNALS))


def _trusted_seed(k: KeypointMatch, p: RepresentationParams) -> Optional[np.ndarray]:
    """The keypoint transform seeds the stroke alignment only when it is trustworthy."""
    return k.transform if k.inliers >= p.align_min_inliers else None


def _reverse_aligned_signals(a: SignatureFeatures, b: SignatureFeatures, p: RepresentationParams) -> StrokeSignals:
    """Stroke signals with the roles swapped: questioned `b` aligned onto specimen `a` (EXP-019).

    The keypoint match is recomputed in the b -> a direction (the Lowe ratio test is not
    symmetric), exactly as compare(b, a) would do, so the reverse alignment gets its own seed.
    """
    return stroke_signals(b.stroke, a.stroke, _trusted_seed(keypoint_similarity(b, a, p), p), p.stroke)


def harmonize_scale(
    a: SignatureFeatures, b: SignatureFeatures, params: Optional[RepresentationParams] = None
) -> Tuple[SignatureFeatures, SignatureFeatures]:
    """Bring two signatures to a common ink size by downsampling the larger one (EXP-021).

    Photometric normalisation works in native pixels (denoising window, paper kernel,
    ink blur), so the same signature scanned at another resolution comes out with
    relatively fatter or thinner strokes (stroke width +35% at 0.5x, -17% at 2x on
    dev). When the ink radius-of-gyration ratio leaves [scale_band_low,
    scale_band_high], the larger image is re-extracted from its source, shrunk so both
    radii match. Only ever downsamples (no invented detail); pairs inside the band,
    which is nearly every same-resolution pair, are returned untouched.
    """
    p = params or DEFAULT_CONFIG.representation
    if a.ink_radius <= 0 or b.ink_radius <= 0:
        return a, b
    ratio = a.ink_radius / b.ink_radius
    if p.scale_band_low <= ratio <= p.scale_band_high:
        return a, b
    if ratio > 1.0:
        return downsampled_features(a, 1.0 / ratio, p), b
    return a, downsampled_features(b, ratio, p)


def compare(
    a: SignatureFeatures,
    b: SignatureFeatures,
    fusion: Optional[FusionModel] = None,
    params: Optional[RepresentationParams] = None,
) -> PairSimilarity:
    """Fuse all similarity signals of specimen `a` vs questioned `b` with the fixed logistic model.

    The two signals read through the ICP stroke alignment (stroke direction, pressure
    pattern) take the better of the a -> b and b -> a alignments, which makes them
    symmetric in `a` and `b` (EXP-019). Every other fused signal is symmetric or nearly
    so already; layout and keypoint keep the specimen -> query direction. Measured on dev:
    CV skilled EER 10.58% -> 9.59% for ~5 ms more per pair (5.4 -> 10.2 ms; one feature
    extraction is ~180 ms).
    """
    f = fusion or DEFAULT_CONFIG.fusion
    p = params or DEFAULT_CONFIG.representation
    a, b = harmonize_scale(a, b, p)
    k = keypoint_similarity(a, b, p)
    alignment = aligned_shape(a, b, k, p)
    unaligned = shape_similarity(a, b)
    used = alignment is not None and alignment.shape_similarity > unaligned
    layout = alignment.shape_similarity if used else unaligned
    query_stroke, seed = _query_stroke_geometry(a, b, k, p)
    strokes = stroke_signals(a.stroke, query_stroke, seed, p.stroke)
    if query_stroke is b.stroke:
        reverse = _reverse_aligned_signals(a, b, p)
    else:
        # Derotated query: the reverse alignment starts from the inverse of the forward seed.
        reverse = stroke_signals(query_stroke, a.stroke, np.linalg.inv(_homogeneous(seed))[:2], p.stroke)
    signals = {
        "layout": layout,
        "keypoint": k.similarity,
        "stroke_direction": max(strokes.direction_agreement, reverse.direction_agreement),
        "slant": strokes.slant,
        "column_profile": strokes.column_profile,
        "row_profile": strokes.row_profile,
        "stroke_width": strokes.stroke_width,
        "pressure_pattern": max(strokes.pressure_pattern, reverse.pressure_pattern),
        "curvature": strokes.curvature,
    }
    logit = fuse_signals(signals, f)
    prob = float(1.0 / (1.0 + np.exp(-logit)))
    return PairSimilarity(shape=layout, keypoint=k, fused_logit=logit, probability=prob, signals=signals,
                          alignment=alignment, alignment_used=used)


@dataclass(frozen=True)
class MultiReferenceSimilarity:
    per_reference: tuple            # PairSimilarity for each reference, in input order
    best_shape_index: int
    best_keypoint_index: int
    fused_logit: float
    probability: float


def compare_multi(
    references: Sequence[SignatureFeatures],
    query: SignatureFeatures,
    fusion: Optional[FusionModel] = None,
    params: Optional[RepresentationParams] = None,
) -> MultiReferenceSimilarity:
    """Signal-wise max aggregation over enrolled references, then fusion.

    Each similarity signal takes its best value over the references independently
    (a genuine signature may match one specimen's layout and another's stroke
    detail). Chosen in EXP-003 over mean/max/median/top-2 of fused scores and
    cohort z-normalisation; ties resolve to the lowest reference index.
    """
    if not references:
        raise ValueError("At least one reference signature is required")
    f = fusion or DEFAULT_CONFIG.fusion
    pairs = tuple(compare(r, query, f, params) for r in references)
    best = {name: max(p.signals[name] for p in pairs) for name in FUSION_SIGNALS}
    i_s = int(np.argmax([p.shape for p in pairs]))
    i_k = int(np.argmax([p.keypoint.similarity for p in pairs]))
    logit = fuse_signals(best, f)
    prob = float(1.0 / (1.0 + np.exp(-logit)))
    return MultiReferenceSimilarity(pairs, i_s, i_k, logit, prob)
