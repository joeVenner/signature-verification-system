"""Deterministic comparison functions over `SignatureFeatures`."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Sequence

import cv2
import numpy as np

from signature_verification_system.src.core.config import (
    DEFAULT_CONFIG,
    FUSION_SIGNALS,
    FusionModel,
    RepresentationParams,
)
from signature_verification_system.src.preprocessing.normalization import canonicalize
from signature_verification_system.src.verification.features import SignatureFeatures, gradient_grid_descriptor
from signature_verification_system.src.verification.stroke_geometry import stroke_signals


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
    compare() is asymmetric by design (specimen `a` → query `b`). Rotation is in
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


def compare(
    a: SignatureFeatures,
    b: SignatureFeatures,
    fusion: Optional[FusionModel] = None,
    params: Optional[RepresentationParams] = None,
) -> PairSimilarity:
    """Fuse all similarity signals of specimen `a` vs questioned `b` with the fixed logistic model."""
    f = fusion or DEFAULT_CONFIG.fusion
    p = params or DEFAULT_CONFIG.representation
    k = keypoint_similarity(a, b, p)
    alignment = aligned_shape(a, b, k, p)
    unaligned = shape_similarity(a, b)
    used = alignment is not None and alignment.shape_similarity > unaligned
    layout = alignment.shape_similarity if used else unaligned
    # The keypoint transform seeds the stroke alignment only when it is trustworthy.
    seed = k.transform if k.inliers >= p.align_min_inliers else None
    strokes = stroke_signals(a.stroke, b.stroke, seed, p.stroke)
    signals = {
        "layout": layout,
        "keypoint": k.similarity,
        "stroke_direction": strokes.direction_agreement,
        "slant": strokes.slant,
        "column_profile": strokes.column_profile,
        "row_profile": strokes.row_profile,
        "stroke_width": strokes.stroke_width,
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
