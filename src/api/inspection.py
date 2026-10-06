"""1:1 signature comparison plus the real pipeline intermediates, for the live console.

Each image is prepared, gated and feature-extracted once (`gate_and_extract`);
the verdict is the verifier's result on those features (exactly what
`/signature/compare` returns) and the intermediates are rendered from the same
features. The fusion is re-run with the verifier's own configuration; the
recomputed log-odds is then checked against the verdict's log-odds and the
signal breakdown is withheld if they ever disagree, so the console can never
show a decomposition that does not add up to the decision.
"""

from __future__ import annotations

import time
from typing import Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

from signature_verification_system.src.api import visuals
from signature_verification_system.src.core.config import FUSION_SIGNALS, RepresentationParams
from signature_verification_system.src.preprocessing.quality import SignatureQuality, quality_message
from signature_verification_system.src.verification.deterministic import (
    DeterministicVerifier,
    GatedSignature,
    gate_and_extract,
)
from signature_verification_system.src.verification.features import (
    SignatureFeatures,
    capture_scale_plan,
    match_capture_scale,
)
from signature_verification_system.src.verification.signature_compare import (
    SCORE_ACCEPT_ANCHOR,
    SCORE_REJECT_ANCHOR,
    SCORE_TAIL_LOGODDS,
    SignatureComparison,
    comparison_from_result,
)
from signature_verification_system.src.verification.similarity import PairSimilarity, compare

LOG_ODDS_DECIMALS = 6   # precision of SignatureComparison.log_odds
SUM_TOLERANCE = 1e-9
EXTRACTION_FAILED_MESSAGE = "No signature could be isolated in this image."
QUALITY_BLOCKED_MESSAGE = "Image did not pass the signature quality gate; no intermediates were computed."


SIGNAL_LABELS: Dict[str, str] = {
    "keypoint": "Keypoint correspondence",
    "stroke_direction": "Stroke direction",
    "slant": "Slant distribution",
    "column_profile": "Horizontal ink profile",
    "row_profile": "Vertical ink profile",
    "stroke_width": "Pen width",
    "layout": "Global layout",
}


class StrokeStats(BaseModel):
    skeleton_points: int
    keypoints: int = Field(..., description="All SIFT keypoints used for matching")
    keypoints_off_stroke: int = Field(
        ..., description="Keypoints whose centre lies off the strokes (coarse-scale blobs); counted, not drawn")
    stroke_width_px: float = Field(..., description="Mean pen width on the comparison canvas")
    ink_density: float
    aspect_ratio: float


class QualityFinding(BaseModel):
    code: str
    message: str = Field(..., description="Plain-language text for the code")


class ImageQualityView(BaseModel):
    """What the signature quality gate found and how the image was routed."""
    passed: bool
    blocking_issues: List[QualityFinding] = Field(default_factory=list)
    warnings: List[QualityFinding] = Field(default_factory=list)
    background_removed: bool = Field(..., description="Ink was extracted from a textured (cheque) background")
    background_texture: float = Field(..., description="Share of background pixels covered by security texture")


class ImageInspection(BaseModel):
    """Intermediates for one input image; fields are None when that stage did not run."""
    width: int
    height: int
    original_png: str
    harmonised_png: Optional[str] = None
    ink_crop_png: Optional[str] = None
    strokes_png: Optional[str] = None
    bbox: Optional[List[int]] = Field(
        None, description="(x, y, w, h) of the signature in the processed image (input x processing_scale)")
    processing_scale: Optional[float] = Field(
        None, description="Processed / input linear scale: > 1 low-resolution upscale, < 1 fine-capture downsample")
    quality: Optional[ImageQualityView] = None
    stats: Optional[StrokeStats] = None
    error: Optional[str] = None


class CaptureScaleMatch(BaseModel):
    """The finer image was re-extracted at the coarser one's capture scale before scoring (EXP-027)."""
    downsampled: Literal["reference", "questioned"]
    factor: float = Field(..., description="Linear downsampling factor applied to that image")
    scored_stroke_width_px: float = Field(..., description="Its canvas pen width after matching, as scored")
    statement: str


class SignalContribution(BaseModel):
    name: str
    label: str
    value: float
    fused: bool
    weight: Optional[float] = None
    contribution: Optional[float] = Field(None, description="weight x value, in log-odds")


class FusionBreakdown(BaseModel):
    signals: List[SignalContribution]
    contributions_total: float = Field(..., description="Sum of weight x value over the fused signals")
    bias: float
    log_odds: float = Field(..., description="bias + contributions_total, as computed by the fusion model")


class AlignmentInspection(BaseModel):
    overlay_png: str
    transform_source: Literal["keypoint_ransac", "none"]
    description: str
    matrix: Optional[List[List[float]]] = None
    rotation_deg: Optional[float] = None
    scale: Optional[float] = None
    keypoint_inliers: int
    tentative_matches: int
    reference_keypoints: int
    questioned_keypoints: int
    layout_alignment_used: bool = Field(..., description="Layout score came from the aligned comparison")


class DecisionMargin(BaseModel):
    nearest_threshold: Literal["MATCH", "NO MATCH"]
    threshold_log_odds: float
    distance_log_odds: float
    side: Literal["above", "below"]
    statement: str


class Thresholds(BaseModel):
    accept_log_odds: Optional[float]
    reject_log_odds: Optional[float]
    score_reject_anchor: float = SCORE_REJECT_ANCHOR
    score_accept_anchor: float = SCORE_ACCEPT_ANCHOR
    score_tail_log_odds: float = SCORE_TAIL_LOGODDS


class Consistency(BaseModel):
    log_odds_matches: Optional[bool] = Field(None, description="Recomputed log-odds equals the verdict's log-odds")
    contributions_sum_matches: Optional[bool] = None


class Timing(BaseModel):
    comparison_ms: float
    visuals_ms: float
    total_ms: float


class SignatureInspection(BaseModel):
    comparison: SignatureComparison
    reference: ImageInspection
    questioned: ImageInspection
    alignment: Optional[AlignmentInspection] = None
    capture_scale: Optional[CaptureScaleMatch] = None
    fusion: Optional[FusionBreakdown] = None
    margin: Optional[DecisionMargin] = None
    thresholds: Thresholds
    consistency: Consistency
    timing: Timing


def _features_or_error(gated: GatedSignature) -> tuple[Optional[SignatureFeatures], Optional[str]]:
    """Features for the visuals, only for images the quality gate lets through (same rule as verify).

    Extraction error texts are logged by `gate_and_extract`, never returned: a
    message could expose internals.
    """
    if not gated.quality.passed:
        return None, QUALITY_BLOCKED_MESSAGE
    if gated.features is None:
        return None, EXTRACTION_FAILED_MESSAGE
    return gated.features, None


def quality_view(quality: SignatureQuality) -> ImageQualityView:
    """Gate result with a plain-language message per code."""
    return ImageQualityView(
        passed=quality.passed,
        blocking_issues=[QualityFinding(code=c, message=quality_message(c)) for c in quality.blocking_issues],
        warnings=[QualityFinding(code=c, message=quality_message(c)) for c in quality.warnings],
        background_removed=quality.background_removed,
        background_texture=quality.background_texture,
    )


def inspect_image(image: np.ndarray, feats: Optional[SignatureFeatures], error: Optional[str],
                  params: RepresentationParams, quality: Optional[SignatureQuality] = None) -> ImageInspection:
    """Render every intermediate that exists for one image."""
    h, w = image.shape[:2]
    base = ImageInspection(width=w, height=h, original_png=visuals.render_original(image), error=error,
                           quality=None if quality is None else quality_view(quality))
    if feats is None:
        return base
    norm, stroke = feats.normalized, feats.stroke
    skeleton = getattr(stroke, "skeleton_points", None)
    canvas_w, canvas_h = params.keypoint_canvas_width, params.keypoint_canvas_height
    on_stroke = visuals.on_stroke_keypoints(norm.ink, feats.keypoints, canvas_w, canvas_h, params.stroke.ink_threshold)
    return base.model_copy(update={
        "harmonised_png": visuals.render_harmonised(norm.gray, norm.bbox),
        "ink_crop_png": visuals.render_ink_crop(norm.ink),
        "strokes_png": visuals.render_strokes(norm.ink, skeleton, feats.keypoints[on_stroke], canvas_w, canvas_h),
        "bbox": [int(v) for v in norm.bbox],
        "processing_scale": round(norm.gray.shape[1] / w, 4),
        "stats": StrokeStats(
            skeleton_points=0 if skeleton is None else int(len(skeleton)),
            keypoints=int(len(feats.keypoints)),
            keypoints_off_stroke=int(len(feats.keypoints) - np.count_nonzero(on_stroke)),
            stroke_width_px=round(float(getattr(stroke, "stroke_width", 0.0)), 3),
            ink_density=round(float(feats.ink_density), 4),
            aspect_ratio=round(float(feats.aspect_ratio), 4),
        ),
    })


def fusion_breakdown(pair: PairSimilarity, verifier: DeterministicVerifier) -> FusionBreakdown:
    """Each fused signal with its weight and contribution; unfused signals listed with no weight."""
    weights = verifier.config.fusion.signal_weights
    rows = [SignalContribution(name=n, label=SIGNAL_LABELS.get(n, n.replace("_", " ").capitalize()),
                               value=round(pair.signals[n], 6), fused=True, weight=weights[n],
                               contribution=round(weights[n] * pair.signals[n], 6))
            for n in FUSION_SIGNALS]
    rows += [SignalContribution(name=n, label=SIGNAL_LABELS.get(n, n.replace("_", " ").capitalize()),
                                value=round(v, 6), fused=False)
             for n, v in sorted(pair.signals.items()) if n not in FUSION_SIGNALS]
    total = sum(weights[n] * pair.signals[n] for n in FUSION_SIGNALS)
    return FusionBreakdown(signals=rows, contributions_total=round(total, 6), bias=verifier.config.fusion.bias,
                           log_odds=round(pair.fused_logit, 6))


def alignment_inspection(ref: SignatureFeatures, que: SignatureFeatures, pair: PairSimilarity,
                         verifier: DeterministicVerifier) -> AlignmentInspection:
    """Overlay using the transform the comparison actually trusted (or none)."""
    p = verifier.config.representation
    k = pair.keypoint
    trusted = k.transform is not None and k.inliers >= p.align_min_inliers
    matrix = np.asarray(k.transform, dtype=np.float64) if trusted else None
    overlay = visuals.render_alignment_overlay(
        ref.stroke.skeleton_points, que.stroke.skeleton_points, que.normalized.ink, matrix,
        p.keypoint_canvas_width, p.keypoint_canvas_height)
    if trusted:
        description = (f"Reference strokes mapped with the RANSAC keypoint transform ({k.inliers} inliers). "
                       "It is the transform tried for the aligned layout score and one of two starting points "
                       "(with the identity) for the stroke-direction alignment; which start won and the "
                       "ICP-refined result are not exposed by the engine. Stroke direction and pressure "
                       "pattern also align the questioned strokes onto the reference and keep the better "
                       "of the two directions.")
        rotation = round(float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))), 2)
        scale = round(float(np.hypot(matrix[0, 0], matrix[1, 0])), 4)
        rows = [[round(float(v), 6) for v in row] for row in matrix]
    else:
        description = (f"No trustworthy keypoint transform ({k.inliers} inliers, {p.align_min_inliers} required): "
                       "strokes shown unaligned; stroke direction was aligned from the identity start only "
                       "(in each direction, keeping the better).")
        rotation = scale = rows = None
    return AlignmentInspection(
        overlay_png=overlay, transform_source="keypoint_ransac" if trusted else "none", description=description,
        matrix=rows, rotation_deg=rotation, scale=scale, keypoint_inliers=k.inliers,
        tentative_matches=k.tentative_matches, reference_keypoints=k.ref_keypoints,
        questioned_keypoints=k.query_keypoints, layout_alignment_used=pair.alignment_used,
    )


def capture_scale_match(ref: SignatureFeatures, que: SignatureFeatures,
                        p: RepresentationParams) -> Optional[CaptureScaleMatch]:
    """What `compare` did about a capture-scale mismatch, or None if it compared the images as extracted.

    Re-runs the same `match_capture_scale` (its re-extraction is cached); a failed
    re-extraction falls back to the unmatched features and is reported as None.
    """
    plan = capture_scale_plan(ref, que)
    if plan is None:
        return None
    ref_is_finer, factor = plan
    matched_ref, matched_que = match_capture_scale(ref, que, p)
    matched, original = (matched_ref, ref) if ref_is_finer else (matched_que, que)
    if matched is original:
        return None
    role = "reference" if ref_is_finer else "questioned"
    other = "questioned" if ref_is_finer else "reference"
    width = round(float(getattr(matched.stroke, "stroke_width", 0.0)), 3)
    return CaptureScaleMatch(
        downsampled=role, factor=factor, scored_stroke_width_px=width,
        statement=(f"Capture scale matched: the {role} image was downsampled ×{factor:.2f} to the {other} "
                   f"image's coarser capture before scoring. Its pen width as scored is {width:.2f} px; "
                   "the Stroke representation stage shows it as submitted."),
    )


def decision_margin(log_odds: Optional[float], reject_t: Optional[float],
                    accept_t: Optional[float]) -> Optional[DecisionMargin]:
    """Distance of the log-odds from the nearest decision threshold (no probabilities)."""
    if log_odds is None or reject_t is None or accept_t is None:
        return None
    to_accept, to_reject = abs(log_odds - accept_t), abs(log_odds - reject_t)
    name, t = ("MATCH", accept_t) if to_accept <= to_reject else ("NO MATCH", reject_t)
    side = "above" if log_odds >= t else "below"
    distance = round(abs(log_odds - t), 2)
    return DecisionMargin(nearest_threshold=name, threshold_log_odds=round(t, 4), distance_log_odds=distance,
                          side=side, statement=f"log-odds {distance:.2f} {side} the {name} threshold ({t:.2f})")


def inspect_signatures(reference: np.ndarray, questioned: np.ndarray,
                       verifier: DeterministicVerifier) -> SignatureInspection:
    """Verdict plus every intermediate of one 1:1 comparison."""
    start = time.perf_counter()
    p = verifier.config.representation
    ref_gated, que_gated = gate_and_extract(reference, p), gate_and_extract(questioned, p)
    comparison = comparison_from_result(verifier.verify_gated(ref_gated, que_gated), verifier, 1)
    compared = time.perf_counter()

    ref_feats, ref_err = _features_or_error(ref_gated)
    que_feats, que_err = _features_or_error(que_gated)
    ref_view = inspect_image(reference, ref_feats, ref_err, p, ref_gated.quality)
    que_view = inspect_image(questioned, que_feats, que_err, p, que_gated.quality)

    alignment = fusion = capture = None
    consistency = Consistency()
    if comparison.band != "INCONCLUSIVE" and ref_feats is not None and que_feats is not None:
        pair = compare(ref_feats, que_feats, verifier.config.fusion, p)
        breakdown = fusion_breakdown(pair, verifier)
        total = breakdown.bias + sum(verifier.config.fusion.signal_weights[n] * pair.signals[n] for n in FUSION_SIGNALS)
        consistency = Consistency(
            log_odds_matches=round(pair.fused_logit, LOG_ODDS_DECIMALS) == comparison.log_odds,
            contributions_sum_matches=bool(abs(total - pair.fused_logit) <= SUM_TOLERANCE),
        )
        alignment = alignment_inspection(ref_feats, que_feats, pair, verifier)
        capture = capture_scale_match(ref_feats, que_feats, p)
        if consistency.log_odds_matches and consistency.contributions_sum_matches:
            fusion = breakdown
    done = time.perf_counter()
    return SignatureInspection(
        comparison=comparison, reference=ref_view, questioned=que_view, alignment=alignment,
        capture_scale=capture, fusion=fusion,
        margin=decision_margin(comparison.log_odds, comparison.reject_threshold, comparison.accept_threshold),
        thresholds=Thresholds(accept_log_odds=comparison.accept_threshold, reject_log_odds=comparison.reject_threshold),
        consistency=consistency,
        timing=Timing(comparison_ms=round((compared - start) * 1000, 1), visuals_ms=round((done - compared) * 1000, 1),
                      total_ms=round((done - start) * 1000, 1)),
    )
