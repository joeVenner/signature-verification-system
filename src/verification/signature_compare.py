"""Signature-only comparison: reference signature(s) vs one questioned signature.

No cheque handling, no amount/mandate/stale policy, no audit ledger: just the
signature verdict, a similarity score and the evidence behind it. Uses exactly
the same validated scoring and thresholds as the full pipeline.

    from signature_verification_system.src.verification.signature_compare import compare_signatures
    report = compare_signatures([ref_img], questioned_img)
    print(report.verdict, report.match_score)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from pydantic import BaseModel, Field

from signature_verification_system.src.core.types import VerificationResult
from signature_verification_system.src.verification.deterministic import DeterministicVerifier
from signature_verification_system.src.verification.explanation import render_text

MAX_REFERENCES = 10

VERDICTS = {
    "ACCEPT": "MATCH",
    "REVIEW": "UNCERTAIN - MANUAL REVIEW",
    "REJECT": "NO MATCH",
    "INCONCLUSIVE": "INCONCLUSIVE - IMAGE QUALITY",
}


class SignatureComparison(BaseModel):
    """Result of comparing a questioned signature with reference signature(s)."""

    verdict: str = Field(..., description="MATCH / UNCERTAIN - MANUAL REVIEW / NO MATCH / INCONCLUSIVE - IMAGE QUALITY")
    band: str = Field(..., description="Underlying decision band: ACCEPT / REVIEW / REJECT / INCONCLUSIVE")
    match_score: Optional[float] = Field(
        None, ge=0.0, le=100.0,
        description="0-100 similarity on a decision-aligned scale: >= 70 is MATCH, < 40 is NO MATCH, "
                    "40-70 needs manual review. Monotone in log-odds; not a probability of genuineness.",
    )
    log_odds: Optional[float] = Field(None, description="Raw fused log-odds the verdict thresholds apply to")
    accept_threshold: Optional[float] = Field(None, description="log-odds at or above which the verdict is MATCH")
    reject_threshold: Optional[float] = Field(None, description="log-odds below which the verdict is NO MATCH")
    references_used: int = Field(..., ge=0, description="Reference signatures that passed the quality gate")
    references_supplied: int = Field(..., ge=0)
    layout_similarity: Optional[float] = Field(None, description="Overall layout & stroke-direction similarity, 0-1")
    stroke_detail_similarity: Optional[float] = Field(None, description="Fine stroke detail (keypoint) similarity")
    consistent_stroke_features: Optional[int] = Field(None, description="Geometrically consistent keypoint matches")
    alignment: Optional[Dict[str, float]] = Field(None, description="Rotation/scale corrected before comparison, if used")
    notes: List[str] = Field(default_factory=list)
    explanation: Dict[str, Any] = Field(default_factory=dict)
    explanation_text: str = ""


SCORE_REJECT_ANCHOR = 40.0   # score at the NO MATCH threshold
SCORE_ACCEPT_ANCHOR = 70.0   # score at the MATCH threshold
SCORE_TAIL_LOGODDS = 6.0     # log-odds beyond a threshold that maps to 0 / 100


def decision_aligned_score(log_odds: Optional[float], reject_t: float, accept_t: float) -> Optional[float]:
    """Map log-odds to 0-100 so that the score can never contradict the verdict.

    Piecewise-linear and monotone through the anchors:
      (reject_t - 6) -> 0, reject_t -> 40, accept_t -> 70, (accept_t + 6) -> 100.
    A plain sigmoid is centred at log-odds 0, but the validated thresholds are
    not, so e.g. a rejected forgery could display "51/100".
    """
    if log_odds is None or not math.isfinite(log_odds):
        return None
    lo, hi = reject_t - SCORE_TAIL_LOGODDS, accept_t + SCORE_TAIL_LOGODDS
    xs = [lo, reject_t, accept_t, hi]
    ys = [0.0, SCORE_REJECT_ANCHOR, SCORE_ACCEPT_ANCHOR, 100.0]
    return round(float(np.interp(log_odds, xs, ys)), 1)


def compare_signatures(
    references: Sequence[np.ndarray],
    questioned: np.ndarray,
    verifier: Optional[DeterministicVerifier] = None,
) -> SignatureComparison:
    """Compare a questioned signature with 1..10 reference signatures of the same person.

    Raises:
        ValueError: no references or more than MAX_REFERENCES supplied.
    """
    if not references:
        raise ValueError("At least one reference signature is required")
    if len(references) > MAX_REFERENCES:
        raise ValueError(f"At most {MAX_REFERENCES} reference signatures are supported")
    v = verifier or DeterministicVerifier()
    result = v.verify(references[0], questioned) if len(references) == 1 else \
        v.verify_against_references(list(references), questioned)
    return comparison_from_result(result, v, len(references))


def comparison_from_result(result: VerificationResult, v: DeterministicVerifier,
                           references_supplied: int) -> SignatureComparison:
    """Signature-only view of a verifier result (the verdict `compare_signatures` returns)."""
    band = result.decision_band or "INCONCLUSIVE"
    multi = (result.reference_count or 0) >= 2
    t = v.config.decision
    verified = band != "INCONCLUSIVE"
    f = result.features
    explanation = result.explanation or {}
    reject_t = t.multi_reject_logit if multi else t.single_reject_logit
    accept_t = t.multi_accept_logit if multi else t.single_accept_logit
    return SignatureComparison(
        verdict=VERDICTS[band],
        band=band,
        match_score=decision_aligned_score(result.match_logit, reject_t, accept_t) if verified else None,
        log_odds=result.match_logit,
        accept_threshold=accept_t if verified else None,
        reject_threshold=reject_t if verified else None,
        references_used=result.reference_count or 0,
        references_supplied=references_supplied,
        layout_similarity=f.shape_similarity if verified else None,
        stroke_detail_similarity=f.keypoint_similarity if verified else None,
        consistent_stroke_features=f.keypoint_inliers if verified else None,
        alignment=explanation.get("alignment"),
        notes=list(result.notes),
        explanation=explanation,
        explanation_text=render_text(explanation) if explanation else "",
    )
