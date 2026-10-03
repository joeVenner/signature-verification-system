"""Pure mapping from a fused match logit to a risk band (EXP-004).

Kept separate from the verifier and the policy engine so the operating points
are a single, testable function of (logit, number of specimens, config).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from signature_verification_system.src.core.config import DEFAULT_CONFIG, DecisionThresholds

ACCEPT = "ACCEPT"
REVIEW = "REVIEW"
REJECT = "REJECT"
INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class BandDecision:
    band: str
    hard_reject: bool          # below every genuine logit seen during selection
    accept_logit: float
    reject_logit: float


def classify_band(
    match_logit: Optional[float], reference_count: int, thresholds: Optional[DecisionThresholds] = None
) -> BandDecision:
    """ACCEPT >= accept cut; REJECT < reject cut; REVIEW in between."""
    t = thresholds or DEFAULT_CONFIG.decision
    multi = reference_count >= 2
    accept = t.multi_accept_logit if multi else t.single_accept_logit
    reject = t.multi_reject_logit if multi else t.single_reject_logit
    hard = t.multi_hard_reject_logit if multi else t.single_hard_reject_logit
    if match_logit is None or reference_count < 1:
        return BandDecision(INCONCLUSIVE, False, accept, reject)
    if match_logit >= accept:
        return BandDecision(ACCEPT, False, accept, reject)
    if match_logit < reject:
        return BandDecision(REJECT, match_logit < hard, accept, reject)
    return BandDecision(REVIEW, False, accept, reject)
