"""Three-Tier Clearing Adjudication Engine with CBUAE Compliance Rules.

Implements deterministic routing:
- GREEN (STP): Auto-clear, zero human touch (< AED 20k, score >= 0.85, positive pay)
- AMBER (L1): Back-office operator queue (AED 20k-100k, or score 0.65-0.85)
- RED (L2): Mandatory four-eyes dual verification / rejection (>= AED 100k, score < 0.65, or mandate violation)

Enforces UAE Central Bank (CBUAE) and UAE Commercial Transactions Law rules:
- Words govern figures (CAR vs LAR)
- Stale cheques (> 6 months) and post-dated cheque vaults
- Standardized ICCS bank return codes
"""

from __future__ import annotations

from typing import List, Optional
from signature_verification_system.src.core.types import (
    DecisionTier,
    AmountTier,
    StandardReturnCode,
    VerificationResult,
    MandateRule,
    DecisionResult,
    IQAMetrics,
)
from signature_verification_system.src.core.config import SystemConfig, DEFAULT_CONFIG
from signature_verification_system.src.adjudication.thresholds import (
    ACCEPT, INCONCLUSIVE, REJECT, REVIEW, classify_band,
)


def _evidence(result: VerificationResult) -> str:
    """Describe the match by its decision quantity (log-odds) when available;
    the balanced-prior probability is over-confident (EXP-004)."""
    if result.match_logit is not None:
        return f"log-odds {result.match_logit:+.2f} ({result.reference_count} specimen(s))"
    return f"score {result.similarity_score:.3f}"


class DecisionEngine:
    """Deterministic policy engine for cheque clearing and signature adjudication."""

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or DEFAULT_CONFIG

    def _signature_band(self, result: VerificationResult) -> tuple[str, bool]:
        """Risk band from the v3 logit operating points; legacy score thresholds
        apply only to results that carry no logit (e.g. externally built)."""
        if result.decision_band == INCONCLUSIVE:
            return INCONCLUSIVE, False
        if result.match_logit is not None and result.reference_count:
            b = classify_band(result.match_logit, result.reference_count, self.config.decision)
            # Upstream stages may only make the band MORE conservative (e.g. the
            # pipeline caps a fallback-zone crop at REVIEW); never less.
            order = {REJECT: 0, REVIEW: 1, ACCEPT: 2}
            if result.decision_band in order and order[result.decision_band] < order[b.band]:
                return result.decision_band, False
            return b.band, b.hard_reject
        cal = self.config.calibration
        s = result.similarity_score
        if s < cal.threshold_amber_min:
            return REJECT, s < cal.threshold_hard_reject
        return (ACCEPT if s >= cal.threshold_green_stp else REVIEW), False

    def classify_amount_tier(self, amount: float) -> AmountTier:
        """Classify transaction amount into CBUAE monetary value tier."""
        amounts = self.config.amounts
        if amount < amounts.floor_stp_limit:
            return AmountTier.LOW_VALUE
        elif amount < amounts.high_value_limit:
            return AmountTier.MEDIUM_VALUE
        else:
            return AmountTier.HIGH_VALUE

    def evaluate(
        self,
        verification_result: VerificationResult,
        amount: float,
        currency: str = "AED",
        car_lar_match: bool = True,
        positive_pay_match: bool = True,
        mandate: Optional[MandateRule] = None,
        detected_signers_count: int = 1,
        iqa_metrics: Optional[IQAMetrics] = None,
        stale_cheque: bool = False,
        post_dated: bool = False,
        suspected_alteration: bool = False,
    ) -> DecisionResult:
        """Evaluate verification results against banking clearing rules.
        
        Pure deterministic decision logic: given identical inputs, always yields
        the exact same tier, action, reasons, and return code.
        """
        cal = self.config.calibration
        policy_ver = self.config.policy_version
        band, hard_reject = self._signature_band(verification_result)
        amount_tier = self.classify_amount_tier(amount)

        score = verification_result.similarity_score
        reasons: List[str] = []
        cbuae_flags: List[str] = []
        return_code: Optional[StandardReturnCode] = None
        requires_four_eyes = False

        # ---------------------------------------------------------
        # 1. Critical Disqualifiers & Legal Compliance Checks
        # ---------------------------------------------------------
        if suspected_alteration:
            cbuae_flags.append("SUSPECTED_MECHANICAL_OR_CHEMICAL_ALTERATION")
            reasons.append("Document alteration detected on ink/paper layer.")
            return DecisionResult(
                tier=DecisionTier.RED,
                action="REJECT",
                amount=amount,
                currency=currency,
                amount_tier=amount_tier,
                similarity_score=score,
                policy_version=policy_ver,
                cbuae_compliance_flags=cbuae_flags,
                reasons=reasons,
                return_code=StandardReturnCode.SUSPECTED_ALTERATION,
                requires_four_eyes=True,
            )

        if stale_cheque:
            cbuae_flags.append("STALE_CHEQUE_PRESENTMENT_EXCEEDS_6_MONTHS")
            reasons.append("Cheque presentment date exceeds 6 months legal validity (UAE CTL).")
            return DecisionResult(
                tier=DecisionTier.RED,
                action="REJECT",
                amount=amount,
                currency=currency,
                amount_tier=amount_tier,
                similarity_score=score,
                policy_version=policy_ver,
                cbuae_compliance_flags=cbuae_flags,
                reasons=reasons,
                return_code=StandardReturnCode.IRREGULAR_SIGNATURE,
                requires_four_eyes=False,
            )

        if iqa_metrics and not iqa_metrics.passed:
            cbuae_flags.append("IQA_ANSI_STANDARD_FAILURE")
            reasons.extend(iqa_metrics.failure_reasons)
            return DecisionResult(
                tier=DecisionTier.RED,
                action="REJECT",
                amount=amount,
                currency=currency,
                amount_tier=amount_tier,
                similarity_score=score,
                policy_version=policy_ver,
                cbuae_compliance_flags=cbuae_flags,
                reasons=reasons,
                return_code=StandardReturnCode.IQA_QUALITY_FAIL,
                requires_four_eyes=False,
            )

        # ---------------------------------------------------------
        # 2. Account Mandate Verification
        # ---------------------------------------------------------
        if mandate is not None:
            if detected_signers_count < mandate.required_signers_count:
                cbuae_flags.append("MANDATE_INSUFFICIENT_SIGNATURES")
                reasons.append(
                    f"Mandate requires {mandate.required_signers_count} signatures, found {detected_signers_count}."
                )
                return DecisionResult(
                    tier=DecisionTier.RED,
                    action="REJECT",
                    amount=amount,
                    currency=currency,
                    amount_tier=amount_tier,
                    similarity_score=score,
                    policy_version=policy_ver,
                    cbuae_compliance_flags=cbuae_flags,
                    reasons=reasons,
                    return_code=StandardReturnCode.MANDATE_INCOMPLETE,
                    requires_four_eyes=True,
                )

            if detected_signers_count == 1 and amount > mandate.max_single_signer_limit:
                cbuae_flags.append("AMOUNT_EXCEEDS_SINGLE_SIGNER_MANDATE")
                reasons.append(
                    f"Amount {currency} {amount:,.2f} exceeds single-signer limit of {currency} {mandate.max_single_signer_limit:,.2f}."
                )
                return DecisionResult(
                    tier=DecisionTier.RED,
                    action="REJECT",
                    amount=amount,
                    currency=currency,
                    amount_tier=amount_tier,
                    similarity_score=score,
                    policy_version=policy_ver,
                    cbuae_compliance_flags=cbuae_flags,
                    reasons=reasons,
                    return_code=StandardReturnCode.AMOUNT_EXCEEDS_MANDATE,
                    requires_four_eyes=True,
                )

        # ---------------------------------------------------------
        # 3. Post-Dated Cheque (PDC) Hold
        # ---------------------------------------------------------
        if post_dated:
            cbuae_flags.append("POST_DATED_CHEQUE_HOLD")
            reasons.append("Post-dated cheque routed to PDC custodian vault until maturity.")
            return DecisionResult(
                tier=DecisionTier.AMBER,
                action="OPERATOR_REVIEW",
                amount=amount,
                currency=currency,
                amount_tier=amount_tier,
                similarity_score=score,
                policy_version=policy_ver,
                cbuae_compliance_flags=cbuae_flags,
                reasons=reasons,
                return_code=None,
                requires_four_eyes=False,
            )

        # ---------------------------------------------------------
        # 4. Inconclusive verification or REJECT band -> RED
        # ---------------------------------------------------------
        if band == INCONCLUSIVE:
            cbuae_flags.append("SIGNATURE_VERIFICATION_INCONCLUSIVE")
            reasons.append("Signature could not be verified (no usable ink / specimen); manual examination required.")
            return DecisionResult(
                tier=DecisionTier.RED,
                action="MANDATORY_FOUR_EYES_ESCALATE",
                amount=amount,
                currency=currency,
                amount_tier=amount_tier,
                similarity_score=score,
                policy_version=policy_ver,
                cbuae_compliance_flags=cbuae_flags,
                reasons=reasons,
                return_code=StandardReturnCode.IRREGULAR_SIGNATURE,
                requires_four_eyes=True,
            )

        if band == REJECT:
            reasons.append(
                f"Signature match {_evidence(verification_result)} falls in the REJECT band of the validated operating point."
            )
            cbuae_flags.append("SIGNATURE_MISMATCH_ESCALATION")

            if hard_reject:
                return_code = StandardReturnCode.SIGNATURE_DIFFERS
                action = "REJECT"
            else:
                return_code = StandardReturnCode.IRREGULAR_SIGNATURE
                action = "MANDATORY_FOUR_EYES_ESCALATE"

            return DecisionResult(
                tier=DecisionTier.RED,
                action=action,
                amount=amount,
                currency=currency,
                amount_tier=amount_tier,
                similarity_score=score,
                policy_version=policy_ver,
                cbuae_compliance_flags=cbuae_flags,
                reasons=reasons,
                return_code=return_code,
                requires_four_eyes=True,
            )

        # ---------------------------------------------------------
        # 5. High-Value Cheques (>= AED 100k) -> Mandatory RED L2
        # ---------------------------------------------------------
        if amount_tier == AmountTier.HIGH_VALUE:
            cbuae_flags.append("CBUAE_HIGH_VALUE_FOUR_EYES_MANDATE")
            reasons.append(
                f"High value presentment {currency} {amount:,.2f} >= {self.config.amounts.high_value_limit:,.2f} requires mandatory dual sign-off."
            )
            return DecisionResult(
                tier=DecisionTier.RED,
                action="MANDATORY_FOUR_EYES_ESCALATE",
                amount=amount,
                currency=currency,
                amount_tier=amount_tier,
                similarity_score=score,
                policy_version=policy_ver,
                cbuae_compliance_flags=cbuae_flags,
                reasons=reasons,
                return_code=None,
                requires_four_eyes=True,
            )

        # ---------------------------------------------------------
        # 6. Straight-Through Processing (STP) -> GREEN
        # ---------------------------------------------------------
        is_green_score = band == ACCEPT
        is_low_amount = amount_tier == AmountTier.LOW_VALUE
        is_data_clean = car_lar_match and positive_pay_match

        if is_green_score and is_low_amount and is_data_clean:
            reasons.append(
                f"Signature match {_evidence(verification_result)} in the ACCEPT band, under floor limit ({currency} {amount:,.2f}). Auto-cleared."
            )
            return DecisionResult(
                tier=DecisionTier.GREEN,
                action="AUTO_CLEAR",
                amount=amount,
                currency=currency,
                amount_tier=amount_tier,
                similarity_score=score,
                policy_version=policy_ver,
                cbuae_compliance_flags=[],
                reasons=reasons,
                return_code=StandardReturnCode.CLEAR,
                requires_four_eyes=False,
            )

        # ---------------------------------------------------------
        # 7. Remaining Cases Route to AMBER (L1 Review)
        # ---------------------------------------------------------
        if not car_lar_match:
            cbuae_flags.append("CAR_LAR_MISMATCH")
            reasons.append("Courtesy amount (figures) does not match legal amount (words).")

        if not positive_pay_match:
            cbuae_flags.append("POSITIVE_PAY_UNCONFIRMED")
            reasons.append("Positive Pay cheque pre-issuance record not found or unconfirmed.")

        if amount_tier == AmountTier.MEDIUM_VALUE:
            reasons.append(
                f"Medium-value presentment ({currency} {amount:,.2f}) requires standard L1 clearing queue review."
            )

        if band == REVIEW:
            reasons.append(
                f"Moderate similarity score {score:.3f} requires visual inspection on clearing console."
            )

        return DecisionResult(
            tier=DecisionTier.AMBER,
            action="OPERATOR_REVIEW",
            amount=amount,
            currency=currency,
            amount_tier=amount_tier,
            similarity_score=score,
            policy_version=policy_ver,
            cbuae_compliance_flags=cbuae_flags,
            reasons=reasons,
            return_code=None,
            requires_four_eyes=False,
        )
