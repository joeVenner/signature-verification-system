"""End-to-end cheque signature verification pipeline (single orchestrator).

    cheque image
      → document IQA (+ deskew)
      → signature localisation and crop
      → signature quality gate
      → normalisation + feature extraction (inside the verifier)
      → comparison against every enrolled specimen (feature-wise max)
      → validated ACCEPT / REVIEW / REJECT band
      → clearing policy (amount tiers, mandate, stale, CAR/LAR, positive pay)
      → explanation + hash-chained audit record

Every stage records what it actually measured; nothing is inferred for display.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from signature_verification_system.src.adjudication.audit_logger import AuditLogger
from signature_verification_system.src.adjudication.decision_engine import DecisionEngine
from signature_verification_system.src.core.config import DEFAULT_CONFIG, SystemConfig
from signature_verification_system.src.core.types import (
    AuditRecord, DecisionResult, DetectionResult, IQAMetrics, VerificationResult,
)
from signature_verification_system.src.detection.locator import SignatureLocator, refine_signature_bbox
from signature_verification_system.src.preprocessing.form_cleaning import clean_printed_rules
from signature_verification_system.src.preprocessing.iqa import assess_image_quality
from signature_verification_system.src.verification.deterministic import DeterministicVerifier


@dataclass
class StageRecord:
    name: str
    status: str                       # PASS | WARN | FAIL | INFO
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0


@dataclass
class PipelineReport:
    stages: List[StageRecord]
    iqa: IQAMetrics
    detection: DetectionResult
    signature_crop: np.ndarray
    crop_bbox: Optional[tuple]
    verification: VerificationResult
    decision: DecisionResult
    audit: Optional[AuditRecord]

    def stage(self, name: str) -> StageRecord:
        return next(s for s in self.stages if s.name == name)


class ChequeVerificationPipeline:
    """Deterministic orchestration of all stages; components are injectable."""

    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        locator: Optional[SignatureLocator] = None,
        verifier: Optional[DeterministicVerifier] = None,
        engine: Optional[DecisionEngine] = None,
        audit_logger: Optional[AuditLogger] = None,
        refine_crop: bool = True,
        clean_rules: bool = True,
    ):
        self.config = config or DEFAULT_CONFIG
        self.locator = locator or SignatureLocator()
        self.verifier = verifier or DeterministicVerifier(config=self.config)
        self.engine = engine or DecisionEngine(config=self.config)
        self.audit_logger = audit_logger
        self.refine_crop = refine_crop  # EXP-009: grow detector box to full ink components
        self.clean_rules = clean_rules  # EXP-009: erase printed rules / dashed borders from the crop

    def run(
        self,
        cheque_image: np.ndarray,
        specimen_images: Sequence[np.ndarray],
        amount: float,
        currency: str = "AED",
        car_lar_match: bool = True,
        positive_pay_match: bool = True,
        stale_days: int = 0,
        document_id: str = "CHQ-DEMO",
        operator_id: Optional[str] = None,
        audit_metadata: Optional[Dict[str, Any]] = None,
    ) -> PipelineReport:
        if cheque_image is None or cheque_image.size == 0:
            raise ValueError("Empty cheque image")
        if not specimen_images:
            raise ValueError("At least one specimen signature is required")
        stages: List[StageRecord] = []

        t = time.perf_counter()
        iqa, deskewed = assess_image_quality(cheque_image, thresholds=self.config.iqa, deskew=True)
        stages.append(StageRecord(
            "DOCUMENT_QUALITY", "PASS" if iqa.passed else "FAIL",
            "Cheque image meets capture standard" if iqa.passed else "; ".join(iqa.failure_reasons),
            {"skew_deg": round(iqa.skew_angle, 2), "blur": round(iqa.blur_score, 1),
             "contrast": round(iqa.contrast_score, 1), "brightness": round(iqa.brightness_score, 3)},
            (time.perf_counter() - t) * 1000))

        t = time.perf_counter()
        detection = self.locator.locate(deskewed, is_cheque=True)
        if detection.best_candidate is not None:
            b = detection.best_candidate.bbox
            bbox = refine_signature_bbox(deskewed, (b.x, b.y, b.w, b.h)) if self.refine_crop else (b.x, b.y, b.w, b.h)
            crop = deskewed[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]].copy()
            stages.append(StageRecord(
                "SIGNATURE_DETECTED", "PASS",
                f"Signature located at x={bbox[0]}, y={bbox[1]}, {bbox[2]}x{bbox[3]}px",
                {"bbox": bbox, "detector_bbox": (b.x, b.y, b.w, b.h),
                 "detector_confidence": round(detection.best_candidate.confidence, 3),
                 "candidates": len(detection.candidates)},
                (time.perf_counter() - t) * 1000))
        else:
            h, w = deskewed.shape[:2]
            y0, y1, x0, x1 = int(0.50 * h), int(0.90 * h), int(0.55 * w), int(0.97 * w)
            crop, bbox = deskewed[y0:y1, x0:x1], (x0, y0, x1 - x0, y1 - y0)
            stages.append(StageRecord(
                "SIGNATURE_DETECTED", "WARN", "No signature candidate found; using standard drawer zone",
                {"bbox": bbox, "candidates": 0}, (time.perf_counter() - t) * 1000))

        if self.clean_rules:
            crop = clean_printed_rules(crop)

        t = time.perf_counter()
        if len(specimen_images) >= 2:
            verification = self.verifier.verify_against_references(list(specimen_images), crop)
        else:
            verification = self.verifier.verify(specimen_images[0], crop)
        elapsed = (time.perf_counter() - t) * 1000
        if detection.best_candidate is None and verification.decision_band == "ACCEPT":
            # Fallback zone crop: no signature was actually located, so never auto-accept.
            verification = verification.model_copy(update={
                "decision_band": "REVIEW",
                "is_match": False,
                "notes": ["Signature not located by the detector; verified a default zone, capped at REVIEW."]
                         + list(verification.notes),
            })
        q = (verification.quality or {}).get("questioned", {})
        stages.append(StageRecord(
            "SIGNATURE_QUALITY", "PASS" if q.get("passed") else "FAIL",
            "Signature fit for automated verification" if q.get("passed")
            else "Signature unfit: " + ", ".join(q.get("blocking_issues", [])),
            {k: q.get(k) for k in ("ink_contrast", "edge_sharpness", "noise_ratio", "blocking_issues", "warnings")},
            0.0))
        if verification.decision_band != "INCONCLUSIVE":
            stages.append(StageRecord(
                "SIGNATURE_ANALYSIS", "INFO", "Measured signals vs. genuine / forgery reference distributions",
                {"evidence": (verification.explanation or {}).get("evidence", [])}, 0.0))
            stages.append(StageRecord(
                "REFERENCE_MATCH", "INFO",
                f"Compared against {verification.reference_count} enrolled specimen(s)",
                {"match_logit": verification.match_logit, "reference_count": verification.reference_count,
                 "timing_note": "elapsed_ms covers the whole verifier: quality gate, features, matching"},
                elapsed))
        stages.append(StageRecord(
            "RISK_BAND", {"ACCEPT": "PASS", "REVIEW": "WARN"}.get(verification.decision_band or "", "FAIL"),
            f"Signature band: {verification.decision_band}",
            {"band_reliability": (verification.explanation or {}).get("band_reliability")}, 0.0))

        t = time.perf_counter()
        decision = self.engine.evaluate(
            verification_result=verification,
            amount=amount,
            currency=currency,
            car_lar_match=car_lar_match,
            positive_pay_match=positive_pay_match,
            stale_cheque=stale_days > 180,
            # Only the best candidate is verified, so it is the only verified signer.
            # (Detector candidates are proposals, not signatures; counting them could
            # satisfy a joint-signature mandate with noise. Code review H3.)
            detected_signers_count=1,
            iqa_metrics=iqa,
        )
        stages.append(StageRecord(
            "FINAL_DECISION", {"GREEN": "PASS", "AMBER": "WARN"}.get(decision.tier.value, "FAIL"),
            f"{decision.tier.value} / {decision.action}",
            {"reasons": decision.reasons, "return_code": decision.return_code.value if decision.return_code else None,
             "requires_four_eyes": decision.requires_four_eyes},
            (time.perf_counter() - t) * 1000))

        audit = None
        if self.audit_logger is not None:
            meta = dict(audit_metadata or {})
            meta.update({"signature_band": verification.decision_band, "match_logit": verification.match_logit,
                         "reference_count": verification.reference_count})
            audit = self.audit_logger.log_decision(
                document_id=document_id, decision=decision, operator_id=operator_id, additional_metadata=meta)
        return PipelineReport(stages, iqa, detection, crop, bbox, verification, decision, audit)
