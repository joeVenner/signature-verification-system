"""Core data models and types for the Signature Verification System.

All schemas are defined using Pydantic v2 for deterministic serialization,
validation, and schema compliance.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict


# ============================================================================
# 1. ENUMS
# ============================================================================

class DecisionTier(str, Enum):
    """3-Tier clearing decision categories."""
    GREEN = "GREEN"  # STP - Straight Through Processing / Auto-clear
    AMBER = "AMBER"  # L1 Operator Queue (Human Review)
    RED = "RED"      # L2 Dual-Signoff Escalation / Rejection


class AmountTier(str, Enum):
    """UAE Central Bank (CBUAE) monetary value bands for risk segmentation."""
    LOW_VALUE = "LOW_VALUE"          # < AED 20,000 (Eligible for STP)
    MEDIUM_VALUE = "MEDIUM_VALUE"    # AED 20,000 - AED 100,000 (Requires L1 review)
    HIGH_VALUE = "HIGH_VALUE"        # >= AED 100,000 (Mandatory Four-Eyes Red tier)


class StandardReturnCode(str, Enum):
    """Standardized bank clearing return codes (CBUAE / ICCS compliant)."""
    CLEAR = "NONE_ACCEPTED"
    SIGNATURE_DIFFERS = "SIG_DIFF_01"              # Signature does not match specimen
    SUSPECTED_ALTERATION = "ALT_SUSP_02"           # Suspected mechanical/chemical alteration
    IRREGULAR_SIGNATURE = "SIG_IRREG_03"           # Incomplete or mutilated signature
    MANDATE_INCOMPLETE = "MANDATE_MISSING_SIG_04"  # Missing second required joint signatory
    AMOUNT_EXCEEDS_MANDATE = "MANDATE_LIMIT_05"    # Signer limit exceeded for amount
    IQA_QUALITY_FAIL = "IQA_REJECT_06"             # Image quality fails ANSI/CBUAE specs


class ZoneType(str, Enum):
    """Types of detected signature zones."""
    PRIMARY_CHEQUE_ZONE = "primary_cheque_zone"
    SECONDARY_CHEQUE_ZONE = "secondary_cheque_zone"
    CONTOUR_CLUSTER = "contour_cluster"
    GENERIC_CANDIDATE = "generic_candidate"


# ============================================================================
# 2. IMAGE QUALITY ASSESSMENT (IQA) MODELS
# ============================================================================

class IQAMetrics(BaseModel):
    """Image Quality Assessment (ANSI X9.100-181 & CBUAE specs)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    skew_angle: float = Field(..., description="Estimated deskew angle in degrees (-45.0 to +45.0)")
    blur_score: float = Field(..., description="Variance of Laplacian focus measure")
    contrast_score: float = Field(..., description="Standard deviation of luminance / RMS contrast")
    brightness_score: float = Field(..., description="Mean normalized pixel intensity in [0.0, 1.0]")
    is_skewed: bool = Field(..., description="True if |skew_angle| exceeds policy threshold")
    is_blurry: bool = Field(..., description="True if blur_score is below threshold")
    is_too_dark: bool = Field(..., description="True if brightness is below minimum threshold")
    is_too_bright: bool = Field(..., description="True if brightness is above maximum threshold")
    passed: bool = Field(..., description="True if all IQA checks satisfy clearing standards")
    failure_reasons: List[str] = Field(default_factory=list, description="List of reasons if IQA failed")


# ============================================================================
# 3. DETECTION & LOCALIZATION MODELS
# ============================================================================

class BoundingBox(BaseModel):
    """Axis-aligned bounding box coordinates [x, y, w, h]."""
    x: int = Field(..., ge=0, description="X coordinate of top-left corner")
    y: int = Field(..., ge=0, description="Y coordinate of top-left corner")
    w: int = Field(..., gt=0, description="Bounding box width")
    h: int = Field(..., gt=0, description="Bounding box height")

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def aspect_ratio(self) -> float:
        return float(self.w) / float(self.h) if self.h > 0 else 0.0

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


class SignatureCandidate(BaseModel):
    """Candidate signature region located on a cheque or generic document."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    bbox: BoundingBox
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of detection")
    zone_type: ZoneType = Field(default=ZoneType.PRIMARY_CHEQUE_ZONE)
    aspect_ratio: float = Field(..., description="Width-to-height ratio")
    stroke_density: float = Field(..., ge=0.0, le=1.0, description="Foreground ink stroke density in crop")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DetectionResult(BaseModel):
    """Complete document detection result containing all candidate signature zones."""
    candidates: List[SignatureCandidate] = Field(default_factory=list)
    best_candidate: Optional[SignatureCandidate] = None
    document_type: str = Field(default="cheque", description="'cheque' or 'generic_document'")
    image_shape: Tuple[int, int] = Field(..., description="Height, Width of source document")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# 4. VERIFICATION MODELS
# ============================================================================

class FeatureBreakdown(BaseModel):
    """Granular multi-feature verification sub-scores."""
    hog_similarity: float = Field(..., ge=0.0, le=1.0, description="HOG orientation gradient cosine similarity")
    hu_moments_similarity: float = Field(..., ge=0.0, le=1.0, description="Hu invariant moments normalized similarity")
    contour_similarity: float = Field(..., ge=0.0, le=1.0, description="Structural contour & aspect similarity")
    skeleton_similarity: float = Field(..., ge=0.0, le=1.0, description="Zhang-Suen skeleton junction/endpoint graph overlap")
    stroke_width_variation_score: float = Field(..., ge=0.0, le=1.0, description="Stroke thickness distribution match")
    hesitation_score: float = Field(..., ge=0.0, le=1.0, description="Tracing tremor / ink-blobbing hesitation index")
    curvature_variation_score: Optional[float] = Field(default=None, description="Curvature variation / micro-jitter analysis score")
    shape_similarity: Optional[float] = Field(default=None, description="v3 global gradient-grid shape cosine similarity")
    keypoint_similarity: Optional[float] = Field(default=None, description="v3 RANSAC-verified SIFT correspondence ratio")
    keypoint_inliers: Optional[int] = Field(default=None, description="Geometrically consistent keypoint matches")
    keypoint_tentative_matches: Optional[int] = Field(default=None, description="Ratio-test keypoint matches before RANSAC")
    fused_logit: Optional[float] = Field(default=None, description="v3 logistic fusion log-odds")


class VerificationResult(BaseModel):
    """Deterministic biometric verification comparison output."""
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Calibrated aggregate match score in [0.0, 1.0]")
    raw_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Pre-calibration raw fusion score")
    calibrated_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Non-linear sigmoid calibrated confidence score")
    is_match: bool = Field(..., description="True if similarity meets the operating threshold")
    features: FeatureBreakdown
    confidence: float = Field(..., ge=0.0, le=1.0, description="Statistical confidence in the comparison")
    notes: List[str] = Field(default_factory=list)
    match_logit: Optional[float] = Field(default=None, description="Unrounded fused log-odds used for decision bands")
    reference_count: Optional[int] = Field(default=None, ge=0, description="Number of usable specimens compared")
    decision_band: Optional[str] = Field(default=None, description="ACCEPT / REVIEW / REJECT / INCONCLUSIVE")
    quality: Optional[Dict[str, Any]] = Field(default=None, description="Quality-gate measurements for questioned and reference images")
    explanation: Optional[Dict[str, Any]] = Field(default=None, description="Evidence-based explanation built from measured signals")


# ============================================================================
# 5. ADJUDICATION & POLICY MODELS
# ============================================================================

class MandateRule(BaseModel):
    """Account mandate configuration for signatories."""
    required_signers_count: int = Field(default=1, ge=1)
    max_single_signer_limit: float = Field(default=100_000.0, description="Max amount in AED for single signer")
    mandate_type: str = Field(default="SINGLE", description="'SINGLE', 'JOINT_ANY_2', 'JOINT_ALL'")


class DecisionResult(BaseModel):
    """Three-tier adjudication policy decision."""
    tier: DecisionTier
    action: str = Field(..., description="'AUTO_CLEAR', 'OPERATOR_REVIEW', 'MANDATORY_FOUR_EYES_ESCALATE', or 'REJECT'")
    amount: float = Field(..., ge=0.0, description="Cheque courtesy amount")
    currency: str = Field(default="AED")
    amount_tier: AmountTier
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    policy_version: str
    cbuae_compliance_flags: List[str] = Field(default_factory=list)
    reasons: List[str] = Field(default_factory=list)
    return_code: Optional[StandardReturnCode] = None
    requires_four_eyes: bool = Field(default=False)


# ============================================================================
# 6. AUDIT & LEDGER MODELS
# ============================================================================

class AuditRecord(BaseModel):
    """Tamper-evident, hash-chained ledger block for legal auditability."""
    record_id: str
    sequence_num: int
    timestamp_utc: str
    document_id: str
    amount: float
    currency: str = "AED"
    similarity_score: float
    decision_tier: DecisionTier
    action: str
    policy_version: str
    model_version: str
    operator_id: Optional[str] = None
    prev_hash: str = Field(..., description="SHA-256 hash of previous block (or 'GENESIS')")
    hash: str = Field(..., description="SHA-256 hash of this record block")
    metadata: Dict[str, Any] = Field(default_factory=dict)
