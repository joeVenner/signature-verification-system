"""Signature Verification System - Core Modular Library.

Components:
- core: Data types, schemas, and versioned configuration policies
- preprocessing: Image quality assessment (IQA), deskewing, adaptive binarization, and ink extraction
- detection: Cheque signature zone and generic document locator
- verification: Deterministic SOTA multi-feature verification engine
- adjudication: Three-tier Green/Amber/Red clearing engine and tamper-evident audit logger
"""

__version__ = "1.0.0"

from signature_verification_system.src.core.types import (
    DecisionTier,
    AmountTier,
    StandardReturnCode,
    ZoneType,
    IQAMetrics,
    BoundingBox,
    SignatureCandidate,
    DetectionResult,
    FeatureBreakdown,
    VerificationResult,
    MandateRule,
    DecisionResult,
    AuditRecord,
)
from signature_verification_system.src.core.config import (
    SystemConfig,
    DEFAULT_CONFIG,
    IQAThresholds,
    AmountThresholds,
    VerificationCalibration,
    FeatureWeights,
    PreprocessingParams,
)
from signature_verification_system.src.preprocessing.iqa import (
    estimate_skew_hough,
    estimate_skew_projection,
    deskew_image,
    calculate_blur_score,
    calculate_contrast,
    calculate_brightness,
    assess_image_quality,
)
from signature_verification_system.src.preprocessing.binarization import (
    sauvola_threshold,
    wolf_threshold,
    suppress_pantograph,
    adaptive_binarize,
)
from signature_verification_system.src.preprocessing.ink_extractor import (
    separate_ink_cielab,
    remove_horizontal_baseline,
    compute_pseudo_dynamic_density,
)
from signature_verification_system.src.detection.locator import SignatureLocator
from signature_verification_system.src.verification.deterministic import DeterministicVerifier
from signature_verification_system.src.adjudication.decision_engine import DecisionEngine
from signature_verification_system.src.adjudication.audit_logger import AuditLogger

__all__ = [
    "__version__",
    "DecisionTier",
    "AmountTier",
    "StandardReturnCode",
    "ZoneType",
    "IQAMetrics",
    "BoundingBox",
    "SignatureCandidate",
    "DetectionResult",
    "FeatureBreakdown",
    "VerificationResult",
    "MandateRule",
    "DecisionResult",
    "AuditRecord",
    "SystemConfig",
    "DEFAULT_CONFIG",
    "IQAThresholds",
    "AmountThresholds",
    "VerificationCalibration",
    "FeatureWeights",
    "PreprocessingParams",
    "estimate_skew_hough",
    "estimate_skew_projection",
    "deskew_image",
    "calculate_blur_score",
    "calculate_contrast",
    "calculate_brightness",
    "assess_image_quality",
    "sauvola_threshold",
    "wolf_threshold",
    "suppress_pantograph",
    "adaptive_binarize",
    "separate_ink_cielab",
    "remove_horizontal_baseline",
    "compute_pseudo_dynamic_density",
    "SignatureLocator",
    "DeterministicVerifier",
    "DecisionEngine",
    "AuditLogger",
]
