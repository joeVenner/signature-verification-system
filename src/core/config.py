"""Versioned configuration parameters, thresholds, and CBUAE compliance policies.
"""

from typing import Dict
from pydantic import BaseModel, Field


class IQAThresholds(BaseModel):
    """Image Quality Assessment tolerances per ANSI X9.100-181 and CBUAE standards."""
    max_skew_degrees: float = Field(default=3.0, description="Max acceptable tilt before rejection")
    min_blur_laplacian_var: float = Field(default=80.0, description="Minimum focus measure (variance of Laplacian)")
    min_contrast_rms: float = Field(default=20.0, description="Minimum root-mean-square contrast")
    min_brightness: float = Field(default=0.15, description="Minimum normalized mean brightness [0,1]")
    max_brightness: float = Field(default=0.985, description="Maximum normalized mean brightness [0,1]")


class AmountThresholds(BaseModel):
    """UAE Central Bank (CBUAE) monetary clearing boundaries in AED."""
    floor_stp_limit: float = Field(default=20_000.0, description="Amount < 20,000 AED qualifies for Green STP")
    high_value_limit: float = Field(default=100_000.0, description="Amount >= 100,000 AED triggers mandatory Four-Eyes Red")
    currency: str = Field(default="AED", description="Default operational currency")


class VerificationCalibration(BaseModel):
    """Biometric similarity score boundaries for decision tier routing."""
    threshold_green_stp: float = Field(default=0.78, description="Min similarity score for Straight-Through-Processing")
    threshold_amber_min: float = Field(default=0.52, description="Min similarity score for Amber L1 Operator queue")
    threshold_hard_reject: float = Field(default=0.45, description="Score below this emits immediate SIGNATURE_DIFFERS")


class DecisionThresholds(BaseModel):
    """ACCEPT / REVIEW / REJECT cut-offs on the fused match logit (v3 verifier).

    Selected by benchmark/select_thresholds.py on all enrolled writers with the
    rule validated under writer-disjoint CV in EXP-004:
      accept      = highest impostor (skilled + random) logit + accept_margin
      reject      = 5th percentile of genuine logits (<= 5% genuine auto-rejected)
      hard_reject = lowest genuine logit observed (below it: SIGNATURE_DIFFERS)
    Separate operating points because one specimen is far less discriminative
    than several (EXP-003).
    """
    accept_margin_logit: float = Field(default=1.0)
    reject_genuine_quantile: float = Field(default=0.05)
    single_accept_logit: float = Field(default=13.9380)
    single_reject_logit: float = Field(default=-0.2775)
    single_hard_reject_logit: float = Field(default=-3.8339)
    multi_accept_logit: float = Field(default=4.6503)
    multi_reject_logit: float = Field(default=1.4271)
    multi_hard_reject_logit: float = Field(default=0.6254)
    selected_on: str = Field(
        default="benchmark/select_thresholds.py on CEDAR-12w (single: 1:1 protocol; multi: 3-specimen protocol)"
    )


class FeatureWeights(BaseModel):
    """Deterministic multi-feature fusion weights (sum normalized to 1.0)."""
    hog_weight: float = Field(default=0.35, description="HOG orientation gradient importance")
    hu_moments_weight: float = Field(default=0.10, description="Hu invariant moments importance")
    contour_weight: float = Field(default=0.15, description="Structural contour geometry importance")
    skeleton_weight: float = Field(default=0.25, description="Zhang-Suen medial skeleton graph importance")
    stroke_width_weight: float = Field(default=0.15, description="Stroke thickness consistency importance")
    curvature_weight: float = Field(default=0.00, description="Curvature variation / micro-jitter consistency importance")
    hesitation_penalty_weight: float = Field(default=0.25, description="Max deduction penalty for hesitation/tremor")


class PreprocessingParams(BaseModel):
    """Adaptive binarization and ink extraction hyperparameters."""
    sauvola_window_size: int = Field(default=25, description="Local neighborhood window size (odd)")
    sauvola_k: float = Field(default=0.2, description="Sauvola threshold scaling factor")
    sauvola_r: float = Field(default=128.0, description="Sauvola dynamic range parameter")
    wolf_window_size: int = Field(default=25, description="Wolf local neighborhood window size")
    wolf_k: float = Field(default=0.2, description="Wolf threshold scaling factor")
    baseline_kernel_width: int = Field(default=35, description="Horizontal kernel width for baseline detection")
    pantograph_min_speckle_area: int = Field(default=6, description="Min pixel area to filter out isolated dots")


class RepresentationParams(BaseModel):
    """Descriptor geometry for the v3 verifier (see src/verification/features.py)."""
    shape_canvas_width: int = Field(default=256, description="Aspect-normalised canvas width for global shape")
    shape_canvas_height: int = Field(default=128, description="Aspect-normalised canvas height for global shape")
    shape_grid_rows: int = Field(default=3, description="Gradient-grid rows")
    shape_grid_cols: int = Field(default=6, description="Gradient-grid columns")
    shape_bins: int = Field(default=12, description="Gradient direction bins over 360 degrees")
    shape_blur_sigma: float = Field(default=2.0, description="Gaussian blur before gradients (tolerates stroke jitter)")
    keypoint_canvas_width: int = Field(default=512, description="Letter-boxed canvas width for SIFT")
    keypoint_canvas_height: int = Field(default=256, description="Letter-boxed canvas height for SIFT")
    keypoint_ratio_test: float = Field(default=0.8, description="Lowe ratio-test threshold")
    keypoint_ransac_px: float = Field(default=12.0, description="RANSAC reprojection tolerance in canvas pixels")
    align_min_inliers: int = Field(default=6, description="Min RANSAC inliers to trust the alignment transform (EXP-014)")
    align_max_rotation_deg: float = Field(default=25.0, description="Max plausible rotation between specimen and query")
    align_min_scale: float = Field(default=0.6, description="Min plausible relative scale")
    align_max_scale: float = Field(default=1.6, description="Max plausible relative scale")
    align_blur_sigma: float = Field(default=4.0, description="Blur for the aligned shape descriptor (2x canvas)")


class FusionModel(BaseModel):
    """Logistic fusion of similarities -> match probability (equal priors).

    Fitted writer-independently by benchmark/fit_fusion.py; never hand-tuned.
    """
    shape_weight: float = Field(default=22.948)
    keypoint_weight: float = Field(default=38.455)
    bias: float = Field(default=-20.058)
    fitted_on: str = Field(
        default="CEDAR-12w 1:1 protocol (73 gen / 98 skilled / 1058 random), balanced L2-LR C=1",
        description="Provenance of the coefficients",
    )


class SystemConfig(BaseModel):
    """Master system configuration pinning policy and model versions."""
    policy_version: str = Field(default="CBUAE-POLICY-2026.09.2", description="Clearing policy regulatory version")
    model_version: str = Field(default="DET-SOTA-v2.0", description="Deterministic verification model version")
    iqa: IQAThresholds = Field(default_factory=IQAThresholds)
    amounts: AmountThresholds = Field(default_factory=AmountThresholds)
    calibration: VerificationCalibration = Field(default_factory=VerificationCalibration)
    weights: FeatureWeights = Field(default_factory=FeatureWeights)
    preprocessing: PreprocessingParams = Field(default_factory=PreprocessingParams)
    representation: RepresentationParams = Field(default_factory=RepresentationParams)
    fusion: FusionModel = Field(default_factory=FusionModel)
    decision: DecisionThresholds = Field(default_factory=DecisionThresholds)


# Default immutable reference configuration
DEFAULT_CONFIG = SystemConfig()
