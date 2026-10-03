"""Preprocessing modules: IQA, binarization, and ink extraction."""

from signature_verification_system.src.preprocessing.iqa import (
    estimate_skew_hough,
    estimate_skew_projection,
    deskew_image,
    calculate_blur_score,
    calculate_contrast,
    calculate_brightness,
    assess_image_quality,
    apply_clahe_enhancement,
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

__all__ = [
    "estimate_skew_hough",
    "estimate_skew_projection",
    "deskew_image",
    "calculate_blur_score",
    "calculate_contrast",
    "calculate_brightness",
    "assess_image_quality",
    "apply_clahe_enhancement",
    "sauvola_threshold",
    "wolf_threshold",
    "suppress_pantograph",
    "adaptive_binarize",
    "separate_ink_cielab",
    "remove_horizontal_baseline",
    "compute_pseudo_dynamic_density",
]
