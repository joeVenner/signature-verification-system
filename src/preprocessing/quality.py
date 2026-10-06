"""Signature quality gate (requirement C, EXP-007).

Decides whether a signature crop is fit for automated verification *before*
any score is trusted. Every check is a measured, contrast-normalised signal;
thresholds sit well below the minimum observed on clean CEDAR enrolment images
(see benchmark/experiments.md EXP-007), so a clean scan never fails the gate.

Blocking issues make the verification INCONCLUSIVE; warnings are reported but
do not block.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import cv2
import numpy as np
from pydantic import BaseModel, Field

from signature_verification_system.src.preprocessing.background import PreparedSignature, prepare_signature
from signature_verification_system.src.preprocessing.normalization import ensure_dark_ink, to_gray

# Thresholds (clean-data minimum / maximum in brackets, 70 CEDAR images)
MIN_INK_CONTRAST = 40.0       # grey levels paper-median minus ink p1        [clean min 70]
MIN_EDGE_SHARPNESS = 1.6      # p99.5 gradient magnitude / ink contrast       [clean min 3.89]
MAX_NOISE_RATIO = 0.15        # paper residual sigma / ink contrast, blocking [clean max 0.020]
WARN_NOISE_RATIO = 0.06       # non-blocking warning
MIN_INK_PIXELS = 150          # ink pixels at the ink mask level
MIN_SIGNATURE_HEIGHT = 16     # px of the ink bounding box
MIN_SIGNATURE_WIDTH = 32
BORDER_MARGIN = 2             # ink within this many px of the edge => possibly cropped


_LOG = logging.getLogger(__name__)

# Plain-language text for every blocking issue and warning code below, for
# user-facing views (the codes stay the machine-readable contract).
QUALITY_MESSAGES = {
    "EMPTY_OR_INVALID_IMAGE": "The image is empty or too small to contain a signature.",
    "PREPROCESSING_FAILED": "The image could not be preprocessed (unsupported format or layout).",
    "IMAGE_DIMENSIONS_UNSUPPORTED": "Image size is not supported: above 16 megapixels, or wider / taller than 20:1. "
                                    "Crop the image to the signature.",
    "INSUFFICIENT_INK_CONTRAST": "The ink is too faint against the paper.",
    "INSUFFICIENT_INK": "Too little ink was found to analyse a signature.",
    "SIGNATURE_TOO_SMALL": "The signature is too small in the image.",
    "IMAGE_TOO_BLURRY": "The image is too blurry: stroke edges are not sharp enough.",
    "BACKGROUND_NOT_SEPARABLE": "A patterned (cheque security) background was detected, "
                                "but the signature ink could not be separated from it.",
    "EXCESSIVE_NOISE": "The background is too noisy to separate the ink reliably.",
    "ELEVATED_NOISE": "The background is noticeably noisy; measurements may be less reliable.",
    "SIGNATURE_MAY_BE_CROPPED": "Ink touches the image edge; the signature may be cut off.",
    "POLARITY_INVERTED_CORRECTED": "Light ink on a dark background was detected and inverted.",
    "TEXTURED_BACKGROUND_REMOVED": "A patterned (cheque security) background was detected; "
                                   "the signature ink was extracted from it before analysis.",
    "LOW_RESOLUTION_UPSCALED": "The crop is low resolution and was upscaled before analysis.",
}


def quality_message(code: str) -> str:
    """Plain-language text for a quality code (a generic sentence for an unknown one)."""
    return QUALITY_MESSAGES.get(code, "A signature quality check flagged this image.")


class SignatureQuality(BaseModel):
    """Measured quality signals for one signature image."""

    passed: bool
    ink_contrast: float = Field(..., description="Paper median minus darkest-1% ink, grey levels")
    edge_sharpness: float = Field(..., description="Contrast-normalised stroke edge steepness")
    noise_ratio: float = Field(..., description="Background noise sigma relative to ink contrast")
    ink_pixels: int
    signature_width: int
    signature_height: int
    polarity_inverted: bool
    background_texture: float = Field(0.0, description="Share of background pixels covered by security texture")
    background_removed: bool = Field(False, description="Ink was extracted from a textured background before measuring")
    blocking_issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


def _blocked(issue: str, inverted: bool = False) -> SignatureQuality:
    """Gate result for an image that is rejected before any signal is measured."""
    return SignatureQuality(
        passed=False, ink_contrast=0.0, edge_sharpness=0.0, noise_ratio=0.0, ink_pixels=0,
        signature_width=0, signature_height=0, polarity_inverted=inverted, blocking_issues=[issue],
    )


def assess_signature_quality(image: Optional[np.ndarray], prepared: Optional[PreparedSignature] = None) -> SignatureQuality:
    """Measure quality signals and apply the gate. Deterministic; never raises.

    On a textured (cheque security) background the signals are measured on the
    extracted ink layer (background.py), so the texture is not mistaken for
    noise; edge sharpness stays on the input because extraction redraws the
    stroke boundaries and would hide blur. If the texture is detected but no
    signature can be separated from it the image is blocked.

    `prepared` is `normalization.prepare_image(image)` if already computed
    (it must come from this exact image).
    """
    if image is None or image.size == 0 or min(image.shape[:2]) < 4:
        return _blocked("EMPTY_OR_INVALID_IMAGE")
    try:
        return _measure(image, prepared)
    except (cv2.error, MemoryError, ValueError):
        # Unsupported layouts / dtypes or an OpenCV failure must make the
        # verification INCONCLUSIVE, never an API error. Details go to the log only.
        _LOG.warning("Signature quality preprocessing failed", exc_info=True)
        return _blocked("PREPROCESSING_FAILED")


def _measure(image: np.ndarray, prepared: Optional[PreparedSignature]) -> SignatureQuality:
    """Quality signals for a non-empty image (may raise on unsupported input)."""
    input_u8, inverted = ensure_dark_ink(to_gray(image))
    if prepared is None:
        prepared = prepare_signature(image, input_u8)
    if not prepared.dimensions_supported:
        return _blocked("IMAGE_DIMENSIONS_UNSUPPORTED", inverted)
    gray_u8 = to_gray(prepared.gate_image) if prepared.background_removed else input_u8
    gray = gray_u8.astype(np.float64)
    paper = float(np.median(gray))
    contrast = paper - float(np.quantile(gray, 0.01))

    residual = gray - cv2.medianBlur(gray_u8, 5).astype(np.float64)
    paper_mask = gray > paper - 0.25 * contrast
    noise_sigma = 1.4826 * float(np.median(np.abs(residual[paper_mask]))) if paper_mask.any() else 0.0
    noise_ratio = noise_sigma / max(contrast, 1.0)

    sharp_src = input_u8.astype(np.float64)
    sharp_contrast = float(np.median(sharp_src)) - float(np.quantile(sharp_src, 0.01))
    gx = cv2.Sobel(sharp_src, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(sharp_src, cv2.CV_64F, 0, 1, ksize=3)
    sharpness = float(np.quantile(np.hypot(gx, gy), 0.995)) / max(sharp_contrast, 1.0)

    ink_mask = gray < paper - 0.5 * contrast if contrast > 0 else np.zeros_like(gray, bool)
    ys, xs = np.nonzero(ink_mask)
    ink_pixels = int(len(ys))
    if ink_pixels:
        width, height = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    else:
        width = height = 0

    blocking: List[str] = []
    warnings: List[str] = []
    if contrast < MIN_INK_CONTRAST:
        blocking.append("INSUFFICIENT_INK_CONTRAST")
    if ink_pixels < MIN_INK_PIXELS:
        blocking.append("INSUFFICIENT_INK")
    elif width < MIN_SIGNATURE_WIDTH or height < MIN_SIGNATURE_HEIGHT:
        blocking.append("SIGNATURE_TOO_SMALL")
    if contrast >= MIN_INK_CONTRAST and sharpness < MIN_EDGE_SHARPNESS:
        blocking.append("IMAGE_TOO_BLURRY")
    if not prepared.separable:
        # The raw texture is what makes the noise ratio high here; one cause, one issue.
        blocking.append("BACKGROUND_NOT_SEPARABLE")
    elif noise_ratio > MAX_NOISE_RATIO:
        blocking.append("EXCESSIVE_NOISE")
    elif noise_ratio > WARN_NOISE_RATIO:
        warnings.append("ELEVATED_NOISE")
    if ink_pixels:
        h, w = gray.shape
        if xs.min() < BORDER_MARGIN or ys.min() < BORDER_MARGIN or xs.max() >= w - BORDER_MARGIN or ys.max() >= h - BORDER_MARGIN:
            warnings.append("SIGNATURE_MAY_BE_CROPPED")
    if inverted:
        warnings.append("POLARITY_INVERTED_CORRECTED")
    if prepared.background_removed:
        warnings.append("TEXTURED_BACKGROUND_REMOVED")
    if prepared.upscale > 1.0:
        warnings.append("LOW_RESOLUTION_UPSCALED")

    return SignatureQuality(
        passed=not blocking,
        ink_contrast=round(contrast, 2),
        edge_sharpness=round(sharpness, 3),
        noise_ratio=round(noise_ratio, 4),
        ink_pixels=ink_pixels,
        signature_width=width,
        signature_height=height,
        polarity_inverted=inverted,
        background_texture=round(prepared.texture.texture_fraction, 4),
        background_removed=prepared.background_removed,
        blocking_issues=blocking,
        warnings=warnings,
    )
