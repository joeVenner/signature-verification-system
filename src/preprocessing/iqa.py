"""Image Quality Assessment (IQA) and Deskewing.

Implements ANSI X9.100-181 and CBUAE image specifications:
- Hough & Radon projection profile deskewing
- Laplacian blur detection
- RMS contrast & Michelson contrast
- Normalized luminance / brightness boundaries
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import cv2
import numpy as np

from signature_verification_system.src.core.types import IQAMetrics
from signature_verification_system.src.core.config import IQAThresholds, DEFAULT_CONFIG


def estimate_skew_hough(gray: np.ndarray, max_angle: float = 15.0) -> float:
    """Estimate image skew angle in degrees using probabilistic Hough line transform.
    
    Returns angle in degrees (positive = counter-clockwise, negative = clockwise).
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=gray.shape[1] // 8, maxLineGap=20)
    
    if lines is None or len(lines) == 0:
        return 0.0

    angles = []
    for line in lines:
        line_sq = np.squeeze(line)
        if line_sq.ndim != 1 or len(line_sq) != 4:
            continue
        x1, y1, x2, y2 = line_sq
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if dx == 0:
            continue
        # In image coords, y increases downwards, so clockwise tilt produces dy/dx > 0.
        # Negate to match standard mathematical/Cartesian rotation angle convention.
        angle = -math.degrees(math.atan2(dy, dx))
        if abs(angle) <= max_angle:
            angles.append(angle)

    if not angles:
        return 0.0

    # Return robust median angle
    return float(np.median(angles))


def estimate_skew_projection(
    gray: np.ndarray,
    angle_range: float = 10.0,
    angle_step: float = 0.5
) -> float:
    """Estimate skew angle via horizontal projection profile variance (Radon-like).
    
    Evaluates trial rotation angles; the rotation that maximizes the variance
    of horizontal row sums corresponds to alignment.
    Returns the document's skew angle in degrees.
    """
    h, w = gray.shape[:2]
    target_w = 600
    if w > target_w:
        scale = target_w / float(w)
        resized = cv2.resize(gray, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        resized = gray

    # Binarize inverted so strokes/text are white (foreground)
    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    angles = np.arange(-angle_range, angle_range + angle_step, angle_step)
    best_score = -1.0
    best_corrective_angle = 0.0

    rh, rw = thresh.shape
    center = (rw // 2, rh // 2)

    for angle in angles:
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(thresh, rot_mat, (rw, rh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # Sum along horizontal rows
        proj = np.sum(rotated, axis=1)
        # Variance of row projection profile
        score = float(np.var(proj))
        if score > best_score:
            best_score = score
            best_corrective_angle = angle

    # The document tilt angle is the opposite of the corrective rotation
    doc_skew = -float(best_corrective_angle)
    return doc_skew


def deskew_image(
    image: np.ndarray,
    angle: float,
    border_value: int | Tuple[int, int, int] = 255
) -> np.ndarray:
    """Rotate image to reverse the document skew angle.
    
    Args:
        image: Source grayscale or color image.
        angle: Estimated document skew angle in degrees to cancel out.
        border_value: Background color for empty canvas margins.
    """
    if abs(angle) < 0.05:
        return image.copy()

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    # Apply -angle rotation to cancel out positive/negative skew
    rot_mat = cv2.getRotationMatrix2D(center, -angle, 1.0)
    
    rotated = cv2.warpAffine(
        image,
        rot_mat,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )
    return rotated


def calculate_blur_score(gray: np.ndarray) -> float:
    """Calculate focus measure using variance of the Laplacian operator."""
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def calculate_contrast(gray: np.ndarray) -> Tuple[float, float]:
    """Calculate RMS contrast (standard deviation) and Michelson contrast.
    
    Returns:
        (rms_contrast, michelson_contrast)
    """
    rms = float(np.std(gray))
    min_val = float(np.min(gray))
    max_val = float(np.max(gray))
    denom = max_val + min_val
    michelson = float((max_val - min_val) / (denom if denom > 0 else 1.0))
    return rms, michelson


def calculate_brightness(gray: np.ndarray) -> float:
    """Calculate normalized mean image brightness in range [0.0, 1.0]."""
    return float(np.mean(gray) / 255.0)



def apply_clahe_enhancement(
    image: np.ndarray,
    clip_limit: float = 3.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Enhance local contrast using Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Supports both single-channel grayscale and multi-channel color (BGR) images.
    For BGR images, CLAHE is applied to the L (luminance) channel in LAB color space.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if len(image.shape) == 2:
        return clahe.apply(image)
    elif len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = clahe.apply(l)
        enhanced_lab = cv2.merge([l_clahe, a, b])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return image.copy()


def assess_image_quality(
    image: np.ndarray,
    thresholds: Optional[IQAThresholds] = None,
    deskew: bool = True
) -> Tuple[IQAMetrics, np.ndarray]:
    """Comprehensive Image Quality Assessment (IQA).
    
    Checks focus, tilt, contrast, and illumination levels per CBUAE and ANSI standards.
    Optionally returns deskewed image.
    
    Returns:
        (IQAMetrics, processed_image)
    """
    cfg = thresholds or DEFAULT_CONFIG.iqa

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 1. Skew analysis (hybrid: projection profile + Hough fallback)
    skew_angle = estimate_skew_projection(gray)
    if abs(skew_angle) < 0.1:
        hough_angle = estimate_skew_hough(gray)
        if abs(hough_angle) > 0.2:
            skew_angle = hough_angle

    is_skewed = abs(skew_angle) > cfg.max_skew_degrees

    # Deskew if requested
    if deskew and abs(skew_angle) >= 0.1:
        processed_image = deskew_image(image, skew_angle)
        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY) if len(processed_image.shape) == 3 else processed_image
    else:
        processed_image = image.copy()

    # 2. Blur / Focus analysis
    blur_score = calculate_blur_score(gray)
    is_blurry = blur_score < cfg.min_blur_laplacian_var

    # 3. Contrast analysis & CLAHE pre-enhancement for low-contrast images
    rms_contrast, _ = calculate_contrast(gray)
    is_low_contrast = rms_contrast < cfg.min_contrast_rms

    if is_low_contrast:
        enhanced_image = apply_clahe_enhancement(processed_image)
        enhanced_gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY) if len(enhanced_image.shape) == 3 else enhanced_image
        enh_rms, _ = calculate_contrast(enhanced_gray)
        if enh_rms > rms_contrast:
            processed_image = enhanced_image
            gray = enhanced_gray
            rms_contrast = enh_rms
            is_low_contrast = rms_contrast < cfg.min_contrast_rms

    # 4. Brightness / Illumination analysis
    brightness = calculate_brightness(gray)
    is_too_dark = brightness < cfg.min_brightness
    is_too_bright = brightness > cfg.max_brightness

    # 5. Failure compilation
    failure_reasons = []
    if is_skewed:
        failure_reasons.append(f"Excessive document skew: {skew_angle:+.2f}° (max: ±{cfg.max_skew_degrees}°)")
    if is_blurry:
        failure_reasons.append(f"Image blur/out of focus: Laplacian variance {blur_score:.1f} < {cfg.min_blur_laplacian_var}")
    if is_low_contrast:
        failure_reasons.append(f"Insufficient contrast: RMS {rms_contrast:.1f} < {cfg.min_contrast_rms}")
    if is_too_dark:
        failure_reasons.append(f"Document underexposed / too dark: brightness {brightness:.2f} < {cfg.min_brightness}")
    if is_too_bright:
        failure_reasons.append(f"Document washed out / overexposed: brightness {brightness:.2f} > {cfg.max_brightness}")

    passed = len(failure_reasons) == 0

    metrics = IQAMetrics(
        skew_angle=round(skew_angle, 2),
        blur_score=round(blur_score, 2),
        contrast_score=round(rms_contrast, 2),
        brightness_score=round(brightness, 3),
        is_skewed=is_skewed,
        is_blurry=is_blurry,
        is_too_dark=is_too_dark,
        is_too_bright=is_too_bright,
        passed=passed,
        failure_reasons=failure_reasons
    )

    return metrics, processed_image
