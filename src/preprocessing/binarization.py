"""Adaptive Document Binarization and Security Pantograph Suppression.

Provides:
- Sauvola adaptive local binarization (Sauvola & Pietikäinen)
- Wolf-Jolion adaptive binarization (Wolf et al., 2002)
- Pantograph / guilloche security pattern suppression
- Connected-component speckle noise elimination
"""

from __future__ import annotations

from typing import Literal
import cv2
import numpy as np

from signature_verification_system.src.core.config import PreprocessingParams, DEFAULT_CONFIG


def sauvola_threshold(
    gray: np.ndarray,
    window_size: int = 25,
    k: float = 0.2,
    r: float = 128.0
) -> np.ndarray:
    """Compute Sauvola adaptive binary thresholding.
    
    Formula: T(x, y) = mean * (1 + k * (std / R - 1))
    Where ink is foreground (255) and background paper is 0.
    
    Args:
        gray: Single-channel grayscale input image uint8.
        window_size: Size of local neighborhood window (must be odd).
        k: Dynamic sensitivity factor (typical: 0.1 - 0.3).
        r: Dynamic range of standard deviation (default 128.0 for 8-bit).
        
    Returns:
        Binary image with foreground ink = 255, background = 0.
    """
    if window_size % 2 == 0:
        window_size += 1

    fimg = gray.astype(np.float64)
    ksize = (window_size, window_size)

    # Local mean and local squared mean using integral boxFilter
    mean = cv2.boxFilter(fimg, cv2.CV_64F, ksize)
    mean_sq = cv2.boxFilter(fimg ** 2, cv2.CV_64F, ksize)
    variance = np.maximum(mean_sq - mean ** 2, 0.0)
    std = np.sqrt(variance)

    thresh = mean * (1.0 + k * (std / r - 1.0))
    # Foreground ink is darker than threshold
    binary = np.zeros_like(gray, dtype=np.uint8)
    binary[fimg < thresh] = 255
    return binary


def wolf_threshold(
    gray: np.ndarray,
    window_size: int = 25,
    k: float = 0.2
) -> np.ndarray:
    """Compute Wolf-Jolion-Chassaing adaptive thresholding.
    
    Normalizes local standard deviation by global maximum standard deviation and
    local mean by global minimum gray value. Resilient against uneven background tint.
    
    Formula: T(x, y) = mean + k * (std / s_max) * (mean - m_min)
    
    Returns:
        Binary image with foreground ink = 255, background = 0.
    """
    if window_size % 2 == 0:
        window_size += 1

    fimg = gray.astype(np.float64)
    ksize = (window_size, window_size)

    mean = cv2.boxFilter(fimg, cv2.CV_64F, ksize)
    mean_sq = cv2.boxFilter(fimg ** 2, cv2.CV_64F, ksize)
    variance = np.maximum(mean_sq - mean ** 2, 0.0)
    std = np.sqrt(variance)

    m_min = float(np.min(fimg))
    s_max = float(np.max(std))
    if s_max <= 0:
        s_max = 1.0

    thresh = mean + k * (std / s_max) * (mean - m_min)
    binary = np.zeros_like(gray, dtype=np.uint8)
    binary[fimg < thresh] = 255
    return binary


def suppress_pantograph(
    binary: np.ndarray,
    min_speckle_area: int = 6,
    morph_cleanup: bool = True
) -> np.ndarray:
    """Suppress isolated pantograph dots, guilloche tint fragments, and printer speckle.
    
    Args:
        binary: Binary image where foreground ink is 255 and background is 0.
        min_speckle_area: Minimum connected-component pixel area to preserve as stroke.
        morph_cleanup: Apply minor morphological opening to smooth jagged edges.
        
    Returns:
        Cleaned binary image (ink = 255, background = 0).
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)

    # stats columns: [LEFT, TOP, WIDTH, HEIGHT, AREA]
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]

        # Ignore tiny isolated dots (pantograph halftones)
        if area < min_speckle_area:
            continue
        
        # Single-pixel width/height dots
        if w <= 1 and h <= 1:
            continue

        cleaned[labels == label] = 255

    if morph_cleanup:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    return cleaned


def adaptive_binarize(
    image: np.ndarray,
    method: Literal["sauvola", "wolf"] = "sauvola",
    suppress_security_patterns: bool = True,
    params: PreprocessingParams | None = None
) -> np.ndarray:
    """Unified entry point for adaptive cheque and document binarization.
    
    Args:
        image: BGR or Grayscale input image.
        method: "sauvola" or "wolf".
        suppress_security_patterns: Enable pantograph and speckle elimination.
        params: Hyperparameter configuration.
        
    Returns:
        Clean binary mask with ink strokes = 255, paper background = 0.
    """
    p = params or DEFAULT_CONFIG.preprocessing

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Pre-filter: bilateral smoothing retains stroke edges while blurring pastel waves
    smoothed = cv2.bilateralFilter(gray, d=5, sigmaColor=30, sigmaSpace=30)

    if method == "wolf":
        binary = wolf_threshold(smoothed, window_size=p.wolf_window_size, k=p.wolf_k)
    else:
        binary = sauvola_threshold(
            smoothed,
            window_size=p.sauvola_window_size,
            k=p.sauvola_k,
            r=p.sauvola_r
        )

    if suppress_security_patterns:
        binary = suppress_pantograph(binary, min_speckle_area=p.pantograph_min_speckle_area)

    return binary
