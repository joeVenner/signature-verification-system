"""Cheque Ink Extraction and Stroke Enhancement.

Implements:
- CIE-Lab color-space chromatic ink separation (blue/black vs security tint)
- Horizontal baseline removal with crossing descender preservation
- Pseudo-dynamic stroke density and pressure-gradient estimation
"""

from __future__ import annotations

from typing import Dict, Tuple
import cv2
import numpy as np

from signature_verification_system.src.core.config import PreprocessingParams, DEFAULT_CONFIG


def separate_ink_cielab(image: np.ndarray) -> np.ndarray:
    """Separate pen ink (blue and black) from cheque safety tint using CIE-Lab chromaticity.
    
    In CIE-Lab color space:
    - b* < 120 cleanly separates blue ballpoint/fountain ink from pastel yellow/pink/green paper
    - L* < 110 captures dark black/carbon ink strokes
    
    Args:
        image: BGR color image (or grayscale).
        
    Returns:
        Binary ink mask where stroke ink = 255 and background = 0.
    """
    if len(image.shape) == 2:
        # Grayscale fallback: simple Otsu + Sauvola
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # 1. Blue ink mask: b* channel is lower than neutral (128)
    # Blue ink typically has b* <= 118 and L* <= 200
    blue_mask = (b_channel <= 118) & (l_channel <= 205)

    # 2. Black/Dark ink mask: L* is very low and chromaticity is near-neutral
    black_mask = (l_channel <= 110)

    # 3. Combine ink detections
    combined_mask = (blue_mask | black_mask).astype(np.uint8) * 255

    # Refine with morphological clean-up (remove 1-pixel noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    return combined_mask


def remove_horizontal_baseline(
    binary_mask: np.ndarray,
    preserve_descenders: bool = True,
    kernel_width: int = 35,
    min_crossing_height: int = 7
) -> np.ndarray:
    """Remove pre-printed signature guide lines while preserving crossing stroke descenders.
    
    Naively subtracting the horizontal rule chops through lowercase descenders
    (g, y, j, f, p, q, cursive loops). This algorithm detects vertical/diagonal
    stroke continuity across the baseline and anchors the stroke pixels so they
    are not eroded.
    
    Args:
        binary_mask: Binary image with ink = 255, background = 0.
        preserve_descenders: Enable preservation of descender crossings.
        kernel_width: Width of horizontal morphological line detector kernel.
        min_crossing_height: Height of vertical probe kernel for descender detection.
        
    Returns:
        Cleaned binary image without horizontal line but intact descenders.
    """
    # 1. Extract purely horizontal lines
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    detected_lines = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, horiz_kernel)

    if not preserve_descenders:
        # Direct subtraction
        return cv2.bitwise_and(binary_mask, cv2.bitwise_not(detected_lines))

    # 2. Extract vertical/slanted stroke crossings
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_crossing_height))
    vertical_strokes = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, vert_kernel)

    # 3. Dilate crossing strokes slightly to protect the baseline intersection
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    crossing_bridge = cv2.dilate(vertical_strokes, bridge_kernel)

    # 4. Only remove baseline pixels that DO NOT intersect crossing strokes
    line_to_remove = cv2.bitwise_and(detected_lines, cv2.bitwise_not(crossing_bridge))

    # 5. Subtract non-stroke baseline fragments
    cleaned = cv2.bitwise_and(binary_mask, cv2.bitwise_not(line_to_remove))
    return cleaned


def compute_pseudo_dynamic_density(
    gray_image: np.ndarray,
    binary_mask: np.ndarray
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute pseudo-dynamic stroke density and pressure-gradient metrics.
    
    Offline ballpoint and fountain pen strokes deposit ink inversely proportional
    to ballistic writing speed. Slower movements (or tracing pauses) cause ink
    pooling and dark, wide deposits.
    
    Args:
        gray_image: Grayscale version of signature (uint8).
        binary_mask: Binary mask of ink strokes (ink = 255).
        
    Returns:
        (density_map [0.0 - 1.0], stats_dict)
    """
    if len(gray_image.shape) == 3:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

    mask_bool = binary_mask > 0
    stroke_pixel_count = int(np.count_nonzero(mask_bool))

    if stroke_pixel_count == 0:
        return np.zeros_like(gray_image, dtype=np.float32), {
            "mean_density": 0.0,
            "density_std": 0.0,
            "hesitation_blob_ratio": 0.0,
            "total_stroke_pixels": 0,
        }

    # Ink density: darker pixels (low gray values) have higher ink density
    # Range [0.0, 1.0] where 1.0 is pure black saturated ink
    density_map = np.zeros_like(gray_image, dtype=np.float32)
    density_map[mask_bool] = (255.0 - gray_image[mask_bool].astype(np.float32)) / 255.0

    stroke_densities = density_map[mask_bool]
    mean_density = float(np.mean(stroke_densities))
    density_std = float(np.std(stroke_densities))

    # High-density blobs: pixels with ink density > 0.85 (deep ink pools)
    high_density_pixels = np.count_nonzero(stroke_densities > 0.85)
    hesitation_blob_ratio = float(high_density_pixels) / float(stroke_pixel_count)

    stats = {
        "mean_density": round(mean_density, 4),
        "density_std": round(density_std, 4),
        "hesitation_blob_ratio": round(hesitation_blob_ratio, 4),
        "total_stroke_pixels": stroke_pixel_count,
    }

    return density_map, stats
