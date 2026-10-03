"""Signature Zone Detection and Localization.

Provides:
- Standard Cheque Signature Zone Detection (ANSI X9 / GCC / UAE Cheque Specifications)
- Primary & Secondary (Joint Mandate) candidate extraction
- Generic Document Fallback Detector (Contour clustering & Density heuristics)
- Descender-safe padded cropping
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import cv2
import numpy as np

from signature_verification_system.src.core.types import (
    BoundingBox,
    SignatureCandidate,
    DetectionResult,
    ZoneType,
)
from signature_verification_system.src.preprocessing.binarization import adaptive_binarize
from signature_verification_system.src.preprocessing.ink_extractor import remove_horizontal_baseline


class SignatureLocator:
    """Locator for signatures on bank cheques and general administrative documents."""

    def __init__(
        self,
        padding_ratio: float = 0.12,
        min_signature_area: int = 500,
        max_aspect_ratio: float = 7.0,
        min_aspect_ratio: float = 0.8
    ):
        self.padding_ratio = padding_ratio
        self.min_signature_area = min_signature_area
        self.max_aspect_ratio = max_aspect_ratio
        self.min_aspect_ratio = min_aspect_ratio

    def locate_cheque_zones(
        self,
        image: np.ndarray,
        binary_mask: Optional[np.ndarray] = None
    ) -> List[SignatureCandidate]:
        """Locate signatures within standard bank cheque zones (Primary + Joint Secondary).
        
        Cheque Geometry (GCC / CBUAE ICCS Standard):
        - Primary Signatory Zone: Bottom-right quadrant [0.55*W to 0.97*W, 0.48*H to 0.88*H]
        - Secondary Signatory Zone: Bottom-center quadrant [0.25*W to 0.65*W, 0.48*H to 0.88*H]
        """
        h, w = image.shape[:2]
        if binary_mask is None:
            binary_mask = adaptive_binarize(image, method="sauvola")

        # Clean baselines so they don't artificially merge unrelated text
        cleaned_mask = remove_horizontal_baseline(binary_mask, preserve_descenders=True)

        candidates: List[SignatureCandidate] = []

        # Define zone definitions: (zone_type, (x_min_r, y_min_r, x_max_r, y_max_r), base_conf)
        zones = [
            (
                ZoneType.PRIMARY_CHEQUE_ZONE,
                (0.55, 0.48, 0.97, 0.88),
                0.95
            ),
            (
                ZoneType.SECONDARY_CHEQUE_ZONE,
                (0.25, 0.48, 0.65, 0.88),
                0.75
            )
        ]

        for zone_type, (xr1, yr1, xr2, yr2), base_conf in zones:
            zx1 = int(xr1 * w)
            zy1 = int(yr1 * h)
            zx2 = int(xr2 * w)
            zy2 = int(yr2 * h)

            zone_mask = cleaned_mask[zy1:zy2, zx1:zx2]
            ink_count = int(np.count_nonzero(zone_mask))

            # If there's enough ink in the zone to constitute a signature
            if ink_count < 150:
                continue

            # Connected components in zone
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(zone_mask, connectivity=8)
            component_boxes = []

            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                cw = stats[i, cv2.CC_STAT_WIDTH]
                ch = stats[i, cv2.CC_STAT_HEIGHT]
                # Filter tiny dots and overly gigantic full-zone fills
                if area >= 25 and cw < (zx2 - zx1) * 0.95:
                    component_boxes.append((
                        stats[i, cv2.CC_STAT_LEFT],
                        stats[i, cv2.CC_STAT_TOP],
                        cw,
                        ch
                    ))

            if not component_boxes:
                continue

            # Compute bounding envelope of components
            bx1 = min(b[0] for b in component_boxes)
            by1 = min(b[1] for b in component_boxes)
            bx2 = max(b[0] + b[2] for b in component_boxes)
            by2 = max(b[1] + b[3] for b in component_boxes)

            # Map back to global cheque coordinates
            gx1 = zx1 + bx1
            gy1 = zy1 + by1
            gw = bx2 - bx1
            gh = by2 - by1

            if gw * gh < self.min_signature_area:
                continue

            # Add descender/ascender safe padding
            pad_x = int(gw * self.padding_ratio)
            pad_y = int(gh * self.padding_ratio)

            final_x = max(0, gx1 - pad_x)
            final_y = max(0, gy1 - pad_y)
            final_w = min(w - final_x, gw + 2 * pad_x)
            final_h = min(h - final_y, gh + 2 * pad_y)

            bbox = BoundingBox(x=final_x, y=final_y, w=final_w, h=final_h)
            crop_density = float(np.count_nonzero(cleaned_mask[final_y:final_y+final_h, final_x:final_x+final_w])) / float(final_w * final_h)

            candidate = SignatureCandidate(
                bbox=bbox,
                confidence=base_conf,
                zone_type=zone_type,
                aspect_ratio=round(bbox.aspect_ratio, 2),
                stroke_density=round(crop_density, 4),
                metadata={"ink_pixel_count": ink_count}
            )
            candidates.append(candidate)

        return candidates

    def locate_generic_document_zones(
        self,
        image: np.ndarray,
        binary_mask: Optional[np.ndarray] = None
    ) -> List[SignatureCandidate]:
        """Fallback locator for generic forms, letters, and documents.
        
        Clusters contours and filters by stroke density and signature geometry.
        """
        h, w = image.shape[:2]
        if binary_mask is None:
            binary_mask = adaptive_binarize(image, method="sauvola")

        cleaned_mask = remove_horizontal_baseline(binary_mask, preserve_descenders=True)

        # Morphological horizontal dilation to merge cursive loops and letters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        clustered = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clustered, connectivity=8)
        candidates: List[SignatureCandidate] = []

        total_doc_area = float(h * w)

        for i in range(1, num_labels):
            bx = stats[i, cv2.CC_STAT_LEFT]
            by = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            if bh <= 0 or bw <= 0:
                continue

            aspect_ratio = float(bw) / float(bh)
            area_ratio = float(area) / total_doc_area

            # Filter heuristics:
            # 1. Signatures have area between 600 and 15% of document
            # 2. Aspect ratio typically 1.0 to 6.0
            # 3. Stroke density is cursive (0.05 to 0.45) rather than solid stamp (0.8+)
            if area < self.min_signature_area or area_ratio > 0.15:
                continue
            if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                continue

            crop_ink = np.count_nonzero(cleaned_mask[by:by+bh, bx:bx+bw])
            density = float(crop_ink) / float(bw * bh)
            if not (0.03 <= density <= 0.50):
                continue

            # Prioritize candidates located in bottom half of page
            vertical_bias = 0.7 if (by / h) > 0.5 else 0.4
            confidence = min(0.95, round(vertical_bias + min(0.25, area / 10000.0), 2))

            pad_x = int(bw * self.padding_ratio)
            pad_y = int(bh * self.padding_ratio)
            fx = max(0, bx - pad_x)
            fy = max(0, by - pad_y)
            fw = min(w - fx, bw + 2 * pad_x)
            fh = min(h - fy, bh + 2 * pad_y)

            candidates.append(SignatureCandidate(
                bbox=BoundingBox(x=fx, y=fy, w=fw, h=fh),
                confidence=confidence,
                zone_type=ZoneType.GENERIC_CANDIDATE,
                aspect_ratio=round(float(fw) / float(fh), 2),
                stroke_density=round(density, 4),
                metadata={"area": area}
            ))

        # Sort descending by confidence and area
        candidates.sort(key=lambda c: (c.confidence, c.bbox.area), reverse=True)
        return candidates

    def locate(
        self,
        image: np.ndarray,
        is_cheque: bool = True
    ) -> DetectionResult:
        """Execute signature localization pipeline on an input document image.
        
        Args:
            image: Document image array (BGR or grayscale).
            is_cheque: True if input document is a bank cheque, False for generic doc.
            
        Returns:
            DetectionResult populated with candidate zones and best candidate.
        """
        h, w = image.shape[:2]
        doc_type = "cheque" if is_cheque else "generic_document"

        if is_cheque:
            candidates = self.locate_cheque_zones(image)
            # If cheque locator found nothing, fallback to generic
            if not candidates:
                candidates = self.locate_generic_document_zones(image)
        else:
            candidates = self.locate_generic_document_zones(image)

        best_candidate = candidates[0] if candidates else None

        return DetectionResult(
            candidates=candidates,
            best_candidate=best_candidate,
            document_type=doc_type,
            image_shape=(h, w),
            metadata={"candidate_count": len(candidates)}
        )

    def extract_crop(
        self,
        image: np.ndarray,
        candidate: SignatureCandidate
    ) -> np.ndarray:
        """Extract cropped signature array from image using candidate bounding box."""
        b = candidate.bbox
        return image[b.y : b.y + b.h, b.x : b.x + b.w].copy()


def refine_signature_bbox(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    search_expand: float = 0.6,
    pad: int = 6,
) -> Tuple[int, int, int, int]:
    """Grow a detector box to the full extent of the ink strokes it touches (EXP-009).

    Detector boxes often clip descenders and trailing strokes. Within a search
    window (box expanded by `search_expand` on each side) the ink is binarised
    and every connected component intersecting the *original* box is merged.
    Single pass, no chaining: iterating let dashed form borders propagate the
    box across the whole signing area (EXP-009). Printed rules, dashes and
    other flat components are never merged. Deterministic; returns (x, y, w, h).
    """
    x, y, w, h = bbox
    H, W = image.shape[:2]
    sx0, sy0 = max(0, int(x - search_expand * w)), max(0, int(y - search_expand * h))
    sx1, sy1 = min(W, int(x + w + search_expand * w)), min(H, int(y + h + search_expand * h))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    window = gray[sy0:sy1, sx0:sx1]
    _, ink = cv2.threshold(window, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    comps = []
    for i in range(1, n):
        cx, cy, cw, ch, area = (int(v) for v in stats[i])
        is_rule = cw >= 0.8 * window.shape[1] or ch <= 6
        if area < 8 or is_rule:
            continue
        comps.append((cx + sx0, cy + sy0, cx + cw + sx0, cy + ch + sy0))
    bx0, by0, bx1, by1 = x, y, x + w, y + h
    for c0, r0, c1, r1 in comps:
        if c0 < x + w and c1 > x and r0 < y + h and r1 > y:
            bx0, by0, bx1, by1 = min(bx0, c0), min(by0, r0), max(bx1, c1), max(by1, r1)
    bx0, by0 = max(0, bx0 - pad), max(0, by0 - pad)
    bx1, by1 = min(W, bx1 + pad), min(H, by1 + pad)
    return bx0, by0, bx1 - bx0, by1 - by0

