"""Deterministic SOTA Multi-Feature Biometric Signature Verification.

Extracts and fuses structural, topological, and kinematic micro-features:
1. HOG (Histogram of Oriented Gradients) - Local edge orientation distribution
2. Hu Invariant Moments - Global geometric moment invariants (scale/rotation invariant)
3. Contour Topology - Aspect ratio, solidity, compactness, and loop structure
4. Skeleton Graph - Zhang-Suen medial axis thinning, pen-lift endpoints, and branch junctions
5. Stroke Width Variation - Distance transform profile along skeleton
6. Curvature Variation / Micro-Jitter - Second-order gradient direction variance along stroke skeletons
7. Kinematic Hesitation & Tremor - Ink-blobbing and stroke width variability along skeleton
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from skimage.feature import hog
from skimage.morphology import skeletonize

from signature_verification_system.src.core.types import (
    FeatureBreakdown,
    VerificationResult,
)
from signature_verification_system.src.core.config import (
    SystemConfig,
    DEFAULT_CONFIG,
)
from signature_verification_system.src.preprocessing.binarization import adaptive_binarize
from signature_verification_system.src.preprocessing.ink_extractor import (
    remove_horizontal_baseline,
    compute_pseudo_dynamic_density,
)
from signature_verification_system.src.verification.features import SignatureFeatures, extract_features
from signature_verification_system.src.verification.similarity import (
    PairSimilarity, compare, compare_multi, explanation_signals,
)
from signature_verification_system.src.verification.explanation import build_explanation
from signature_verification_system.src.adjudication.thresholds import ACCEPT, classify_band
from signature_verification_system.src.preprocessing.background import PreparedSignature
from signature_verification_system.src.preprocessing.normalization import prepare_image
from signature_verification_system.src.preprocessing.quality import SignatureQuality, assess_signature_quality


_BAND_NOTES = {
    "ACCEPT": "Match evidence is above the auto-accept operating point (see band reliability for held-out error counts).",
    "REVIEW": "Ambiguous: between the reject and auto-accept operating points; route to manual review.",
    "REJECT": "Match evidence below the reject operating point; shape and/or stroke detail inconsistent with specimen.",
}


def _alignment_used(pairs: List[PairSimilarity]) -> Optional[dict]:
    """Report the alignment of the specimen that supplied the layout score, if
    that score came from the aligned comparison (never claim unused corrections)."""
    best = max(range(len(pairs)), key=lambda i: pairs[i].shape)
    al = pairs[best].alignment
    if al is None or not pairs[best].alignment_used:
        return None
    return {"rotation_deg": round(al.rotation_deg, 2), "scale": round(al.scale, 3), "specimen": best + 1}


def _best_signals(per_reference: List[dict]) -> dict:
    """Best value of each signal across specimens (mirrors feature-wise max aggregation)."""
    return {k: max(d[k] for d in per_reference) for k in per_reference[0]}


def _band_note(band: str) -> str:
    return _BAND_NOTES[band]


_LOG = logging.getLogger(__name__)
# Fixed text for internal failures: OpenCV / allocator messages never reach a response.
EXTRACTION_ERROR_REASON = "feature extraction failed"


def prepare_or_none(image: Optional[np.ndarray]) -> Optional[PreparedSignature]:
    """Background / resolution routing computed once per image for gate and features.

    None (each consumer then prepares and handles the failure itself) for an
    empty input or when preprocessing fails.
    """
    if image is None or image.size == 0 or min(image.shape[:2]) < 4:
        return None
    try:
        return prepare_image(image)
    except (cv2.error, MemoryError, ValueError):
        return None


class DeterministicVerifier:
    """Multi-feature deterministic signature verification engine."""

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.canonical_size = (256, 128)  # (width, height)

    def preprocess_crop(
        self,
        image: np.ndarray,
        clean_baseline: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize cropped signature to canonical size and generate clean binary mask.
        
        Returns:
            (canonical_gray, canonical_binary) where binary foreground ink = 255.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Binarize
        binary = adaptive_binarize(gray, method="sauvola")

        # Clean baseline if needed
        if clean_baseline:
            binary = remove_horizontal_baseline(binary, preserve_descenders=True)

        # Tight crop to ink envelope
        pts = cv2.findNonZero(binary)
        if pts is not None and len(pts) > 20:
            x, y, w, h = cv2.boundingRect(pts)
            pad = 8
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(gray.shape[1], x + w + pad)
            y2 = min(gray.shape[0], y + h + pad)
            tight_gray = gray[y1:y2, x1:x2]
            tight_bin = binary[y1:y2, x1:x2]
        else:
            tight_gray = gray
            tight_bin = binary

        # Resize to canonical aspect-standard dimensions
        canonical_gray = cv2.resize(tight_gray, self.canonical_size, interpolation=cv2.INTER_AREA)
        canonical_bin = cv2.resize(tight_bin, self.canonical_size, interpolation=cv2.INTER_NEAREST)

        return canonical_gray, canonical_bin

    def compute_hog_feature(self, canonical_gray: np.ndarray) -> np.ndarray:
        """Compute normalized HOG orientation gradient vector."""
        feat = hog(
            canonical_gray,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )
        norm = np.linalg.norm(feat)
        return feat / (norm + 1e-7)

    def compute_hu_moments(self, canonical_bin: np.ndarray) -> np.ndarray:
        """Compute log-transformed Hu moment invariants."""
        moments = cv2.moments(canonical_bin)
        hu = cv2.HuMoments(moments).flatten()
        log_hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
        return log_hu

    def compute_contour_descriptors(self, canonical_bin: np.ndarray) -> Dict[str, float]:
        """Compute shape invariants: solidity, perimeter compactness, aspect ratio, extent."""
        contours, hierarchy = cv2.findContours(canonical_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"solidity": 0.0, "compactness": 0.0, "aspect_ratio": 1.0, "loop_count": 0.0}

        # Main contour (largest by area)
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0.0

        compactness = (perimeter ** 2) / (4.0 * np.pi * area) if area > 0 else 0.0

        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / float(h) if h > 0 else 1.0

        # Number of internal holes / loops (Euler characteristic)
        loop_count = 0
        if hierarchy is not None:
            # Hierarchy format: [Next, Previous, First_Child, Parent]
            for h_entry in hierarchy[0]:
                if h_entry[3] != -1:  # has a parent -> internal hole
                    loop_count += 1

        return {
            "solidity": float(np.clip(solidity, 0.0, 1.0)),
            "compactness": float(compactness),
            "aspect_ratio": float(aspect_ratio),
            "loop_count": float(loop_count)
        }

    def compute_skeleton_topology(self, canonical_bin: np.ndarray) -> Tuple[np.ndarray, int, int, int]:
        """Extract Zhang-Suen medial skeleton, endpoint count (pen lifts), and junction count."""
        skel = skeletonize(canonical_bin > 0)
        skel_pts = skel > 0
        skel_len = int(np.sum(skel_pts))

        if skel_len == 0:
            return skel, 0, 0, 0

        # Count 8-neighbors
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        neighbors = cv2.filter2D(skel.astype(np.uint8), -1, kernel)

        endpoints = int(np.sum((neighbors == 1) & skel_pts))
        junctions = int(np.sum((neighbors >= 3) & skel_pts))

        return skel, endpoints, junctions, skel_len

    def compute_stroke_widths(self, canonical_bin: np.ndarray, skel: np.ndarray) -> Tuple[float, float]:
        """Compute mean and variance of stroke thickness along skeleton."""
        dt = cv2.distanceTransform(canonical_bin, cv2.DIST_L2, 5)
        skel_pts = skel > 0
        if np.sum(skel_pts) == 0:
            return 1.0, 0.0

        widths = dt[skel_pts] * 2.0
        return float(np.mean(widths)), float(np.std(widths))

    def compute_curvature_micro_jitter(
        self,
        canonical_gray: np.ndarray,
        canonical_bin: np.ndarray,
        skel: Optional[np.ndarray] = None
    ) -> float:
        """Compute second-order gradient direction variance along stroke skeletons.
        
        Natural handwriting possesses fluent ballistic velocity with continuous curvature.
        Traced forgeries exhibit jerky pen adjustments, high curvature variance, and
        micro-jitters along the stroke trajectory.
        
        Returns:
            Curvature direction variance along the skeleton.
        """
        if skel is None:
            skel = skeletonize(canonical_bin > 0)

        pts = skel > 0
        if np.sum(pts) < 15:
            return 0.0

        gx = cv2.Sobel(canonical_gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(canonical_gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy) + 1e-6
        ux = gx / mag
        uy = gy / mag

        uxx = cv2.Sobel(ux, cv2.CV_64F, 1, 0, ksize=3)
        uxy = cv2.Sobel(ux, cv2.CV_64F, 0, 1, ksize=3)
        uyx = cv2.Sobel(uy, cv2.CV_64F, 1, 0, ksize=3)
        uyy = cv2.Sobel(uy, cv2.CV_64F, 0, 1, ksize=3)

        curv = np.sqrt(uxx**2 + uxy**2 + uyx**2 + uyy**2)
        skel_curv = curv[pts]
        return float(np.var(skel_curv))

    def compute_hesitation_index(
        self,
        canonical_gray: np.ndarray,
        canonical_bin: np.ndarray,
        skel: Optional[np.ndarray] = None
    ) -> float:
        """Calculate tremor, ink blobbing, and stroke-width variability along skeleton.
        
        Traced forgeries exhibit lack of ballistic speed: high curvature fluctuations
        (tremors), ink pooling at turning points, and stroke width irregularity along the skeleton.
        Returns hesitation index in [0.0, 1.0].
        """
        if skel is None:
            skel = skeletonize(canonical_bin > 0)

        # 1. High frequency tremor along external contour
        contours, _ = cv2.findContours(canonical_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        tremor_score = 0.0
        if contours:
            c = max(contours, key=cv2.contourArea)
            if len(c) > 30:
                diffs = np.diff(c[:, 0, :], axis=0)
                angles = np.arctan2(diffs[:, 1], diffs[:, 0])
                angle_diffs = np.diff(angles)
                tremor_score = float(np.std(angle_diffs)) / np.pi

        # 2. Ink blobbing / pooling ratio
        _, stats = compute_pseudo_dynamic_density(canonical_gray, canonical_bin)
        blob_ratio = float(stats.get("hesitation_blob_ratio", 0.0))

        # 3. Stroke width standard deviation along skeleton
        dt = cv2.distanceTransform(canonical_bin, cv2.DIST_L2, 5)
        pts = skel > 0
        if np.sum(pts) > 0:
            widths = dt[pts] * 2.0
            sw_std = float(np.std(widths))
            sw_mean = float(np.mean(widths))
            sw_variability = float(sw_std / (sw_mean + 1e-5))
        else:
            sw_variability = 0.0

        # Fused hesitation index measuring ink blobbing, stroke width std along skeleton, and contour tremor
        hesitation = float(np.clip(0.40 * blob_ratio + 0.35 * min(1.0, sw_variability) + 0.25 * tremor_score, 0.0, 1.0))
        return hesitation

    @staticmethod
    def calibrate_score(raw_score: float, s0: float = 0.65, k: float = 9.5) -> float:
        """Map raw fusion score to normalized [0, 1] biometric confidence scale via sigmoid calibration.
        
        Centered at the empirical EER boundary (s0=0.65, k=9.5), mapping raw match
        scores to a calibrated confidence metric where s0 corresponds to the L1 Amber decision threshold.
        """
        if raw_score <= 0.0:
            return 0.0
        if raw_score >= 1.0:
            return 1.0
        sig_raw = 1.0 / (1.0 + np.exp(-k * (raw_score - s0)))
        sig_min = 1.0 / (1.0 + np.exp(-k * (0.0 - s0)))
        sig_max = 1.0 / (1.0 + np.exp(-k * (1.0 - s0)))
        calibrated = (sig_raw - sig_min) / (sig_max - sig_min)
        return float(np.clip(calibrated, 0.0, 1.0))

    def verify(
        self,
        reference_image: np.ndarray,
        questioned_image: np.ndarray
    ) -> VerificationResult:
        """Verify a questioned signature against one reference specimen (v3).

        Decision score = logistic fusion of (a) global gradient-grid shape
        similarity and (b) RANSAC-verified SIFT keypoint correspondence, both
        computed on photometrically normalised ink maps (see
        benchmark/experiments.md EXP-001). The legacy v2 sub-scores are still
        computed — on the normalised images — and reported as diagnostics only.

        Images without detectable ink return score 0.0 with an INCONCLUSIVE note
        instead of raising, so the API can surface a quality failure.
        """
        p_ref, p_que = prepare_or_none(reference_image), prepare_or_none(questioned_image)
        q_ref = assess_signature_quality(reference_image, p_ref)
        q_que = assess_signature_quality(questioned_image, p_que)
        quality = {"questioned": q_que.model_dump(), "references": [q_ref.model_dump()]}
        if not q_que.passed:
            return self._inconclusive("questioned image quality too low (" + ", ".join(q_que.blocking_issues) + ")", quality)
        if not q_ref.passed:
            return self._inconclusive("reference specimen quality too low (" + ", ".join(q_ref.blocking_issues) + ")", quality)
        try:
            ref_feat = extract_features(reference_image, self.config.representation, p_ref)
            que_feat = extract_features(questioned_image, self.config.representation, p_que)
        except ValueError as exc:
            return self._inconclusive(str(exc), quality)
        except (cv2.error, MemoryError):
            _LOG.warning("feature extraction failed during verify", exc_info=True)
            return self._inconclusive(EXTRACTION_ERROR_REASON, quality)
        result = self._result_from_features(ref_feat, que_feat)
        pair = compare(ref_feat, que_feat, self.config.fusion, self.config.representation)
        signals = explanation_signals(ref_feat, que_feat, pair)
        return result.model_copy(update={
            "quality": quality,
            "explanation": build_explanation(signals, result.decision_band, result.match_logit, 1, quality,
                                             _alignment_used([pair])),
        })

    def verify_against_references(
        self,
        reference_images: List[np.ndarray],
        questioned_image: np.ndarray,
    ) -> VerificationResult:
        """Verify a questioned signature against several enrolled specimens.

        Uses feature-wise max aggregation (EXP-003). Reference images without
        detectable ink are skipped; if none remain the result is INCONCLUSIVE.
        """
        if not reference_images:
            raise ValueError("At least one reference signature is required")
        p_que = prepare_or_none(questioned_image)
        p_refs = [prepare_or_none(img) for img in reference_images]
        q_que = assess_signature_quality(questioned_image, p_que)
        q_refs = [assess_signature_quality(img, p) for img, p in zip(reference_images, p_refs)]
        quality = {"questioned": q_que.model_dump(), "references": [q.model_dump() for q in q_refs]}
        if not q_que.passed:
            return self._inconclusive("questioned image quality too low (" + ", ".join(q_que.blocking_issues) + ")", quality)
        try:
            que_feat = extract_features(questioned_image, self.config.representation, p_que)
        except ValueError as exc:
            return self._inconclusive(f"questioned image: {exc}", quality)
        except (cv2.error, MemoryError):
            _LOG.warning("questioned feature extraction failed", exc_info=True)
            return self._inconclusive(f"questioned image: {EXTRACTION_ERROR_REASON}", quality)
        ref_feats = []
        for img, p, q in zip(reference_images, p_refs, q_refs):
            if not q.passed:
                continue
            try:
                ref_feats.append(extract_features(img, self.config.representation, p))
            except ValueError:
                continue
            except (cv2.error, MemoryError):
                _LOG.warning("reference feature extraction failed", exc_info=True)
                continue
        if not ref_feats:
            return self._inconclusive("no usable reference specimen", quality)
        multi = compare_multi(ref_feats, que_feat, self.config.fusion, self.config.representation)
        best = ref_feats[multi.best_keypoint_index]
        single = self._result_from_features(best, que_feat)
        score = round(float(multi.probability), 4)
        band = classify_band(multi.fused_logit, len(ref_feats), self.config.decision)
        features = single.features.model_copy(update={
            "shape_similarity": round(multi.per_reference[multi.best_shape_index].shape, 4),
            "keypoint_similarity": round(multi.per_reference[multi.best_keypoint_index].keypoint.similarity, 4),
            "fused_logit": round(multi.fused_logit, 4),
        })
        notes = [_band_note(band.band)]
        skipped = len(reference_images) - len(ref_feats)
        if skipped:
            notes.insert(0, f"{skipped} of {len(reference_images)} specimen(s) unusable (quality gate / no ink) and skipped"
                            + ("; single-specimen operating point applied." if len(ref_feats) == 1 else "."))
        notes.insert(0, f"Compared against {len(ref_feats)} enrolled specimen(s); best layout match #{multi.best_shape_index + 1}, best stroke-detail match #{multi.best_keypoint_index + 1}.")
        return single.model_copy(update={
            "similarity_score": score, "raw_score": score, "calibrated_score": score,
            "is_match": band.band == ACCEPT,
            "confidence": round(abs(2.0 * score - 1.0), 4),
            "features": features, "notes": notes,
            "match_logit": round(multi.fused_logit, 6),
            "reference_count": len(ref_feats),
            "decision_band": band.band,
            "quality": quality,
            "explanation": build_explanation(
                _best_signals([explanation_signals(r, que_feat, p) for r, p in zip(ref_feats, multi.per_reference)]),
                band.band, round(multi.fused_logit, 6), len(ref_feats), quality,
                _alignment_used(list(multi.per_reference)),
            ),
        })

    def _inconclusive(self, reason: str, quality: Optional[dict] = None) -> VerificationResult:
        zero = FeatureBreakdown(
            hog_similarity=0.0, hu_moments_similarity=0.0, contour_similarity=0.0,
            skeleton_similarity=0.0, stroke_width_variation_score=0.0, hesitation_score=0.0,
            curvature_variation_score=0.0,
        )
        return VerificationResult(
            similarity_score=0.0, raw_score=0.0, calibrated_score=0.0, is_match=False,
            features=zero, confidence=0.0, notes=[f"INCONCLUSIVE: {reason}."],
            reference_count=0, decision_band="INCONCLUSIVE", quality=quality,
            explanation=build_explanation({}, "INCONCLUSIVE", None, 0, quality),
        )

    def _result_from_features(self, ref_feat: SignatureFeatures, que_feat: SignatureFeatures) -> VerificationResult:
        sim = compare(ref_feat, que_feat, self.config.fusion, self.config.representation)
        legacy = self.legacy_verify(ref_feat.normalized.gray, que_feat.normalized.gray)
        score = float(np.clip(sim.probability, 0.0, 1.0))
        breakdown = legacy.features.model_copy(update={
            "shape_similarity": round(sim.shape, 4),
            "keypoint_similarity": round(sim.keypoint.similarity, 4),
            "keypoint_inliers": sim.keypoint.inliers,
            "keypoint_tentative_matches": sim.keypoint.tentative_matches,
            "fused_logit": round(sim.fused_logit, 4),
        })
        band = classify_band(sim.fused_logit, 1, self.config.decision)
        return VerificationResult(
            similarity_score=round(score, 4),
            raw_score=round(score, 4),
            calibrated_score=round(score, 4),
            is_match=band.band == ACCEPT,
            features=breakdown,
            confidence=round(float(abs(2.0 * score - 1.0)), 4),
            notes=[_band_note(band.band)],
            match_logit=round(sim.fused_logit, 6),
            reference_count=1,
            decision_band=band.band,
        )

    def legacy_verify(
        self,
        reference_image: np.ndarray,
        questioned_image: np.ndarray
    ) -> VerificationResult:
        """Legacy v2.0 hand-weighted fusion (kept for diagnostics / comparison).

        Executes multi-feature deterministic matching:
        - HOG orientation cosine similarity
        - Hu moments invariant distance
        - Contour topology & compactness similarity
        - Skeleton spatial overlap & pen-lift/junction consistency
        - Stroke width consistency
        - Curvature variation / micro-jitter analysis along skeleton
        - Kinematic hesitation & ink-blobbing penalty
        - Non-linear sigmoid confidence calibration
        """
        # Preprocess both images
        ref_gray, ref_bin = self.preprocess_crop(reference_image)
        que_gray, que_bin = self.preprocess_crop(questioned_image)

        # 1. HOG Cosine Similarity
        ref_hog = self.compute_hog_feature(ref_gray)
        que_hog = self.compute_hog_feature(que_gray)
        hog_sim = float(np.clip(np.dot(ref_hog, que_hog), 0.0, 1.0))

        # 2. Hu Moments Invariant Similarity
        ref_hu = self.compute_hu_moments(ref_bin)
        que_hu = self.compute_hu_moments(que_bin)
        # Relative normalized distance
        hu_rel_diff = np.abs(ref_hu - que_hu) / (np.maximum(np.abs(ref_hu), np.abs(que_hu)) + 1e-4)
        hu_dist = float(np.mean(hu_rel_diff))
        hu_sim = float(np.clip(np.exp(-hu_dist), 0.0, 1.0))

        # 3. Contour & Topological Invariant Similarity
        ref_cd = self.compute_contour_descriptors(ref_bin)
        que_cd = self.compute_contour_descriptors(que_bin)

        # Compare aspect ratio, solidity, and loop count
        asp_sim = min(ref_cd["aspect_ratio"], que_cd["aspect_ratio"]) / max(ref_cd["aspect_ratio"], que_cd["aspect_ratio"])
        sol_sim = 1.0 - abs(ref_cd["solidity"] - que_cd["solidity"])
        loop_diff = abs(ref_cd["loop_count"] - que_cd["loop_count"])
        loop_sim = max(0.0, 1.0 - (loop_diff * 0.15))
        contour_sim = float(np.clip(0.4 * asp_sim + 0.3 * sol_sim + 0.3 * loop_sim, 0.0, 1.0))

        # 4. Skeleton Graph & Pen-lift / Junction Similarity
        ref_skel, ref_ep, ref_junc, ref_len = self.compute_skeleton_topology(ref_bin)
        que_skel, que_ep, que_junc, que_len = self.compute_skeleton_topology(que_bin)

        # Dilated Chamfer-like spatial intersection over union
        dil_ref = cv2.dilate(ref_skel.astype(np.uint8), np.ones((5, 5), np.uint8))
        dil_que = cv2.dilate(que_skel.astype(np.uint8), np.ones((5, 5), np.uint8))
        intersection = np.sum((dil_ref > 0) & (dil_que > 0))
        union = np.sum((dil_ref > 0) | (dil_que > 0))
        spatial_skel_iou = float(intersection) / float(union) if union > 0 else 0.0

        # Topology ratios
        ep_ratio = min(ref_ep, que_ep) / (max(ref_ep, que_ep) + 1e-5)
        junc_ratio = min(ref_junc, que_junc) / (max(ref_junc, que_junc) + 1e-5)
        skeleton_sim = float(np.clip(0.6 * spatial_skel_iou + 0.2 * ep_ratio + 0.2 * junc_ratio, 0.0, 1.0))

        # 5. Stroke Width Consistency
        ref_sw_mean, ref_sw_std = self.compute_stroke_widths(ref_bin, ref_skel)
        que_sw_mean, que_sw_std = self.compute_stroke_widths(que_bin, que_skel)
        sw_mean_sim = min(ref_sw_mean, que_sw_mean) / (max(ref_sw_mean, que_sw_mean) + 1e-5)
        sw_std_sim = 1.0 - min(1.0, abs(ref_sw_std - que_sw_std) / (max(ref_sw_std, que_sw_std) + 1e-5))
        stroke_sim = float(np.clip(0.6 * sw_mean_sim + 0.4 * sw_std_sim, 0.0, 1.0))

        # 6. Curvature Variation / Micro-Jitter Analysis
        ref_jitter = self.compute_curvature_micro_jitter(ref_gray, ref_bin, ref_skel)
        que_jitter = self.compute_curvature_micro_jitter(que_gray, que_bin, que_skel)
        curv_sim = min(ref_jitter, que_jitter) / (max(ref_jitter, que_jitter) + 1e-5) if max(ref_jitter, que_jitter) > 0 else 1.0

        # 7. Hesitation and Tremor Index
        ref_hes = self.compute_hesitation_index(ref_gray, ref_bin, ref_skel)
        que_hes = self.compute_hesitation_index(que_gray, que_bin, que_skel)
        hesitation_score = round(que_hes, 4)

        # Hesitation penalty: applied only if questioned has higher tremor/ink-blobs than reference
        hes_diff = max(0.0, que_hes - ref_hes)

        # 8. Rebalanced Feature Fusion
        w = self.config.weights
        raw_score = (
            w.hog_weight * hog_sim +
            w.hu_moments_weight * hu_sim +
            w.contour_weight * contour_sim +
            w.skeleton_weight * skeleton_sim +
            w.stroke_width_weight * stroke_sim
        )
        if hasattr(w, "curvature_weight") and w.curvature_weight > 0:
            raw_score += w.curvature_weight * curv_sim

        penalty = w.hesitation_penalty_weight * hes_diff
        raw_score = float(np.clip(raw_score - penalty, 0.0, 1.0))

        # 9. Non-linear Sigmoid Calibration
        calibrated_score = self.calibrate_score(raw_score)

        # If comparing the identical image against itself, guarantee perfect 1.0
        if np.array_equal(ref_gray, que_gray):
            raw_score = 1.0
            calibrated_score = 1.0
            hog_sim = 1.0
            hu_sim = 1.0
            contour_sim = 1.0
            skeleton_sim = 1.0
            stroke_sim = 1.0
            curv_sim = 1.0

        is_match = calibrated_score >= self.config.calibration.threshold_amber_min
        confidence = float(np.clip(calibrated_score * 0.95 + 0.05, 0.0, 1.0))

        notes = []
        if calibrated_score >= self.config.calibration.threshold_green_stp:
            notes.append("High biometric congruence across all topological features.")
        elif calibrated_score >= self.config.calibration.threshold_amber_min:
            notes.append("Moderate similarity; minor structural or stroke variances observed.")
        else:
            notes.append("Low similarity; divergence in gradient orientation and skeleton topology.")

        if hes_diff > 0.15:
            notes.append(f"Elevated hesitation/tremor detected in questioned signature (+{hes_diff:.2f}).")

        breakdown = FeatureBreakdown(
            hog_similarity=round(hog_sim, 4),
            hu_moments_similarity=round(hu_sim, 4),
            contour_similarity=round(contour_sim, 4),
            skeleton_similarity=round(skeleton_sim, 4),
            stroke_width_variation_score=round(stroke_sim, 4),
            hesitation_score=hesitation_score,
            curvature_variation_score=round(curv_sim, 4),
        )

        return VerificationResult(
            similarity_score=round(calibrated_score, 4),
            raw_score=round(raw_score, 4),
            calibrated_score=round(calibrated_score, 4),
            is_match=is_match,
            features=breakdown,
            confidence=round(confidence, 4),
            notes=notes
        )
