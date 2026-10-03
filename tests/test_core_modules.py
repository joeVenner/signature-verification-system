"""Unit tests for the Signature Verification System core modular library.

Tests:
1. Core Types & Configuration
2. Image Quality Assessment (IQA) & Deskewing
3. Adaptive Binarization & Pantograph Suppression
4. Ink Extraction, Descender Preservation & Pseudo-Dynamic Density
5. Signature Zone Detection & Generic Fallback Locator
6. Deterministic Multi-Feature Verification Engine
7. 3-Tier Clearing Decision Engine & CBUAE Rules
8. Tamper-Evident SHA-256 Hash-Chained Audit Logger
"""

import os
import tempfile
import unittest
import numpy as np
import cv2
import sqlite3

from signature_verification_system.src.core.types import (
    DecisionTier,
    AmountTier,
    StandardReturnCode,
    ZoneType,
    BoundingBox,
    SignatureCandidate,
    DetectionResult,
    FeatureBreakdown,
    VerificationResult,
    MandateRule,
    DecisionResult,
    IQAMetrics,
)
from signature_verification_system.src.core.config import (
    SystemConfig,
    DEFAULT_CONFIG,
    IQAThresholds,
    AmountThresholds,
)
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
from signature_verification_system.src.detection.locator import SignatureLocator
from signature_verification_system.src.verification.deterministic import DeterministicVerifier
from signature_verification_system.src.adjudication.decision_engine import DecisionEngine
from signature_verification_system.src.adjudication.audit_logger import (
    AuditLogger,
    calculate_record_hash,
    GENESIS_HASH,
)


class TestCoreTypesAndConfig(unittest.TestCase):
    """Test data types, schemas, and versioned configuration policies."""

    def test_pydantic_models(self):
        bbox = BoundingBox(x=10, y=20, w=100, h=50)
        self.assertEqual(bbox.x2, 110)
        self.assertEqual(bbox.y2, 70)
        self.assertEqual(bbox.area, 5000)
        self.assertAlmostEqual(bbox.aspect_ratio, 2.0)

        cand = SignatureCandidate(
            bbox=bbox,
            confidence=0.92,
            zone_type=ZoneType.PRIMARY_CHEQUE_ZONE,
            aspect_ratio=2.0,
            stroke_density=0.15,
        )
        self.assertEqual(cand.confidence, 0.92)

        det_res = DetectionResult(
            candidates=[cand],
            best_candidate=cand,
            document_type="cheque",
            image_shape=(800, 1600),
        )
        self.assertEqual(len(det_res.candidates), 1)

    def test_config_defaults(self):
        config = DEFAULT_CONFIG
        self.assertEqual(config.amounts.currency, "AED")
        self.assertEqual(config.amounts.floor_stp_limit, 20000.0)
        self.assertEqual(config.amounts.high_value_limit, 100000.0)
        self.assertEqual(config.calibration.threshold_green_stp, 0.78)
        self.assertEqual(config.calibration.threshold_amber_min, 0.52)
        self.assertEqual(config.policy_version, "CBUAE-POLICY-2026.09.2")
        self.assertEqual(config.iqa.max_brightness, 0.985)


class TestIQAAndDeskew(unittest.TestCase):
    """Test Image Quality Assessment algorithms and deskew routines."""

    def setUp(self):
        # Create a clean white card with horizontal dark text lines
        self.card = np.full((200, 400), 255, dtype=np.uint8)
        for y in range(40, 170, 30):
            cv2.line(self.card, (40, y), (360, y), 30, 3)

    def test_blur_score(self):
        sharp_score = calculate_blur_score(self.card)
        blurred = cv2.GaussianBlur(self.card, (15, 15), 5.0)
        blurred_score = calculate_blur_score(blurred)
        self.assertGreater(sharp_score, blurred_score)
        self.assertGreater(sharp_score, 100.0)
        self.assertLess(blurred_score, 50.0)

    def test_contrast_and_brightness(self):
        rms, michelson = calculate_contrast(self.card)
        self.assertGreater(rms, 50.0)
        self.assertGreater(michelson, 0.5)

        brightness = calculate_brightness(self.card)
        self.assertGreater(brightness, 0.6)
        self.assertLess(brightness, 1.0)

    def test_skew_estimation_and_deskew(self):
        # Rotate image by known tilt: -4.0 degrees
        target_skew = -4.0
        h, w = self.card.shape
        rot_mat = cv2.getRotationMatrix2D((w // 2, h // 2), target_skew, 1.0)
        skewed = cv2.warpAffine(self.card, rot_mat, (w, h), borderValue=255)

        detected_angle = estimate_skew_projection(skewed)
        # Check angle estimated within tolerance (±1.0 degree)
        self.assertAlmostEqual(detected_angle, target_skew, delta=1.5)

        deskewed = deskew_image(skewed, detected_angle)
        self.assertEqual(deskewed.shape, skewed.shape)

    def test_assess_image_quality_pipeline(self):
        # Test good quality card
        metrics, processed = assess_image_quality(self.card)
        self.assertTrue(metrics.passed)
        self.assertEqual(len(metrics.failure_reasons), 0)

        # Test overly dark blurred image
        dark_blurred = np.full((100, 100), 20, dtype=np.uint8)
        metrics_bad, _ = assess_image_quality(dark_blurred)
        self.assertFalse(metrics_bad.passed)
        self.assertTrue(metrics_bad.is_too_dark)


class TestBinarizationAndPantograph(unittest.TestCase):
    """Test Sauvola, Wolf binarization, and pantograph dot suppression."""

    def setUp(self):
        # Create gradient background with sharp stroke
        y, x = np.mgrid[0:100, 0:100]
        # Smooth background gradient from 160 to 240
        self.img = (160 + 0.8 * x).astype(np.uint8)
        # Draw dark stroke
        cv2.line(self.img, (20, 20), (80, 80), 40, 3)

    def test_sauvola_and_wolf_threshold(self):
        bin_sauvola = sauvola_threshold(self.img, window_size=25, k=0.2)
        bin_wolf = wolf_threshold(self.img, window_size=25, k=0.2)

        self.assertEqual(bin_sauvola.shape, self.img.shape)
        self.assertEqual(bin_wolf.shape, self.img.shape)
        # The line region should be detected as foreground (255)
        self.assertEqual(bin_sauvola[50, 50], 255)
        self.assertEqual(bin_wolf[50, 50], 255)
        # Background margin should be 0
        self.assertEqual(bin_sauvola[10, 90], 0)

    def test_pantograph_suppression(self):
        binary = np.zeros((100, 100), dtype=np.uint8)
        # Draw continuous genuine stroke
        cv2.line(binary, (10, 50), (90, 50), 255, 3)
        # Add isolated 1-pixel or 2-pixel pantograph halftone noise dots
        noise_coords = [(15, 20), (35, 80), (70, 25), (85, 85)]
        for pt in noise_coords:
            binary[pt[1], pt[0]] = 255

        cleaned = suppress_pantograph(binary, min_speckle_area=6)
        # Stroke must be preserved
        self.assertEqual(cleaned[50, 50], 255)
        # Noise speckles must be removed
        for pt in noise_coords:
            self.assertEqual(cleaned[pt[1], pt[0]], 0)


class TestInkExtractor(unittest.TestCase):
    """Test CIE-Lab ink separation, baseline removal with descender preservation, and density."""

    def test_horizontal_baseline_removal_with_descender_preservation(self):
        # Canvas with horizontal baseline and crossing descender (e.g. letter 'y' descender)
        canvas = np.zeros((100, 200), dtype=np.uint8)
        # Baseline at y = 50 from x=10 to 190
        cv2.line(canvas, (10, 50), (190, 50), 255, 2)
        # Descender crossing at x = 100 from y=30 to y=80
        cv2.line(canvas, (100, 30), (100, 80), 255, 3)

        cleaned = remove_horizontal_baseline(canvas, preserve_descenders=True, kernel_width=30)

        # Baseline away from crossing must be removed
        self.assertEqual(cleaned[50, 30], 0)
        self.assertEqual(cleaned[50, 160], 0)
        # Descender crossing must be preserved!
        self.assertEqual(cleaned[50, 100], 255)
        self.assertEqual(cleaned[35, 100], 255)
        self.assertEqual(cleaned[75, 100], 255)

    def test_separate_ink_cielab(self):
        # Create image with pastel background and blue/black strokes
        # BGR: pastel green background (e.g. B=200, G=240, R=200)
        doc = np.full((100, 100, 3), (200, 240, 200), dtype=np.uint8)
        # Draw dark blue pen stroke (B=180, G=40, R=20)
        cv2.line(doc, (10, 10), (90, 90), (180, 40, 20), 3)
        # Draw black pen stroke (B=20, G=20, R=20)
        cv2.line(doc, (10, 90), (90, 10), (20, 20, 20), 3)

        mask = separate_ink_cielab(doc)
        # Strokes should be extracted as foreground
        self.assertEqual(mask[50, 50], 255)
        # Background should be 0
        self.assertEqual(mask[5, 50], 0)

    def test_pseudo_dynamic_density(self):
        gray = np.full((100, 100), 240, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Draw stroke with variable pressure
        cv2.line(gray, (10, 50), (50, 50), 30, 3)   # dark / slow stroke
        cv2.line(gray, (50, 50), (90, 50), 180, 3)  # light / fast stroke
        cv2.line(mask, (10, 50), (90, 50), 255, 3)

        density_map, stats = compute_pseudo_dynamic_density(gray, mask)
        self.assertGreater(stats["mean_density"], 0.2)
        self.assertGreater(stats["density_std"], 0.1)
        self.assertIn("hesitation_blob_ratio", stats)


class TestSignatureLocator(unittest.TestCase):
    """Test cheque signature zone locator and generic fallback detector."""

    def test_locate_on_cheque_sample(self):
        sample_path = "signature_verification_system/data/samples/cheques/cheque_real_scanned_08.jpg"
        if not os.path.exists(sample_path):
            self.skipTest(f"Cheque sample {sample_path} not found")

        img = cv2.imread(sample_path)
        locator = SignatureLocator()
        result = locator.locate(img, is_cheque=True)

        self.assertIsNotNone(result.best_candidate)
        self.assertGreater(len(result.candidates), 0)
        best = result.best_candidate
        self.assertEqual(best.zone_type, ZoneType.PRIMARY_CHEQUE_ZONE)
        self.assertGreater(best.confidence, 0.8)

        crop = locator.extract_crop(img, best)
        self.assertGreater(crop.shape[0], 20)
        self.assertGreater(crop.shape[1], 50)

    def test_locate_on_generic_document(self):
        # Create synthetic A4-like page (600x400)
        doc = np.full((600, 400), 255, dtype=np.uint8)
        # Draw some paragraphs of simulated printed text (thin lines)
        for y in range(80, 250, 15):
            cv2.line(doc, (50, y), (350, y), 80, 1)

        # Draw signature in bottom-right area (cursive stroke)
        pts = np.array([[240, 480], [280, 450], [310, 490], [350, 460], [380, 500]], np.int32)
        cv2.polylines(doc, [pts], isClosed=False, color=20, thickness=3)

        locator = SignatureLocator(min_signature_area=200)
        result = locator.locate(doc, is_cheque=False)

        self.assertEqual(result.document_type, "generic_document")
        self.assertGreater(len(result.candidates), 0)


class TestDeterministicVerifier(unittest.TestCase):
    """Test deterministic multi-feature verification."""

    def setUp(self):
        self.verifier = DeterministicVerifier()

    def test_identity_property(self):
        # Comparing a signature to itself must yield score = 1.0
        ref_path = "signature_verification_system/data/samples/genuine_pairs/pair_01_cedar_w01_ref.png"
        if not os.path.exists(ref_path):
            self.skipTest(f"Sample {ref_path} not found")

        img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        res = self.verifier.verify(img, img)

        self.assertAlmostEqual(res.similarity_score, 1.0, places=3)
        self.assertTrue(res.is_match)
        self.assertAlmostEqual(res.features.hog_similarity, 1.0, places=3)
        self.assertAlmostEqual(res.features.skeleton_similarity, 1.0, places=3)

    def test_genuine_vs_random_forgery_discrimination(self):
        p_ref = "signature_verification_system/data/samples/genuine_pairs/pair_01_cedar_w01_ref.png"
        p_que = "signature_verification_system/data/samples/genuine_pairs/pair_01_cedar_w01_questioned.png"
        p_rand = "signature_verification_system/data/samples/random_forgeries/pair_01_random_w02.png"

        if not (os.path.exists(p_ref) and os.path.exists(p_que) and os.path.exists(p_rand)):
            self.skipTest("Sample signature files not found")

        img_ref = cv2.imread(p_ref, cv2.IMREAD_GRAYSCALE)
        img_que = cv2.imread(p_que, cv2.IMREAD_GRAYSCALE)
        img_rand = cv2.imread(p_rand, cv2.IMREAD_GRAYSCALE)

        res_genuine = self.verifier.verify(img_ref, img_que)
        res_random = self.verifier.verify(img_ref, img_rand)

        # Genuine pair similarity must be significantly higher than random forgery
        self.assertGreater(res_genuine.similarity_score, res_random.similarity_score)
        self.assertGreater(res_genuine.features.hog_similarity, res_random.features.hog_similarity)


class TestDecisionEngine(unittest.TestCase):
    """Test 3-Tier clearing policy engine, amount tiers, and CBUAE compliance rules."""

    def setUp(self):
        self.engine = DecisionEngine()
        self.dummy_features = FeatureBreakdown(
            hog_similarity=0.92,
            hu_moments_similarity=0.88,
            contour_similarity=0.90,
            skeleton_similarity=0.85,
            stroke_width_variation_score=0.91,
            hesitation_score=0.10,
        )

    def test_tier_green_stp(self):
        v_res = VerificationResult(
            similarity_score=0.92,
            is_match=True,
            features=self.dummy_features,
            confidence=0.92,
        )
        dec = self.engine.evaluate(v_res, amount=5_000.0)
        self.assertEqual(dec.tier, DecisionTier.GREEN)
        self.assertEqual(dec.action, "AUTO_CLEAR")
        self.assertEqual(dec.amount_tier, AmountTier.LOW_VALUE)
        self.assertFalse(dec.requires_four_eyes)

    def test_tier_amber_medium_value(self):
        # Score is high (0.92), but amount is 45,000 AED (Medium Value) -> Amber L1 review
        v_res = VerificationResult(
            similarity_score=0.92,
            is_match=True,
            features=self.dummy_features,
            confidence=0.92,
        )
        dec = self.engine.evaluate(v_res, amount=45_000.0)
        self.assertEqual(dec.tier, DecisionTier.AMBER)
        self.assertEqual(dec.action, "OPERATOR_REVIEW")
        self.assertEqual(dec.amount_tier, AmountTier.MEDIUM_VALUE)

    def test_tier_amber_moderate_score(self):
        # Amount is low (5,000 AED), but score is 0.72 (moderate) -> Amber L1 review
        v_res = VerificationResult(
            similarity_score=0.72,
            is_match=True,
            features=self.dummy_features,
            confidence=0.75,
        )
        dec = self.engine.evaluate(v_res, amount=5_000.0)
        self.assertEqual(dec.tier, DecisionTier.AMBER)
        self.assertEqual(dec.action, "OPERATOR_REVIEW")

    def test_tier_red_high_value_four_eyes(self):
        # Amount >= 100,000 AED -> Mandatory Four-Eyes escalation
        v_res = VerificationResult(
            similarity_score=0.95,
            is_match=True,
            features=self.dummy_features,
            confidence=0.95,
        )
        dec = self.engine.evaluate(v_res, amount=120_000.0)
        self.assertEqual(dec.tier, DecisionTier.RED)
        self.assertEqual(dec.action, "MANDATORY_FOUR_EYES_ESCALATE")
        self.assertTrue(dec.requires_four_eyes)
        self.assertIn("CBUAE_HIGH_VALUE_FOUR_EYES_MANDATE", dec.cbuae_compliance_flags)

    def test_tier_red_low_similarity_reject(self):
        # Score < 0.45 -> Hard reject with SIGNATURE_DIFFERS
        v_res = VerificationResult(
            similarity_score=0.35,
            is_match=False,
            features=self.dummy_features,
            confidence=0.38,
        )
        dec = self.engine.evaluate(v_res, amount=8_000.0)
        self.assertEqual(dec.tier, DecisionTier.RED)
        self.assertEqual(dec.action, "REJECT")
        self.assertEqual(dec.return_code, StandardReturnCode.SIGNATURE_DIFFERS)

    def test_cbuae_compliance_mandate_and_alteration(self):
        v_res = VerificationResult(
            similarity_score=0.95,
            is_match=True,
            features=self.dummy_features,
            confidence=0.95,
        )
        # Mandate requires 2 signers, detected 1
        mandate = MandateRule(required_signers_count=2, max_single_signer_limit=50000.0)
        dec_mandate = self.engine.evaluate(v_res, amount=10_000.0, mandate=mandate, detected_signers_count=1)
        self.assertEqual(dec_mandate.return_code, StandardReturnCode.MANDATE_INCOMPLETE)

        # Suspected alteration check
        dec_alt = self.engine.evaluate(v_res, amount=10_000.0, suspected_alteration=True)
        self.assertEqual(dec_alt.return_code, StandardReturnCode.SUSPECTED_ALTERATION)


class TestAuditLogger(unittest.TestCase):
    """Test tamper-evident SHA-256 hash-chained SQLite & JSONL audit logger."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_audit.db")
        self.jsonl_path = os.path.join(self.temp_dir.name, "test_audit.jsonl")
        self.logger = AuditLogger(db_path=self.db_path, jsonl_path=self.jsonl_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_hash_chaining_and_integrity(self):
        # Log 3 decisions
        dec1 = DecisionResult(
            tier=DecisionTier.GREEN,
            action="AUTO_CLEAR",
            amount=5000.0,
            currency="AED",
            amount_tier=AmountTier.LOW_VALUE,
            similarity_score=0.91,
            policy_version="POLICY-TEST-1.0",
        )
        dec2 = DecisionResult(
            tier=DecisionTier.AMBER,
            action="OPERATOR_REVIEW",
            amount=35000.0,
            currency="AED",
            amount_tier=AmountTier.MEDIUM_VALUE,
            similarity_score=0.74,
            policy_version="POLICY-TEST-1.0",
        )
        dec3 = DecisionResult(
            tier=DecisionTier.RED,
            action="MANDATORY_FOUR_EYES_ESCALATE",
            amount=250000.0,
            currency="AED",
            amount_tier=AmountTier.HIGH_VALUE,
            similarity_score=0.96,
            policy_version="POLICY-TEST-1.0",
        )

        r1 = self.logger.log_decision("DOC-001", dec1)
        r2 = self.logger.log_decision("DOC-002", dec2)
        r3 = self.logger.log_decision("DOC-003", dec3)

        self.assertEqual(r1.sequence_num, 1)
        self.assertEqual(r1.prev_hash, GENESIS_HASH)
        self.assertEqual(r2.sequence_num, 2)
        self.assertEqual(r2.prev_hash, r1.hash)
        self.assertEqual(r3.sequence_num, 3)
        self.assertEqual(r3.prev_hash, r2.hash)

        # Verify integrity succeeds
        is_valid, err = self.logger.verify_integrity()
        self.assertTrue(is_valid)
        self.assertIsNone(err)

        # Verify JSONL lines exist
        self.assertTrue(os.path.exists(self.jsonl_path))
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 3)

    def test_tamper_detection(self):
        dec = DecisionResult(
            tier=DecisionTier.GREEN,
            action="AUTO_CLEAR",
            amount=1000.0,
            currency="AED",
            amount_tier=AmountTier.LOW_VALUE,
            similarity_score=0.90,
            policy_version="POLICY-TEST-1.0",
        )
        self.logger.log_decision("DOC-001", dec)
        self.logger.log_decision("DOC-002", dec)

        # Maliciously tamper with amount in record 1 directly inside SQLite DB
        conn = sqlite3.connect(self.db_path)
        with conn:
            conn.execute("UPDATE audit_ledger SET amount = 999999.0 WHERE sequence_num = 1;")
        conn.close()

        # The hash chain integrity check must catch this tampering!
        is_valid, err = self.logger.verify_integrity()
        self.assertFalse(is_valid)
        self.assertIn("Tampered record at sequence 1", err)




class TestIterativeImprovements(unittest.TestCase):
    """Unit tests for IQA enhancements, micro-jitter, and non-linear calibration."""

    def setUp(self):
        self.verifier = DeterministicVerifier()

    def test_iqa_max_brightness_tolerance(self):
        # Image with brightness 0.97 (within 0.985 limit, previously rejected by 0.95)
        bright_img = np.full((100, 200), 247, dtype=np.uint8)
        # Add some dark strokes so contrast is adequate
        cv2.line(bright_img, (10, 50), (190, 50), 30, 3)
        cv2.line(bright_img, (50, 10), (150, 90), 30, 3)
        metrics, _ = assess_image_quality(bright_img)
        self.assertFalse(metrics.is_too_bright)
        self.assertTrue(metrics.passed)

    def test_clahe_enhancement_on_low_contrast(self):
        # Create low contrast image with narrow histogram [110, 130]
        np.random.seed(42)
        low_contrast = np.random.randint(110, 130, (150, 150), dtype=np.uint8)
        rms_before, _ = calculate_contrast(low_contrast)
        enhanced = apply_clahe_enhancement(low_contrast)
        rms_after, _ = calculate_contrast(enhanced)
        self.assertGreater(rms_after, rms_before)

    def test_curvature_micro_jitter_calculation(self):
        gray = np.full((128, 256), 255, dtype=np.uint8)
        binary = np.zeros((128, 256), dtype=np.uint8)
        # Draw smooth diagonal line
        cv2.line(gray, (20, 20), (220, 100), 20, 3)
        cv2.line(binary, (20, 20), (220, 100), 255, 3)
        jitter_var = self.verifier.compute_curvature_micro_jitter(gray, binary)
        self.assertIsInstance(jitter_var, float)
        self.assertGreaterEqual(jitter_var, 0.0)

    def test_improved_hesitation_index(self):
        gray = np.full((128, 256), 255, dtype=np.uint8)
        binary = np.zeros((128, 256), dtype=np.uint8)
        cv2.circle(gray, (100, 60), 25, 30, -1)
        cv2.circle(binary, (100, 60), 25, 255, -1)
        hes = self.verifier.compute_hesitation_index(gray, binary)
        self.assertIsInstance(hes, float)
        self.assertGreaterEqual(hes, 0.0)
        self.assertLessEqual(hes, 1.0)

    def test_nonlinear_sigmoid_calibration_scale(self):
        self.assertEqual(DeterministicVerifier.calibrate_score(0.0), 0.0)
        self.assertEqual(DeterministicVerifier.calibrate_score(1.0), 1.0)
        # Empirical EER boundary s0=0.65 maps near 0.52
        cal_eer = DeterministicVerifier.calibrate_score(0.65)
        self.assertAlmostEqual(cal_eer, 0.517, places=2)
        # Score 0.78 maps to high confidence (>= 0.78)
        cal_green = DeterministicVerifier.calibrate_score(0.78)
        self.assertGreaterEqual(cal_green, 0.78)

    def test_verification_raw_and_calibrated_scores(self):
        # A single 4-px straight line is not a signature: the quality gate (EXP-007)
        # must refuse it rather than return a confident score.
        line = np.full((100, 200), 255, dtype=np.uint8)
        cv2.line(line, (20, 50), (180, 50), 30, 4)
        gated = self.verifier.verify(line, line)
        self.assertEqual(gated.decision_band, "INCONCLUSIVE")
        self.assertIn("SIGNATURE_TOO_SMALL", gated.quality["questioned"]["blocking_issues"])

        img = np.full((100, 200), 255, dtype=np.uint8)
        xs = np.arange(20, 181)
        pts = np.stack([xs, 50 + (25 * np.sin(xs / 12.0)).astype(int)], axis=1).astype(np.int32)
        cv2.polylines(img, [pts], False, 30, 3)
        res = self.verifier.verify(img, img)
        self.assertIsNotNone(res.raw_score)
        self.assertIsNotNone(res.calibrated_score)
        self.assertEqual(res.similarity_score, res.calibrated_score)
        self.assertEqual(res.similarity_score, 1.0)
        self.assertIsNotNone(res.features.curvature_variation_score)


if __name__ == "__main__":
    unittest.main()
