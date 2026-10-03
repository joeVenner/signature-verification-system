"""Deterministic Verification, Extreme Edge Cases, and Audit Log Tamper Resistance Tests.

Comprehensive validation suite for:
1. 10-run determinism test: exact bitwise float equality (0.00000000 variance) on all scores and features.
2. Edge Case 1: Blank / all-white images (handled gracefully without div-by-zero, returns rejection / low score or IQA failure).
3. Edge Case 2: Solid black images (handled gracefully without div-by-zero, returns rejection / low score or IQA failure).
4. Edge Case 3: Micro-resolution image (e.g. 16x16, 8x8 pixels).
5. Edge Case 4: Severely noisy / salt-and-pepper images.
6. Edge Case 5: High-value mandate tier escalation (150,000 AED cheque triggering Red Four-Eyes review even with high score).
7. Edge Case 6: Audit log tamper resistance (insert 5 blocks, verify chain, corrupt block #2, assert validation fails with exact block # identified).
"""

from __future__ import annotations

import os
import sys
import tempfile
import sqlite3
import unittest
import numpy as np
import cv2

# Ensure project and workspace paths are resolvable
_curr_dir = os.path.dirname(os.path.abspath(__file__))
_proj_dir = os.path.abspath(os.path.join(_curr_dir, ".."))
_workspace_dir = os.path.abspath(os.path.join(_proj_dir, ".."))
for p in [_proj_dir, _workspace_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from signature_verification_system.src.core.types import (
    DecisionTier,
    AmountTier,
    StandardReturnCode,
    FeatureBreakdown,
    VerificationResult,
    DecisionResult,
    IQAMetrics,
    MandateRule,
)
from signature_verification_system.src.core.config import (
    SystemConfig,
    DEFAULT_CONFIG,
    IQAThresholds,
    AmountThresholds,
)
from signature_verification_system.src.preprocessing.iqa import (
    assess_image_quality,
    calculate_blur_score,
    calculate_contrast,
    calculate_brightness,
)
from signature_verification_system.src.verification.deterministic import DeterministicVerifier
from signature_verification_system.src.adjudication.decision_engine import DecisionEngine
from signature_verification_system.src.adjudication.audit_logger import (
    AuditLogger,
    GENESIS_HASH,
    calculate_record_hash,
)


def create_synthetic_signature_image(seed: int = 42, width: int = 300, height: int = 150) -> np.ndarray:
    """Generate a clean synthetic signature stroke on white paper with high determinism."""
    canvas = np.full((height, width), 255, dtype=np.uint8)
    rng = np.random.RandomState(seed)

    # Generate a smooth signature-like spline curve
    points = []
    x = 30
    y = height // 2
    points.append((x, y))
    for i in range(12):
        x += rng.randint(15, 25)
        y = int(np.clip(y + rng.randint(-35, 35), 20, height - 20))
        points.append((x, y))

    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [pts], isClosed=False, color=20, thickness=2, lineType=cv2.LINE_AA)

    # Add a cross flourish
    cv2.line(canvas, (40, height // 2 + 10), (width - 40, height // 2 - 10), 20, 2, cv2.LINE_AA)
    # Add a loop
    cv2.circle(canvas, (width // 2, height // 2), 15, 20, 2, cv2.LINE_AA)
    return canvas


class TestDeterminismAndReproducibility(unittest.TestCase):
    """Test bitwise exact reproducibility and 0.00000000 variance across multiple runs."""

    def setUp(self):
        self.verifier = DeterministicVerifier()
        self.engine = DecisionEngine()

    def test_ten_run_determinism_on_real_samples(self):
        """10-run determinism test on dataset samples: exact float equality across all dimensions."""
        ref_path = "signature_verification_system/data/samples/genuine_pairs/pair_01_cedar_w01_ref.png"
        que_path = "signature_verification_system/data/samples/genuine_pairs/pair_01_cedar_w01_questioned.png"

        if os.path.exists(ref_path) and os.path.exists(que_path):
            img_ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            img_que = cv2.imread(que_path, cv2.IMREAD_GRAYSCALE)
        else:
            # Fallback to deterministic synthetic pair
            img_ref = create_synthetic_signature_image(seed=101)
            img_que = create_synthetic_signature_image(seed=102)

        runs = []
        for run_idx in range(10):
            res = self.verifier.verify(img_ref, img_que)
            score_vector = (
                res.similarity_score,
                res.raw_score,
                res.calibrated_score,
                res.confidence,
                res.features.hog_similarity,
                res.features.hu_moments_similarity,
                res.features.contour_similarity,
                res.features.skeleton_similarity,
                res.features.stroke_width_variation_score,
                res.features.hesitation_score,
                res.features.curvature_variation_score,
            )
            runs.append(score_vector)

        # 1. Exact bitwise float equality check against baseline run
        baseline = runs[0]
        for run_idx, score_vector in enumerate(runs[1:], start=2):
            self.assertEqual(
                score_vector,
                baseline,
                f"Determinism violation on run {run_idx}: {score_vector} != {baseline}",
            )
            # Verify individual float components are bitwise identical
            for dim_idx, (v_test, v_base) in enumerate(zip(score_vector, baseline)):
                self.assertEqual(v_test, v_base, f"Dim {dim_idx} differed on run {run_idx}")

        # 2. Verify variance is 0.00000000 (below machine precision threshold)
        arr = np.array(runs, dtype=np.float64)
        variances = np.var(arr, axis=0)
        for dim_idx, var_val in enumerate(variances):
            self.assertAlmostEqual(
                float(var_val),
                0.0,
                delta=1e-15,
                msg=f"Feature dimension {dim_idx} has non-zero variance: {var_val:.16e}",
            )

    def test_ten_run_determinism_on_synthetic_pairs(self):
        """10-run determinism test on procedural inputs: verify 0.00000000 variance."""
        img1 = create_synthetic_signature_image(seed=999)
        img2 = create_synthetic_signature_image(seed=888)

        runs = []
        for _ in range(10):
            res = self.verifier.verify(img1, img2)
            runs.append([
                res.similarity_score,
                res.raw_score,
                res.calibrated_score,
                res.confidence,
                res.features.hog_similarity,
                res.features.hu_moments_similarity,
                res.features.contour_similarity,
                res.features.skeleton_similarity,
                res.features.stroke_width_variation_score,
                res.features.hesitation_score,
                res.features.curvature_variation_score,
            ])

        baseline = runs[0]
        for idx, r in enumerate(runs[1:], start=2):
            self.assertEqual(r, baseline, f"Synthetic determinism mismatch at run {idx}")

        runs_arr = np.array(runs, dtype=np.float64)
        variances = np.var(runs_arr, axis=0)
        np.testing.assert_allclose(
            variances,
            0.0,
            atol=1e-15,
            err_msg="Determinism failed: feature scores exhibited non-zero variance over 10 runs",
        )

    def test_ten_run_iqa_determinism(self):
        """10-run determinism test on Image Quality Assessment."""
        sample = create_synthetic_signature_image(seed=303)
        metrics_runs = []
        for _ in range(10):
            metrics, _ = assess_image_quality(sample, deskew=True)
            metrics_runs.append((
                metrics.skew_angle,
                metrics.blur_score,
                metrics.contrast_score,
                metrics.brightness_score,
                metrics.passed,
            ))

        baseline = metrics_runs[0]
        for idx, m in enumerate(metrics_runs[1:], start=2):
            self.assertEqual(m, baseline, f"IQA run {idx} differed from baseline")

    def test_ten_run_decision_engine_determinism(self):
        """10-run determinism test on clearing decision evaluation."""
        features = FeatureBreakdown(
            hog_similarity=0.88,
            hu_moments_similarity=0.85,
            contour_similarity=0.89,
            skeleton_similarity=0.86,
            stroke_width_variation_score=0.90,
            hesitation_score=0.08,
            curvature_variation_score=0.87,
        )
        ver_res = VerificationResult(
            similarity_score=0.88,
            raw_score=0.88,
            calibrated_score=0.88,
            is_match=True,
            features=features,
            confidence=0.89,
        )

        decisions = []
        for _ in range(10):
            dec = self.engine.evaluate(ver_res, amount=25_000.0, currency="AED")
            decisions.append((
                dec.tier,
                dec.action,
                dec.amount_tier,
                dec.similarity_score,
                dec.requires_four_eyes,
                tuple(dec.reasons),
            ))

        baseline_dec = decisions[0]
        for idx, d in enumerate(decisions[1:], start=2):
            self.assertEqual(d, baseline_dec, f"DecisionEngine run {idx} differed from baseline")


class TestExtremeEdgeCases(unittest.TestCase):
    """Test system stability, graceful failure, and strict rejection on pathological inputs."""

    def setUp(self):
        self.verifier = DeterministicVerifier()
        self.engine = DecisionEngine()
        self.genuine_sample = create_synthetic_signature_image(seed=777)

    def test_edge_case_1_blank_all_white_image(self):
        """Edge case 1: Blank / all-white images handled gracefully without div-by-zero.
        
        Must return rejection / low score or IQA failure.
        """
        white_img = np.full((128, 256), 255, dtype=np.uint8)

        # 1. Image Quality Assessment must fail on blank image
        metrics, _ = assess_image_quality(white_img)
        self.assertFalse(metrics.passed, "Blank white image must fail IQA")
        # Check reasons for overexposure/washed out or blur/contrast
        reasons_str = " ".join(metrics.failure_reasons).lower()
        self.assertTrue(
            "washed out" in reasons_str or "contrast" in reasons_str or "blur" in reasons_str or "overexposed" in reasons_str,
            f"Expected contrast/overexposure rejection, got: {metrics.failure_reasons}",
        )

        # 2. DeterministicVerifier must not crash with ZeroDivisionError
        try:
            res = self.verifier.verify(self.genuine_sample, white_img)
        except ZeroDivisionError:
            self.fail("Verifier raised ZeroDivisionError on all-white image")

        # Must reject or yield extremely low score
        self.assertFalse(res.is_match)
        self.assertLess(res.similarity_score, 0.20, f"Expected low score for blank image, got {res.similarity_score}")

        # 3. Both images blank must also execute safely without div-by-zero
        try:
            res_both = self.verifier.verify(white_img, white_img)
            self.assertIsNotNone(res_both)
        except ZeroDivisionError:
            self.fail("Verifier raised ZeroDivisionError when comparing blank with blank")

    def test_edge_case_2_solid_black_image(self):
        """Edge case 2: Solid black images handled gracefully without div-by-zero."""
        black_img = np.zeros((128, 256), dtype=np.uint8)

        # 1. IQA must fail on black image due to darkness/underexposure
        metrics, _ = assess_image_quality(black_img)
        self.assertFalse(metrics.passed, "Solid black image must fail IQA")
        self.assertTrue(metrics.is_too_dark, "Solid black image must be flagged as too dark")

        # 2. DeterministicVerifier must handle safely without crashing
        try:
            res = self.verifier.verify(self.genuine_sample, black_img)
        except Exception as e:
            self.fail(f"Verifier crashed on solid black image: {type(e).__name__}: {e}")

        # Must reject
        self.assertFalse(res.is_match)
        self.assertLess(res.similarity_score, 0.20, f"Expected low score for black image, got {res.similarity_score}")

        # 3. Black vs black image
        try:
            res_both = self.verifier.verify(black_img, black_img)
            self.assertIsNotNone(res_both)
        except Exception as e:
            self.fail(f"Verifier crashed comparing black to black: {e}")

    def test_edge_case_3_micro_resolution_image(self):
        """Edge case 3: Micro-resolution image (e.g. 16x16, 8x8 pixels) handled gracefully."""
        # Micro 16x16 image
        micro_16 = np.full((16, 16), 255, dtype=np.uint8)
        micro_16[4:12, 4:12] = 0  # tiny dark patch

        # 1. IQA runs without crashing
        try:
            metrics_16, _ = assess_image_quality(micro_16)
            self.assertIsNotNone(metrics_16)
        except Exception as e:
            self.fail(f"IQA crashed on 16x16 image: {e}")

        # 2. Verifier runs without crash
        try:
            res_16 = self.verifier.verify(self.genuine_sample, micro_16)
            self.assertFalse(res_16.is_match)
            self.assertLess(res_16.similarity_score, 0.30)
        except Exception as e:
            self.fail(f"Verifier crashed on 16x16 micro-image: {e}")

        # Micro 8x8 image
        micro_8 = np.full((8, 8), 255, dtype=np.uint8)
        micro_8[2:6, 2:6] = 0
        try:
            res_8 = self.verifier.verify(self.genuine_sample, micro_8)
            self.assertFalse(res_8.is_match)
        except Exception as e:
            self.fail(f"Verifier crashed on 8x8 micro-image: {e}")

        # Micro vs micro
        try:
            res_mm = self.verifier.verify(micro_16, micro_8)
            self.assertIsNotNone(res_mm)
        except Exception as e:
            self.fail(f"Verifier crashed on micro vs micro: {e}")

    def test_edge_case_4_severely_noisy_salt_and_pepper_images(self):
        """Edge case 4: Severely noisy / salt-and-pepper images handled gracefully."""
        # Create severe salt-and-pepper noise canvas
        noisy_img = np.full((128, 256), 255, dtype=np.uint8)
        rng = np.random.RandomState(404)
        noise_mask_black = rng.rand(128, 256) < 0.25  # 25% pepper
        noise_mask_white = rng.rand(128, 256) < 0.25  # 25% salt
        noisy_img[noise_mask_black] = 0
        noisy_img[noise_mask_white] = 255

        # 1. IQA runs safely
        try:
            metrics, _ = assess_image_quality(noisy_img)
            self.assertIsNotNone(metrics)
        except Exception as e:
            self.fail(f"IQA crashed on salt-and-pepper image: {e}")

        # 2. DeterministicVerifier runs safely and rejects matching against clean signature
        try:
            res = self.verifier.verify(self.genuine_sample, noisy_img)
            self.assertFalse(res.is_match)
            self.assertLess(res.similarity_score, 0.40)
        except Exception as e:
            self.fail(f"Verifier crashed on salt-and-pepper image: {e}")

    def test_edge_case_5_high_value_mandate_escalation(self):
        """Edge case 5: High-value mandate tier escalation (150,000 AED cheque).
        
        Even with a near-perfect similarity score (0.98), cheque must trigger
        Red Four-Eyes review per CBUAE and bank mandate policies.
        """
        high_match_features = FeatureBreakdown(
            hog_similarity=0.98,
            hu_moments_similarity=0.97,
            contour_similarity=0.98,
            skeleton_similarity=0.96,
            stroke_width_variation_score=0.97,
            hesitation_score=0.01,
            curvature_variation_score=0.96,
        )
        high_match_ver = VerificationResult(
            similarity_score=0.98,
            raw_score=0.98,
            calibrated_score=0.98,
            is_match=True,
            features=high_match_features,
            confidence=0.98,
        )

        amount = 150_000.0  # AED 150,000 >= 100,000 threshold

        dec = self.engine.evaluate(high_match_ver, amount=amount, currency="AED")

        # Strict assertion of Red tier escalation and four-eyes review requirement
        self.assertEqual(
            dec.tier,
            DecisionTier.RED,
            f"Expected RED tier for 150,000 AED cheque, got {dec.tier}",
        )
        self.assertEqual(
            dec.action,
            "MANDATORY_FOUR_EYES_ESCALATE",
            f"Expected MANDATORY_FOUR_EYES_ESCALATE, got {dec.action}",
        )
        self.assertEqual(
            dec.amount_tier,
            AmountTier.HIGH_VALUE,
            f"Expected HIGH_VALUE tier, got {dec.amount_tier}",
        )
        self.assertTrue(
            dec.requires_four_eyes,
            "High-value 150,000 AED cheque MUST require four-eyes sign-off",
        )
        # Verify explanatory reason is logged
        reasons_joined = " ".join(dec.reasons)
        self.assertIn("150,000.00", reasons_joined)
        self.assertTrue(
            "dual sign-off" in reasons_joined or "mandatory" in reasons_joined,
            f"Expected dual sign-off mention, got: {dec.reasons}",
        )

        # Contrast with Low-Value same score cheque (e.g. 5,000 AED) which should AUTO_CLEAR in GREEN
        dec_low = self.engine.evaluate(high_match_ver, amount=5_000.0, currency="AED")
        self.assertEqual(dec_low.tier, DecisionTier.GREEN)
        self.assertEqual(dec_low.action, "AUTO_CLEAR")
        self.assertFalse(dec_low.requires_four_eyes)


class TestAuditLogTamperResistance(unittest.TestCase):
    """Test cryptographic SHA-256 hash chaining and tamper identification."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "audit_tamper_test.db")
        self.jsonl_path = os.path.join(self.temp_dir.name, "audit_tamper_test.jsonl")
        self.logger = AuditLogger(db_path=self.db_path, jsonl_path=self.jsonl_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_edge_case_6_insert_5_blocks_corrupt_block_2_assert_fails_at_block_2(self):
        """Edge case 6: Audit log tamper resistance:
        
        Insert 5 blocks, verify chain, corrupt block #2, assert validation
        fails with exact block # identified.
        """
        # 1. Insert 5 distinct blocks into the ledger
        records = []
        for i in range(1, 6):
            doc_id = f"CHQ-UAE-2026-{1000 + i}"
            amount = 10_000.0 * i
            tier = DecisionTier.GREEN if i < 3 else (DecisionTier.AMBER if i < 5 else DecisionTier.RED)
            action = "AUTO_CLEAR" if tier == DecisionTier.GREEN else ("OPERATOR_REVIEW" if tier == DecisionTier.AMBER else "MANDATORY_FOUR_EYES_ESCALATE")
            score = 0.95 - (i * 0.05)

            dec = DecisionResult(
                tier=tier,
                action=action,
                amount=amount,
                currency="AED",
                amount_tier=AmountTier.LOW_VALUE if amount < 20000 else (AmountTier.MEDIUM_VALUE if amount < 100000 else AmountTier.HIGH_VALUE),
                similarity_score=round(score, 4),
                policy_version="CBUAE-CHQ-V2.1",
            )
            rec = self.logger.log_decision(doc_id, dec, operator_id=f"TELLER-{i:03d}")
            records.append(rec)

        self.assertEqual(len(records), 5)
        # Check sequence numbers are 1, 2, 3, 4, 5
        self.assertEqual([r.sequence_num for r in records], [1, 2, 3, 4, 5])

        # 2. Verify complete chain integrity initially passes
        is_valid_initial, err_initial = self.logger.verify_integrity()
        self.assertTrue(is_valid_initial, "Initial 5-block chain must pass integrity check")
        self.assertIsNone(err_initial, "Initial check must not produce errors")

        # Also verify cryptographic linking between consecutive blocks
        self.assertEqual(records[0].prev_hash, GENESIS_HASH)
        for idx in range(1, 5):
            self.assertEqual(
                records[idx].prev_hash,
                records[idx - 1].hash,
                f"Block {idx + 1} prev_hash must strictly match block {idx} hash",
            )

        # 3. Corrupt block #2 in the SQLite database (modify the amount field)
        corrupted_seq = 2
        conn = sqlite3.connect(self.db_path)
        with conn:
            # Change block #2 amount from 20000.0 to 999999.0 without updating hash
            conn.execute(
                "UPDATE audit_ledger SET amount = 999999.0 WHERE sequence_num = ?;",
                (corrupted_seq,)
            )
        conn.close()

        # 4. Assert validation fails with EXACT block # identified
        is_valid_corrupt, err_corrupt = self.logger.verify_integrity()
        self.assertFalse(
            is_valid_corrupt,
            "Audit log validation MUST fail after block #2 was corrupted",
        )
        self.assertIsNotNone(err_corrupt, "Audit log verification must return error description")

        # Check that sequence number 2 is explicitly and unambiguously identified
        self.assertIn(
            f"sequence {corrupted_seq}",
            err_corrupt,
            f"Error message must specify sequence {corrupted_seq}. Got: '{err_corrupt}'",
        )
        self.assertIn("Tampered record", err_corrupt)

    def test_corrupt_prev_hash_breaks_chain(self):
        """Verify modifying the prev_hash pointer is caught as broken chain at exact sequence."""
        for i in range(1, 4):
            dec = DecisionResult(
                tier=DecisionTier.GREEN,
                action="AUTO_CLEAR",
                amount=5000.0,
                currency="AED",
                amount_tier=AmountTier.LOW_VALUE,
                similarity_score=0.92,
                policy_version="CBUAE-CHQ-V2.1",
            )
            self.logger.log_decision(f"DOC-CHAIN-{i}", dec)

        # Tamper prev_hash of block 3
        conn = sqlite3.connect(self.db_path)
        with conn:
            conn.execute(
                "UPDATE audit_ledger SET prev_hash = '1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff' WHERE sequence_num = 3;"
            )
        conn.close()

        is_valid, err = self.logger.verify_integrity()
        self.assertFalse(is_valid)
        self.assertIn("sequence 3", err)
        self.assertIn("Broken chain", err)


if __name__ == "__main__":
    unittest.main(verbosity=2)
