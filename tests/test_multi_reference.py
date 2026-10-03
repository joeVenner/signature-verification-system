"""Regression tests for multi-reference verification (EXP-003)."""

import unittest
from pathlib import Path

import cv2
import numpy as np

from signature_verification_system.src.verification.deterministic import DeterministicVerifier

DATA = Path(__file__).resolve().parent.parent / "data" / "samples"


def _img(rel: str) -> np.ndarray:
    img = cv2.imread(str(DATA / rel))
    if img is None:
        raise unittest.SkipTest(f"missing sample {rel}")
    return img


class TestMultiReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v = DeterministicVerifier()
        # cedar writer 05: turns 1-3 enrolled, turn 4 held out
        cls.refs = [
            _img("genuine_pairs/pair_09_cedar_w05_ref.png"),
            _img("genuine_pairs/pair_09_cedar_w05_questioned.png"),
            _img("genuine_pairs/pair_10_cedar_w05_ref.png"),
        ]
        cls.genuine = _img("genuine_pairs/pair_10_cedar_w05_questioned.png")
        cls.skilled = _img("skilled_forgeries/pair_09_cedar_w05_forgery01.png")
        cls.other_writer = _img("genuine_pairs/pair_17_cedar_w09_ref.png")

    def test_genuine_outscores_forgeries(self):
        g = self.v.verify_against_references(self.refs, self.genuine).similarity_score
        s = self.v.verify_against_references(self.refs, self.skilled).similarity_score
        r = self.v.verify_against_references(self.refs, self.other_writer).similarity_score
        self.assertGreater(g, s)
        self.assertGreater(g, r)

    def test_deterministic_repeat(self):
        a = self.v.verify_against_references(self.refs, self.skilled)
        b = DeterministicVerifier().verify_against_references(self.refs, self.skilled)
        self.assertEqual(a.model_dump(), b.model_dump())

    def test_reference_order_does_not_change_score(self):
        a = self.v.verify_against_references(self.refs, self.genuine).similarity_score
        b = self.v.verify_against_references(self.refs[::-1], self.genuine).similarity_score
        self.assertEqual(a, b)

    def test_empty_reference_list_raises(self):
        with self.assertRaises(ValueError):
            self.v.verify_against_references([], self.genuine)

    def test_blank_references_are_inconclusive(self):
        blank = np.full((128, 256), 255, np.uint8)
        res = self.v.verify_against_references([blank, blank], self.genuine)
        self.assertFalse(res.is_match)
        self.assertEqual(res.similarity_score, 0.0)
        self.assertTrue(res.notes[0].startswith("INCONCLUSIVE"))

    def test_blank_reference_is_skipped(self):
        blank = np.full((128, 256), 255, np.uint8)
        with_blank = self.v.verify_against_references([blank] + self.refs, self.genuine).similarity_score
        without = self.v.verify_against_references(self.refs, self.genuine).similarity_score
        self.assertEqual(with_blank, without)


if __name__ == "__main__":
    unittest.main()


class TestPolarityInvariance(unittest.TestCase):
    """EXP-005: light-ink-on-dark inputs are flipped before normalisation."""

    def test_inverted_query_scores_like_original(self):
        v = DeterministicVerifier()
        ref = _img("genuine_pairs/pair_09_cedar_w05_ref.png")
        que = _img("genuine_pairs/pair_09_cedar_w05_questioned.png")
        normal = v.verify(ref, que)
        inverted = v.verify(ref, 255 - que)
        self.assertEqual(normal.decision_band, inverted.decision_band)
        self.assertAlmostEqual(normal.match_logit, inverted.match_logit, delta=0.5)

    def test_polarity_flag_reported(self):
        from signature_verification_system.src.preprocessing.normalization import normalize_signature
        que = _img("genuine_pairs/pair_09_cedar_w05_questioned.png")
        self.assertFalse(normalize_signature(que).polarity_inverted)
        self.assertTrue(normalize_signature(255 - que).polarity_inverted)


class TestQualityGate(unittest.TestCase):
    """EXP-007: degraded inputs become INCONCLUSIVE, never a confident verdict."""

    @classmethod
    def setUpClass(cls):
        cls.v = DeterministicVerifier()
        cls.ref = _img("genuine_pairs/pair_09_cedar_w05_ref.png")
        cls.que = _img("genuine_pairs/pair_09_cedar_w05_questioned.png")

    def test_clean_pair_passes_gate(self):
        res = self.v.verify(self.ref, self.que)
        self.assertNotEqual(res.decision_band, "INCONCLUSIVE")
        self.assertTrue(res.quality["questioned"]["passed"])

    def test_heavy_blur_is_inconclusive(self):
        blurred = cv2.GaussianBlur(self.que, (0, 0), 3.0)
        res = self.v.verify(self.ref, blurred)
        self.assertEqual(res.decision_band, "INCONCLUSIVE")
        self.assertFalse(res.is_match)

    def test_washed_out_ink_is_inconclusive(self):
        faint = np.clip((self.que.astype(float) - self.que.mean()) * 0.25 + self.que.mean(), 0, 255).astype(np.uint8)
        res = self.v.verify(self.ref, faint)
        self.assertEqual(res.decision_band, "INCONCLUSIVE")
        self.assertIn("INSUFFICIENT_INK_CONTRAST", res.quality["questioned"]["blocking_issues"])

    def test_bad_reference_is_dropped_in_multi_mode(self):
        bad = cv2.GaussianBlur(self.ref, (0, 0), 3.0)
        refs = [bad, self.ref, _img("genuine_pairs/pair_10_cedar_w05_ref.png")]
        res = self.v.verify_against_references(refs, self.que)
        self.assertEqual(res.reference_count, 2)
        self.assertFalse(res.quality["references"][0]["passed"])

    def test_engine_routes_inconclusive_to_red_four_eyes(self):
        from signature_verification_system.src.adjudication.decision_engine import DecisionEngine
        res = self.v.verify(self.ref, cv2.GaussianBlur(self.que, (0, 0), 3.0))
        dec = DecisionEngine().evaluate(res, amount=1000.0)
        self.assertEqual(dec.tier.value, "RED")
        self.assertTrue(dec.requires_four_eyes)
        self.assertIn("SIGNATURE_VERIFICATION_INCONCLUSIVE", dec.cbuae_compliance_flags)


class TestExplanation(unittest.TestCase):
    """EXP-008: explanations are derived from measured signals only."""

    @classmethod
    def setUpClass(cls):
        cls.v = DeterministicVerifier()
        cls.refs = [
            _img("genuine_pairs/pair_09_cedar_w05_ref.png"),
            _img("genuine_pairs/pair_09_cedar_w05_questioned.png"),
            _img("genuine_pairs/pair_10_cedar_w05_ref.png"),
        ]

    def test_evidence_values_match_result_features(self):
        res = self.v.verify(self.refs[0], self.refs[1])
        ev = {e["signal"]: e for e in res.explanation["evidence"]}
        self.assertAlmostEqual(ev["shape_similarity"]["value"], res.features.shape_similarity, places=3)
        self.assertAlmostEqual(ev["keypoint_similarity"]["value"], res.features.keypoint_similarity, places=3)
        self.assertEqual(res.explanation["decision_band"], res.decision_band)

    def test_weak_signals_never_presented_as_evidence(self):
        res = self.v.verify(self.refs[0], self.refs[1])
        for item in res.explanation["evidence"]:
            self.assertGreaterEqual(item["discriminative_auc"], 0.80)
        for item in res.explanation["observations"]:
            self.assertLess(item["discriminative_auc"], 0.80)

    def test_inconclusive_has_no_evidence_or_preprocessing_claims(self):
        res = self.v.verify(self.refs[0], cv2.GaussianBlur(self.refs[1], (0, 0), 3.0))
        self.assertEqual(res.explanation["evidence"], [])
        self.assertEqual(res.explanation["preprocessing"], [])

    def test_render_is_deterministic(self):
        from signature_verification_system.src.verification.explanation import render_text
        q = _img("skilled_forgeries/pair_09_cedar_w05_forgery01.png")
        a = render_text(self.v.verify_against_references(self.refs, q).explanation)
        b = render_text(DeterministicVerifier().verify_against_references(self.refs, q).explanation)
        self.assertEqual(a, b)


class TestRequiredRegressionCases(unittest.TestCase):
    """Representative cases from the project brief that are not covered elsewhere."""

    @classmethod
    def setUpClass(cls):
        cls.v = DeterministicVerifier()
        cls.refs = [
            _img("genuine_pairs/pair_09_cedar_w05_ref.png"),
            _img("genuine_pairs/pair_09_cedar_w05_questioned.png"),
            _img("genuine_pairs/pair_10_cedar_w05_ref.png"),
        ]
        cls.genuine = _img("genuine_pairs/pair_10_cedar_w05_questioned.png")

    def _rotate(self, img, deg):
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
        paper = tuple(int(x) for x in np.median(img.reshape(-1, 3), axis=0))
        return cv2.warpAffine(img, m, (w, h), borderValue=paper)

    def test_genuine_with_rotation_is_not_rejected(self):
        res = self.v.verify_against_references(self.refs, self._rotate(self.genuine, 5))
        self.assertIn(res.decision_band, ("ACCEPT", "REVIEW"))

    def test_genuine_with_scale_change_is_not_rejected(self):
        h, w = self.genuine.shape[:2]
        scaled = cv2.resize(self.genuine, (int(w * 0.75), int(h * 0.75)), interpolation=cv2.INTER_AREA)
        res = self.v.verify_against_references(self.refs, scaled)
        self.assertIn(res.decision_band, ("ACCEPT", "REVIEW"))

    def test_different_writer_is_rejected(self):
        res = self.v.verify_against_references(self.refs, _img("genuine_pairs/pair_17_cedar_w09_ref.png"))
        self.assertEqual(res.decision_band, "REJECT")

    def test_near_identical_lookalike_is_never_auto_accepted(self):
        # Known limitation (EXP-005): synthetic writer B mimics writer A's layout and
        # outscores A's own genuine variant. The operating point must still not auto-accept it.
        a = _img("genuine_pairs/pair_00_synthetic_ref.png")
        b = _img("random_forgeries/pair_00_synthetic_questioned.png")
        res = self.v.verify(a, b)
        self.assertNotEqual(res.decision_band, "ACCEPT")
        self.assertFalse(res.is_match)


class TestAlignment(unittest.TestCase):
    """EXP-014: RANSAC-aligned layout comparison and truthful reporting of it."""

    def test_rotated_genuine_reports_measured_alignment_and_is_not_rejected(self):
        v = DeterministicVerifier()
        refs = [_img("genuine_pairs/pair_09_cedar_w05_ref.png"), _img("genuine_pairs/pair_09_cedar_w05_questioned.png"),
                _img("genuine_pairs/pair_10_cedar_w05_ref.png")]
        q = _img("genuine_pairs/pair_10_cedar_w05_questioned.png")
        h, w = q.shape[:2]
        rot = cv2.warpAffine(q, cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1.0), (w, h),
                             borderValue=tuple(int(x) for x in np.median(q.reshape(-1, 3), axis=0)))
        res = v.verify_against_references(refs, rot)
        self.assertNotEqual(res.decision_band, "REJECT")
        al = res.explanation["alignment"]
        self.assertIsNotNone(al)
        self.assertGreater(abs(al["rotation_deg"]), 3.0)

    def test_inconclusive_never_claims_alignment(self):
        v = DeterministicVerifier()
        ref = _img("genuine_pairs/pair_09_cedar_w05_ref.png")
        res = v.verify(ref, cv2.GaussianBlur(_img("genuine_pairs/pair_09_cedar_w05_questioned.png"), (0, 0), 3.0))
        self.assertIsNone(res.explanation["alignment"])


class TestAlignedShapeUnit(unittest.TestCase):
    """Direct tests of similarity.aligned_shape (code review EXP-014 item 2)."""

    @classmethod
    def setUpClass(cls):
        from signature_verification_system.src.verification.features import extract_features
        cls.extract = staticmethod(extract_features)
        cls.img = _img("genuine_pairs/pair_10_cedar_w05_questioned.png")
        cls.base = extract_features(cls.img)

    def _rot(self, deg):
        h, w = self.img.shape[:2]
        paper = tuple(int(x) for x in np.median(self.img.reshape(-1, 3), axis=0))
        return cv2.warpAffine(self.img, cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0), (w, h), borderValue=paper)

    def test_recovers_rotation_and_raises_layout_score(self):
        from signature_verification_system.src.verification.similarity import compare, shape_similarity
        rotated = self.extract(self._rot(10))
        pair = compare(self.base, rotated)
        self.assertIsNotNone(pair.alignment)
        # getRotationMatrix2D(+10) turns the image counter-clockwise on screen => about -10 deg here
        self.assertAlmostEqual(abs(pair.alignment.rotation_deg), 10.0, delta=3.0)
        self.assertTrue(pair.alignment_used)
        self.assertGreater(pair.shape, shape_similarity(self.base, rotated))

    def test_no_alignment_without_enough_inliers(self):
        from signature_verification_system.src.verification.similarity import KeypointMatch, aligned_shape
        weak = KeypointMatch(0.0, 3, 4, 100, 100, np.float64([[1, 0, 0], [0, 1, 0]]))
        self.assertIsNone(aligned_shape(self.base, self.base, weak))

    def test_no_alignment_for_implausible_or_nan_transform(self):
        from signature_verification_system.src.verification.similarity import KeypointMatch, aligned_shape
        c, s = np.cos(np.radians(60)), np.sin(np.radians(60))
        big_rot = KeypointMatch(0.5, 50, 60, 100, 100, np.float64([[c, -s, 0], [s, c, 0]]))
        nan_t = KeypointMatch(0.5, 50, 60, 100, 100, np.full((2, 3), np.nan))
        self.assertIsNone(aligned_shape(self.base, self.base, big_rot))
        self.assertIsNone(aligned_shape(self.base, self.base, nan_t))

    def test_compare_is_deterministic(self):
        from signature_verification_system.src.verification.similarity import compare
        rotated = self.extract(self._rot(12))
        a, b = compare(self.base, rotated), compare(self.base, rotated)
        self.assertEqual(a.fused_logit, b.fused_logit)
        self.assertEqual(a.alignment, b.alignment)
