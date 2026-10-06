"""Capture-resolution matching by pen width (src/verification/capture_scale.py, EXP-026/027)."""

import base64
import tempfile
import unittest
import unittest.mock
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from signature_verification_system.src.verification import capture_scale as cs
from signature_verification_system.src.verification import features
from signature_verification_system.src.verification.features import extract_features, match_capture_scale
from signature_verification_system.src.verification.similarity import compare


def signature(scale: float = 1.0, pen: int = 2) -> np.ndarray:
    """Synthetic signature (dark ink on white, uint8 BGR); `scale` multiplies size and pen width alike."""
    w, h = int(round(600 * scale)), int(round(260 * scale))
    canvas = np.full((h, w), 255, np.uint8)
    t = np.linspace(0.0, 1.0, 600)
    for k in range(3):
        x = (50 + 500 * t) * scale
        y = (100 + 25 * k + 55 * np.sin(2 * np.pi * (1.5 + k) * t + k) * np.exp(-3 * (t - 0.3 - 0.2 * k) ** 2)) * scale
        pts = np.round(np.stack([x, y], 1) * 16).astype(np.int32)
        cv2.polylines(canvas, [pts], False, 30, max(1, int(round(pen * scale))), cv2.LINE_AA, shift=4)
    cv2.ellipse(canvas, (int(170 * scale), int(80 * scale)), (int(38 * scale), int(26 * scale)), 0, 0, 360, 30,
                max(1, int(round(pen * scale))), cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


class PenWidthTest(unittest.TestCase):
    def test_measures_a_drawn_stroke(self):
        ink = np.zeros((60, 200))
        ink[25:31, 10:190] = 1.0                          # 6 px thick bar
        pen, ink_width = cs.capture_widths(ink, 1.0)
        self.assertAlmostEqual(pen, 6.0, delta=1.0)
        self.assertAlmostEqual(ink_width, 6.0, delta=0.5)

    def test_divides_by_working_scale(self):
        ink = np.zeros((60, 200))
        ink[25:31, 10:190] = 1.0
        native, halved = cs.capture_widths(ink, 1.0), cs.capture_widths(ink, 2.0)
        self.assertAlmostEqual(halved[0], native[0] / 2.0)
        self.assertAlmostEqual(halved[1], native[1] / 2.0)

    def test_no_strokes_is_zero(self):
        self.assertEqual(cs.capture_widths(np.zeros((20, 20)), 1.0), (0.0, 0.0))


class RulesTest(unittest.TestCase):
    def test_high_resolution_factor_only_above_band(self):
        self.assertIsNone(cs.high_resolution_factor(cs.PEN_WIDTH_MAX_PX))
        self.assertIsNone(cs.high_resolution_factor(4.0))
        self.assertAlmostEqual(cs.high_resolution_factor(10.4), cs.PEN_WIDTH_TARGET_PX / 10.4)
        self.assertEqual(cs.high_resolution_factor(1000.0), cs.MIN_RESAMPLE_FACTOR)

    def test_snap_factor_grid_and_clamps(self):
        self.assertIsNone(cs.snap_factor(cs.MATCHED_RATIO))
        self.assertIsNone(cs.snap_factor(0.98))                       # snaps to 1.0: already matched
        self.assertEqual(cs.snap_factor(0.481), 0.5)
        self.assertEqual(cs.snap_factor(0.96), 1.0 - cs.FACTOR_STEP)
        self.assertEqual(cs.snap_factor(0.05), cs.MIN_RESAMPLE_FACTOR)

    def test_model_capture_scale_inverts_the_blur_floor(self):
        for scale in (0.5, 1.0, 2.0):
            width = cs.FLOOR_MODEL_A * scale + cs.FLOOR_MODEL_B
            self.assertAlmostEqual(cs.model_capture_scale(width, 1.0), scale)
        self.assertEqual(cs.model_capture_scale(1.0, 1.0), 0.0)        # below the floor: clamped

    def test_pair_factor_needs_a_coarse_image(self):
        self.assertIsNone(cs.pair_factor(6.0, 1.0, cs.INK_WIDTH_MIN_PX, 1.0))  # both in band: untouched
        self.assertIsNone(cs.pair_factor(6.0, 1.0, 0.0, 1.0))
        self.assertIsNone(cs.pair_factor(0.0, 1.0, 4.0, 1.0))
        self.assertIsNone(cs.pair_factor(4.5, 1.0, 4.49, 1.0))                 # same capture: matched

    def test_pair_factor_follows_the_model(self):
        half = cs.FLOOR_MODEL_A * 0.5 + cs.FLOOR_MODEL_B              # ink width of a 0.5x capture
        self.assertEqual(cs.pair_factor(6.0, 1.0, half, 1.0), 0.5)     # in-band partner counts as scale 1
        quarter = cs.FLOOR_MODEL_A * 0.25 + cs.FLOOR_MODEL_B
        self.assertEqual(cs.pair_factor(half, 1.0, quarter, 1.0), 0.5)  # both coarse: ratio of scales

    def test_downsample_never_upsamples(self):
        img = np.zeros((40, 80, 3), np.uint8)
        self.assertEqual(cs.downsample(img, 0.5).shape, (20, 40, 3))
        self.assertEqual(cs.downsample(img, 0.01).shape[:2], (8, 8))
        for bad in (0.0, 1.0, 2.0, -0.5):
            with self.assertRaises(ValueError):
                cs.downsample(img, bad)


class ExtractionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = signature(1.0)
        cls.fine = signature(3.0)
        cls.coarse = cv2.resize(signature(1.0), (208, 90), interpolation=cv2.INTER_AREA)
        cls.f_native = extract_features(cls.native)
        cls.f_fine = extract_features(cls.fine)
        cls.f_coarse = extract_features(cls.coarse)

    def test_in_band_image_is_used_as_given(self):
        self.assertLessEqual(self.f_native.pen_width_px, cs.PEN_WIDTH_MAX_PX)
        self.assertGreaterEqual(self.f_native.ink_width_px, cs.INK_WIDTH_MIN_PX)
        self.assertIs(self.f_native.source_image, self.native)

    def test_fine_capture_is_brought_into_band(self):
        from signature_verification_system.src.core.config import DEFAULT_CONFIG
        from signature_verification_system.src.preprocessing.normalization import normalize_signature
        from signature_verification_system.src.verification.features import _describe

        self.assertLess(self.f_fine.source_image.shape[1], self.fine.shape[1])
        self.assertLessEqual(self.f_fine.pen_width_px, cs.PEN_WIDTH_MAX_PX)
        unmatched = _describe(normalize_signature(self.fine), self.fine, (0.0, 0.0, 1.0), DEFAULT_CONFIG.representation)
        native = self.f_native.stroke.stroke_width
        self.assertLess(abs(np.log(self.f_fine.stroke.stroke_width / native)),
                        abs(np.log(unmatched.stroke.stroke_width / native)))

    def test_in_band_pair_is_untouched(self):
        a, b = match_capture_scale(self.f_native, self.f_fine)
        self.assertIs(a, self.f_native)
        self.assertIs(b, self.f_fine)

    def test_in_band_pair_compares_bit_identically(self):
        from dataclasses import replace
        # Without a source image matching cannot run: that is the unmatched path.
        unmatched = compare(replace(self.f_native, source_image=None), replace(self.f_fine, source_image=None))
        matched = compare(self.f_native, self.f_fine)
        self.assertEqual(matched.fused_logit, unmatched.fused_logit)
        self.assertEqual(matched.signals, unmatched.signals)

    def test_finer_partner_of_a_coarse_image_is_made_coarser(self):
        self.assertLess(self.f_coarse.ink_width_px, cs.INK_WIDTH_MIN_PX)
        for ref, que, flip in ((self.f_native, self.f_coarse, False), (self.f_coarse, self.f_native, True)):
            a, b = match_capture_scale(ref, que)
            matched, kept = (b, a) if flip else (a, b)
            self.assertIs(kept, self.f_coarse)
            self.assertLess(matched.source_image.shape[1], self.native.shape[1])
            self.assertLess(matched.ink_width_px, self.f_native.ink_width_px)

    def test_features_without_source_are_left_alone(self):
        from dataclasses import replace
        bare = replace(self.f_native, source_image=None)
        a, b = match_capture_scale(bare, self.f_coarse)
        self.assertIs(a, bare)
        self.assertIs(b, self.f_coarse)

    def test_compare_is_deterministic_and_cache_independent(self):
        first = compare(self.f_native, self.f_coarse).fused_logit
        features._coarse_cache.clear()
        second = compare(self.f_native, self.f_coarse).fused_logit
        third = compare(self.f_native, self.f_coarse).fused_logit
        self.assertEqual(first, second)
        self.assertEqual(second, third)

    def test_repeated_matching_is_deterministic(self):
        features._coarse_cache.clear()
        runs = [match_capture_scale(self.f_native, self.f_coarse)[0] for _ in range(3)]
        features._coarse_cache.clear()
        runs.append(match_capture_scale(self.f_native, self.f_coarse)[0])
        for other in runs[1:]:
            np.testing.assert_array_equal(other.source_image, runs[0].source_image)
            np.testing.assert_array_equal(other.shape_descriptor, runs[0].shape_descriptor)
            np.testing.assert_array_equal(other.keypoints, runs[0].keypoints)
            self.assertEqual(other.ink_width_px, runs[0].ink_width_px)

    def test_coarse_cache_is_bounded_lru(self):
        from signature_verification_system.src.core.config import DEFAULT_CONFIG

        p = DEFAULT_CONFIG.representation
        saved = features._COARSE_CACHE_SIZE
        features._coarse_cache.clear()
        features._COARSE_CACHE_SIZE = 2
        try:
            for factor in (0.9, 0.8, 0.7):
                features._coarser(self.f_native, factor, p)
            self.assertEqual(len(features._coarse_cache), 2)
            self.assertEqual([key[3] for key in features._coarse_cache], [0.8, 0.7])   # oldest evicted
            hit = features._coarser(self.f_native, 0.8, p)
            self.assertIs(hit, features._coarser(self.f_native, 0.8, p))
            self.assertEqual([key[3] for key in features._coarse_cache], [0.7, 0.8])   # hit refreshed
        finally:
            features._COARSE_CACHE_SIZE = saved
            features._coarse_cache.clear()

def _unmatched(a, b):
    """compare() on the unmatched path (no source image, so matching cannot run)."""
    return compare(replace(a, source_image=None), replace(b, source_image=None))


class FallbackTest(unittest.TestCase):
    """Matching is an accuracy aid: any re-extraction failure falls back to the original features."""

    @classmethod
    def setUpClass(cls):
        cls.f_native = extract_features(signature(1.0))
        cls.f_coarse = extract_features(cv2.resize(signature(1.0), (208, 90), interpolation=cv2.INTER_AREA))
        cls.expected = _unmatched(cls.f_native, cls.f_coarse)

    def test_resample_errors_compare_unmatched(self):
        for exc in (ValueError("no ink"), cv2.error("opencv detail"), MemoryError("allocator detail")):
            with self.subTest(error=type(exc).__name__):
                boom = unittest.mock.Mock(side_effect=exc)
                with unittest.mock.patch.object(features, "_coarser", boom), \
                        self.assertLogs(features.__name__, level="INFO"):
                    a, b = match_capture_scale(self.f_native, self.f_coarse)
                    result = compare(self.f_native, self.f_coarse)
                boom.assert_called()
                self.assertIs(a, self.f_native)
                self.assertIs(b, self.f_coarse)
                self.assertEqual(result.fused_logit, self.expected.fused_logit)
                self.assertEqual(result.signals, self.expected.signals)

    def test_failed_high_resolution_pass_keeps_the_previous_one(self):
        fine = signature(3.0)
        reference = extract_features(fine)
        self.assertLess(reference.source_image.shape[1], fine.shape[1])    # the pass normally runs
        real = features.normalize_signature
        calls = []

        def second_call_fails(image, prepared=None):
            calls.append(image.shape)
            if len(calls) > 1:
                raise ValueError("No signature ink detected")
            return real(image, prepared)

        with unittest.mock.patch.object(features, "normalize_signature", second_call_fails):
            kept = extract_features(fine)
        self.assertGreater(len(calls), 1)
        self.assertIs(kept.source_image, fine)                            # first (full-size) pass kept
        self.assertGreater(kept.pen_width_px, cs.PEN_WIDTH_MAX_PX)

    def test_unexpected_errors_still_propagate(self):
        with unittest.mock.patch.object(features, "_coarser", unittest.mock.Mock(side_effect=KeyError("bug"))):
            with self.assertRaises(KeyError):
                match_capture_scale(self.f_native, self.f_coarse)


class FallbackHttpTest(unittest.TestCase):
    """A capture-scale failure inside compare() never becomes an HTTP 500."""

    SECRET = "secret internal allocator detail"

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        from signature_verification_system.src.adjudication.audit_logger import AuditLogger
        from signature_verification_system.src.api.app import create_app

        cls.tmp = tempfile.TemporaryDirectory()
        db = Path(cls.tmp.name) / "audit.db"
        cls.client = TestClient(create_app(audit_logger=AuditLogger(str(db), str(db.with_suffix(".jsonl"))),
                                           samples=None))

        def b64(image):
            ok, buf = cv2.imencode(".png", image)
            assert ok
            return base64.b64encode(buf.tobytes()).decode("ascii")

        cls.ref = b64(signature(1.0))
        cls.que = b64(cv2.resize(signature(1.0), (208, 90), interpolation=cv2.INTER_AREA))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_verify_compare_and_inspect_do_not_fail(self):
        for exc in (cv2.error(self.SECRET), MemoryError(self.SECRET)):
            with self.subTest(error=type(exc).__name__):
                boom = unittest.mock.Mock(side_effect=exc)
                with unittest.mock.patch.object(features, "_coarser", boom):
                    responses = {
                        "verify": self.client.post("/api/v1/verify", json={"ref_image": self.ref,
                                                                           "test_image": self.que}),
                        "compare_multi": self.client.post("/api/v1/signature/compare", json={
                            "reference_images": [self.ref, self.ref], "questioned_image": self.que}),
                        "inspect": self.client.post("/api/v1/signature/inspect", json={
                            "reference_images": [self.ref], "questioned_image": self.que}),
                    }
                boom.assert_called()                                       # matching was attempted
                for name, r in responses.items():
                    self.assertEqual(r.status_code, 200, f"{name}: {r.text}")
                    self.assertNotIn(self.SECRET, r.text, name)
                self.assertNotEqual(responses["verify"].json()["verification"]["decision_band"], "INCONCLUSIVE")
                self.assertNotEqual(responses["compare_multi"].json()["band"], "INCONCLUSIVE")
                self.assertNotEqual(responses["inspect"].json()["comparison"]["band"], "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
