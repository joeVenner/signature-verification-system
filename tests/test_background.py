"""Textured cheque backgrounds: detection, ink extraction, quality gate (EXP-025).

All images are synthetic and seeded: a known stroke mask drawn over a dense
wavy guilloche, then downscaled to a ~200 px field crop.
"""

import unittest

import cv2
import numpy as np

from signature_verification_system.src.preprocessing.background import (
    TEXTURED_MIN_FRACTION,
    LOWRES_MAX_WIDTH,
    LOWRES_TARGET_WIDTH,
    MAX_WORK_PIXELS,
    extract_ink_layer,
    measure_background_texture,
    prepare_signature,
    upscale_factor,
)
from signature_verification_system.src.preprocessing.isolation import isolate_signature_ink
from signature_verification_system.src.preprocessing.normalization import (
    INK_MASK_LEVEL,
    harmonize_photometric,
    normalize_signature,
)
from signature_verification_system.src.preprocessing.quality import assess_signature_quality
from signature_verification_system.src.verification.deterministic import DeterministicVerifier

CANVAS_H, CANVAS_W = 270, 640
PAPER = 238.0
INK = 45.0


def _strokes() -> np.ndarray:
    """Signature-like stroke coverage in [0, 1] on the full-size canvas."""
    canvas = np.zeros((CANVAS_H, CANVAS_W), np.uint8)
    t = np.linspace(0.0, 1.0, 400)
    for k in range(3):
        x = 60 + 520 * t
        y = 110 + 25 * k + 60 * np.sin(2 * np.pi * (1.5 + k) * t + k) * np.exp(-3 * (t - 0.3 - 0.2 * k) ** 2)
        cv2.polylines(canvas, [np.stack([x, y], 1).astype(np.int32)], False, 255, 7, cv2.LINE_AA)
    cv2.ellipse(canvas, (180, 90), (40, 28), 0, 0, 360, 255, 7, cv2.LINE_AA)
    return canvas.astype(np.float64) / 255.0


def _guilloche(depth: float = 0.35) -> np.ndarray:
    """Paper with one family of wavy lines (period ~4 px after downscaling to 208 px)."""
    yy, xx = np.mgrid[0:CANVAS_H, 0:CANVAS_W].astype(np.float64)
    phase = 2 * np.pi * (yy + 8 * np.sin(2 * np.pi * xx / 110.0)) / 12.0
    return PAPER * (1.0 - depth * np.clip(np.cos(phase), 0.0, 1.0) ** 1.5)


def _capture(full: np.ndarray, width: int, seed: int) -> np.ndarray:
    h = round(CANVAS_H * width / CANVAS_W)
    small = cv2.resize(full, (width, h), interpolation=cv2.INTER_AREA)
    noise = np.random.default_rng(seed).normal(0.0, 3.0, small.shape)
    return np.clip(small + noise, 0, 255).astype(np.uint8)


def signature(textured: bool = True, width: int = 208, seed: int = 0, strokes: bool = True):
    """(uint8 image, bool ground-truth stroke mask) at the requested width."""
    a = _strokes() if strokes else np.zeros((CANVAS_H, CANVAS_W))
    paper = _guilloche() if textured else np.full((CANVAS_H, CANVAS_W), PAPER)
    img = _capture(paper * (1.0 - a) + INK * a, width, seed)
    gt = cv2.resize(a, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA) > 0.5
    return img, gt


def verifier_ink_mask(image: np.ndarray) -> np.ndarray:
    """Mask the verifier crops on: normalisation darkness above INK_MASK_LEVEL."""
    darkness = (255.0 - harmonize_photometric(image).astype(np.float64)) / 255.0
    return isolate_signature_ink(darkness, INK_MASK_LEVEL) > INK_MASK_LEVEL


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.shape != gt.shape:
        pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA) > 0
    return float((pred & gt).sum()) / float((pred | gt).sum())


class TestTextureDetector(unittest.TestCase):
    def test_guilloche_is_textured(self):
        for width in (208, 640):
            report = measure_background_texture(signature(True, width)[0])
            self.assertTrue(report.is_textured, (width, report))

    def test_clean_and_ruled_paper_are_not_textured(self):
        img, _ = signature(textured=False, width=640)
        self.assertFalse(measure_background_texture(img).is_textured)
        ruled = img.copy()
        ruled[18::36] = (ruled[18::36] * 0.75).astype(np.uint8)   # clearance-score ruled_lines style
        ruled[19::36] = (ruled[19::36] * 0.75).astype(np.uint8)
        self.assertFalse(measure_background_texture(ruled).is_textured)

    def test_extracted_layer_is_not_textured_again(self):
        img = signature(True, 208)[0]
        prepared = prepare_signature(img, img)
        self.assertTrue(prepared.background_removed)
        self.assertFalse(measure_background_texture(prepared.work_image).is_textured)


class TestInkExtraction(unittest.TestCase):
    def test_extraction_recovers_stroke_mask(self):
        for width in (208, 640):
            img, gt = signature(True, width)
            before = iou(verifier_ink_mask(img), gt)
            after = iou(verifier_ink_mask(prepare_signature(img, img).work_image), gt)
            self.assertGreaterEqual(after, 0.8, (width, before, after))
            self.assertLess(before, 0.7, (width, before, after))

    def test_background_becomes_white_paper_and_ink_keeps_its_darkness(self):
        img, gt = signature(True, 640)
        layer = extract_ink_layer(img)
        self.assertIsNotNone(layer)
        far = cv2.dilate(gt.astype(np.uint8), np.ones((9, 9), np.uint8)) == 0
        self.assertGreater(float(np.mean(layer[far] == 255)), 0.99)
        core = cv2.erode(gt.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        self.assertGreater(float(np.std(layer[core])), 0.0)   # not binarised
        self.assertLess(float(np.median(layer[core])), 100.0)

    def test_pure_texture_is_not_separable(self):
        texture_only, _ = signature(True, 640, strokes=False)
        self.assertIsNone(extract_ink_layer(texture_only))

    def test_deterministic(self):
        img, _ = signature(True, 208)
        a, b = prepare_signature(img, img), prepare_signature(img.copy(), img.copy())
        self.assertEqual(a.work_image.tobytes(), b.work_image.tobytes())
        self.assertEqual(a.gate_image.tobytes(), b.gate_image.tobytes())


class TestRouting(unittest.TestCase):
    def test_clean_full_size_image_is_passed_through_unchanged(self):
        img, _ = signature(textured=False, width=640)
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        prepared = prepare_signature(bgr, img)
        self.assertIs(prepared.work_image, bgr)
        self.assertIs(prepared.gate_image, bgr)
        self.assertEqual(prepared.upscale, 1.0)

    def test_small_crop_is_upscaled_but_gated_at_input_resolution(self):
        img, _ = signature(textured=False, width=200)
        self.assertLess(img.shape[1], LOWRES_MAX_WIDTH)
        prepared = prepare_signature(img, img)
        self.assertEqual(prepared.work_image.shape[1], LOWRES_TARGET_WIDTH)
        self.assertIs(prepared.gate_image, img)
        self.assertFalse(prepared.background_removed)


class TestDimensionLimits(unittest.TestCase):
    def test_extreme_aspect_ratio_is_not_upscaled_and_is_blocked_fast(self):
        import time

        img = np.full((16, 400_000), 240, np.uint8)
        img[4:12, 1000:5000] = 30
        start = time.perf_counter()
        prepared = prepare_signature(img, img)
        q = assess_signature_quality(img)
        elapsed = time.perf_counter() - start
        self.assertFalse(prepared.dimensions_supported)
        self.assertIs(prepared.work_image, img)
        self.assertEqual(prepared.upscale, 1.0)
        self.assertFalse(q.passed)
        self.assertEqual(q.blocking_issues, ["IMAGE_DIMENSIONS_UNSUPPORTED"])
        self.assertLess(elapsed, 2.0)

    def test_upscale_never_exceeds_the_work_pixel_cap(self):
        for h, w in ((16, 255), (300, 255), (4000, 250), (16, 3_000_000)):
            scale = upscale_factor(h, w)
            self.assertLessEqual(h * w * scale * scale, max(MAX_WORK_PIXELS, h * w))
        self.assertEqual(upscale_factor(4000, 250), 1.0)
        self.assertEqual(upscale_factor(100, 200), LOWRES_TARGET_WIDTH / 200)


def sparse_signature(scale: float) -> tuple:
    """Signature shrunk by `scale` in the top-left of a full guilloche crop (640 px)."""
    small = cv2.resize(_strokes(), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    a = np.zeros((CANVAS_H, CANVAS_W))
    a[10:10 + small.shape[0], 20:20 + small.shape[1]] = small
    img = _capture(_guilloche() * (1.0 - a) + INK * a, CANVAS_W, 0)
    return img, a > 0.5


class TestEdgeCases(unittest.TestCase):
    def test_sparse_ink_is_still_routed(self):
        img, gt = sparse_signature(0.25)
        self.assertLess(float(gt.mean()), 0.01)          # ink covers < 1% of the crop
        prepared = prepare_signature(img, img)
        self.assertTrue(prepared.texture.is_textured)
        self.assertTrue(prepared.background_removed)
        self.assertGreaterEqual(iou(verifier_ink_mask(prepared.work_image), gt), 0.6)

    def test_tiny_images_do_not_raise(self):
        for h, w in ((4, 4), (4, 300), (15, 15), (15, 300), (300, 15)):
            img = np.full((h, w), 240, np.uint8)
            img[h // 2, :] = 20
            prepared = prepare_signature(img, img)
            self.assertIs(prepared.work_image, img)
            self.assertFalse(assess_signature_quality(img).passed)

    def test_uniform_images_are_blocked(self):
        for value in (255, 0):
            img = np.full((90, 210), value, np.uint8)
            self.assertFalse(assess_signature_quality(img).passed)
            with self.assertRaises(ValueError):
                normalize_signature(img)

    def test_channel_layouts_and_uint16_take_the_textured_route(self):
        gray, _ = signature(True, 208)
        variants = {
            "bgr": cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
            "bgra": cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA),
            "uint16": gray.astype(np.uint16) * 257,
        }
        for name, img in variants.items():
            q = assess_signature_quality(img)
            self.assertTrue(q.passed, (name, q.blocking_issues))
            self.assertTrue(q.background_removed, name)
            self.assertGreater(normalize_signature(img).ink_pixel_count, 0, name)

    def test_inverted_polarity_with_texture(self):
        q = assess_signature_quality(255 - signature(True, 208)[0])
        self.assertTrue(q.passed, q.blocking_issues)
        self.assertTrue(q.background_removed)
        self.assertIn("POLARITY_INVERTED_CORRECTED", q.warnings)

    def test_textured_fraction_boundary(self):
        from signature_verification_system.src.preprocessing.background import _DarknessAnalysis, _texture_report

        n = 10_000
        for share, expected in ((TEXTURED_MIN_FRACTION, True), (TEXTURED_MIN_FRACTION - 0.001, False)):
            darkness = np.zeros(n)
            darkness[: int(round(share * n))] = 0.5
            report = _texture_report(_DarknessAnalysis(darkness, 1.0, np.ones(n, bool)))
            self.assertIs(report.is_textured, expected, share)


class TestFailureHandling(unittest.TestCase):
    def test_gate_never_raises_on_preprocessing_failure(self):
        from unittest import mock

        def boom(*_args, **_kwargs):
            raise cv2.error("internal detail")

        with mock.patch("signature_verification_system.src.preprocessing.quality.prepare_signature", boom):
            q = assess_signature_quality(signature(True, 208)[0])
        self.assertFalse(q.passed)
        self.assertEqual(q.blocking_issues, ["PREPROCESSING_FAILED"])

    def test_gate_blocks_unsupported_dtype(self):
        q = assess_signature_quality(np.zeros((90, 210), np.float64))
        self.assertEqual(q.blocking_issues, ["PREPROCESSING_FAILED"])

    def test_opencv_failure_in_normalisation_is_inconclusive_without_details(self):
        from unittest import mock

        def boom(*_args, **_kwargs):
            raise cv2.error("internal detail")

        reference, _ = signature(textured=False, width=640, seed=1)
        questioned, _ = signature(textured=True, width=208, seed=2)
        with mock.patch("signature_verification_system.src.preprocessing.normalization.prepare_signature", boom):
            result = DeterministicVerifier().verify(reference, questioned)
        self.assertEqual(result.decision_band, "INCONCLUSIVE")
        self.assertNotIn("internal detail", result.model_dump_json())


class TestQualityGateOnTexture(unittest.TestCase):
    def test_legible_signature_on_guilloche_passes(self):
        q = assess_signature_quality(signature(True, 208)[0])
        self.assertTrue(q.passed, q.blocking_issues)
        self.assertTrue(q.background_removed)
        self.assertIn("TEXTURED_BACKGROUND_REMOVED", q.warnings)
        self.assertLess(q.noise_ratio, 0.06)

    def test_pure_noise_stays_blocked(self):
        rng = np.random.default_rng(7)
        for img in (np.clip(rng.normal(180, 40, (90, 210)), 0, 255).astype(np.uint8),
                    rng.integers(0, 256, (90, 210)).astype(np.uint8)):
            q = assess_signature_quality(img)
            self.assertFalse(q.passed)
            self.assertIn("EXCESSIVE_NOISE", q.blocking_issues)

    def test_texture_with_only_dark_specks_is_blocked(self):
        img, _ = signature(True, 208, strokes=False)
        specks = np.random.default_rng(3).random(img.shape) < 0.01
        img[specks] = 30
        q = assess_signature_quality(img)
        self.assertFalse(q.passed)
        self.assertIn("BACKGROUND_NOT_SEPARABLE", q.blocking_issues)

    def test_verifier_gives_a_verdict_on_a_textured_field_crop(self):
        reference, _ = signature(textured=False, width=640, seed=1)
        questioned, _ = signature(textured=True, width=208, seed=2)
        result = DeterministicVerifier().verify(reference, questioned)
        self.assertNotEqual(result.decision_band, "INCONCLUSIVE", result.notes)
        self.assertTrue(result.quality["questioned"]["background_removed"])


class TestTexturedRouteDeterminism(unittest.TestCase):
    """The textured + upscaled route must be byte-identical in and across processes."""

    SCRIPT = (
        "import sys, cv2\n"
        "from signature_verification_system.src.verification.signature_compare import compare_signatures\n"
        "ref = cv2.imread(sys.argv[1]); q = cv2.imread(sys.argv[2])\n"
        "sys.stdout.write(compare_signatures([ref], q).model_dump_json())\n"
    )

    def setUp(self):
        self.reference = cv2.cvtColor(signature(textured=False, width=640, seed=1)[0], cv2.COLOR_GRAY2BGR)
        self.questioned = cv2.cvtColor(signature(textured=True, width=208, seed=2)[0], cv2.COLOR_GRAY2BGR)

    def test_repeated_runs_identical(self):
        from signature_verification_system.src.verification.signature_compare import compare_signatures

        outputs = {compare_signatures([self.reference], self.questioned).model_dump_json() for _ in range(20)}
        self.assertEqual(len(outputs), 1)

    def test_fresh_interpreters_match_in_process_result(self):
        import os
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        import signature_verification_system
        from signature_verification_system.src.verification.signature_compare import compare_signatures

        expected = compare_signatures([self.reference], self.questioned).model_dump_json()
        package_dir = Path(list(signature_verification_system.__path__)[0])
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(package_dir.parent), str(package_dir.resolve().parent)])}
        with tempfile.TemporaryDirectory() as tmp:
            ref_path, q_path = Path(tmp) / "ref.png", Path(tmp) / "q.png"
            cv2.imwrite(str(ref_path), self.reference)
            cv2.imwrite(str(q_path), self.questioned)
            for _ in range(2):
                out = subprocess.run([sys.executable, "-c", self.SCRIPT, str(ref_path), str(q_path)],
                                     env=env, capture_output=True, text=True, check=True).stdout
                self.assertEqual(out, expected)


if __name__ == "__main__":
    unittest.main()
