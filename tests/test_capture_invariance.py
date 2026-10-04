"""Rotation and resolution invariance of the 1:1 comparison (EXP-021).

Synthetic, in-process images only (no dataset, no network). The questioned image
is rotated on an expanded white canvas or re-sampled, exactly like the capture
conditions of benchmark/clearance_score.py.
"""

import unittest

import cv2
import numpy as np

from signature_verification_system.src.core.config import DEFAULT_CONFIG
from signature_verification_system.src.preprocessing.normalization import (
    canonicalize,
    letterbox_transform,
    rotate_ink,
)
from signature_verification_system.src.verification.features import (
    downsampled_features,
    extract_features,
)
from signature_verification_system.src.verification.similarity import compare, derotation_angle, harmonize_scale
from signature_verification_system.tests.test_stroke_geometry import draw_signature

PARAMS = DEFAULT_CONFIG.representation
NO_INVARIANCE = PARAMS.model_copy(update={"derotate_min_deg": 1e9, "scale_band_low": 0.0, "scale_band_high": 1e9})


def rotate_expanded(image: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate on an expanded white canvas so no ink is clipped (as the benchmark does)."""
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(image, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))


def resample(image: np.ndarray, factor: float) -> np.ndarray:
    h, w = image.shape[:2]
    interpolation = cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, (round(w * factor), round(h * factor)), interpolation=interpolation)


class TestGeometryHelpers(unittest.TestCase):
    def test_rotate_ink_matrix_tracks_ink(self):
        ink = np.zeros((60, 200))
        ink[20:24, 150:154] = 1.0          # one blob far from the centre
        ink[30:34, 10:14] = 1.0            # second blob so the crop is non-trivial
        rotated, matrix = rotate_ink(ink, 25.0)
        expected = matrix @ np.array([151.5, 21.5, 1.0])
        ys, xs = np.nonzero(rotated > 0.5)
        near = np.hypot(xs - expected[0], ys - expected[1]) < 4
        self.assertGreater(int(near.sum()), 4)

    def test_rotate_ink_round_trip_keeps_crop_size(self):
        ink = canonicalize(extract_features(draw_signature(3)).normalized.ink, 400, 140, keep_aspect=True)
        back, _ = rotate_ink(rotate_ink(ink, 20.0)[0], -20.0)
        original, _ = rotate_ink(ink, 0.0)
        self.assertLess(abs(back.shape[1] - original.shape[1]), 6)
        self.assertLess(abs(back.shape[0] - original.shape[0]), 6)

    def test_letterbox_transform_matches_canonicalize(self):
        ink = np.zeros((50, 300))
        ink[10:14, 250:254] = 1.0
        canvas = canonicalize(ink, 512, 256, keep_aspect=True)
        expected = letterbox_transform(ink.shape, 512, 256) @ np.array([251.5, 11.5, 1.0])
        ys, xs = np.mgrid[0:canvas.shape[0], 0:canvas.shape[1]]
        x, y = (xs * canvas).sum() / canvas.sum(), (ys * canvas).sum() / canvas.sum()
        self.assertLess(np.hypot(x - expected[0], y - expected[1]), 1.5)


class TestRotationInvariance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specimen = extract_features(draw_signature(5))
        cls.upright = extract_features(draw_signature(5, variation=0.02, variation_seed=1))

    def test_derotation_restores_signals(self):
        reference = compare(self.specimen, self.upright)
        for degrees in (20.0, -15.0):
            rotated = extract_features(rotate_expanded(draw_signature(5, variation=0.02, variation_seed=1), degrees))
            fixed = compare(self.specimen, rotated)
            self.assertIsNotNone(derotation_angle(fixed.keypoint))
            plain = compare(self.specimen, rotated, params=NO_INVARIANCE)
            self.assertGreater(fixed.fused_logit, plain.fused_logit)
            self.assertLess(abs(fixed.signals["slant"] - reference.signals["slant"]), 0.04)
            self.assertLess(abs(fixed.fused_logit - reference.fused_logit), abs(plain.fused_logit - reference.fused_logit))

    def test_derotation_angle_sign_matches_capture_rotation(self):
        rotated = extract_features(rotate_expanded(draw_signature(5), 20.0))
        angle = derotation_angle(compare(self.specimen, rotated).keypoint)
        self.assertIsNotNone(angle)
        self.assertAlmostEqual(angle, -20.0, delta=3.0)   # rotate_ink(ink, -20) undoes a +20 capture

    def test_small_rotation_is_left_alone(self):
        rotated = extract_features(rotate_expanded(draw_signature(5), 3.0))
        self.assertIsNone(derotation_angle(compare(self.specimen, rotated).keypoint))

    def test_rotated_comparison_is_deterministic(self):
        rotated = extract_features(rotate_expanded(draw_signature(5), 20.0))
        self.assertEqual(compare(self.specimen, rotated).fused_logit, compare(self.specimen, rotated).fused_logit)


class TestScaleHarmonisation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = draw_signature(9)
        cls.specimen = extract_features(cls.image)

    def test_radius_of_gyration_scales_with_image(self):
        half = extract_features(resample(self.image, 0.5))
        self.assertAlmostEqual(half.ink_radius / self.specimen.ink_radius, 0.5, delta=0.03)

    def test_pairs_inside_band_are_untouched(self):
        similar = extract_features(resample(self.image, 0.9))
        a, b = harmonize_scale(self.specimen, similar)
        self.assertIs(a, self.specimen)
        self.assertIs(b, similar)

    def test_larger_image_is_downsampled_to_match(self):
        for factor in (0.5, 2.0):
            query = extract_features(resample(self.image, factor))
            a, b = harmonize_scale(self.specimen, query)
            larger, untouched = (a, b) if factor < 1 else (b, a)
            self.assertIs(untouched, query if factor < 1 else self.specimen)
            self.assertAlmostEqual(larger.ink_radius / untouched.ink_radius, 1.0, delta=0.05)

    def test_downsampling_never_upsamples(self):
        self.assertIs(downsampled_features(self.specimen, 1.5), self.specimen)

    def test_unreadable_downsample_keeps_native_features(self):
        speck = np.full((400, 400, 3), 255, np.uint8)
        cv2.circle(speck, (200, 200), 3, (0, 0, 0), -1)
        cv2.circle(speck, (230, 200), 3, (0, 0, 0), -1)
        features = extract_features(speck)
        self.assertIs(downsampled_features(features, 0.02), features)

    def test_harmonised_stroke_width_matches(self):
        for factor in (0.5, 2.0):
            query = extract_features(resample(self.image, factor))
            on = compare(self.specimen, query).signals["stroke_width"]
            off = compare(self.specimen, query, params=NO_INVARIANCE).signals["stroke_width"]
            self.assertGreater(on, off)
            self.assertGreater(on, 0.9)


if __name__ == "__main__":
    unittest.main()
