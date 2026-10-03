"""Tests for the stroke-level signals and the fusion that uses them (EXP-015).

All images are synthetic and generated in-process, so these tests need no dataset
and no network. They check behaviour (alignment, ordering, guards, determinism),
not tuned accuracy numbers: accuracy is measured by benchmark/run_benchmark.py.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from signature_verification_system.src.core.config import (
    DEFAULT_CONFIG,
    FUSION_SIGNALS,
    FusionModel,
    StrokeParams,
)
from signature_verification_system.src.preprocessing.normalization import canonicalize
from signature_verification_system.src.verification.features import extract_features
from signature_verification_system.src.verification.similarity import (
    compare,
    compare_multi,
    fuse_signals,
)
from signature_verification_system.src.verification.stroke_geometry import (
    MAX_SKELETON_POINTS,
    SLANT_BINS,
    SLANT_MAX_EMD,
    StrokeGeometry,
    _banded_dtw,
    _circular_emd,
    _similarity_fit,
    direction_signal,
    extract_stroke_geometry,
    stroke_signals,
)

PARAMS = DEFAULT_CONFIG.representation.stroke
CANVAS = (DEFAULT_CONFIG.representation.keypoint_canvas_width, DEFAULT_CONFIG.representation.keypoint_canvas_height)


def draw_signature(seed: int, variation: float = 0.0, variation_seed: int = 0, angle: float = 0.0) -> np.ndarray:
    """Seeded synthetic signature (BGR uint8, 900x300, dark ink on white).

    `seed` fixes the "writer" (stroke count, slants, amplitudes); `variation`
    perturbs those parameters the way a writer varies between signings.
    """
    writer = np.random.default_rng(seed)
    wobble = np.random.default_rng(1_000_003 * (variation_seed + 1) + seed)
    gray = np.full((300, 900), 255, np.uint8)
    for stroke in range(5):
        x_start = 80 + stroke * 150 + int(writer.integers(-15, 15))
        amplitude = float(writer.uniform(30, 90))
        frequency = float(writer.uniform(0.05, 0.14))
        slant = float(writer.uniform(-0.8, 0.8))
        phase = float(writer.uniform(0, 6.28))
        amplitude *= 1 + variation * wobble.normal()
        frequency *= 1 + variation * wobble.normal()
        slant += variation * wobble.normal()
        x = np.linspace(0, 170, 80)
        y = 150 + amplitude * np.sin(frequency * x + phase) + slant * (x - 85)
        points = np.stack([x_start + x, y], axis=1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(gray, [points], False, 20, 3, cv2.LINE_AA)
    if angle:
        rotation = cv2.getRotationMatrix2D((450, 150), angle, 1.0)
        gray = cv2.warpAffine(gray, rotation, (900, 300), borderValue=255)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def geometry_of(image: np.ndarray) -> StrokeGeometry:
    return extract_features(image).stroke


def signals(specimen: np.ndarray, questioned: np.ndarray):
    a, b = extract_features(specimen), extract_features(questioned)
    pair = compare(a, b)
    seed = pair.keypoint.transform if pair.keypoint.inliers >= DEFAULT_CONFIG.representation.align_min_inliers else None
    return stroke_signals(a.stroke, b.stroke, seed, PARAMS)


class TestStrokeGeometryExtraction(unittest.TestCase):
    def test_descriptor_shapes_and_normalisation(self):
        g = geometry_of(draw_signature(1))
        self.assertEqual(g.skeleton_points.shape[1], 2)
        self.assertEqual(len(g.orientations), len(g.skeleton_points))
        self.assertGreater(len(g.skeleton_points), 100)
        self.assertTrue(np.all((g.orientations >= 0) & (g.orientations < np.pi)))
        self.assertAlmostEqual(float(g.column_profile.sum()), 1.0, places=6)
        self.assertAlmostEqual(float(g.row_profile.sum()), 1.0, places=6)
        self.assertAlmostEqual(float(g.slant_histogram.sum()), 1.0, places=6)
        self.assertGreater(g.stroke_width, 1.0)

    def test_empty_ink_gives_empty_geometry_without_error(self):
        g = extract_stroke_geometry(np.zeros((40, 80)), PARAMS, *CANVAS)
        self.assertEqual(len(g.skeleton_points), 0)
        self.assertEqual(g.stroke_width, 0.0)

    def test_extraction_is_bit_identical_across_calls(self):
        image = draw_signature(2, variation=0.05, variation_seed=1)
        a, b = geometry_of(image), geometry_of(image)
        for name in ("skeleton_points", "orientations", "column_profile", "row_profile", "slant_histogram"):
            self.assertTrue(np.array_equal(getattr(a, name), getattr(b, name)), name)
        self.assertEqual(a.stroke_width, b.stroke_width)


class TestStrokeSignals(unittest.TestCase):
    def test_identical_signature_scores_near_maximum(self):
        image = draw_signature(3)
        s = signals(image, image)
        self.assertGreater(s.direction_agreement, 0.8)
        self.assertGreater(s.slant, 0.99)
        self.assertGreater(s.column_profile, 0.95)
        self.assertGreater(s.row_profile, 0.95)
        self.assertAlmostEqual(s.stroke_width, 1.0, places=6)

    def test_same_writer_beats_different_writer_on_direction(self):
        specimen = draw_signature(4)
        same = signals(specimen, draw_signature(4, variation=0.03, variation_seed=1))
        other = signals(specimen, draw_signature(5))
        self.assertGreater(same.direction_agreement, other.direction_agreement)

    def test_alignment_tolerates_rotation(self):
        specimen = draw_signature(6)
        rotated = signals(specimen, draw_signature(6, angle=8.0))
        other = signals(specimen, draw_signature(7, angle=8.0))
        self.assertGreater(rotated.direction_agreement, other.direction_agreement)
        self.assertGreater(rotated.direction_agreement, 0.4)

    def test_signals_stay_in_range(self):
        s = signals(draw_signature(8), draw_signature(9))
        for value in (s.slant, s.column_profile, s.row_profile, s.stroke_width):
            self.assertTrue(0.0 <= value <= 1.0, value)
        self.assertTrue(-1.0 <= s.direction_agreement <= 1.0)

    def test_degenerate_geometry_returns_zero_direction(self):
        empty = extract_stroke_geometry(np.zeros((40, 80)), PARAMS, *CANVAS)
        full = geometry_of(draw_signature(10))
        self.assertEqual(direction_signal(empty, full, None, PARAMS), 0.0)
        self.assertEqual(direction_signal(full, empty, None, PARAMS), 0.0)

    def test_unusable_keypoint_transform_is_ignored_safely(self):
        a, b = geometry_of(draw_signature(11)), geometry_of(draw_signature(11, variation=0.02, variation_seed=2))
        without = direction_signal(a, b, None, PARAMS)
        nonsense = np.array([[1.0, 0.0, 900.0], [0.0, 1.0, 900.0]])   # sends everything off-canvas
        with_bad_seed = direction_signal(a, b, nonsense, PARAMS)
        self.assertGreater(with_bad_seed, 0.5 * without)   # identity start still wins

    def test_signals_are_deterministic(self):
        specimen, questioned = draw_signature(12), draw_signature(12, variation=0.04, variation_seed=3)
        self.assertEqual(signals(specimen, questioned), signals(specimen, questioned))


class TestSlantRange(unittest.TestCase):
    """Regression: the slant signal once ran to -8 for strongly different slants."""

    def test_circular_emd_is_bounded_by_its_declared_maximum(self):
        for left, right in ((0, SLANT_BINS // 2), (0, 1), (3, 12), (5, 5)):
            p, q = np.zeros(SLANT_BINS), np.zeros(SLANT_BINS)
            p[left], q[right] = 1.0, 1.0
            emd = _circular_emd(p, q)
            self.assertLessEqual(emd, SLANT_MAX_EMD + 1e-9)
            self.assertGreaterEqual(emd, 0.0)
        opposite_p, opposite_q = np.zeros(SLANT_BINS), np.zeros(SLANT_BINS)
        opposite_p[0], opposite_q[SLANT_BINS // 2] = 1.0, 1.0
        self.assertAlmostEqual(_circular_emd(opposite_p, opposite_q), SLANT_MAX_EMD)

    def test_slant_stays_in_unit_interval_for_extreme_pairs(self):
        horizontal = np.full((200, 600, 3), 255, np.uint8)
        cv2.line(horizontal, (30, 100), (570, 100), (20, 20, 20), 4)
        vertical = np.full((200, 600, 3), 255, np.uint8)
        cv2.line(vertical, (300, 10), (300, 190), (20, 20, 20), 4)
        dot = np.full((200, 600, 3), 255, np.uint8)
        cv2.circle(dot, (300, 100), 6, (20, 20, 20), -1)
        for a, b in ((horizontal, vertical), (vertical, horizontal), (horizontal, dot), (dot, vertical)):
            self.assertTrue(0.0 <= signals(a, b).slant <= 1.0)


class TestAlignmentPrimitives(unittest.TestCase):
    def test_similarity_fit_recovers_a_known_transform(self):
        rng = np.random.default_rng(0)
        src = rng.uniform(0, 100, size=(40, 2))
        angle, scale, shift = np.radians(20.0), 1.3, np.array([12.0, -7.0])
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        dst = scale * src @ rotation.T + shift
        fit_scale, fit_rotation, fit_shift = _similarity_fit(src, dst)
        self.assertAlmostEqual(fit_scale, scale, places=6)
        np.testing.assert_allclose(fit_rotation, rotation, atol=1e-9)
        np.testing.assert_allclose(fit_shift, shift, atol=1e-8)

    def test_similarity_fit_never_returns_a_reflection(self):
        rng = np.random.default_rng(1)
        src = rng.uniform(0, 100, size=(40, 2))
        mirrored = src * np.array([-1.0, 1.0])
        _, rotation, _ = _similarity_fit(src, mirrored)
        self.assertGreater(np.linalg.det(rotation), 0.0)

    def test_dtw_is_zero_for_identical_profiles_and_forgives_a_shift(self):
        x = np.sin(np.linspace(0, 6, 80)) + 1.5
        shifted = np.roll(x, 3)
        self.assertEqual(_banded_dtw(x, x, 8), 0.0)
        self.assertLess(_banded_dtw(x, shifted, 8), float(np.abs(x - shifted).sum() / (2 * len(x))))


class TestRobustness(unittest.TestCase):
    def test_dense_noise_skeleton_is_capped_deterministically(self):
        noise = np.random.default_rng(5).uniform(0.0, 1.0, size=(300, 600))
        first = extract_stroke_geometry(noise, PARAMS, *CANVAS)
        second = extract_stroke_geometry(noise, PARAMS, *CANVAS)
        self.assertLessEqual(len(first.skeleton_points), MAX_SKELETON_POINTS)
        self.assertTrue(np.array_equal(first.skeleton_points, second.skeleton_points))

    def test_fusion_model_rejects_missing_signal_weights(self):
        with self.assertRaises(ValueError):
            FusionModel(signal_weights={"keypoint": 1.0})


class TestFusion(unittest.TestCase):
    def test_default_weights_cover_exactly_the_fusion_signals(self):
        self.assertEqual(set(DEFAULT_CONFIG.fusion.signal_weights), set(FUSION_SIGNALS))
        self.assertTrue(all(w > 0 for w in DEFAULT_CONFIG.fusion.signal_weights.values()),
                        "every fused signal measures similarity, so its weight must be positive")

    def test_threshold_ordering(self):
        d = DEFAULT_CONFIG.decision
        self.assertLess(d.single_hard_reject_logit, d.single_reject_logit)
        self.assertLess(d.single_reject_logit, d.single_accept_logit)
        self.assertLess(d.multi_hard_reject_logit, d.multi_reject_logit)
        self.assertLess(d.multi_reject_logit, d.multi_accept_logit)

    def test_compare_reports_every_signal_and_matches_manual_fusion(self):
        pair = compare(extract_features(draw_signature(13)), extract_features(draw_signature(13, 0.03, 1)))
        self.assertIsNotNone(pair.signals)
        for name in FUSION_SIGNALS:
            self.assertIn(name, pair.signals)
        self.assertEqual(pair.fused_logit, fuse_signals(pair.signals, DEFAULT_CONFIG.fusion))

    def test_genuine_variant_outscores_different_writer(self):
        specimen = extract_features(draw_signature(14))
        genuine = compare(specimen, extract_features(draw_signature(14, 0.03, 1))).fused_logit
        impostor = compare(specimen, extract_features(draw_signature(15))).fused_logit
        self.assertGreater(genuine, impostor)

    def test_custom_fusion_weights_are_honoured(self):
        a, b = extract_features(draw_signature(16)), extract_features(draw_signature(16, 0.03, 1))
        zero = FusionModel(signal_weights={name: 0.0 for name in FUSION_SIGNALS}, bias=1.25)
        self.assertEqual(compare(a, b, zero).fused_logit, 1.25)

    def test_multi_reference_is_order_invariant_and_matches_single_for_one(self):
        specimens = [extract_features(draw_signature(17, 0.03, k)) for k in range(3)]
        query = extract_features(draw_signature(17, 0.03, 9))
        forward = compare_multi(specimens, query).fused_logit
        backward = compare_multi(list(reversed(specimens)), query).fused_logit
        self.assertEqual(forward, backward)
        self.assertEqual(compare_multi(specimens[:1], query).fused_logit, compare(specimens[0], query).fused_logit)

    def test_signalwise_max_is_at_least_every_single_reference(self):
        specimens = [extract_features(draw_signature(18, 0.05, k)) for k in range(3)]
        query = extract_features(draw_signature(18, 0.05, 7))
        multi = compare_multi(specimens, query).fused_logit
        for specimen in specimens:
            self.assertGreaterEqual(multi + 1e-9, compare(specimen, query).fused_logit)

    def test_empty_reference_list_rejected(self):
        with self.assertRaises(ValueError):
            compare_multi([], extract_features(draw_signature(19)))


class TestCrossProcessDeterminism(unittest.TestCase):
    """The same two files give a byte-identical comparison report in fresh interpreters.

    Hermetic counterpart of the dataset-based check in test_pipeline.py: it uses
    synthetic signatures, so it also runs where the sample images are absent.
    """

    SCRIPT = (
        "import cv2, sys\n"
        "from signature_verification_system.src.verification.signature_compare import compare_signatures\n"
        "refs = [cv2.imread(p) for p in sys.argv[2:4]]\n"
        "report = compare_signatures(refs[:int(sys.argv[1])], cv2.imread(sys.argv[4]))\n"
        "sys.stdout.write(report.model_dump_json())\n"
    )

    def _run_three(self, reference_count: int):
        root = Path(__file__).resolve().parent.parent.parent
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for name, image in (("ref_a", draw_signature(20)), ("ref_b", draw_signature(20, 0.03, 1)),
                                ("query", draw_signature(20, 0.03, 2))):
                path = str(Path(tmp) / f"{name}.png")
                cv2.imwrite(path, image)
                paths.append(path)
            return [
                subprocess.run([sys.executable, "-c", self.SCRIPT, str(reference_count), *paths],
                               cwd=root, capture_output=True, text=True, check=True).stdout
                for _ in range(3)
            ]

    def test_single_specimen_report_is_identical_across_processes(self):
        outs = self._run_three(1)
        self.assertTrue(outs[0])
        self.assertEqual(outs[0], outs[1])
        self.assertEqual(outs[1], outs[2])

    def test_two_specimen_report_is_identical_across_processes(self):
        outs = self._run_three(2)
        self.assertTrue(outs[0])
        self.assertEqual(outs[0], outs[1])
        self.assertEqual(outs[1], outs[2])


class TestStrokeParams(unittest.TestCase):
    def test_defaults_are_sane(self):
        p = StrokeParams()
        self.assertGreater(p.icp_iterations, 0)
        self.assertTrue(0.0 < p.icp_trim_quantile <= 1.0)
        self.assertLess(p.icp_min_step_scale, 1.0 < p.icp_max_step_scale)
        self.assertGreaterEqual(p.icp_point_stride, 1)


if __name__ == "__main__":
    unittest.main()
