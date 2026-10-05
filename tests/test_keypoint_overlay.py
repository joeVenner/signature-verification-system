"""Regression: keypoint markers in the "Stroke representation" PNG sit on the strokes.

The markers are located by decoding the rendered PNG (not by re-running the
selection), then measured against ink computed from the normalised ink map, so
the test checks what the console actually displays. Synthetic images only.
"""

import base64
import unittest

import cv2
import numpy as np

from signature_verification_system.src.api import visuals
from signature_verification_system.src.api.inspection import inspect_image
from signature_verification_system.src.core.config import DEFAULT_CONFIG
from signature_verification_system.src.preprocessing.normalization import canonicalize
from signature_verification_system.src.verification.features import extract_features

# Drawn centres are <= 3 px (the ring radius) from ink by construction; overlapping
# rings decode as one component whose centroid can land ~1 px further out.
MAX_MARKER_TO_INK_PX = 4.0
RING_CORE_LEVEL = 180       # template pixels at least this bright are drawn solid, not anti-aliased
RED_MARGIN = 40             # keypoint red exceeds green / blue by far more than any grey pixel


def _strokes(paper: np.ndarray, ink_level: int, seed: int) -> np.ndarray:
    """Cursive-like strokes with closed loops (where SIFT blob centres land) on `paper`."""
    rng = np.random.default_rng(seed)
    height, width = paper.shape
    t = np.linspace(0.0, 1.0, 500)
    xs = 30 + t * (width - 60)
    ys = height / 2 + 30 * np.sin(2 * np.pi * 4 * t + rng.uniform(0, np.pi))
    pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(paper, [pts], False, ink_level, 4, cv2.LINE_AA)
    for cx in (0.25, 0.5, 0.75):
        cv2.ellipse(paper, (int(width * cx), height // 3), (24, 30), 0, 0, 360, ink_level, 4, cv2.LINE_AA)
    cv2.line(paper, (40, height // 5), (width - 40, height // 5 + 6), ink_level, 3, cv2.LINE_AA)
    return cv2.cvtColor(paper, cv2.COLOR_GRAY2BGR)


def clean_signature(seed: int = 3) -> np.ndarray:
    return _strokes(np.full((220, 520), 245, np.uint8), 20, seed)


def dark_background_signature(seed: int = 3) -> np.ndarray:
    """Dark, textured paper (noise plus low-frequency blotches) with near-black ink."""
    rng = np.random.default_rng(seed)
    shape = (220, 520)
    texture = 75 + rng.normal(0, 10, shape) + cv2.GaussianBlur(rng.normal(0, 40, shape), (0, 0), 6)
    return _strokes(np.clip(texture, 0, 255).astype(np.uint8), 5, seed)


def marker_centres(png_b64: str) -> np.ndarray:
    """(x, y) centres of the keypoint rings drawn in a strokes PNG."""
    img = cv2.imdecode(np.frombuffer(base64.b64decode(png_b64), np.uint8), cv2.IMREAD_COLOR).astype(np.int64)
    blue, green, red = img[..., 0], img[..., 1], img[..., 2]
    red_mask = ((red - green > RED_MARGIN) & (red - blue > RED_MARGIN)).astype(np.uint8)
    size = 2 * visuals.KEYPOINT_RING_RADIUS + 3
    centre = size // 2
    template = np.zeros((size, size), np.uint8)
    cv2.circle(template, (centre, centre), visuals.KEYPOINT_RING_RADIUS, 255, 1, cv2.LINE_AA)
    # A pixel survives erosion only if the whole solid ring around it is red.
    hits = cv2.erode(red_mask, (template >= RING_CORE_LEVEL).astype(np.uint8), anchor=(centre, centre),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0)
    count, _, _, centroids = cv2.connectedComponentsWithStats(hits)
    return centroids[1:count]


class KeypointOverlayPlacementTest(unittest.TestCase):
    params = DEFAULT_CONFIG.representation

    def _check(self, image: np.ndarray) -> None:
        p = self.params
        feats = extract_features(image, p)
        view = inspect_image(image, feats, None, p)
        centres = marker_centres(view.strokes_png)

        stroke = canonicalize(feats.normalized.ink, p.keypoint_canvas_width, p.keypoint_canvas_height,
                              keep_aspect=True) > p.stroke.ink_threshold
        distance = cv2.distanceTransform((~stroke).astype(np.uint8), cv2.DIST_L2, 5)
        xy = np.round(centres).astype(np.int64)
        to_ink = distance[xy[:, 1], xy[:, 0]]

        self.assertGreater(len(centres), 5, "no keypoint markers found in the rendered PNG")
        self.assertLessEqual(float(to_ink.max()), MAX_MARKER_TO_INK_PX,
                             f"markers off-stroke at {xy[to_ink > MAX_MARKER_TO_INK_PX].tolist()}")
        self.assertEqual(view.stats.keypoints, len(feats.keypoints))
        self.assertGreaterEqual(view.stats.keypoints_off_stroke, 0)
        self.assertLess(view.stats.keypoints_off_stroke, view.stats.keypoints)

    def test_markers_on_strokes_clean_background(self):
        self._check(clean_signature())

    def test_markers_on_strokes_dark_textured_background(self):
        self._check(dark_background_signature())

    def test_keypoints_used_for_matching_are_unchanged(self):
        image = clean_signature()
        before = extract_features(image, self.params).keypoints.copy()
        inspect_image(image, extract_features(image, self.params), None, self.params)
        self.assertTrue(np.array_equal(extract_features(image, self.params).keypoints, before))


class OnStrokeKeypointsTest(unittest.TestCase):
    def test_distance_threshold(self):
        ink = np.zeros((40, 80))
        ink[18:22, :] = 1.0                       # horizontal stroke, fills the 80x40 canvas exactly
        points = np.array([[40, 20], [40, 23], [40, 24], [40, 26]], np.float32)
        mask = visuals.on_stroke_keypoints(ink, points, 80, 40, 0.35, max_distance_px=3.0)
        self.assertEqual(mask.tolist(), [True, True, True, False])

    def test_empty_and_blank_inputs(self):
        self.assertEqual(len(visuals.on_stroke_keypoints(np.ones((4, 4)), None, 8, 8, 0.35)), 0)
        blank = visuals.on_stroke_keypoints(np.zeros((4, 4)), np.array([[1.0, 1.0]]), 8, 8, 0.35)
        self.assertEqual(blank.tolist(), [False])


if __name__ == "__main__":
    unittest.main()
