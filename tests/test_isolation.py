"""Tests for signature-ink isolation from printed rules and text (EXP-016).

All images are synthetic and generated in-process (no dataset, no network).
They check behaviour: printed context is removed, every part of the
handwriting (detached dots, separate names, straight underline flourishes)
is kept, and clean inputs pass through untouched.
"""

import unittest

import cv2
import numpy as np

from signature_verification_system.src.preprocessing.isolation import (
    isolate_signature_ink,
    remove_horizontal_rules,
)
from signature_verification_system.src.preprocessing.normalization import (
    INK_MASK_LEVEL,
    harmonize_photometric,
    normalize_signature,
)

INK = (40, 40, 40)
SIG_H, SIG_W = 260, 520


def _signature(canvas: np.ndarray, x0: int, y0: int, underline: bool = False) -> None:
    """Draw a deterministic cursive-like signature with detached parts."""
    xs = np.arange(0, 300)
    ys = 70 + 45 * np.sin(xs / 17.0) + 0.1 * xs
    first = np.stack([x0 + 20 + xs, y0 + ys], axis=1).astype(np.int32)
    cv2.polylines(canvas, [first], False, INK, 3, cv2.LINE_AA)
    cv2.circle(canvas, (x0 + 150, y0 + 15), 4, INK, -1, cv2.LINE_AA)               # i-dot
    cv2.ellipse(canvas, (x0 + 410, y0 + 110), (45, 60), 20, 0, 330, INK, 3, cv2.LINE_AA)  # surname
    if underline:
        # Straight flourish continuing the last stroke, as in many CEDAR samples.
        cv2.line(canvas, (x0 + 318, y0 + 130), (x0 + 330, y0 + 205), INK, 3, cv2.LINE_AA)
        cv2.line(canvas, (x0 + 330, y0 + 205), (x0 + 500, y0 + 205), INK, 3, cv2.LINE_AA)


def _clean(underline: bool = False) -> np.ndarray:
    img = np.full((SIG_H, SIG_W, 3), 255, np.uint8)
    _signature(img, 0, 20, underline)
    return img


def _darkness(img: np.ndarray) -> np.ndarray:
    return (255.0 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)) / 255.0


def _harmonized_darkness(img: np.ndarray) -> np.ndarray:
    return (255.0 - harmonize_photometric(img).astype(np.float64)) / 255.0


def _cheque_context(sig: np.ndarray) -> np.ndarray:
    """Signature on a page with a printed caption and a form rule far above it."""
    h, w = sig.shape[:2]
    page = np.full((2 * h, 2 * w, 3), 255, np.uint8)
    page[h:, w // 2:w // 2 + w] = sig
    cv2.putText(page, "PAY TO THE ORDER OF", (40, h // 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, INK, 2, cv2.LINE_AA)
    cv2.line(page, (40, h // 2), (2 * w - 40, h // 2), (60, 60, 60), 2)
    return page


def _ruled(sig: np.ndarray) -> np.ndarray:
    out = sig.astype(np.float64)
    for y in range(18, sig.shape[0], 36):
        out[y:y + 2] *= np.array([1.0, 0.80, 0.70])
    return np.clip(out, 0, 255).astype(np.uint8)


class TestIsolationKeepsHandwriting(unittest.TestCase):
    def test_clean_signature_passes_through_unchanged(self):
        darkness = _darkness(_clean())
        self.assertIs(isolate_signature_ink(darkness, INK_MASK_LEVEL), darkness)

    def test_straight_underline_flourish_is_kept(self):
        darkness = _darkness(_clean(underline=True))
        self.assertIs(isolate_signature_ink(darkness, INK_MASK_LEVEL), darkness)

    def test_rule_touching_image_edge_needs_full_length(self):
        darkness = np.zeros((60, 400))
        darkness[30:33, 0:150] = 0.8           # stroke from the edge, < half the width
        darkness[5:50, 140:146] = 0.8          # joined to handwriting
        self.assertIs(remove_horizontal_rules(darkness, INK_MASK_LEVEL), darkness)


class TestIsolationRemovesPrint(unittest.TestCase):
    def test_caption_and_form_rule_do_not_set_the_crop(self):
        clean = normalize_signature(_clean())
        page = normalize_signature(_cheque_context(_clean()))
        self.assertLessEqual(abs(page.bbox[2] - clean.bbox[2]), 3)
        self.assertLessEqual(abs(page.bbox[3] - clean.bbox[3]), 3)
        self.assertGreaterEqual(page.bbox[1], SIG_H)     # crop starts in the signature half

    def test_ruled_lines_are_removed_from_the_ink_map(self):
        clean = _harmonized_darkness(_clean())
        ruled = _harmonized_darkness(_ruled(_clean()))
        before = float(np.abs(ruled - clean).mean())
        after = float(np.abs(isolate_signature_ink(ruled, INK_MASK_LEVEL) - clean).mean())
        self.assertLess(after, 0.1 * before)    # measured: 0.0056 -> 0.0004

    def test_faint_rule_crossed_by_strokes_is_removed(self):
        darkness = np.zeros((80, 400))
        darkness[40:42, :] = 0.12              # ruled line, fainter than ink
        darkness[10:70, 100:104] = 0.8         # strokes crossing it
        darkness[10:70, 250:254] = 0.8
        cleaned = remove_horizontal_rules(darkness, INK_MASK_LEVEL)
        self.assertEqual(float(cleaned[40:42, :90].max()), 0.0)
        np.testing.assert_array_equal(cleaned[10:70, 100:104] > INK_MASK_LEVEL, True)

    def test_all_text_input_is_not_emptied(self):
        page = np.full((200, 700, 3), 255, np.uint8)
        cv2.putText(page, "PAY TO THE ORDER OF", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, INK, 2, cv2.LINE_AA)
        self.assertGreater(normalize_signature(page).ink_pixel_count, 100)

    def test_lone_straight_stroke_is_not_emptied(self):
        page = np.full((200, 600, 3), 255, np.uint8)
        cv2.line(page, (30, 100), (570, 100), INK, 4)
        self.assertGreater(normalize_signature(page).ink_pixel_count, 100)

    def test_isolation_is_deterministic(self):
        darkness = _darkness(_cheque_context(_clean()))
        a = isolate_signature_ink(darkness, INK_MASK_LEVEL)
        b = isolate_signature_ink(darkness.copy(), INK_MASK_LEVEL)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, darkness))


if __name__ == "__main__":
    unittest.main()
