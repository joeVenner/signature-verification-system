"""Clean-path golden values: cheque-background routing must not touch clean scans.

The hashes and quality signals below were recorded with origin/main (bcd00ac,
before background routing existed) on the synthetic fixtures of
test_determinism_100. Clean images at least LOWRES_MAX_WIDTH wide must give the
same normalised ink map, crop box and quality signals. The values depend on
the pinned numpy / OpenCV builds; re-record them on main if those change.
"""

import hashlib
import unittest

import numpy as np

from signature_verification_system.src.preprocessing.normalization import normalize_signature
from signature_verification_system.src.preprocessing.quality import assess_signature_quality
from signature_verification_system.tests.test_determinism_100 import draw_signature, tint_and_resize

MAIN_KEYS = ("passed", "ink_contrast", "edge_sharpness", "noise_ratio", "ink_pixels", "signature_width",
             "signature_height", "polarity_inverted", "blocking_issues", "warnings")

GOLDEN = {
    "plain": (
        "2a9e6eedf5d93873d97de8def52d61e353280e4628dd71ed6015f9e77cc1d8e9", (91, 25, 767, 243),
        {"blocking_issues": [], "edge_sharpness": 4.044, "ink_contrast": 235.0, "ink_pixels": 18707,
         "noise_ratio": 0.0, "passed": True, "polarity_inverted": False, "signature_height": 247,
         "signature_width": 768, "warnings": []},
    ),
    "tinted_x1.5": (
        "74a1e72e20720bbb12bc0f04903fcff8a53bad135095852ad52d590697137828", (139, 33, 1147, 374),
        {"blocking_issues": [], "edge_sharpness": 3.014, "ink_contrast": 217.0, "ink_pixels": 42477,
         "noise_ratio": 0.0, "passed": True, "polarity_inverted": False, "signature_height": 383,
         "signature_width": 1151, "warnings": []},
    ),
    "tinted_x0.4": (
        "44f11a7bd4e049dd5c80cddf09ff2e84c5f1ad7531736dfc5dfcd05466b84188", (31, 24, 311, 71),
        {"blocking_issues": [], "edge_sharpness": 4.268, "ink_contrast": 217.0, "ink_pixels": 2788,
         "noise_ratio": 0.0, "passed": True, "polarity_inverted": False, "signature_height": 69,
         "signature_width": 309, "warnings": []},
    ),
}


def fixtures() -> dict:
    return {
        "plain": draw_signature(7),
        "tinted_x1.5": tint_and_resize(draw_signature(7, perturbation=0.03), 1.5),
        "tinted_x0.4": tint_and_resize(draw_signature(31), 0.4),   # 360 px wide: still >= LOWRES_MAX_WIDTH
    }


class TestCleanPathMatchesMain(unittest.TestCase):
    def test_normalisation_and_quality_identical_to_main(self):
        for name, image in fixtures().items():
            ink_sha, bbox, quality = GOLDEN[name]
            norm = normalize_signature(image)
            self.assertEqual(hashlib.sha256(np.ascontiguousarray(norm.ink).tobytes()).hexdigest(), ink_sha, name)
            self.assertEqual(norm.bbox, bbox, name)
            q = assess_signature_quality(image).model_dump()
            self.assertEqual({k: q[k] for k in MAIN_KEYS}, quality, name)
            self.assertFalse(q["background_removed"], name)


if __name__ == "__main__":
    unittest.main()
