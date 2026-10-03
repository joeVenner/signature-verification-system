"""Determinism contract: the same image pair through the public 1:1 comparison
must yield byte-identical output on every one of 100 runs.

All images are synthetic and built with fixed parameters (no dataset, no network).
Half of the runs use a fresh DeterministicVerifier, half a shared one, to expose
hidden state in either construction path.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import numpy as np

import signature_verification_system
from signature_verification_system.src.verification.deterministic import DeterministicVerifier
from signature_verification_system.src.verification.signature_compare import compare_signatures

RUNS = 100
CREAM_BGR = (200, 235, 250)


def draw_signature(writer_id: int, perturbation: float = 0.0, angle: float = 0.0) -> np.ndarray:
    """Deterministic multi-stroke signature (BGR uint8, 900x300, dark ink on white).

    The rng only builds the fixture; `perturbation` shifts stroke geometry by a
    fixed fraction, `angle` rotates the whole canvas in degrees.
    """
    rng = np.random.default_rng(writer_id)
    gray = np.full((300, 900), 255, np.uint8)
    for stroke in range(5):
        x_start = 80 + stroke * 150 + int(rng.integers(-15, 15))
        amplitude = float(rng.uniform(30, 90)) * (1 + perturbation)
        frequency = float(rng.uniform(0.05, 0.14)) * (1 - perturbation / 2)
        slant = float(rng.uniform(-0.8, 0.8)) + perturbation
        phase = float(rng.uniform(0, 6.28))
        x = np.linspace(0, 170, 80)
        y = 150 + amplitude * np.sin(frequency * x + phase) + slant * (x - 85)
        points = np.stack([x_start + x, y], axis=1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(gray, [points], False, 20, 3, cv2.LINE_AA)
    if angle:
        rotation = cv2.getRotationMatrix2D((450, 150), angle, 1.0)
        gray = cv2.warpAffine(gray, rotation, (900, 300), borderValue=255)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def tint_and_resize(image: np.ndarray, scale: float) -> np.ndarray:
    """Resize and multiply onto a cream-coloured paper tone (ink stays dark)."""
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    paper = np.array(CREAM_BGR, np.float32) / 255.0
    return np.clip(resized.astype(np.float32) * paper, 0, 255).astype(np.uint8)


def genuine_pair() -> Tuple[np.ndarray, np.ndarray]:
    return draw_signature(7), draw_signature(7, perturbation=0.04, angle=2.0)


def different_pair() -> Tuple[np.ndarray, np.ndarray]:
    return draw_signature(7), draw_signature(31)


def tinted_resized_pair() -> Tuple[np.ndarray, np.ndarray]:
    return draw_signature(7), tint_and_resize(draw_signature(7, perturbation=0.03), 1.5)


class TestRepeatedComparisonIsIdentical(unittest.TestCase):
    def _assert_identical(self, make_pair: Callable[[], Tuple[np.ndarray, np.ndarray]]) -> None:
        reference, questioned = make_pair()
        shared = DeterministicVerifier()
        results = []
        for run in range(RUNS):
            verifier = shared if run % 2 else DeterministicVerifier()
            results.append(compare_signatures([reference], questioned, verifier=verifier))
        serialized = [r.model_dump_json() for r in results]
        self.assertEqual(len(serialized), RUNS)
        self.assertEqual(len(set(serialized)), 1)
        self.assertEqual(len({r.match_score for r in results}), 1)
        self.assertEqual(len({r.log_odds for r in results}), 1)
        self.assertEqual(len({r.verdict for r in results}), 1)

    def test_genuine_like_pair_100_runs(self) -> None:
        self._assert_identical(genuine_pair)

    def test_different_signature_pair_100_runs(self) -> None:
        self._assert_identical(different_pair)

    def test_tinted_background_resized_pair_100_runs(self) -> None:
        self._assert_identical(tinted_resized_pair)


class TestCrossProcessRepeatability(unittest.TestCase):
    SCRIPT = (
        "import sys, cv2\n"
        "from signature_verification_system.src.verification.signature_compare import compare_signatures\n"
        "ref = cv2.imread(sys.argv[1]); q = cv2.imread(sys.argv[2])\n"
        "sys.stdout.write(compare_signatures([ref], q).model_dump_json())\n"
    )

    def test_three_fresh_interpreters_match_in_process_result(self) -> None:
        reference, questioned = genuine_pair()
        expected = compare_signatures([reference], questioned).model_dump_json()
        # The package may be a namespace package (no __file__); use its path entry.
        package_dir = Path(list(signature_verification_system.__path__)[0])
        unresolved_parent = str(package_dir.parent)
        package_parent = str(package_dir.resolve().parent)
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([unresolved_parent, package_parent])}
        with tempfile.TemporaryDirectory() as tmp:
            ref_path, q_path = Path(tmp) / "ref.png", Path(tmp) / "q.png"
            cv2.imwrite(str(ref_path), reference)  # PNG is lossless
            cv2.imwrite(str(q_path), questioned)
            outputs: List[str] = [
                subprocess.run(
                    [sys.executable, "-c", self.SCRIPT, str(ref_path), str(q_path)],
                    env=env, capture_output=True, text=True, check=True,
                ).stdout
                for _ in range(3)
            ]
        for output in outputs:
            self.assertEqual(json.loads(output), json.loads(expected))
            self.assertEqual(output, expected)


if __name__ == "__main__":
    unittest.main()
