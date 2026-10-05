"""Hermetic tests for the live-console endpoints: /signature/inspect, /samples and the static UI.

Synthetic signatures only (no dataset needed); the sample gallery is pointed at a
temporary directory.
"""

import base64
import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from signature_verification_system.src.adjudication.audit_logger import AuditLogger
from signature_verification_system.src.api.app import create_app
from signature_verification_system.src.api.samples import load_catalog
from signature_verification_system.src.core.config import FUSION_SIGNALS

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def synthetic_signature(seed: int, width: int = 420, height: int = 180) -> np.ndarray:
    """Deterministic cursive-like strokes on slightly textured paper (BGR)."""
    rng = np.random.default_rng(seed)
    paper = np.full((height, width), 238, np.uint8)
    paper = cv2.add(paper, rng.integers(0, 12, (height, width), dtype=np.uint8))
    t = np.linspace(0.0, 1.0, 400)
    phase = rng.uniform(0, np.pi, 3)
    xs = 30 + t * (width - 60)
    ys = height / 2 + 35 * np.sin(2 * np.pi * (3 * t) + phase[0]) + 12 * np.sin(2 * np.pi * 9 * t + phase[1])
    pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(paper, [pts], False, 25, 3, cv2.LINE_AA)
    loop_x = int(width * (0.3 + 0.1 * np.cos(phase[2])))
    cv2.ellipse(paper, (loop_x, height // 2), (22, 40), 20, 0, 330, 25, 3, cv2.LINE_AA)
    cv2.line(paper, (40, height - 40), (width - 50, height - 48), 25, 2, cv2.LINE_AA)
    return cv2.cvtColor(paper, cv2.COLOR_GRAY2BGR)


def to_b64(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _logger(directory: str, name: str) -> AuditLogger:
    db = Path(directory) / f"{name}.db"
    return AuditLogger(str(db), str(db.with_suffix(".jsonl")))


def strip_timing(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k != "timing"}


class InspectEndpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        logger = _logger(cls.tmp.name, "audit")
        cls.client = TestClient(create_app(audit_logger=logger, samples=None))
        cls.ref = to_b64(synthetic_signature(1))
        cls.que = to_b64(synthetic_signature(1))   # identical writer pattern
        cls.other = to_b64(synthetic_signature(7))
        cls.body = {"reference_images": [cls.ref], "questioned_image": cls.other}
        cls.response = cls.client.post("/api/v1/signature/inspect", json=cls.body)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_schema_and_scored(self):
        self.assertEqual(self.response.status_code, 200, self.response.text)
        data = self.response.json()
        for key in ("comparison", "reference", "questioned", "alignment", "fusion", "margin",
                    "thresholds", "consistency", "timing"):
            self.assertIn(key, data)
        # The synthetic pair must actually be scored, or the checks below are vacuous.
        self.assertNotEqual(data["comparison"]["band"], "INCONCLUSIVE")
        self.assertTrue(data["consistency"]["log_odds_matches"])
        self.assertTrue(data["consistency"]["contributions_sum_matches"])
        self.assertIsNotNone(data["fusion"])
        self.assertIn(data["alignment"]["transform_source"], ("keypoint_ransac", "none"))
        self.assertGreater(data["timing"]["total_ms"], 0.0)
        self.assertEqual(data["thresholds"]["accept_log_odds"], data["comparison"]["accept_threshold"])

    def test_contributions_compose_log_odds(self):
        fusion = self.response.json()["fusion"]
        fused = [s for s in fusion["signals"] if s["fused"]]
        self.assertEqual([s["name"] for s in fused], list(FUSION_SIGNALS))
        for s in fused:
            self.assertAlmostEqual(s["contribution"], s["weight"] * s["value"], places=4)
        self.assertAlmostEqual(fusion["contributions_total"], sum(s["contribution"] for s in fused), places=4)
        self.assertAlmostEqual(fusion["bias"] + fusion["contributions_total"], fusion["log_odds"], places=5)
        self.assertAlmostEqual(fusion["log_odds"], self.response.json()["comparison"]["log_odds"], places=6)
        for s in fusion["signals"]:
            if not s["fused"]:
                self.assertIsNone(s["weight"])
                self.assertIsNone(s["contribution"])

    def test_visuals_are_valid_pngs(self):
        data = self.response.json()
        images = [data["alignment"]["overlay_png"]]
        for side in ("reference", "questioned"):
            images += [data[side][k] for k in ("original_png", "harmonised_png", "ink_crop_png", "strokes_png")]
        for b64 in images:
            raw = base64.b64decode(b64)
            self.assertTrue(raw.startswith(PNG_MAGIC))
            decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
            self.assertIsNotNone(decoded)
            self.assertLessEqual(max(decoded.shape[:2]), 640)

    def test_stroke_stats_report_off_stroke_keypoints(self):
        for side in ("reference", "questioned"):
            self.assertIsInstance(self.response.json()[side]["stats"]["keypoints_off_stroke"], int)

    def test_identical_to_compare_endpoint(self):
        for questioned in (self.que, self.other):
            body = {"reference_images": [self.ref], "questioned_image": questioned}
            inspected = self.client.post("/api/v1/signature/inspect", json=body).json()["comparison"]
            compared = self.client.post("/api/v1/signature/compare", json=body).json()
            self.assertEqual(inspected, compared)

    def test_deterministic_across_calls(self):
        first = strip_timing(self.response.json())
        for _ in range(2):
            again = self.client.post("/api/v1/signature/inspect", json=self.body).json()
            self.assertEqual(json.dumps(strip_timing(again), sort_keys=True), json.dumps(first, sort_keys=True))

    def test_multipart_matches_json(self):
        files = [("reference_images", ("r.png", base64.b64decode(self.ref), "image/png")),
                 ("questioned_image", ("q.png", base64.b64decode(self.other), "image/png"))]
        r = self.client.post("/api/v1/signature/inspect", files=files)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(strip_timing(r.json()), strip_timing(self.response.json()))

    def test_blank_image_is_inconclusive_without_breakdown(self):
        blank = to_b64(np.full((180, 420, 3), 245, np.uint8))
        r = self.client.post("/api/v1/signature/inspect",
                             json={"reference_images": [self.ref], "questioned_image": blank})
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["comparison"]["band"], "INCONCLUSIVE")
        self.assertIsNone(data["fusion"])
        self.assertIsNone(data["margin"])
        self.assertIsNone(data["alignment"])
        self.assertTrue(base64.b64decode(data["questioned"]["original_png"]).startswith(PNG_MAGIC))

    def test_rejects_multiple_or_missing_references(self):
        r = self.client.post("/api/v1/signature/inspect",
                             json={"reference_images": [self.ref, self.ref], "questioned_image": self.que})
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/api/v1/signature/inspect", json={"reference_images": [], "questioned_image": self.que})
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/api/v1/signature/inspect",
                             json={"reference_images": ["bm90IGFuIGltYWdl"], "questioned_image": self.que})
        self.assertEqual(r.status_code, 400)

    def test_ui_served_and_api_not_shadowed(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/html", r.headers["content-type"])
        self.assertEqual(self.client.get("/health").status_code, 200)


class SamplesEndpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name) / "samples"
        genuine, skilled = root / "genuine_pairs", root / "skilled_forgeries"
        genuine.mkdir(parents=True)
        skilled.mkdir()
        for directory, names in ((genuine, ("a.png", "b.png")), (skilled, ("a.png", "f.png"))):
            for name in names:
                cv2.imwrite(str(directory / name), synthetic_signature(len(name) + len(str(directory))))
        (Path(cls.tmp.name) / "secret.png").write_bytes(b"\x89PNG\r\n\x1a\nSECRET")
        (genuine / "manifest.json").write_text(json.dumps([
            {"author": "w1", "ref_image": "a.png", "questioned_image": "b.png"},
            {"author": "w1", "ref_image": "../../secret.png", "questioned_image": "b.png"},
            {"author": "w1", "ref_image": "/etc/passwd", "questioned_image": "b.png"},
            {"author": "w1", "ref_image": "missing.png", "questioned_image": "b.png"},
            "not-an-object",
        ]))
        (skilled / "manifest.json").write_text(json.dumps([
            {"target_author": "w1", "ref_image": "a.png", "questioned_image": "f.png"},
        ]))
        cls.root = root
        logger = _logger(cls.tmp.name, "audit")
        cls.client = TestClient(create_app(audit_logger=logger, samples=load_catalog(str(root))))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_lists_only_safe_pairs_with_labels(self):
        r = self.client.get("/api/v1/samples")
        self.assertEqual(r.status_code, 200)
        samples = r.json()["samples"]
        self.assertEqual([s["dataset_label"] for s in samples], ["Genuine", "Skilled forgery"])
        self.assertNotIn(str(self.root), r.text)
        image = self.client.get(f"/api/v1/samples/{samples[0]['reference_image_id']}")
        self.assertEqual(image.status_code, 200)
        self.assertTrue(image.content.startswith(PNG_MAGIC))
        self.assertNotIn(b"SECRET", image.content)

    def test_rejects_traversal_and_unknown_ids(self):
        for bad in ("..%2F..%2Fsecret.png", "..", "-1", "1e3", "999", "0x1", "%2Fetc%2Fpasswd", "a.png", " 1"):
            r = self.client.get(f"/api/v1/samples/{bad}")
            self.assertIn(r.status_code, (404, 405), bad)
            self.assertNotIn(b"SECRET", r.content)
        r = self.client.get("/api/v1/samples/../../secret.png")
        self.assertNotIn(b"SECRET", r.content)

    def test_disabled_when_env_unset(self):
        with unittest.mock.patch.dict("os.environ", {}, clear=False) as env:
            env.pop("SIGVERIFY_SAMPLES_DIR", None)
            client = TestClient(create_app(audit_logger=_logger(self.tmp.name, "a2")))
        self.assertEqual(client.get("/api/v1/samples").status_code, 404)
        self.assertEqual(client.get("/api/v1/samples/0").status_code, 404)

    def test_nonexistent_dir_disables(self):
        self.assertIsNone(load_catalog(str(Path(self.tmp.name) / "nope")))
        self.assertIsNone(load_catalog(None))


if __name__ == "__main__":
    unittest.main()
