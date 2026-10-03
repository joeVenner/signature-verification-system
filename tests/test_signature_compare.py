"""Signature-only comparison mode: function, API endpoint and CLI."""

import base64
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import cv2
import numpy as np

from signature_verification_system.src.verification.signature_compare import (
    compare_signatures, decision_aligned_score,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "samples"
REFS = ["genuine_pairs/pair_09_cedar_w05_ref.png", "genuine_pairs/pair_09_cedar_w05_questioned.png",
        "genuine_pairs/pair_10_cedar_w05_ref.png"]
GENUINE = "genuine_pairs/pair_10_cedar_w05_questioned.png"
FORGERY = "skilled_forgeries/pair_09_cedar_w05_forgery01.png"
OTHER = "genuine_pairs/pair_17_cedar_w09_ref.png"


def _img(rel):
    img = cv2.imread(str(DATA / rel))
    if img is None:
        raise unittest.SkipTest(rel)
    return img


class TestCompareFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.refs = [_img(r) for r in REFS]

    def test_genuine_forgery_other(self):
        g = compare_signatures(self.refs, _img(GENUINE))
        f = compare_signatures(self.refs, _img(FORGERY))
        o = compare_signatures(self.refs, _img(OTHER))
        self.assertEqual(g.verdict, "MATCH")
        self.assertEqual(f.verdict, "NO MATCH")
        self.assertEqual(o.verdict, "NO MATCH")
        self.assertGreater(g.match_score, f.match_score)
        self.assertGreater(f.match_score, o.match_score)

    def test_score_never_contradicts_verdict(self):
        for q in (GENUINE, FORGERY, OTHER):
            for refs in (self.refs, self.refs[:1]):
                r = compare_signatures(refs, _img(q))
                if r.band == "ACCEPT":
                    self.assertGreaterEqual(r.match_score, 70.0)
                elif r.band == "REJECT":
                    self.assertLess(r.match_score, 40.0)
                elif r.band == "REVIEW":
                    self.assertTrue(40.0 <= r.match_score < 70.0)

    def test_score_mapping_anchors_and_monotonic(self):
        self.assertEqual(decision_aligned_score(1.0, 1.0, 5.0), 40.0)
        self.assertEqual(decision_aligned_score(5.0, 1.0, 5.0), 70.0)
        self.assertEqual(decision_aligned_score(-100.0, 1.0, 5.0), 0.0)
        self.assertEqual(decision_aligned_score(100.0, 1.0, 5.0), 100.0)
        xs = np.linspace(-10, 15, 200)
        ys = [decision_aligned_score(x, 1.0, 5.0) for x in xs]
        self.assertTrue(all(b >= a for a, b in zip(ys, ys[1:])))
        self.assertIsNone(decision_aligned_score(float("nan"), 1.0, 5.0))

    def test_poor_quality_is_inconclusive_without_score(self):
        r = compare_signatures(self.refs[:1], cv2.GaussianBlur(_img(GENUINE), (0, 0), 3.0))
        self.assertEqual(r.band, "INCONCLUSIVE")
        self.assertIsNone(r.match_score)

    def test_input_limits(self):
        with self.assertRaises(ValueError):
            compare_signatures([], _img(GENUINE))
        with self.assertRaises(ValueError):
            compare_signatures(self.refs * 4, _img(GENUINE))

    def test_deterministic(self):
        a = compare_signatures(self.refs, _img(FORGERY)).model_dump()
        b = compare_signatures(self.refs, _img(FORGERY)).model_dump()
        self.assertEqual(a, b)


class TestCompareApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from signature_verification_system.src.adjudication.audit_logger import AuditLogger
        from signature_verification_system.src.api.app import create_app
        cls.tmp = tempfile.TemporaryDirectory()
        db = Path(cls.tmp.name) / "a.db"
        cls.logger = AuditLogger(str(db), str(db.with_suffix(".jsonl")))
        cls.client = TestClient(create_app(audit_logger=cls.logger))
        cls.b64 = {p: base64.b64encode((DATA / p).read_bytes()).decode() for p in REFS + [GENUINE, FORGERY]}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_json_compare(self):
        r = self.client.post("/api/v1/signature/compare", json={
            "reference_images": [self.b64[p] for p in REFS], "questioned_image": self.b64[GENUINE]})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["verdict"], "MATCH")
        self.assertEqual(body["references_used"], 3)
        self.assertIn("explanation_text", body)

    def test_multipart_compare(self):
        files = [("reference_images", (p, (DATA / p).read_bytes(), "image/png")) for p in REFS]
        files.append(("questioned_image", ("q.png", (DATA / FORGERY).read_bytes(), "image/png")))
        r = self.client.post("/api/v1/signature/compare", files=files)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["verdict"], "NO MATCH")

    def test_writes_no_audit_record(self):
        before = self.logger.get_record_count()
        self.client.post("/api/v1/signature/compare", json={
            "reference_images": [self.b64[REFS[0]]], "questioned_image": self.b64[GENUINE]})
        self.assertEqual(self.logger.get_record_count(), before)

    def test_missing_references_rejected(self):
        r = self.client.post("/api/v1/signature/compare", json={"reference_images": [], "questioned_image": self.b64[GENUINE]})
        self.assertEqual(r.status_code, 400)


class TestCompareCli(unittest.TestCase):
    def test_cli_json(self):
        from signature_verification_system.scripts.demo_cli import main
        out = StringIO()
        with redirect_stdout(out):
            code = main(["compare", "--reference", *[str(DATA / p) for p in REFS],
                         "--questioned", str(DATA / FORGERY), "--json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue())["verdict"], "NO MATCH")


if __name__ == "__main__":
    unittest.main()
