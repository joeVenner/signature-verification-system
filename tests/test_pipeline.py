"""Regression tests for the end-to-end cheque pipeline and form cleaning (EXP-009)."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from signature_verification_system.src.adjudication.audit_logger import AuditLogger
from signature_verification_system.src.detection.locator import refine_signature_bbox
from signature_verification_system.src.pipeline import ChequeVerificationPipeline
from signature_verification_system.src.preprocessing.form_cleaning import clean_printed_rules
from signature_verification_system.scripts.demo_showcase import SPECIMENS, TEMPLATE, compose_demo_cheque

DATA = Path(__file__).resolve().parent.parent / "data" / "samples"


def _img(rel: str) -> np.ndarray:
    img = cv2.imread(str(DATA / rel))
    if img is None:
        raise unittest.SkipTest(f"missing sample {rel}")
    return img


class TestFormCleaning(unittest.TestCase):
    def test_rule_line_removed_and_crossing_stroke_kept(self):
        crop = np.full((120, 300), 250, np.uint8)
        cv2.line(crop, (10, 100), (290, 100), 20, 3)       # printed rule
        cv2.line(crop, (150, 20), (150, 115), 20, 3)       # stroke crossing the rule
        for x in range(10, 290, 14):                       # dashed border
            cv2.line(crop, (x, 8), (x + 7, 8), 60, 2)
        out = clean_printed_rules(crop)
        self.assertGreater(int(out[100, 60]), 200, "rule pixel should be painted as paper")
        self.assertGreater(int(out[8, 12]), 200, "dash should be removed")
        self.assertLess(int(out[60, 150]), 100, "signature stroke must survive")

    def test_clean_crop_unchanged(self):
        crop = _img("genuine_pairs/pair_09_cedar_w05_ref.png")
        out = clean_printed_rules(crop)
        changed = np.mean(np.abs(out.astype(int) - crop.astype(int)) > 30)
        self.assertLess(changed, 0.01)

    def test_refine_does_not_chain_across_dashed_border(self):
        img = np.full((400, 900, 3), 250, np.uint8)
        for x in range(20, 880, 14):
            cv2.line(img, (x, 30), (x + 7, 30), (60, 60, 60), 2)
        cv2.ellipse(img, (300, 200), (120, 60), 0, 0, 360, (20, 20, 20), 3)
        x, y, w, h = refine_signature_bbox(img, (220, 160, 120, 60))
        self.assertLess(w, 300)
        self.assertGreater(y, 40)


class TestChequePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        db = Path(cls.tmp.name) / "ledger.db"
        cls.pipeline = ChequeVerificationPipeline(audit_logger=AuditLogger(str(db), str(db.with_suffix(".jsonl"))))
        cls.template = cv2.imread(str(TEMPLATE))
        cls.specimens = [_img(s) for s in SPECIMENS]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _run(self, signature: str, amount: float = 4750.0, blur: float = 0.0):
        cheque = compose_demo_cheque(self.template, _img(signature))
        if blur:
            cheque = cv2.GaussianBlur(cheque, (0, 0), blur)
        return self.pipeline.run(cheque, self.specimens, amount=amount)

    def test_genuine_auto_clears(self):
        r = self._run("genuine_pairs/pair_10_cedar_w05_questioned.png")
        self.assertEqual(r.verification.decision_band, "ACCEPT")
        self.assertEqual(r.decision.tier.value, "GREEN")
        self.assertIsNotNone(r.audit)

    def test_skilled_forgery_rejected(self):
        r = self._run("skilled_forgeries/pair_09_cedar_w05_forgery01.png")
        self.assertEqual(r.decision.tier.value, "RED")
        self.assertNotEqual(r.verification.decision_band, "ACCEPT")

    def test_high_value_genuine_requires_four_eyes(self):
        r = self._run("genuine_pairs/pair_10_cedar_w05_questioned.png", amount=150_000.0)
        self.assertEqual(r.verification.decision_band, "ACCEPT")
        self.assertTrue(r.decision.requires_four_eyes)

    def test_blurred_capture_never_auto_clears(self):
        r = self._run("genuine_pairs/pair_10_cedar_w05_questioned.png", blur=6.0)
        self.assertEqual(r.decision.tier.value, "RED")

    def test_pipeline_deterministic(self):
        a = self._run("skilled_forgeries/pair_09_cedar_w05_forgery01.png").verification.model_dump()
        b = self._run("skilled_forgeries/pair_09_cedar_w05_forgery01.png").verification.model_dump()
        self.assertEqual(a, b)


class TestApiHardening(unittest.TestCase):
    """EXP-010: API uses the pipeline, accepts several specimens, refuses unsafe input."""

    @classmethod
    def setUpClass(cls):
        import base64
        from fastapi.testclient import TestClient
        from signature_verification_system.src.api.app import create_app
        cls.tmp = tempfile.TemporaryDirectory()
        db = Path(cls.tmp.name) / "api.db"
        cls.client = TestClient(create_app(audit_logger=AuditLogger(str(db), str(db.with_suffix(".jsonl")))))
        cheque = compose_demo_cheque(cv2.imread(str(TEMPLATE)), _img("genuine_pairs/pair_10_cedar_w05_questioned.png"))
        enc = lambda im: base64.b64encode(cv2.imencode(".png", im)[1].tobytes()).decode()
        cls.cheque_b64 = enc(cheque)
        cls.spec_b64 = [enc(_img(s)) for s in SPECIMENS]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_multi_specimen_cheque_auto_clears(self):
        r = self.client.post("/api/v1/cheque/process", json={
            "cheque_image": self.cheque_b64, "specimen_image": self.spec_b64[0],
            "additional_specimens": self.spec_b64[1:], "amount": 4750.0})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["verification"]["reference_count"], 3)
        self.assertEqual(body["verification"]["decision_band"], "ACCEPT")
        self.assertEqual(body["decision"]["tier"], "GREEN")
        self.assertIn("FINAL_DECISION", [s["name"] for s in body["stages"]])

    def test_server_file_path_is_not_read(self):
        path = str(DATA / "genuine_pairs" / "pair_09_cedar_w05_ref.png")
        r = self.client.post("/api/v1/cheque/process", json={
            "cheque_image": path, "specimen_image": path, "amount": 10.0})
        self.assertEqual(r.status_code, 400)

    def test_oversized_payload_rejected(self):
        import importlib
        api = importlib.import_module("signature_verification_system.src.api.app")
        r = self.client.post("/api/v1/cheque/process", json={
            "cheque_image": "A" * ((api.MAX_IMAGE_BYTES * 4) // 3 + 4096),
            "specimen_image": self.spec_b64[0], "amount": 10.0})
        self.assertEqual(r.status_code, 413)

    def test_too_many_specimens_rejected(self):
        r = self.client.post("/api/v1/cheque/process", json={
            "cheque_image": self.cheque_b64, "specimen_image": self.spec_b64[0],
            "additional_specimens": self.spec_b64[:1] * 10, "amount": 10.0})
        self.assertEqual(r.status_code, 400)

    def test_png_decompression_bomb_rejected_before_decode(self):
        import base64
        bomb = cv2.imencode(".png", np.zeros((8000, 8000), np.uint8))[1].tobytes()   # 64 MP, tiny file
        self.assertLess(len(bomb), 2_000_000)
        r = self.client.post("/api/v1/cheque/process", json={
            "cheque_image": base64.b64encode(bomb).decode(), "specimen_image": self.spec_b64[0], "amount": 1.0})
        self.assertEqual(r.status_code, 413)

    def test_nan_and_negative_amount_rejected(self):
        for amount in ("NaN", -5):
            r = self.client.post(
                "/api/v1/cheque/process",
                content=('{"cheque_image": "%s", "specimen_image": "%s", "amount": %s}'
                         % (self.cheque_b64, self.spec_b64[0], amount)).encode(),
                headers={"content-type": "application/json"})
            self.assertEqual(r.status_code, 400, amount)

    def test_non_object_json_rejected(self):
        r = self.client.post("/api/v1/cheque/process", json=["not", "an", "object"])
        self.assertEqual(r.status_code, 400)

    def test_account_number_masked_in_ledger(self):
        r = self.client.post("/api/v1/cheque/process", json={
            "cheque_image": self.cheque_b64, "specimen_image": self.spec_b64[0],
            "account_no": "1234567890", "amount": 10.0})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["audit_record"]["metadata"]["account_no"], "******7890")


class TestCrossProcessDeterminism(unittest.TestCase):
    """Identical input → bit-identical output in independent interpreters."""

    SCRIPT = (
        "import cv2, json, sys\n"
        "from signature_verification_system.src.verification.deterministic import DeterministicVerifier\n"
        "d = 'signature_verification_system/data/samples/'\n"
        "refs = [cv2.imread(d + p) for p in ('genuine_pairs/pair_09_cedar_w05_ref.png',\n"
        "        'genuine_pairs/pair_09_cedar_w05_questioned.png', 'genuine_pairs/pair_10_cedar_w05_ref.png')]\n"
        "q = cv2.imread(d + 'skilled_forgeries/pair_09_cedar_w05_forgery01.png')\n"
        "r = DeterministicVerifier().verify_against_references(refs, q)\n"
        "sys.stdout.write(json.dumps(r.model_dump(), sort_keys=True))\n"
    )

    def test_three_fresh_processes_agree(self):
        import subprocess
        import sys
        root = Path(__file__).resolve().parent.parent.parent
        outs = [
            subprocess.run([sys.executable, "-c", self.SCRIPT], cwd=root, capture_output=True, text=True, check=True).stdout
            for _ in range(3)
        ]
        self.assertTrue(outs[0])
        self.assertEqual(outs[0], outs[1])
        self.assertEqual(outs[1], outs[2])
