"""Automated tests for FastAPI REST API service and CLI demo tool.

Tests:
1. API Health endpoint (GET /health)
2. API IQA endpoint (POST /api/v1/iqa - multipart & base64 JSON)
3. API Verification endpoint (POST /api/v1/verify - multipart & base64 JSON)
4. API Policy adjudication & CBUAE compliance rules
5. API Cheque processing full pipeline (POST /api/v1/cheque/process)
6. API Audit trail endpoints (GET /api/v1/audit/recent & GET /api/v1/audit/verify-chain)
7. CLI verify subcommand
8. CLI cheque subcommand
9. CLI audit subcommand (table display & SHA-256 chain verification)
10. CLI benchmark subcommand (batch verification & report generation)
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from signature_verification_system.src.adjudication.audit_logger import AuditLogger
from signature_verification_system.src.api.app import create_app
from signature_verification_system.src.core.config import DEFAULT_CONFIG
from signature_verification_system.src.core.types import (
    AmountTier,
    DecisionTier,
    StandardReturnCode,
)
from signature_verification_system.scripts.demo_cli import main as cli_main


class TestFastAPIEndpoints(unittest.TestCase):
    """Test suite for FastAPI REST API service."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.db_path = os.path.join(cls.temp_dir.name, "api_audit.db")
        cls.jsonl_path = os.path.join(cls.temp_dir.name, "api_audit.jsonl")
        cls.audit_logger = AuditLogger(db_path=cls.db_path, jsonl_path=cls.jsonl_path)
        cls.app = create_app(audit_logger=cls.audit_logger)
        cls.client = TestClient(cls.app)

        # Dataset paths
        cls.data_dir = Path(__file__).resolve().parent.parent / "data" / "samples"
        cls.ref_sig_path = str(cls.data_dir / "genuine_pairs" / "pair_00_synthetic_ref.png")
        cls.que_sig_path = str(cls.data_dir / "genuine_pairs" / "pair_00_synthetic_questioned.png")
        cls.cheque_path = str(cls.data_dir / "cheques" / "cheque_benchmark_synth_9000.jpg")

        with open(cls.ref_sig_path, "rb") as f:
            cls.ref_bytes = f.read()
        with open(cls.que_sig_path, "rb") as f:
            cls.que_bytes = f.read()
        with open(cls.cheque_path, "rb") as f:
            cls.cheque_bytes = f.read()

        cls.ref_b64 = base64.b64encode(cls.ref_bytes).decode("ascii")
        cls.que_b64 = base64.b64encode(cls.que_bytes).decode("ascii")
        cls.cheque_b64 = base64.b64encode(cls.cheque_bytes).decode("ascii")

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    # 1. Health
    def test_health_endpoint(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("api", data["versions"])
        self.assertIn("policy", data["versions"])
        self.assertIn("model", data["versions"])
        self.assertTrue(len(data["timestamp"]) > 10)

    # 2. IQA Multipart
    def test_iqa_endpoint_multipart(self):
        # A valid cheque image passes ANSI X9.100-181 specifications
        res = self.client.post(
            "/api/v1/iqa",
            files={"image": ("cheque.jpg", self.cheque_bytes, "image/jpeg")},
        )
        self.assertEqual(res.status_code, 200)
        metrics = res.json()
        self.assertIn("skew_angle", metrics)
        self.assertIn("blur_score", metrics)
        self.assertIn("contrast_score", metrics)
        self.assertIn("brightness_score", metrics)
        self.assertIn("passed", metrics)
        self.assertTrue(metrics["passed"])
        self.assertGreater(metrics["blur_score"], 100.0)

    # 3. IQA JSON Base64
    def test_iqa_endpoint_base64_json(self):
        res = self.client.post(
            "/api/v1/iqa",
            json={"image": self.cheque_b64},
        )
        self.assertEqual(res.status_code, 200)
        metrics = res.json()
        self.assertIn("blur_score", metrics)
        self.assertTrue(metrics["passed"])

    def test_iqa_endpoint_corrupted_payload(self):
        res = self.client.post(
            "/api/v1/iqa",
            files={"image": ("bad.png", b"not-a-valid-image", "image/png")},
        )
        self.assertEqual(res.status_code, 400)

    # 4. Verify Multipart
    def test_verify_endpoint_multipart(self):
        # Testing genuine identical signature verification
        res = self.client.post(
            "/api/v1/verify",
            files={
                "ref_image": ("ref.png", self.ref_bytes, "image/png"),
                "test_image": ("ref.png", self.ref_bytes, "image/png"),
            },
            data={
                "amount": "5000.0",
                "currency": "AED",
                "cheque_no": "CHQ-7788",
                "car_lar_match": "true",
                "positive_pay_match": "true",
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()

        v = data["verification"]
        d = data["decision"]
        a = data["audit_record"]

        self.assertEqual(v["similarity_score"], 1.0)
        self.assertTrue(v["is_match"])
        self.assertEqual(d["tier"], DecisionTier.GREEN.value)
        self.assertEqual(d["action"], "AUTO_CLEAR")
        self.assertEqual(d["amount_tier"], AmountTier.LOW_VALUE.value)
        self.assertEqual(a["document_id"], "CHQ-7788")
        self.assertEqual(len(a["hash"]), 64)

    # 5. Verify JSON Base64
    def test_verify_endpoint_base64_json(self):
        res = self.client.post(
            "/api/v1/verify",
            json={
                "ref_image": self.ref_b64,
                "test_image": self.que_b64,
                "amount": 10000.0,
                "currency": "AED",
                "cheque_no": "CHQ-8899",
                "car_lar_match": True,
                "positive_pay_match": True,
                "stale_days": 10,
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("verification", data)
        self.assertIn("decision", data)
        self.assertIn("audit_record", data)
        self.assertGreater(data["verification"]["similarity_score"], 0.70)

    # 6. Policy Adjudication & Compliance Triggers
    def test_verify_high_value_escalation(self):
        # Amount >= 100,000 AED triggers mandatory Four-Eyes Red tier
        res = self.client.post(
            "/api/v1/verify",
            json={
                "ref_image": self.ref_b64,
                "test_image": self.ref_b64,
                "amount": 250000.0,
                "currency": "AED",
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["decision"]["tier"], DecisionTier.RED.value)
        self.assertEqual(data["decision"]["action"], "MANDATORY_FOUR_EYES_ESCALATE")
        self.assertTrue(data["decision"]["requires_four_eyes"])
        self.assertIn("CBUAE_HIGH_VALUE_FOUR_EYES_MANDATE", data["decision"]["cbuae_compliance_flags"])

    def test_verify_stale_cheque_rejection(self):
        # Stale cheque presentment > 180 days triggers Red reject
        res = self.client.post(
            "/api/v1/verify",
            json={
                "ref_image": self.ref_b64,
                "test_image": self.ref_b64,
                "amount": 5000.0,
                "stale_days": 195,
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["decision"]["tier"], DecisionTier.RED.value)
        self.assertEqual(data["decision"]["action"], "REJECT")
        self.assertIn("STALE_CHEQUE_PRESENTMENT_EXCEEDS_6_MONTHS", data["decision"]["cbuae_compliance_flags"])

    # 7. Cheque Processing Full Pipeline
    def test_cheque_process_multipart(self):
        res = self.client.post(
            "/api/v1/cheque/process",
            files={
                "cheque_image": ("cheque.jpg", self.cheque_bytes, "image/jpeg"),
                "specimen_image": ("spec.png", self.ref_bytes, "image/png"),
            },
            data={
                "amount": "12500.0",
                "cheque_no": "CHQ-9000",
                "currency": "AED",
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertIn("iqa", data)
        self.assertIn("detection", data)
        self.assertIn("verification", data)
        self.assertIn("decision", data)
        self.assertIn("audit_record", data)

        self.assertTrue(data["iqa"]["passed"])
        self.assertGreaterEqual(len(data["detection"]["candidates"]), 1)
        self.assertTrue(data["crop_extracted"])
        self.assertEqual(data["audit_record"]["document_id"], "CHQ-9000")

    def test_cheque_process_base64_json(self):
        res = self.client.post(
            "/api/v1/cheque/process",
            json={
                "cheque_image": self.cheque_b64,
                "specimen_image": self.ref_b64,
                "amount": 8000.0,
                "cheque_no": "CHQ-9001",
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["crop_extracted"])
        self.assertIn(data["decision"]["tier"], [DecisionTier.GREEN.value, DecisionTier.AMBER.value, DecisionTier.RED.value])

    # 8. Audit Recent and Verify Chain
    def test_audit_endpoints(self):
        # Ensure at least one record exists in test ledger
        self.client.post(
            "/api/v1/verify",
            json={
                "ref_image": self.ref_b64,
                "test_image": self.ref_b64,
                "amount": 5000.0,
                "cheque_no": "CHQ-INIT-001",
            },
        )
        # 1. Fetch recent records
        res_rec = self.client.get("/api/v1/audit/recent?limit=5")
        self.assertEqual(res_rec.status_code, 200)
        records = res_rec.json()
        self.assertIsInstance(records, list)
        self.assertGreater(len(records), 0)

        # Verify descending order
        if len(records) > 1:
            self.assertGreater(records[0]["sequence_num"], records[1]["sequence_num"])

        # 2. Verify SHA-256 chain integrity
        res_chain = self.client.get("/api/v1/audit/verify-chain")
        self.assertEqual(res_chain.status_code, 200)
        chain_data = res_chain.json()
        self.assertTrue(chain_data["is_valid"])
        self.assertGreaterEqual(chain_data["total_records"], len(records))
        self.assertIsNone(chain_data["error"])


class TestCLITool(unittest.TestCase):
    """Test suite for Demo CLI subcommands via both function invocation and subprocess."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.cli_script = str(Path(__file__).resolve().parent.parent / "scripts" / "demo_cli.py")
        cls.data_dir = str(Path(__file__).resolve().parent.parent / "data" / "samples")
        cls.ref_sig = os.path.join(cls.data_dir, "genuine_pairs", "pair_00_synthetic_ref.png")
        cls.que_sig = os.path.join(cls.data_dir, "genuine_pairs", "pair_00_synthetic_questioned.png")
        cls.cheque_img = os.path.join(cls.data_dir, "cheques", "cheque_benchmark_synth_9000.jpg")
        cls.db_path = os.path.join(cls.temp_dir.name, "cli_test_audit.db")

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_cli_help(self):
        ret = subprocess.run(
            [sys.executable, self.cli_script, "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(ret.returncode, 0)
        self.assertIn("verify", ret.stdout)
        self.assertIn("cheque", ret.stdout)
        self.assertIn("audit", ret.stdout)
        self.assertIn("benchmark", ret.stdout)

    def test_cli_verify_subcommand(self):
        ret = subprocess.run(
            [
                sys.executable,
                self.cli_script,
                "verify",
                "--ref", self.ref_sig,
                "--test", self.que_sig,
                "--amount", "7500",
                "--cheque-no", "554433",
                "--db-path", self.db_path,
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(ret.returncode, 0)
        self.assertIn("Biometric Similarity Score", ret.stdout)
        self.assertIn("Multi-Feature Biometric Breakdown", ret.stdout)
        self.assertIn("HOG Gradient Orientation", ret.stdout)
        self.assertIn("Immutable Tamper-Evident Ledger Entry", ret.stdout)

    def test_cli_cheque_subcommand(self):
        crop_path = os.path.join(self.temp_dir.name, "extracted_crop.png")
        ret = subprocess.run(
            [
                sys.executable,
                self.cli_script,
                "cheque",
                "--cheque", self.cheque_img,
                "--specimen", self.ref_sig,
                "--amount", "18000",
                "--cheque-no", "9000",
                "--save-crop", crop_path,
                "--db-path", self.db_path,
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(ret.returncode, 0)
        self.assertIn("Image Quality Assessment", ret.stdout)
        self.assertIn("Signature Zone Detection", ret.stdout)
        self.assertIn("Audit Ledger Block Committed", ret.stdout)
        self.assertTrue(os.path.exists(crop_path))

    def test_cli_audit_subcommand(self):
        # 1. Test recent records display
        ret_recent = subprocess.run(
            [
                sys.executable,
                self.cli_script,
                "audit",
                "--limit", "5",
                "--db-path", self.db_path,
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(ret_recent.returncode, 0)
        self.assertIn("Recent Audit Ledger Records", ret_recent.stdout)

        # 2. Test chain integrity verification
        ret_chain = subprocess.run(
            [
                sys.executable,
                self.cli_script,
                "audit",
                "--verify-chain",
                "--db-path", self.db_path,
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(ret_chain.returncode, 0)
        self.assertIn("SHA-256 HASH-CHAIN INTEGRITY VERIFIED", ret_chain.stdout)

    def test_cli_benchmark_subcommand(self):
        report_path = os.path.join(self.temp_dir.name, "benchmark_report.json")
        ret = subprocess.run(
            [
                sys.executable,
                self.cli_script,
                "benchmark",
                "--data-dir", self.data_dir,
                "--category", "all",
                "--limit", "2",
                "--save-report", report_path,
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(ret.returncode, 0)
        self.assertIn("Benchmark Performance Summary", ret.stdout)
        self.assertIn("Final Benchmark Statistics", ret.stdout)
        self.assertTrue(os.path.exists(report_path))

        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("overall_accuracy_percent", data)
        self.assertIn("categories", data)
        self.assertEqual(len(data["categories"]), 3)

    def test_cli_verify_missing_file_error(self):
        ret = subprocess.run(
            [
                sys.executable,
                self.cli_script,
                "verify",
                "--ref", "/non/existent/file.png",
                "--test", self.que_sig,
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(ret.returncode, 1)
        self.assertIn("Error", ret.stdout)


if __name__ == "__main__":
    unittest.main()
