"""FastAPI REST API Service for Signature Verification and Cheque Clearing.

Endpoints:
- GET  /health: System health, engine and regulatory policy versions, timestamp
- POST /api/v1/iqa: Image Quality Assessment metrics (focus, skew, contrast, brightness)
- POST /api/v1/verify: 1-to-1 biometric signature verification and adjudication
- POST /api/v1/cheque/process: End-to-end cheque clearing pipeline (IQA, zone detection, crop, verify, adjudicate, audit)
- GET  /api/v1/audit/recent: Retrieve recent immutable audit trail records
- GET  /api/v1/audit/verify-chain: Verify SHA-256 cryptographic hash-chain integrity of audit ledger
- POST /api/v1/signature/inspect: 1:1 comparison plus the real pipeline intermediates (live console)
- GET  /api/v1/samples, /api/v1/samples/{id}: labelled sample gallery (only if SIGVERIFY_SAMPLES_DIR is set)
- GET  /: live verification console (static UI)
"""

from __future__ import annotations

import base64
import io
import math
import os

from signature_verification_system.envfile import load_env_file

# `.env` first, so its values reach OpenCV when cv2 is imported below (shell variables win).
load_env_file()
# Defence in depth against decompression bombs: OpenCV's own decode ceiling.
# Must be set before cv2 is first imported in the process.
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", "50000000")
import uuid
from dataclasses import asdict
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError

from signature_verification_system.src.core.config import DEFAULT_CONFIG, SystemConfig
from signature_verification_system.src.core.types import (
    AuditRecord,
    DecisionResult,
    DetectionResult,
    IQAMetrics,
    VerificationResult,
)
from signature_verification_system.src.detection.locator import SignatureLocator
from signature_verification_system.src.preprocessing.iqa import assess_image_quality
from signature_verification_system.src.verification.deterministic import DeterministicVerifier
from signature_verification_system.src.adjudication.decision_engine import DecisionEngine
from signature_verification_system.src.adjudication.audit_logger import AuditLogger
from signature_verification_system.src.pipeline import ChequeVerificationPipeline
from signature_verification_system.src.verification.signature_compare import SignatureComparison, compare_signatures
from signature_verification_system.src.api.inspection import SignatureInspection, inspect_signatures
from signature_verification_system.src.api.samples import (
    IMAGE_SUFFIXES,
    SampleCatalog,
    SampleList,
    catalog_from_env,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Input limits: reject oversized payloads before decoding (decompression-bomb guard).
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
MAX_ADDITIONAL_SPECIMENS = 9
MAX_COMPARE_REFERENCES = 10
DEFAULT_AUDIT_DB_PATH = "audit_ledger.db"
DEFAULT_AUDIT_JSONL_PATH = "audit_ledger.jsonl"
MAX_REQUEST_BYTES = 64 * 1024 * 1024   # whole-request cap (Content-Length); enforce at the proxy too
# The console renders images from data:/blob: URLs and loads only Google Fonts; nothing else is allowed.
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; img-src 'self' data: blob:; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com"
)
MAX_TEXT_FIELD = 64
CURRENCY_PATTERN = r"^[A-Z]{3}$"
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


# ============================================================================
# API Response & Request Schemas
# ============================================================================

class HealthResponse(BaseModel):
    status: str = "healthy"
    versions: Dict[str, str]
    timestamp: str


class VerifyResponse(BaseModel):
    verification: VerificationResult
    decision: DecisionResult
    audit_record: AuditRecord


class ChequeProcessResponse(BaseModel):
    iqa: IQAMetrics
    detection: DetectionResult
    verification: VerificationResult
    decision: DecisionResult
    audit_record: AuditRecord
    crop_extracted: bool = True
    crop_bbox: Optional[List[int]] = None
    stages: List[Dict[str, Any]] = Field(default_factory=list, description="Stage-by-stage pipeline report")


class VerifyChainResponse(BaseModel):
    is_valid: bool
    total_records: int
    error: Optional[str] = None
    verified_at_utc: str


class _ClearingFields(BaseModel):
    """Common, strictly validated clearing inputs (no NaN/inf, bounded text)."""
    model_config = ConfigDict(allow_inf_nan=False, extra="ignore")
    amount: float = Field(default=0.0, ge=0.0, le=1e12)
    currency: str = Field(default="AED", pattern=CURRENCY_PATTERN)
    cheque_no: Optional[str] = Field(default=None, max_length=MAX_TEXT_FIELD)
    account_no: Optional[str] = Field(default=None, max_length=MAX_TEXT_FIELD)
    car_lar_match: bool = True
    positive_pay_match: bool = True
    stale_days: int = Field(default=0, ge=0, le=36500)
    operator_id: Optional[str] = Field(default=None, max_length=MAX_TEXT_FIELD)
    document_id: Optional[str] = Field(default=None, max_length=MAX_TEXT_FIELD)


class Base64VerifyRequest(_ClearingFields):
    ref_image: str = Field(..., validation_alias=AliasChoices("ref_image", "reference_image"),
                           description="Base64 encoded reference signature specimen")
    test_image: str = Field(..., validation_alias=AliasChoices("test_image", "questioned_image"),
                            description="Base64 encoded questioned signature crop")


class Base64ChequeProcessRequest(_ClearingFields):
    cheque_image: str = Field(..., description="Base64 encoded full cheque image")
    specimen_image: str = Field(..., validation_alias=AliasChoices("specimen_image", "ref_image"),
                                description="Base64 encoded specimen signature")
    additional_specimens: List[str] = Field(default_factory=list, max_length=MAX_ADDITIONAL_SPECIMENS,
                                            description="Further base64 specimens (feature-wise max aggregation)")


class Base64SignatureCompareRequest(BaseModel):
    """Signature-only comparison: no cheque, no clearing policy, no audit record."""
    model_config = ConfigDict(extra="ignore")
    reference_images: List[str] = Field(..., min_length=1, max_length=MAX_COMPARE_REFERENCES,
                                        description="Base64 reference signature(s) of the account holder")
    questioned_image: str = Field(..., description="Base64 signature to verify")


class Base64SignatureInspectRequest(BaseModel):
    """Same fields as the compare request, restricted to exactly one reference (1:1)."""
    model_config = ConfigDict(extra="ignore")
    reference_images: List[str] = Field(..., min_length=1, max_length=1,
                                        description="Exactly one base64 reference signature")
    questioned_image: str = Field(..., description="Base64 signature to verify")


class Base64IQARequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image string")


# ============================================================================
# Image Decoding Helpers
# ============================================================================

ACCEPTED_FORMATS = {"PNG", "JPEG", "TIFF", "BMP"}


def _check_header_dimensions(raw_bytes: bytes) -> None:
    """Read width/height from the file header only (Pillow opens lazily) and
    reject oversized or unsupported images *before* any pixel decoding."""
    try:
        with Image.open(io.BytesIO(raw_bytes)) as probe:
            fmt, (width, height) = probe.format, probe.size
    except Image.DecompressionBombError:
        raise HTTPException(status_code=413,
                            detail=f"Image exceeds {MAX_IMAGE_PIXELS:,} pixel limit.")
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Unsupported or corrupt image. Accepted formats: PNG, JPEG, TIFF, BMP.")
    if fmt not in ACCEPTED_FORMATS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Unsupported image format. Accepted formats: PNG, JPEG, TIFF, BMP.")
    if width * height > MAX_IMAGE_PIXELS:
        raise HTTPException(status_code=413,
                            detail=f"Image exceeds {MAX_IMAGE_PIXELS:,} pixel limit.")


def decode_image_bytes(raw_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes into an OpenCV BGR numpy array."""
    if not raw_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provided image byte stream is empty.",
        )
    if len(raw_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit.",
        )
    _check_header_dimensions(raw_bytes)
    arr = np.frombuffer(raw_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to decode image from byte buffer. Ensure file format is valid PNG, JPEG, or TIFF.",
        )
    return image


def format_document_id(doc_id: Optional[str], cheque_no: Optional[str], prefix: str = "CHQ") -> str:
    if doc_id:
        return doc_id
    if cheque_no:
        clean = cheque_no.strip()
        if clean.upper().startswith(f"{prefix}-"):
            return clean
        return f"{prefix}-{clean}"
    return f"{prefix}-{uuid.uuid4().hex[:10].upper()}"


def decode_base64_image(b64_string: str) -> np.ndarray:
    """Decode a base64 encoded string (with or without data URI header) to an OpenCV array."""
    if not b64_string or not isinstance(b64_string, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image base64 string is missing or empty.",
        )
    if len(b64_string) > (MAX_IMAGE_BYTES * 4) // 3 + 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit.",
        )
    clean_b64 = b64_string.split(",", 1)[1] if "," in b64_string else b64_string
    try:
        raw_bytes = base64.b64decode(clean_b64.strip())
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid base64 image data.",
        )
    return decode_image_bytes(raw_bytes)


async def resolve_image(
    file_candidate: Optional[UploadFile],
    str_candidate: Optional[str] = None
) -> np.ndarray:
    """Extract and decode image from either an UploadFile or a base64 string.

    Server-side filesystem paths are deliberately NOT accepted: a client must
    never be able to make the service read local files (path traversal).
    """
    if file_candidate is not None:
        content = await file_candidate.read(MAX_IMAGE_BYTES + 1)
        if content:
            return decode_image_bytes(content)
    if str_candidate:
        return decode_base64_image(str_candidate)
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="No valid image data supplied (neither multipart file nor base64 string provided).",
    )


async def parse_json_model(request: Request, model: type) -> Any:
    """Parse and validate a JSON body against a Pydantic model (400 on failure)."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object.")
    try:
        return model.model_validate(body)
    except ValidationError as exc:
        fields = sorted({".".join(str(p) for p in e["loc"]) for e in exc.errors()})
        raise HTTPException(status_code=400, detail=f"Invalid or missing fields: {', '.join(fields)}")


def validate_form_fields(amount: float, currency: str, stale_days: int, texts: List[Optional[str]]) -> None:
    """Apply the same constraints as `_ClearingFields` to multipart form inputs."""
    try:
        _ClearingFields(amount=amount, currency=currency, stale_days=stale_days)
    except ValidationError:
        raise HTTPException(status_code=400, detail="Invalid amount, currency or stale_days.")
    if not math.isfinite(amount) or any(t is not None and len(t) > MAX_TEXT_FIELD for t in texts):
        raise HTTPException(status_code=400, detail="Invalid form field.")


def mask_account(account_no: Optional[str]) -> Optional[str]:
    """Keep only the last 4 characters: the audit chain is immutable, so raw PII must not enter it."""
    if not account_no:
        return account_no
    return "*" * max(0, len(account_no) - 4) + account_no[-4:]


# ============================================================================
# Application Factory
# ============================================================================

def create_app(
    config: Optional[SystemConfig] = None,
    audit_logger: Optional[AuditLogger] = None,
    decision_engine: Optional[DecisionEngine] = None,
    verifier: Optional[DeterministicVerifier] = None,
    locator: Optional[SignatureLocator] = None,
    samples: Optional[SampleCatalog] = None,
) -> FastAPI:
    """Instantiate and configure the FastAPI application.

    `samples` overrides the sample gallery; by default it is loaded from the
    SIGVERIFY_SAMPLES_DIR environment variable (gallery disabled when unset).
    """
    sample_catalog = samples if samples is not None else catalog_from_env()
    app_config = config or DEFAULT_CONFIG
    logger = audit_logger or AuditLogger(
        db_path=os.environ.get("SIGVERIFY_AUDIT_DB_PATH", DEFAULT_AUDIT_DB_PATH),
        jsonl_path=os.environ.get("SIGVERIFY_AUDIT_JSONL_PATH", DEFAULT_AUDIT_JSONL_PATH),
        config=app_config,
    )
    engine = decision_engine or DecisionEngine(config=app_config)
    sig_verifier = verifier or DeterministicVerifier(config=app_config)
    sig_locator = locator or SignatureLocator()
    cheque_pipeline = ChequeVerificationPipeline(
        config=app_config, locator=sig_locator, verifier=sig_verifier, engine=engine, audit_logger=logger,
    )

    app = FastAPI(
        title="Automated Cheque & Signature Verification API",
        description="ANSI X9.100-181 & CBUAE ICCS compliant automated clearing and biometric verification service.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Explicit origin allow-list from the environment; no credentials (the API is
    # unauthenticated, see SECURITY notes in benchmark/experiments.md EXP-010).
    origins = [o.strip() for o in os.environ.get("SIGVERIFY_CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def limit_request_size(request: Request, call_next):
        declared = request.headers.get("content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > MAX_REQUEST_BYTES):
            return JSONResponse(status_code=413, content={"detail": "Request body too large."})
        return await call_next(request)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        return response

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, __: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": "Invalid input for verification."})

    # Attach components to state for external reference or testing
    app.state.config = app_config
    app.state.audit_logger = logger
    app.state.decision_engine = engine
    app.state.verifier = sig_verifier
    app.state.locator = sig_locator
    app.state.pipeline = cheque_pipeline

    # ------------------------------------------------------------------------
    # 1. Health Endpoint
    # ------------------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def get_health() -> HealthResponse:
        """Check service health and inspect loaded model and policy versions."""
        return HealthResponse(
            status="healthy",
            versions={
                "api": "1.0.0",
                "policy": app_config.policy_version,
                "model": app_config.model_version,
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # ------------------------------------------------------------------------
    # 2. Image Quality Assessment (IQA) Endpoint
    # ------------------------------------------------------------------------
    @app.post("/api/v1/iqa", response_model=IQAMetrics, tags=["Preprocessing & IQA"])
    async def run_iqa(
        request: Request,
        image: Optional[UploadFile] = File(None),
        file: Optional[UploadFile] = File(None),
    ) -> IQAMetrics:
        """Assess input image against ANSI X9.100-181 & CBUAE quality standards.
        
        Evaluates blur (Laplacian variance), contrast, brightness, and skew tilt angle.
        Supports multipart file upload or JSON payload containing base64 image.
        """
        content_type = request.headers.get("content-type", "")
        img_array: Optional[np.ndarray] = None

        if "application/json" in content_type:
            try:
                body = await request.json()
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid JSON body.")
            if not isinstance(body, dict):
                raise HTTPException(status_code=400, detail="JSON body must be an object.")
            b64_str = body.get("image") or body.get("image_base64")
            if not b64_str:
                raise HTTPException(status_code=400, detail="Missing 'image' or 'image_base64' string in JSON body.")
            img_array = decode_base64_image(b64_str)
        else:
            upload = image or file
            if upload is not None:
                content = await upload.read(MAX_IMAGE_BYTES + 1)
                if content:
                    img_array = decode_image_bytes(content)

        if img_array is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No image uploaded. Provide multipart file ('image' or 'file') or JSON body with 'image'.",
            )

        metrics, _ = await run_in_threadpool(assess_image_quality, img_array, app_config.iqa, False)
        return metrics

    # ------------------------------------------------------------------------
    # 3. 1-to-1 Verification & Adjudication Endpoint
    # ------------------------------------------------------------------------
    @app.post("/api/v1/verify", response_model=VerifyResponse, tags=["Biometric Verification"])
    async def verify_signature_crops(
        request: Request,
        ref_image: Optional[UploadFile] = File(None),
        test_image: Optional[UploadFile] = File(None),
        amount: float = Form(0.0),
        currency: str = Form("AED"),
        cheque_no: Optional[str] = Form(None),
        account_no: Optional[str] = Form(None),
        car_lar_match: bool = Form(True),
        positive_pay_match: bool = Form(True),
        stale_days: int = Form(0),
        operator_id: Optional[str] = Form(None),
        document_id: Optional[str] = Form(None),
    ) -> VerifyResponse:
        """Compare questioned signature against reference specimen and adjudicate clearing.
        
        Accepts multipart/form-data or JSON payload with base64 encoded images.
        Evaluates 6 deterministic topological/kinematic biometric feature channels.
        Persists decision to tamper-evident SHA-256 hash-chained ledger.
        """
        content_type = request.headers.get("content-type", "")

        if "application/json" in content_type:
            req = await parse_json_model(request, Base64VerifyRequest)
            ref_array = decode_base64_image(req.ref_image)
            test_array = decode_base64_image(req.test_image)
            req_amount, req_currency = req.amount, req.currency
            req_cheque_no, req_account_no = req.cheque_no, req.account_no
            req_car_lar, req_positive_pay = req.car_lar_match, req.positive_pay_match
            req_stale_days, req_operator_id, req_document_id = req.stale_days, req.operator_id, req.document_id
        else:
            # Multipart upload
            validate_form_fields(amount, currency, stale_days, [cheque_no, account_no, operator_id, document_id])
            ref_array = await resolve_image(ref_image)
            test_array = await resolve_image(test_image)
            req_amount = amount
            req_currency = currency
            req_cheque_no = cheque_no
            req_account_no = account_no
            req_car_lar = car_lar_match
            req_positive_pay = positive_pay_match
            req_stale_days = stale_days
            req_operator_id = operator_id
            req_document_id = document_id

        # 1. Biometric verification (CPU-bound: run off the event loop)
        verification_result = await run_in_threadpool(sig_verifier.verify, ref_array, test_array)

        # 2. Decision engine adjudication (CBUAE 180-day stale rule: > 180 days is stale)
        is_stale = req_stale_days > 180
        decision_result = engine.evaluate(
            verification_result=verification_result,
            amount=req_amount,
            currency=req_currency,
            car_lar_match=req_car_lar,
            positive_pay_match=req_positive_pay,
            stale_cheque=is_stale,
            detected_signers_count=1,
        )

        # 3. Cryptographic hash-chained audit logging
        doc_id = format_document_id(req_document_id, req_cheque_no, prefix="CHQ")
        audit_record = logger.log_decision(
            document_id=doc_id,
            decision=decision_result,
            operator_id=req_operator_id,
            additional_metadata={
                "cheque_no": req_cheque_no,
                "account_no": mask_account(req_account_no),
                "stale_days": req_stale_days,
                "car_lar_match": req_car_lar,
                "positive_pay_match": req_positive_pay,
                "confidence": verification_result.confidence,
            },
        )

        return VerifyResponse(
            verification=verification_result,
            decision=decision_result,
            audit_record=audit_record,
        )

    # ------------------------------------------------------------------------
    # 4. End-to-End Cheque Processing Pipeline
    # ------------------------------------------------------------------------
    @app.post("/api/v1/cheque/process", response_model=ChequeProcessResponse, tags=["Cheque Clearing Pipeline"])
    async def process_cheque(
        request: Request,
        cheque_image: Optional[UploadFile] = File(None),
        specimen_image: Optional[UploadFile] = File(None),
        additional_specimens: Optional[List[UploadFile]] = File(None),
        amount: float = Form(0.0),
        currency: str = Form("AED"),
        cheque_no: Optional[str] = Form(None),
        account_no: Optional[str] = Form(None),
        car_lar_match: bool = Form(True),
        positive_pay_match: bool = Form(True),
        stale_days: int = Form(0),
        operator_id: Optional[str] = Form(None),
        document_id: Optional[str] = Form(None),
    ) -> ChequeProcessResponse:
        """Full end-to-end cheque processing pipeline.
        
        Steps executed:
        1. Image Quality Assessment (IQA) & Deskewing
        2. Signature Zone Detection & Candidate Extraction
        3. Signature Crop Extraction (with descender-safe padding)
        4. Deterministic Biometric Verification vs Specimen
        5. Policy Adjudication (STP / L1 Amber / L2 Red)
        6. Append-Only SHA-256 Hash-Chained Audit Ledger
        """
        content_type = request.headers.get("content-type", "")

        if "application/json" in content_type:
            req = await parse_json_model(request, Base64ChequeProcessRequest)
            cheque_array = decode_base64_image(req.cheque_image)
            specimen_array = decode_base64_image(req.specimen_image)
            extra_specimens = [decode_base64_image(b) for b in req.additional_specimens]
            req_amount, req_currency = req.amount, req.currency
            req_cheque_no, req_account_no = req.cheque_no, req.account_no
            req_car_lar, req_positive_pay = req.car_lar_match, req.positive_pay_match
            req_stale_days, req_operator_id, req_document_id = req.stale_days, req.operator_id, req.document_id
        else:
            validate_form_fields(amount, currency, stale_days, [cheque_no, account_no, operator_id, document_id])
            uploads = [f for f in (additional_specimens or []) if f is not None]
            if len(uploads) > MAX_ADDITIONAL_SPECIMENS:
                raise HTTPException(status_code=400, detail=f"At most {MAX_ADDITIONAL_SPECIMENS} additional specimens.")
            cheque_array = await resolve_image(cheque_image)
            specimen_array = await resolve_image(specimen_image)
            extra_specimens = [await resolve_image(f) for f in uploads]
            req_amount = amount
            req_currency = currency
            req_cheque_no = cheque_no
            req_account_no = account_no
            req_car_lar = car_lar_match
            req_positive_pay = positive_pay_match
            req_stale_days = stale_days
            req_operator_id = operator_id
            req_document_id = document_id

        specimens = [specimen_array] + extra_specimens
        doc_id = format_document_id(req_document_id, req_cheque_no, prefix="CHQ")
        report = await run_in_threadpool(
            cheque_pipeline.run,
            cheque_array,
            specimens,
            amount=req_amount,
            currency=req_currency,
            car_lar_match=req_car_lar,
            positive_pay_match=req_positive_pay,
            stale_days=req_stale_days,
            document_id=doc_id,
            operator_id=req_operator_id,
            audit_metadata={
                "cheque_no": req_cheque_no,
                "account_no": mask_account(req_account_no),
                "stale_days": req_stale_days,
                "car_lar_match": req_car_lar,
                "positive_pay_match": req_positive_pay,
                "specimen_count": len(specimens),
            },
        )
        crop_extracted = report.detection.best_candidate is not None
        return ChequeProcessResponse(
            iqa=report.iqa,
            detection=report.detection,
            verification=report.verification,
            decision=report.decision,
            audit_record=report.audit,
            crop_extracted=crop_extracted,
            crop_bbox=list(report.crop_bbox) if report.crop_bbox else None,
            stages=[asdict(st) for st in report.stages],
        )

    # ------------------------------------------------------------------------
    # 4b. Signature-only comparison (no cheque, no policy, no ledger)
    # ------------------------------------------------------------------------
    @app.post("/api/v1/signature/compare", response_model=SignatureComparison, tags=["Biometric Verification"])
    async def compare_signature_only(
        request: Request,
        reference_images: Optional[List[UploadFile]] = File(None),
        questioned_image: Optional[UploadFile] = File(None),
    ) -> SignatureComparison:
        """Compare one questioned signature with 1-10 reference signatures.

        Returns the verdict (MATCH / UNCERTAIN - MANUAL REVIEW / NO MATCH /
        INCONCLUSIVE - IMAGE QUALITY), a 0-100 match score, per-signal
        similarities and an evidence-based explanation. Applies no amount,
        mandate or stale-cheque rules and writes nothing to the audit ledger.
        """
        if "application/json" in request.headers.get("content-type", ""):
            req = await parse_json_model(request, Base64SignatureCompareRequest)
            refs = [decode_base64_image(b) for b in req.reference_images]
            questioned = decode_base64_image(req.questioned_image)
        else:
            uploads = [f for f in (reference_images or []) if f is not None]
            if not uploads or len(uploads) > MAX_COMPARE_REFERENCES:
                raise HTTPException(status_code=400,
                                    detail=f"Provide 1-{MAX_COMPARE_REFERENCES} 'reference_images' files.")
            if questioned_image is None:
                raise HTTPException(status_code=400, detail="Provide a 'questioned_image' file.")
            refs = [await resolve_image(f) for f in uploads]
            questioned = await resolve_image(questioned_image)
        return await run_in_threadpool(compare_signatures, refs, questioned, sig_verifier)

    @app.post("/api/v1/signature/inspect", response_model=SignatureInspection, tags=["Biometric Verification"])
    async def inspect_signature_pair(
        request: Request,
        reference_images: Optional[List[UploadFile]] = File(None),
        questioned_image: Optional[UploadFile] = File(None),
    ) -> SignatureInspection:
        """1:1 comparison returning the `/signature/compare` result plus pipeline visuals.

        Same inputs and limits as `/signature/compare`, but exactly one reference.
        Visuals are PNGs (base64) rendered from the intermediates of the same
        deterministic pipeline; the signal breakdown is included only when its
        log-odds provably equals the verdict's.
        """
        if "application/json" in request.headers.get("content-type", ""):
            req = await parse_json_model(request, Base64SignatureInspectRequest)
            ref = decode_base64_image(req.reference_images[0])
            questioned = decode_base64_image(req.questioned_image)
        else:
            uploads = [f for f in (reference_images or []) if f is not None]
            if len(uploads) != 1:
                raise HTTPException(status_code=400, detail="Provide exactly one 'reference_images' file.")
            if questioned_image is None:
                raise HTTPException(status_code=400, detail="Provide a 'questioned_image' file.")
            ref = await resolve_image(uploads[0])
            questioned = await resolve_image(questioned_image)
        return await run_in_threadpool(inspect_signatures, ref, questioned, sig_verifier)

    # ------------------------------------------------------------------------
    # 4c. Sample gallery (operator-configured directory only)
    # ------------------------------------------------------------------------
    @app.get("/api/v1/samples", response_model=SampleList, tags=["Samples"])
    async def list_samples() -> SampleList:
        """Labelled sample pairs; 404 when SIGVERIFY_SAMPLES_DIR is not configured."""
        if sample_catalog is None:
            raise HTTPException(status_code=404, detail="Sample gallery is disabled.")
        return SampleList(samples=list(sample_catalog.pairs))

    @app.get("/api/v1/samples/{sample_id}", tags=["Samples"])
    async def get_sample_image(sample_id: str) -> FileResponse:
        """One sample image by its server-generated id (never a path)."""
        path = sample_catalog.image(sample_id) if sample_catalog is not None else None
        if path is None:
            raise HTTPException(status_code=404, detail="Sample not found.")
        return FileResponse(path, media_type=IMAGE_SUFFIXES[path.suffix.lower()],
                            headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "private, max-age=3600"})

    # ------------------------------------------------------------------------
    # 5. Audit Ledger Endpoints
    # ------------------------------------------------------------------------
    @app.get("/api/v1/audit/recent", response_model=List[AuditRecord], tags=["Audit & Governance"])
    async def get_recent_audit_records(
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> List[AuditRecord]:
        """Retrieve recent immutable audit ledger blocks."""
        return logger.get_recent_records(limit=limit, offset=offset)

    @app.get("/api/v1/audit/verify-chain", response_model=VerifyChainResponse, tags=["Audit & Governance"])
    async def verify_audit_chain_integrity() -> VerifyChainResponse:
        """Verify the complete SHA-256 tamper-evident hash chain across all audit blocks."""
        is_valid, err = logger.verify_integrity()
        total_count = logger.get_record_count()
        return VerifyChainResponse(
            is_valid=is_valid,
            total_records=total_count,
            error=err,
            verified_at_utc=datetime.now(timezone.utc).isoformat(),
        )

    # Mounted last so it never shadows the API routes above.
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="console")
    return app


# Default singleton application instance
app = create_app()
