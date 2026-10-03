#!/usr/bin/env python3
"""Command Line Interface (CLI) for Signature Verification and Cheque Adjudication.

Subcommands:
  - verify: Compare two signature crops (1-to-1 biometric matching)
  - cheque: End-to-end cheque processing pipeline (IQA, crop, verify, adjudicate, audit)
  - audit: Display recent audit ledger entries or verify cryptographic chain integrity
  - benchmark: Run batch evaluation across dataset folders and print statistical summary
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Ensure project and workspace roots are in Python path
_current_file = Path(__file__).resolve()
_project_dir = _current_file.parent.parent
_workspace_dir = _project_dir.parent
for _p in [str(_project_dir), str(_workspace_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from signature_verification_system.src.core.config import DEFAULT_CONFIG, SystemConfig
    from signature_verification_system.src.core.types import (
        AmountTier,
        DecisionTier,
        StandardReturnCode,
    )
    from signature_verification_system.src.preprocessing.iqa import assess_image_quality
    from signature_verification_system.src.detection.locator import SignatureLocator
    from signature_verification_system.src.verification.deterministic import DeterministicVerifier
    from signature_verification_system.src.adjudication.decision_engine import DecisionEngine
    from signature_verification_system.src.adjudication.audit_logger import AuditLogger
    from signature_verification_system.src.pipeline import ChequeVerificationPipeline
    from signature_verification_system.src.verification.explanation import render_text
    from signature_verification_system.src.verification.signature_compare import compare_signatures
except ImportError:
    # Alternative import path if run from inside src directory
    from src.core.config import DEFAULT_CONFIG, SystemConfig
    from src.core.types import AmountTier, DecisionTier, StandardReturnCode
    from src.preprocessing.iqa import assess_image_quality
    from src.detection.locator import SignatureLocator
    from src.verification.deterministic import DeterministicVerifier
    from src.adjudication.decision_engine import DecisionEngine
    from src.adjudication.audit_logger import AuditLogger
    from src.pipeline import ChequeVerificationPipeline
    from src.verification.explanation import render_text
    from src.verification.signature_compare import compare_signatures

console = Console()


def get_tier_style(tier: Union[DecisionTier, str]) -> Tuple[str, str]:
    """Return color and styled text for clearing decision tier."""
    t_str = tier.value if hasattr(tier, "value") else str(tier)
    if t_str == "GREEN":
        return "green", "[bold green]GREEN (STP Auto-Clear)[/bold green]"
    elif t_str == "AMBER":
        return "yellow", "[bold yellow]AMBER (L1 Operator Queue)[/bold yellow]"
    elif t_str == "RED":
        return "red", "[bold red]RED (L2 Escalate / Reject)[/bold red]"
    return "white", t_str


def format_document_id(doc_id: Optional[str], cheque_no: Optional[str], prefix: str = "CHQ") -> str:
    if doc_id:
        return doc_id
    if cheque_no:
        clean = cheque_no.strip()
        if clean.upper().startswith(f"{prefix}-"):
            return clean
        return f"{prefix}-{clean}"
    return f"{prefix}-CLI-{os.getpid()}"


def find_default_data_dir() -> Optional[str]:
    """Locate benchmark data folder across possible relative paths."""
    candidates = [
        _project_dir / "data" / "samples",
        _workspace_dir / "signature_verification_system" / "data" / "samples",
        Path("data/samples"),
        Path("signature_verification_system/data/samples"),
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return str(c.resolve())
    return None


# ============================================================================
# Subcommand 1: Verify
# ============================================================================

def cmd_verify(args: argparse.Namespace) -> int:
    """Compare two signature crops and adjudicate clearing decision."""
    ref_path = args.ref
    test_path = args.test or args.questioned

    if not os.path.exists(ref_path):
        console.print(f"[bold red]Error:[/bold red] Reference image not found: {ref_path}")
        return 1
    if not os.path.exists(test_path):
        console.print(f"[bold red]Error:[/bold red] Questioned image not found: {test_path}")
        return 1

    ref_img = cv2.imread(ref_path)
    test_img = cv2.imread(test_path)
    if ref_img is None or test_img is None:
        console.print("[bold red]Error:[/bold red] Failed to read one or both signature images.")
        return 1

    config = DEFAULT_CONFIG
    verifier = DeterministicVerifier(config=config)
    engine = DecisionEngine(config=config)
    db_path = args.db_path or "audit_ledger.db"
    jsonl_path = os.path.splitext(db_path)[0] + ".jsonl"
    logger = AuditLogger(db_path=db_path, jsonl_path=jsonl_path, config=config)

    t0 = time.perf_counter()
    v_res = verifier.verify(reference_image=ref_img, questioned_image=test_img)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    is_stale = args.stale_days > 180
    decision = engine.evaluate(
        verification_result=v_res,
        amount=args.amount,
        currency=args.currency,
        car_lar_match=not args.no_car_lar_match,
        positive_pay_match=not args.no_positive_pay,
        stale_cheque=is_stale,
    )

    doc_id = format_document_id(getattr(args, "document_id", None), args.cheque_no, prefix="CHQ")
    audit_rec = logger.log_decision(
        document_id=doc_id,
        decision=decision,
        operator_id=args.operator_id,
        additional_metadata={
            "ref_path": ref_path,
            "test_path": test_path,
            "cheque_no": args.cheque_no,
            "account_no": args.account_no,
            "latency_ms": round(latency_ms, 2),
        },
    )

    tier_color, tier_badge = get_tier_style(decision.tier)

    # 1. Header panel
    console.print(Panel(
        f"[bold cyan]1-to-1 Biometric Signature Verification[/bold cyan]\n"
        f"Reference:  [dim]{ref_path}[/dim]\n"
        f"Questioned: [dim]{test_path}[/dim]\n"
        f"Inference Latency: [bold]{latency_ms:.2f} ms[/bold]",
        title="[bold blue]Verification Analysis[/bold blue]",
        border_style="cyan"
    ))

    # 2. Decision Summary Table
    summary_table = Table(title="Clearing Adjudication Summary", header_style="bold magenta")
    summary_table.add_column("Metric / Field", style="dim")
    summary_table.add_column("Value")

    match_str = "[bold green]MATCH[/bold green]" if v_res.is_match else "[bold red]NON-MATCH / REJECT[/bold red]"
    score_style = "green" if v_res.similarity_score >= 0.85 else ("yellow" if v_res.similarity_score >= 0.65 else "red")

    summary_table.add_row("Biometric Similarity Score", f"[{score_style}]{v_res.similarity_score:.4f} ({v_res.similarity_score * 100:.1f}%)[/{score_style}]")
    summary_table.add_row("Match Status", match_str)
    summary_table.add_row("Statistical Confidence", f"{v_res.confidence:.3f}")
    summary_table.add_row("Adjudicated Decision Tier", tier_badge)
    summary_table.add_row("Clearing Action", f"[{tier_color}]{decision.action}[/{tier_color}]")
    summary_table.add_row("Transaction Amount", f"{decision.currency} {decision.amount:,.2f} ({decision.amount_tier.value})")
    summary_table.add_row("Return Code", str(decision.return_code.value if decision.return_code else "CLEAR (Accepted)"))
    summary_table.add_row("Four-Eyes Sign-off Mandatory", "Yes" if decision.requires_four_eyes else "No")
    if decision.cbuae_compliance_flags:
        summary_table.add_row("CBUAE Flags", ", ".join(decision.cbuae_compliance_flags))
    if decision.reasons:
        summary_table.add_row("Adjudication Reasons", " | ".join(decision.reasons))

    console.print(summary_table)

    # 3. Feature Breakdown Table
    feat_table = Table(title="Multi-Feature Biometric Breakdown", header_style="bold cyan")
    feat_table.add_column("Feature Channel", style="cyan")
    feat_table.add_column("Weight", justify="right")
    feat_table.add_column("Score", justify="right")
    feat_table.add_column("Assessment")

    f = v_res.features
    w = config.weights
    feat_table.add_row("HOG Gradient Orientation", f"{w.hog_weight:.2f}", f"{f.hog_similarity:.4f}", "[green]Consistent[/green]" if f.hog_similarity >= 0.7 else "[yellow]Divergent[/yellow]")
    feat_table.add_row("Hu Invariant Moments", f"{w.hu_moments_weight:.2f}", f"{f.hu_moments_similarity:.4f}", "[green]Consistent[/green]" if f.hu_moments_similarity >= 0.7 else "[yellow]Divergent[/yellow]")
    feat_table.add_row("Contour & Topology", f"{w.contour_weight:.2f}", f"{f.contour_similarity:.4f}", "[green]Consistent[/green]" if f.contour_similarity >= 0.7 else "[yellow]Divergent[/yellow]")
    feat_table.add_row("Zhang-Suen Skeleton Graph", f"{w.skeleton_weight:.2f}", f"{f.skeleton_similarity:.4f}", "[green]Consistent[/green]" if f.skeleton_similarity >= 0.7 else "[yellow]Divergent[/yellow]")
    feat_table.add_row("Stroke Width Consistency", f"{w.stroke_width_weight:.2f}", f"{f.stroke_width_variation_score:.4f}", "[green]Uniform[/green]" if f.stroke_width_variation_score >= 0.7 else "[yellow]Irregular[/yellow]")
    feat_table.add_row("Hesitation / Tremor Index", f"-{w.hesitation_penalty_weight:.2f}", f"{f.hesitation_score:.4f}", "[green]Natural Flow[/green]" if f.hesitation_score < 0.25 else "[red]Tremor / Blobbing Detected[/red]")

    console.print(feat_table)

    # 4. Audit Block Panel
    console.print(Panel(
        f"Sequence Number: [bold]{audit_rec.sequence_num}[/bold]\n"
        f"Record ID:       [dim]{audit_rec.record_id}[/dim]\n"
        f"Prev SHA-256:    [dim]{audit_rec.prev_hash}[/dim]\n"
        f"Block SHA-256:   [bold green]{audit_rec.hash}[/bold green]\n"
        f"Ledger DB:       [dim]{db_path}[/dim]",
        title="[bold green]Immutable Tamper-Evident Ledger Entry[/bold green]",
        border_style="green"
    ))

    return 0


# ============================================================================
# Subcommand 2: Cheque
# ============================================================================

def cmd_cheque(args: argparse.Namespace) -> int:
    """Process full cheque against specimen signature through full clearing pipeline."""
    cheque_path = args.cheque
    specimen_paths = list(args.specimen)
    specimen_path = ", ".join(specimen_paths)

    if not os.path.exists(cheque_path):
        console.print(f"[bold red]Error:[/bold red] Cheque image not found: {cheque_path}")
        return 1
    missing = [p for p in specimen_paths if not os.path.exists(p)]
    if missing:
        console.print(f"[bold red]Error:[/bold red] Specimen signature image not found: {missing[0]}")
        return 1

    cheque_img = cv2.imread(cheque_path)
    specimen_imgs = [cv2.imread(p) for p in specimen_paths]
    if cheque_img is None or any(im is None for im in specimen_imgs):
        console.print("[bold red]Error:[/bold red] Failed to load cheque or specimen image.")
        return 1

    config = DEFAULT_CONFIG
    db_path = args.db_path or "audit_ledger.db"
    jsonl_path = os.path.splitext(db_path)[0] + ".jsonl"
    logger = AuditLogger(db_path=db_path, jsonl_path=jsonl_path, config=config)
    pipeline = ChequeVerificationPipeline(config=config, audit_logger=logger)

    t0 = time.perf_counter()
    doc_id = format_document_id(getattr(args, "document_id", None), args.cheque_no, prefix="CHQ")
    report = pipeline.run(
        cheque_img,
        specimen_imgs,
        amount=args.amount,
        currency=args.currency,
        car_lar_match=not args.no_car_lar_match,
        positive_pay_match=not args.no_positive_pay,
        stale_days=args.stale_days,
        document_id=doc_id,
        operator_id=args.operator_id,
        audit_metadata={
            "cheque_path": cheque_path,
            "specimen_paths": specimen_paths,
            "cheque_no": args.cheque_no,
            "account_no": args.account_no,
        },
    )
    total_latency_ms = (time.perf_counter() - t0) * 1000.0
    iqa_metrics, detection_res, v_res = report.iqa, report.detection, report.verification
    decision, audit_rec, crop = report.decision, report.audit, report.signature_crop

    if args.save_crop:
        cv2.imwrite(args.save_crop, crop)
        console.print(f"[dim]Saved extracted signature crop to: {args.save_crop}[/dim]")

    tier_color, tier_badge = get_tier_style(decision.tier)

    # 1. Header
    console.print(Panel(
        f"[bold cyan]End-to-End Cheque Processing & Clearing Pipeline[/bold cyan]\n"
        f"Cheque Document: [dim]{cheque_path}[/dim]\n"
        f"Specimen Image:  [dim]{specimen_path}[/dim]\n"
        f"Total Latency:   [bold]{total_latency_ms:.2f} ms[/bold]",
        title="[bold blue]ICCS Clearing Pipeline[/bold blue]",
        border_style="cyan"
    ))

    # 2. IQA Table
    iqa_table = Table(title="1. Image Quality Assessment (ANSI X9.100-181 & CBUAE)", header_style="bold green")
    iqa_table.add_column("Metric")
    iqa_table.add_column("Measured Value")
    iqa_table.add_column("Specification Boundary")
    iqa_table.add_column("Status")

    iqa_status_badge = "[bold green]PASS[/bold green]" if iqa_metrics.passed else "[bold red]FAIL[/bold red]"
    iqa_table.add_row("Skew Tilt Angle", f"{iqa_metrics.skew_angle:+.2f}°", f"±{config.iqa.max_skew_degrees:.1f}°", "[green]PASS[/green]" if not iqa_metrics.is_skewed else "[red]FAIL[/red]")
    iqa_table.add_row("Laplacian Blur Variance", f"{iqa_metrics.blur_score:.1f}", f">= {config.iqa.min_blur_laplacian_var:.1f}", "[green]PASS[/green]" if not iqa_metrics.is_blurry else "[red]FAIL[/red]")
    iqa_table.add_row("RMS Contrast", f"{iqa_metrics.contrast_score:.1f}", f">= {config.iqa.min_contrast_rms:.1f}", "[green]PASS[/green]")
    iqa_table.add_row("Mean Luminance / Brightness", f"{iqa_metrics.brightness_score:.3f}", f"[{config.iqa.min_brightness:.2f}, {config.iqa.max_brightness:.2f}]", "[green]PASS[/green]" if not (iqa_metrics.is_too_dark or iqa_metrics.is_too_bright) else "[red]FAIL[/red]")
    iqa_table.add_row("Overall IQA Status", "-", "-", iqa_status_badge)
    console.print(iqa_table)

    # 3. Detection Table
    det_table = Table(title="2. Signature Zone Detection & Localization", header_style="bold yellow")
    det_table.add_column("Zone / Candidate")
    det_table.add_column("Bounding Box [x, y, w, h]")
    det_table.add_column("Confidence")
    det_table.add_column("Stroke Density")
    det_table.add_column("Selected")

    if detection_res.candidates:
        for idx, cand in enumerate(detection_res.candidates):
            b = cand.bbox
            is_best = (cand == detection_res.best_candidate)
            det_table.add_row(
                f"Candidate {idx + 1} ({cand.zone_type.value})",
                f"[{b.x}, {b.y}, {b.w}, {b.h}]",
                f"{cand.confidence:.2f}",
                f"{cand.stroke_density:.3f}",
                "[bold green]YES (Primary)[/bold green]" if is_best else "No",
            )
    else:
        det_table.add_row("Fallback ROI", "[Default bottom-right]", "0.50", "N/A", "[yellow]Fallback[/yellow]")
    console.print(det_table)

    # 4. Verification & Clearing Summary Table
    clr_table = Table(title="3. Biometric Verification & Clearing Adjudication", header_style="bold magenta")
    clr_table.add_column("Field", style="dim")
    clr_table.add_column("Adjudication Result")

    clr_table.add_row("Signature Band", f"{v_res.decision_band} (log-odds {v_res.match_logit:+.2f}, {v_res.reference_count} specimen(s))"
                      if v_res.match_logit is not None else str(v_res.decision_band))
    clr_table.add_row("Adjudicated Clearing Tier", tier_badge)
    clr_table.add_row("Workflow Action", f"[{tier_color}]{decision.action}[/{tier_color}]")
    clr_table.add_row("Courtesy Amount", f"{decision.currency} {decision.amount:,.2f} ({decision.amount_tier.value})")
    clr_table.add_row("Return Code", str(decision.return_code.value if decision.return_code else "CLEAR (Accepted)"))
    clr_table.add_row("Dual Sign-off Required", "Yes" if decision.requires_four_eyes else "No")
    if decision.cbuae_compliance_flags:
        clr_table.add_row("Compliance Flags", ", ".join(decision.cbuae_compliance_flags))
    if decision.reasons:
        clr_table.add_row("Adjudication Rationale", " | ".join(decision.reasons))
    console.print(clr_table)
    if v_res.explanation:
        console.print(Panel(render_text(v_res.explanation), title="Evidence-based explanation", border_style="magenta"))

    # 5. Audit Confirmation Panel
    console.print(Panel(
        f"Sequence Number: [bold]{audit_rec.sequence_num}[/bold]\n"
        f"Document ID:     [bold]{audit_rec.document_id}[/bold]\n"
        f"Record ID:       [dim]{audit_rec.record_id}[/dim]\n"
        f"Previous Block:  [dim]{audit_rec.prev_hash[:16]}...{audit_rec.prev_hash[-8:]}[/dim]\n"
        f"Current Digest:  [bold green]{audit_rec.hash}[/bold green]",
        title="[bold green]Audit Ledger Block Committed[/bold green]",
        border_style="green"
    ))

    return 0


# ============================================================================
# Subcommand: Signature-only comparison
# ============================================================================

def cmd_compare(args: argparse.Namespace) -> int:
    """Compare a questioned signature with reference signature(s); no cheque, no policy, no ledger."""
    paths = list(args.reference) + [args.questioned]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        console.print(f"[bold red]Error:[/bold red] Image not found: {missing[0]}")
        return 1
    refs = [cv2.imread(p) for p in args.reference]
    questioned = cv2.imread(args.questioned)
    if questioned is None or any(r is None for r in refs):
        console.print("[bold red]Error:[/bold red] Failed to decode one of the images.")
        return 1
    report = compare_signatures(refs, questioned)
    if args.json:
        print(report.model_dump_json(indent=2))
        return 0
    style = {"ACCEPT": "bold white on green", "REVIEW": "bold black on yellow",
             "REJECT": "bold white on red", "INCONCLUSIVE": "bold white on grey37"}[report.band]
    table = Table(show_header=False, box=None)
    table.add_row("Verdict", f"[{style}] {report.verdict} [/]")
    if report.match_score is not None:
        table.add_row("Similarity score", f"[bold]{report.match_score:.1f} / 100[/bold]  "
                                          f"(≥ 70 MATCH · 40–70 review · < 40 NO MATCH)")
        table.add_row("Log-odds", f"{report.log_odds:+.2f}  (MATCH ≥ {report.accept_threshold:+.2f}, "
                                  f"NO MATCH < {report.reject_threshold:+.2f})")
        table.add_row("Layout similarity", f"{report.layout_similarity:.3f}")
        table.add_row("Stroke-detail similarity", f"{report.stroke_detail_similarity:.3f} "
                                                  f"({report.consistent_stroke_features} consistent stroke features)")
    table.add_row("References used", f"{report.references_used} of {report.references_supplied}")
    console.print(Panel(table, title="[bold cyan]Signature Comparison (signature only)[/bold cyan]", border_style="cyan"))
    if report.explanation_text:
        console.print(Panel(report.explanation_text, title="Evidence", border_style="magenta"))
    return 0


# ============================================================================
# Subcommand 3: Audit
# ============================================================================

def cmd_audit(args: argparse.Namespace) -> int:
    """Inspect recent audit ledger blocks or verify cryptographic chain integrity."""
    db_path = args.db_path or "audit_ledger.db"
    jsonl_path = os.path.splitext(db_path)[0] + ".jsonl"

    if not os.path.exists(db_path):
        console.print(f"[bold yellow]Notice:[/bold yellow] Audit ledger database '{db_path}' does not exist yet. Initializing empty ledger.")
    
    logger = AuditLogger(db_path=db_path, jsonl_path=jsonl_path)

    # Verify cryptographic integrity
    if args.verify_chain:
        is_valid, err = logger.verify_integrity()
        count = logger.get_record_count()
        if is_valid:
            console.print(Panel(
                f"[bold green]✔ SHA-256 HASH-CHAIN INTEGRITY VERIFIED[/bold green]\n\n"
                f"Ledger Database: [dim]{db_path}[/dim]\n"
                f"Total Records:   [bold]{count}[/bold]\n"
                f"Status:          All blocks correctly chained to GENESIS with zero tamper detected.",
                title="[bold green]Cryptographic Chain Integrity[/bold green]",
                border_style="green"
            ))
            return 0
        else:
            console.print(Panel(
                f"[bold red]✖ TAMPERING OR CORRUPTION DETECTED[/bold red]\n\n"
                f"Ledger Database: [dim]{db_path}[/dim]\n"
                f"Violation:       {err}",
                title="[bold red]Ledger Integrity Compromised[/bold red]",
                border_style="red"
            ))
            return 1

    # Otherwise display recent records
    records = logger.get_recent_records(limit=args.limit, offset=args.offset)
    total_count = logger.get_record_count()

    table = Table(
        title=f"Recent Audit Ledger Records ({len(records)} shown of {total_count} total)",
        header_style="bold cyan"
    )
    table.add_column("Seq #", justify="right", style="bold")
    table.add_column("Timestamp (UTC)", style="dim")
    table.add_column("Document ID")
    table.add_column("Amount", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Tier")
    table.add_column("Action")
    table.add_column("Return Code", style="dim")
    table.add_column("SHA-256 Digest", style="dim")

    for r in records:
        _, tier_badge = get_tier_style(r.decision_tier)
        score_style = "green" if r.similarity_score >= 0.85 else ("yellow" if r.similarity_score >= 0.65 else "red")
        ret_code = r.metadata.get("return_code") or "-"
        hash_preview = f"{r.hash[:8]}...{r.hash[-6:]}"
        table.add_row(
            str(r.sequence_num),
            r.timestamp_utc[:19].replace("T", " "),
            r.document_id,
            f"{r.currency} {r.amount:,.2f}",
            f"[{score_style}]{r.similarity_score:.3f}[/{score_style}]",
            tier_badge,
            r.action,
            ret_code,
            hash_preview,
        )

    console.print(table)
    return 0


# ============================================================================
# Subcommand 4: Benchmark
# ============================================================================

def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run batch verification across sample dataset folders and print statistical report."""
    base_data_dir = args.data_dir or find_default_data_dir()
    if not base_data_dir or not os.path.exists(base_data_dir):
        console.print(f"[bold red]Error:[/bold red] Dataset directory not found: {args.data_dir or '(auto-detect)'}")
        return 1

    config = DEFAULT_CONFIG
    verifier = DeterministicVerifier(config=config)
    threshold = args.threshold or config.calibration.threshold_amber_min

    categories = ["genuine_pairs", "skilled_forgeries", "random_forgeries"]
    if args.category and args.category != "all":
        if args.category in categories:
            categories = [args.category]
        else:
            console.print(f"[bold red]Error:[/bold red] Invalid category '{args.category}'. Choose from: all, {', '.join(categories)}")
            return 1

    console.print(Panel(
        f"[bold cyan]Biometric Signature Benchmark Evaluation[/bold cyan]\n"
        f"Dataset Base:   [dim]{base_data_dir}[/dim]\n"
        f"Categories:     [bold]{', '.join(categories)}[/bold]\n"
        f"Match Threshold:[bold] {threshold:.2f}[/bold]\n"
        f"Max Limit/Cat:  [bold]{args.limit or 'All'}[/bold]",
        title="[bold blue]Benchmark Configuration[/bold blue]",
        border_style="cyan"
    ))

    overall_results: Dict[str, Any] = {}
    table = Table(title="Benchmark Performance Summary", header_style="bold magenta")
    table.add_column("Category", style="cyan")
    table.add_column("Ground Truth")
    table.add_column("Pairs", justify="right")
    table.add_column("Mean Score ± SD", justify="center")
    table.add_column("Min / Max", justify="center")
    table.add_column("Accuracy / Metric", justify="right")
    table.add_column("Status")

    total_pairs = 0
    total_correct = 0
    total_latency_ms = 0.0

    for cat in categories:
        cat_dir = os.path.join(base_data_dir, cat)
        manifest_path = os.path.join(cat_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            console.print(f"[yellow]Warning:[/yellow] Manifest not found in {cat_dir}, skipping.")
            continue

        with open(manifest_path, "r", encoding="utf-8") as f:
            pairs = json.load(f)

        if args.limit and args.limit > 0:
            pairs = pairs[:args.limit]

        scores: List[float] = []
        is_genuine_class = (cat == "genuine_pairs")
        correct_count = 0

        for item in pairs:
            ref_path = os.path.join(cat_dir, item["ref_image"])
            que_path = os.path.join(cat_dir, item["questioned_image"])
            ref_img = cv2.imread(ref_path)
            que_img = cv2.imread(que_path)
            if ref_img is None or que_img is None:
                continue

            t_start = time.perf_counter()
            v_res = verifier.verify(reference_image=ref_img, questioned_image=que_img)
            total_latency_ms += (time.perf_counter() - t_start) * 1000.0

            scores.append(v_res.similarity_score)

            if is_genuine_class:
                # Genuine should be >= threshold
                if v_res.similarity_score >= threshold:
                    correct_count += 1
            else:
                # Forgery should be < threshold
                if v_res.similarity_score < threshold:
                    correct_count += 1

        if not scores:
            continue

        n_pairs = len(scores)
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        min_score = float(np.min(scores))
        max_score = float(np.max(scores))
        acc_rate = (correct_count / n_pairs) * 100.0 if n_pairs > 0 else 0.0

        total_pairs += n_pairs
        total_correct += correct_count

        overall_results[cat] = {
            "pairs_evaluated": n_pairs,
            "mean_score": round(mean_score, 4),
            "std_dev": round(std_score, 4),
            "min_score": round(min_score, 4),
            "max_score": round(max_score, 4),
            "correct_predictions": correct_count,
            "accuracy_percent": round(acc_rate, 2),
        }

        metric_name = "TPR (Genuine Acceptance)" if is_genuine_class else "TNR (Forgery Rejection)"
        gt_label = "Genuine" if is_genuine_class else "Forgery"
        status_text = "[bold green]PASS[/bold green]" if acc_rate >= 80.0 else "[bold yellow]ACCEPTABLE[/bold yellow]"

        table.add_row(
            cat,
            gt_label,
            str(n_pairs),
            f"{mean_score:.3f} ± {std_score:.3f}",
            f"{min_score:.3f} / {max_score:.3f}",
            f"{acc_rate:.1f}% ({metric_name})",
            status_text,
        )

    console.print(table)

    overall_accuracy = (total_correct / total_pairs) * 100.0 if total_pairs > 0 else 0.0
    avg_latency = total_latency_ms / total_pairs if total_pairs > 0 else 0.0

    summary_panel = (
        f"Total Pairs Evaluated:   [bold]{total_pairs}[/bold]\n"
        f"Total Correct:           [bold]{total_correct}[/bold]\n"
        f"Overall System Accuracy: [bold green]{overall_accuracy:.2f}%[/bold green]\n"
        f"Avg Inference Latency:   [bold]{avg_latency:.2f} ms/pair[/bold] (~{1000.0 / avg_latency if avg_latency > 0 else 0:.0f} checks/sec)"
    )
    console.print(Panel(summary_panel, title="[bold green]Final Benchmark Statistics[/bold green]", border_style="green"))

    if args.save_report:
        report_data = {
            "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "threshold": threshold,
            "total_pairs": total_pairs,
            "overall_accuracy_percent": round(overall_accuracy, 2),
            "average_latency_ms": round(avg_latency, 2),
            "categories": overall_results,
        }
        with open(args.save_report, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        console.print(f"[dim]Saved JSON benchmark report to: {args.save_report}[/dim]")

    return 0


# ============================================================================
# Argument Parser Setup & Entrypoint
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="CLI tool for Signature Verification and Cheque Processing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # 1. verify
    v_parser = subparsers.add_parser("verify", help="Compare two signature crops (1-to-1 match)")
    v_parser.add_argument("--ref", required=True, help="Path to reference signature specimen")
    v_parser.add_argument("--test", "--questioned", dest="test", required=True, help="Path to questioned signature image")
    v_parser.add_argument("--amount", type=float, default=5000.0, help="Courtesy amount in currency")
    v_parser.add_argument("--currency", default="AED", help="Operational currency code")
    v_parser.add_argument("--cheque-no", default=None, help="Cheque identification number")
    v_parser.add_argument("--account-no", default=None, help="Drawer account number")
    v_parser.add_argument("--operator-id", default=None, help="Bank operator ID")
    v_parser.add_argument("--stale-days", type=int, default=0, help="Days elapsed since cheque date")
    v_parser.add_argument("--no-car-lar-match", action="store_true", help="Flag CAR/LAR figure/word mismatch")
    v_parser.add_argument("--no-positive-pay", action="store_true", help="Flag positive pay mismatch")
    v_parser.add_argument("--db-path", default="audit_ledger.db", help="Path to SQLite audit ledger")

    # 2. cheque
    c_parser = subparsers.add_parser("cheque", help="Process full cheque image through complete clearing pipeline")
    c_parser.add_argument("--cheque", required=True, help="Path to full cheque image")
    c_parser.add_argument("--specimen", required=True, nargs="+", help="Path(s) to enrolled specimen signature image(s); 3+ enables straight-through clearing")
    c_parser.add_argument("--amount", type=float, default=15000.0, help="Courtesy amount in currency")
    c_parser.add_argument("--currency", default="AED", help="Operational currency code")
    c_parser.add_argument("--cheque-no", default=None, help="Cheque identification number")
    c_parser.add_argument("--account-no", default=None, help="Drawer account number")
    c_parser.add_argument("--operator-id", default=None, help="Bank operator ID")
    c_parser.add_argument("--stale-days", type=int, default=0, help="Days elapsed since cheque date")
    c_parser.add_argument("--no-car-lar-match", action="store_true", help="Flag CAR/LAR mismatch")
    c_parser.add_argument("--no-positive-pay", action="store_true", help="Flag positive pay mismatch")
    c_parser.add_argument("--save-crop", default=None, help="Optional path to save extracted signature crop")
    c_parser.add_argument("--db-path", default="audit_ledger.db", help="Path to SQLite audit ledger")

    # 3. audit
    s_parser = subparsers.add_parser("compare", help="Signature only: compare a questioned signature with reference(s)")
    s_parser.add_argument("--reference", required=True, nargs="+", help="Reference signature image(s) of the account holder")
    s_parser.add_argument("--questioned", required=True, help="Signature image to verify")
    s_parser.add_argument("--json", action="store_true", help="Print the full result as JSON")

    d_parser = subparsers.add_parser("demo", help="Judge demo: composed cheques through the full pipeline")
    d_parser.add_argument("--out", default="demo_output", help="Directory for storyboard PNGs")

    a_parser = subparsers.add_parser("audit", help="Inspect audit records or verify SHA-256 chain integrity")
    a_parser.add_argument("--verify-chain", action="store_true", help="Verify SHA-256 hash-chain integrity")
    a_parser.add_argument("--recent", action="store_true", default=True, help="Display recent audit entries")
    a_parser.add_argument("--limit", type=int, default=10, help="Max records to display")
    a_parser.add_argument("--offset", type=int, default=0, help="Records offset")
    a_parser.add_argument("--db-path", default="audit_ledger.db", help="Path to SQLite audit ledger")

    # 4. benchmark
    b_parser = subparsers.add_parser("benchmark", help="Run batch verification on benchmark dataset folders")
    b_parser.add_argument("--data-dir", default=None, help="Base path to dataset samples")
    b_parser.add_argument("--category", choices=["all", "genuine_pairs", "skilled_forgeries", "random_forgeries"], default="all")
    b_parser.add_argument("--limit", type=int, default=None, help="Limit number of pairs per category")
    b_parser.add_argument("--threshold", type=float, default=0.65, help="Biometric match threshold")
    b_parser.add_argument("--save-report", default=None, help="Save JSON report file")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        return 0

    if args.subcommand == "verify":
        return cmd_verify(args)
    elif args.subcommand == "cheque":
        return cmd_cheque(args)
    elif args.subcommand == "compare":
        return cmd_compare(args)
    elif args.subcommand == "demo":
        from signature_verification_system.scripts.demo_showcase import main as showcase_main
        return showcase_main(["--out", args.out])
    elif args.subcommand == "audit":
        return cmd_audit(args)
    elif args.subcommand == "benchmark":
        return cmd_benchmark(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
