#!/usr/bin/env python3
"""Judge-facing demo: cheque → signature → quality → analysis → match → risk → decision.

    python signature_verification_system/scripts/demo_showcase.py [--out DIR]

HONESTY NOTE (printed in the demo): the sample cheques have no enrolled
specimen signatures, so each demo cheque is COMPOSED by placing a real CEDAR
questioned signature (genuine or skilled forgery, unmodified scan pixels,
multiplicative ink blend) into the signature box of a sample cheque. Everything
after composition (localisation, quality gate, verification, decision, audit)
is the production pipeline running for real. Scenario writer and images are
fixed for reproducibility; dataset-level performance is in benchmark/.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from signature_verification_system.src.adjudication.audit_logger import AuditLogger  # noqa: E402
from signature_verification_system.src.pipeline import ChequeVerificationPipeline, PipelineReport  # noqa: E402
from signature_verification_system.src.preprocessing.normalization import normalize_signature  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "samples"
TEMPLATE = DATA / "cheques" / "cheque_procedural_pantograph_1001.png"
SIGNING_AREA = (1470, 625, 2240, 786)  # x0, y0, x1, y1 above the signatory line of TEMPLATE
SPECIMENS = [  # CEDAR writer 05, signing turns 1-3 (enrolled on file)
    "genuine_pairs/pair_09_cedar_w05_ref.png",
    "genuine_pairs/pair_09_cedar_w05_questioned.png",
    "genuine_pairs/pair_10_cedar_w05_ref.png",
]
SCENARIOS = [
    {"id": "A", "title": "Genuine signature, routine amount",
     "signature": "genuine_pairs/pair_10_cedar_w05_questioned.png", "truth": "genuine (CEDAR w05, turn 4, not enrolled)",
     "amount": 4_750.0},
    {"id": "B", "title": "Skilled forgery of the same signature",
     "signature": "skilled_forgeries/pair_09_cedar_w05_forgery01.png", "truth": "skilled forgery (CEDAR w05, forgery 1)",
     "amount": 4_750.0},
    {"id": "C", "title": "Genuine signature, badly blurred capture",
     "signature": "genuine_pairs/pair_10_cedar_w05_questioned.png", "truth": "genuine, capture blurred (Gaussian σ=6 on cheque)",
     "amount": 4_750.0, "blur_sigma": 6.0},
    {"id": "D", "title": "Genuine signature, high-value cheque",
     "signature": "genuine_pairs/pair_10_cedar_w05_questioned.png", "truth": "genuine (CEDAR w05, turn 4)",
     "amount": 150_000.0},
]

console = Console()
STATUS_STYLE = {"PASS": "bold green", "WARN": "bold yellow", "FAIL": "bold red", "INFO": "bold cyan"}
TIER_STYLE = {"GREEN": "bold white on green", "AMBER": "bold black on yellow", "RED": "bold white on red"}


def compose_demo_cheque(template: np.ndarray, signature_scan: np.ndarray) -> np.ndarray:
    """Clear the template's signing area and lay the scan's ink onto the paper.

    Uses the unmodified scan pixels as ink transmission (scan / paper level),
    multiplied onto the cheque, at native scale unless the area is too small.
    """
    x0, y0, x1, y1 = SIGNING_AREA
    out = template.copy()
    region = cv2.cvtColor(out[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    mask = np.zeros(out.shape[:2], np.uint8)
    mask[y0:y1, x0:x1] = (region < 200).astype(np.uint8) * 255
    out = cv2.inpaint(out, cv2.dilate(mask, np.ones((5, 5), np.uint8)), 5, cv2.INPAINT_TELEA)

    gray = cv2.cvtColor(signature_scan, cv2.COLOR_BGR2GRAY).astype(np.float64)
    bx, by, bw, bh = normalize_signature(signature_scan).bbox
    trans = np.clip(gray[by:by + bh, bx:bx + bw] / max(float(np.median(gray)), 1.0), 0.0, 1.0)
    scale = min(1.0, (y1 - y0 - 8) / trans.shape[0], (x1 - x0 - 60) / trans.shape[1])
    if scale < 1.0:
        trans = cv2.resize(trans, (int(trans.shape[1] * scale), int(trans.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    px, py = x0 + 50, y1 - 6 - trans.shape[0]
    roi = out[py:py + trans.shape[0], px:px + trans.shape[1]].astype(np.float64)
    out[py:py + trans.shape[0], px:px + trans.shape[1]] = np.round(roi * trans[..., None]).astype(np.uint8)
    return out


def _thumb(img: np.ndarray, h: int) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.resize(img, (max(1, int(img.shape[1] * h / img.shape[0])), h), interpolation=cv2.INTER_AREA)


def storyboard(cheque: np.ndarray, report: PipelineReport, specimens: List[np.ndarray], title: str) -> np.ndarray:
    """Single PNG: cheque with detected box, crop, specimens, measured result banner."""
    vis = cheque.copy()
    if report.crop_bbox:
        x, y, w, h = report.crop_bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 140, 255), 6)
    top = _thumb(vis, 420)
    crop_t = _thumb(report.signature_crop, 160)
    specs = np.hstack([cv2.copyMakeBorder(_thumb(s, 160), 0, 0, 6, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for s in specimens])
    row = np.hstack([crop_t, np.full((160, 40, 3), 255, np.uint8), specs])
    width = max(top.shape[1], row.shape[1])

    def pad(a: np.ndarray) -> np.ndarray:
        return cv2.copyMakeBorder(a, 0, 0, 0, width - a.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))

    tier = report.decision.tier.value
    colour = {"GREEN": (60, 160, 60), "AMBER": (0, 190, 240), "RED": (40, 40, 210)}[tier]
    banner = np.full((90, width, 3), colour, np.uint8)
    v = report.verification
    text = f"{title} | band {v.decision_band} | log-odds {f"{v.match_logit:+.2f}" if v.match_logit is not None else "n/a"} | {tier} {report.decision.action}"
    cv2.putText(banner, text, (14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
    labels = np.full((34, width, 3), 255, np.uint8)
    cv2.putText(labels, "Extracted from cheque", (4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(labels, "Enrolled specimens", (crop_t.shape[1] + 46, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 1, cv2.LINE_AA)
    note = np.full((34, width, 3), 255, np.uint8)
    cv2.putText(note, "Demo cheque composed from a real CEDAR signature scan; all analysis below is live pipeline output.",
                (4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 1, cv2.LINE_AA)
    return np.vstack([banner, pad(top), note, labels, pad(row)])


def render(report: PipelineReport, scenario: Dict, cheque_name: str) -> None:
    console.rule(f"[bold]Scenario {scenario['id']} — {scenario['title']}")
    console.print(f"[dim]Ground truth (not shown to the system): {scenario['truth']} · amount AED {scenario['amount']:,.2f}[/dim]")
    console.print(f"[dim]Cheque: composed on {cheque_name}[/dim]")
    flow = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    flow.add_column("Stage")
    flow.add_column("Status")
    flow.add_column("Measured result")
    for s in report.stages:
        flow.add_row(s.name.replace("_", " "), f"[{STATUS_STYLE[s.status]}]{s.status}[/]", s.summary)
    console.print(flow)
    v = report.verification
    expl = v.explanation or {}
    if expl.get("evidence"):
        ev = Table(title="Evidence (measured value vs genuine / forgery medians)", box=None)
        for col in ("Signal", "Value", "Genuine med.", "Forgery med.", "AUC", "Status"):
            ev.add_column(col)
        for it in expl["evidence"] + expl.get("observations", []):
            weak = it["discriminative_auc"] < 0.80
            ev.add_row(it["label"] + (" [dim](weak)[/dim]" if weak else ""), f"{it['value']:.3f}", f"{it['genuine_median']:.3f}",
                       f"{it['skilled_forgery_median']:.3f}", f"{it['discriminative_auc']:.2f}",
                       {"CONSISTENT": "[green]✓ consistent[/]", "BORDERLINE": "[yellow]~ borderline[/]",
                        "INCONSISTENT": "[red]✗ inconsistent[/]"}[it["status"]])
        console.print(ev)
    rel = expl.get("band_reliability")
    if rel:
        console.print(f"Band reliability ({rel['mode']}, held-out writers): {rel['validation_trials_in_band']} validation trials landed in "
                      f"{v.decision_band}: {rel['of_which_genuine']} genuine · {rel['of_which_skilled_forgery']} skilled forgery · "
                      f"{rel['of_which_random_forgery']} other writer")
    d = report.decision
    body = "\n".join(f"• {r}" for r in d.reasons) or "•"
    if d.return_code is not None:
        body += f"\nReturn code: {d.return_code.value}"
    if report.audit:
        body += f"\n\nAudit block #{report.audit.sequence_num}  sha256 {report.audit.hash[:16]}…  prev {report.audit.prev_hash[:16]}…"
    console.print(Panel(body, title=f"[{TIER_STYLE[d.tier.value]}] {d.tier.value} · {d.action} [/]", border_style="white"))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="demo_output", help="Directory for storyboard PNGs")
    ap.add_argument("--db-path", default=None, help="Audit ledger path (default: temporary, not the real ledger)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tmp = None
    if args.db_path is None:
        tmp = tempfile.TemporaryDirectory()
        db = Path(tmp.name) / "demo_ledger.db"
    else:
        db = Path(args.db_path)
    logger = AuditLogger(db_path=str(db), jsonl_path=str(db.with_suffix(".jsonl")))
    pipeline = ChequeVerificationPipeline(audit_logger=logger)

    template = cv2.imread(str(TEMPLATE))
    specimens = [cv2.imread(str(DATA / s)) for s in SPECIMENS]
    console.print(Panel(
        "Deterministic cheque signature verification · no ML weights · every number below is measured\n"
        "Demo cheques are COMPOSED: a real CEDAR signature scan is placed into a sample cheque's signature box.\n"
        f"Account holder on file: CEDAR writer 05, {len(specimens)} enrolled specimens.",
        title="[bold cyan]Signature Verification — Live Demo[/]"))
    summary = []
    for sc in SCENARIOS:
        cheque = compose_demo_cheque(template, cv2.imread(str(DATA / sc["signature"])))
        if sc.get("blur_sigma"):
            cheque = cv2.GaussianBlur(cheque, (0, 0), sc["blur_sigma"])
        report = pipeline.run(cheque, specimens, amount=sc["amount"], document_id=f"CHQ-DEMO-{sc['id']}",
                              audit_metadata={"demo_scenario": sc["id"], "composed_cheque": True})
        render(report, sc, TEMPLATE.name)
        cv2.imwrite(str(out / f"scenario_{sc['id']}.png"), storyboard(cheque, report, specimens, f"Scenario {sc['id']}"))
        summary.append((sc, report))

    table = Table(title="Demo summary")
    for col in ("Scenario", "Ground truth", "Signature band", "Log-odds", "Clearing decision"):
        table.add_column(col)
    for sc, r in summary:
        lo = r.verification.match_logit
        table.add_row(sc["id"] + " " + sc["title"], sc["truth"], r.verification.decision_band,
                      f"{lo:+.2f}" if lo is not None else "—", f"{r.decision.tier.value} / {r.decision.action}")
    console.print(table)
    ok, msg = logger.verify_integrity()
    console.print(f"Audit chain integrity: {'[green]VALID[/]' if ok else '[red]BROKEN[/]'} ({msg})")
    console.print(f"[dim]Storyboards written to {out.resolve()}[/dim]")
    if tmp:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
