"""Evidence-based verification explanation (requirement A, EXP-008).

Every statement is a measured value placed against empirical reference
distributions (src/verification/evidence_reference.json, produced by
benchmark/fit_evidence_reference.py). Nothing here is free text about the
signature: if a signal was not measured, it is not mentioned.

Status per signal:
  CONSISTENT    value >= 25th percentile of genuine pairs
  INCONSISTENT  value <= median of skilled-forgery pairs
  BORDERLINE    in between
Signals whose genuine-vs-skilled AUC is < 0.80 are reported as
"observations" (weak, not decisive), never as evidence.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

_REFERENCE_PATH = Path(__file__).with_name("evidence_reference.json")
EVIDENCE_MIN_AUC = 0.80

LABELS = {
    "shape_similarity": "Overall layout & stroke directions",
    "keypoint_similarity": "Fine stroke detail (local keypoints)",
    "keypoint_inliers": "Geometrically consistent stroke features",
    "proportion_agreement": "Signature proportions (width : height)",
    "ink_density_agreement": "Ink coverage density",
}


@lru_cache(maxsize=1)
def load_reference() -> Dict[str, Any]:
    return json.loads(_REFERENCE_PATH.read_text())


def _status(value: float, genuine: Dict[str, float], skilled: Dict[str, float]) -> str:
    if value >= genuine["p25"]:
        return "CONSISTENT"
    if value <= skilled["p50"]:
        return "INCONSISTENT"
    return "BORDERLINE"


def build_explanation(
    signals: Dict[str, float],
    band: str,
    match_logit: Optional[float],
    reference_count: int,
    quality: Optional[Dict[str, Any]] = None,
    alignment: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Structured explanation from measured signals; deterministic and JSON-safe.

    `alignment` (rotation_deg, scale) is reported only when the RANSAC-aligned
    layout comparison was actually the one used in the score (EXP-014).
    """
    ref = load_reference()
    evidence: List[Dict[str, Any]] = []
    observations: List[Dict[str, Any]] = []
    for name in sorted(signals):
        if name not in ref["signals"]:
            continue
        r = ref["signals"][name]
        item = {
            "signal": name,
            "label": LABELS.get(name, name),
            "value": round(float(signals[name]), 4),
            "status": _status(float(signals[name]), r["genuine"], r["skilled"]),
            "genuine_median": r["genuine"]["p50"],
            "skilled_forgery_median": r["skilled"]["p50"],
            "discriminative_auc": r["auc_genuine_vs_skilled"],
        }
        (evidence if r["auc_genuine_vs_skilled"] >= EVIDENCE_MIN_AUC else observations).append(item)

    mode = "multi" if reference_count >= 2 else "single"
    reliability = None
    if band in ("ACCEPT", "REVIEW", "REJECT"):
        comp = ref["band_reliability"][mode][band]
        total = sum(comp.values())
        reliability = {
            "mode": f"{mode}-specimen",
            "validation_trials_in_band": total,
            "of_which_genuine": comp.get("genuine", 0),
            "of_which_skilled_forgery": comp.get("skilled", 0),
            "of_which_random_forgery": comp.get("random", 0),
            "basis": ("2-fold writer-disjoint CV on 12 CEDAR writers (6 per fold): cut-offs re-selected on one "
                      "fold, counted on the other; production cut-offs use all 12 writers. Random-forgery "
                      "counts cover same-fold pairs only. Small sample: indicative, not a guarantee."),
        }

    verified = bool(signals)
    preprocessing = ["Background flattened and ink level normalised (2-pass flat-field)",
                     "Scanner noise suppressed (non-local means)"] if verified else []
    q_section: Dict[str, Any] = {}
    if quality:
        qq = quality.get("questioned") or {}
        if verified and qq.get("polarity_inverted"):
            preprocessing.append("Inverted polarity detected and corrected")
        q_section = {
            "questioned_passed": qq.get("passed"),
            "ink_contrast": qq.get("ink_contrast"),
            "edge_sharpness": qq.get("edge_sharpness"),
            "noise_ratio": qq.get("noise_ratio"),
            "blocking_issues": qq.get("blocking_issues", []),
            "warnings": qq.get("warnings", []),
            "references_passed": sum(1 for r in quality.get("references", []) if r.get("passed")),
            "references_total": len(quality.get("references", [])),
            "reference_issues": [
                {"specimen": i + 1, "blocking_issues": r.get("blocking_issues", [])}
                for i, r in enumerate(quality.get("references", [])) if not r.get("passed")
            ],
        }

    return {
        "decision_band": band,
        "match_logit": match_logit,
        "reference_count": reference_count,
        "band_reliability": reliability,
        "evidence": evidence,
        "signal_source": "best value per signal across specimens" if reference_count >= 2 else "single specimen",
        "observations": observations,
        "image_quality": q_section,
        "preprocessing": preprocessing,
        "alignment": alignment if verified else None,
    }


_MARK = {"CONSISTENT": "✓", "BORDERLINE": "~", "INCONSISTENT": "✗"}


def render_text(explanation: Dict[str, Any]) -> str:
    """Human-readable report; every line traces to a field of `explanation`.

    The balanced-prior match probability is deliberately not shown: it is
    over-confident (EXP-004). Log-odds plus the empirical band composition on
    held-out writers is the honest statement of confidence.
    """
    e = explanation
    lines = ["VERIFICATION RESULT", "-------------------", f"Decision: {e['decision_band']}"]
    if e.get("match_logit") is not None:
        lines.append(f"Match log-odds: {e['match_logit']:+.2f}   specimens compared: {e['reference_count']}")
    r = e.get("band_reliability")
    if r:
        lines.append(
            f"Band reliability ({r['mode']}): of {r['validation_trials_in_band']} held-out validation trials in this band, "
            f"{r['of_which_genuine']} genuine / {r['of_which_skilled_forgery']} skilled forgery / {r['of_which_random_forgery']} other writer"
        )
        lines.append(f"  ({r['basis']})")
    if e["evidence"]:
        src = " — best value per signal across specimens" if e.get("signal_source", "").startswith("best") else ""
        lines += ["", f"Evidence (measured vs. genuine-pair and skilled-forgery medians){src}:"]
        for it in e["evidence"]:
            lines.append(f" {_MARK[it['status']]} {it['label']}: {it['value']:.3f} "
                         f"(genuine {it['genuine_median']:.3f} / forgery {it['skilled_forgery_median']:.3f}) {it['status']}")
    if e["observations"]:
        lines += ["", "Supporting observations (weak discriminators, not decisive):"]
        for it in e["observations"]:
            lines.append(f" {_MARK[it['status']]} {it['label']}: {it['value']:.3f} (AUC {it['discriminative_auc']:.2f})")
    q = e.get("image_quality") or {}
    if q:
        lines += ["", "Image quality:"]
        mark = {True: "✓", False: "✗"}.get(q.get("questioned_passed"), "?")
        lines.append(f" {mark} Questioned image: contrast {q.get('ink_contrast')}, "
                     f"sharpness {q.get('edge_sharpness')}, noise ratio {q.get('noise_ratio')}")
        for issue in q.get("blocking_issues", []):
            lines.append(f" ✗ {issue}")
        for w in q.get("warnings", []):
            lines.append(f" ! {w}")
        lines.append(f" Specimens usable: {q.get('references_passed')}/{q.get('references_total')}")
        for ri in q.get("reference_issues", []):
            lines.append(f" ✗ Specimen #{ri['specimen']}: {', '.join(ri['blocking_issues']) or 'unusable'}")
    if e["preprocessing"]:
        lines += ["", "Preprocessing applied:"] + [f" • {p}" for p in e["preprocessing"]]
        al = e.get("alignment")
        if al:
            turn = "clockwise" if al["rotation_deg"] > 0 else "counter-clockwise"
            lines.append(f" • Geometric alignment applied: {abs(al['rotation_deg']):.1f}° {turn}, scale ×{al['scale']:.2f} "
                         f"(specimen #{al['specimen']} aligned onto the questioned signature; includes natural slant difference)")
        else:
            lines.append(" • No geometric alignment applied (unaligned layout comparison scored higher or was unavailable)")
    else:
        lines += ["", "No similarity score was computed: image quality is insufficient for a reliable verdict."]
    return "\n".join(lines)
