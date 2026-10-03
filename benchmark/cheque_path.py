#!/usr/bin/env python3
"""Cheque-path fidelity benchmark (EXP-009).

    python -m signature_verification_system.benchmark.cheque_path

For every 3-specimen trial query (genuine / skilled), compose the query scan
onto the demo cheque template (scripts/demo_showcase.compose_demo_cheque), run
the full pipeline (IQA → locate → crop → gate → verify), and compare with
verifying the clean scan directly. Reports band agreement and logit shift:
how much the localisation/crop stage degrades verification. Deterministic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from signature_verification_system.benchmark.protocol import load_images, writer_dependent_trials
from signature_verification_system.scripts.demo_showcase import TEMPLATE, compose_demo_cheque
from signature_verification_system.src.pipeline import ChequeVerificationPipeline
from signature_verification_system.src.verification.deterministic import DeterministicVerifier

PKG = Path(__file__).resolve().parent.parent


def run(refine_crop: bool = True, clean_rules: bool = True) -> dict:
    images = load_images(PKG / "data" / "samples")
    px = {s.image_id: cv2.imread(s.path) for s in images}
    template = cv2.imread(str(TEMPLATE))
    pipeline = ChequeVerificationPipeline(refine_crop=refine_crop, clean_rules=clean_rules)
    verifier = DeterministicVerifier()
    trials = [t for t in writer_dependent_trials(images) if t.label in ("genuine", "skilled")]
    rows = []
    for t in trials:
        refs = [px[r] for r in t.ref_ids]
        direct = verifier.verify_against_references(refs, px[t.query_id])
        rep = pipeline.run(compose_demo_cheque(template, px[t.query_id]), refs, amount=1000.0)
        v = rep.verification
        rows.append({"label": t.label, "direct_band": direct.decision_band, "cheque_band": v.decision_band,
                     "direct_logit": direct.match_logit, "cheque_logit": v.match_logit,
                     "crop_bbox": list(rep.crop_bbox) if rep.crop_bbox else None})
    out = {}
    for lab in ("genuine", "skilled"):
        rs = [r for r in rows if r["label"] == lab]
        shifts = [r["cheque_logit"] - r["direct_logit"] for r in rs if r["cheque_logit"] is not None]
        out[lab] = {
            "n": len(rs),
            "band_agreement": sum(r["direct_band"] == r["cheque_band"] for r in rs) / len(rs),
            "cheque_accept": sum(r["cheque_band"] == "ACCEPT" for r in rs) / len(rs),
            "cheque_reject": sum(r["cheque_band"] == "REJECT" for r in rs) / len(rs),
            "cheque_inconclusive": sum(r["cheque_band"] == "INCONCLUSIVE" for r in rs) / len(rs),
            "direct_accept": sum(r["direct_band"] == "ACCEPT" for r in rs) / len(rs),
            "direct_reject": sum(r["direct_band"] == "REJECT" for r in rs) / len(rs),
            "mean_logit_shift": float(np.mean(shifts)) if shifts else None,
            "mean_abs_logit_shift": float(np.mean(np.abs(shifts))) if shifts else None,
        }
    return {"summary": out, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PKG / "benchmark" / "results" / "cheque_path.json"))
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--no-clean", action="store_true")
    args = ap.parse_args()
    res = run(refine_crop=not args.no_refine, clean_rules=not args.no_clean)
    Path(args.out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    for lab, s in res["summary"].items():
        print(f"{lab:8s} n={s['n']:3d} band agreement={s['band_agreement']:.2f} | direct A/X={s['direct_accept']:.2f}/{s['direct_reject']:.2f} "
              f"| cheque A/X/INC={s['cheque_accept']:.2f}/{s['cheque_reject']:.2f}/{s['cheque_inconclusive']:.2f} "
              f"| logit shift mean={s['mean_logit_shift']:+.2f} |abs|={s['mean_abs_logit_shift']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
