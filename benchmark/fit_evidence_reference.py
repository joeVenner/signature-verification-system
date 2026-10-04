#!/usr/bin/env python3
"""Empirical reference distributions for explanation evidence (EXP-008).

    python -m signature_verification_system.benchmark.fit_evidence_reference

For each measured comparison signal, record quantiles over genuine and
skilled-forgery 1:1 pairs. The explanation layer states where a new
comparison falls relative to these, so every sentence is a measured fact
("layout similarity 0.81 — typical of genuine pairs"), never a free-text guess.
Writes src/verification/evidence_reference.json. Deterministic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from signature_verification_system.benchmark import metrics as M
from signature_verification_system.benchmark.conditions import CONDITIONS
from signature_verification_system.benchmark.protocol import load_images, one_to_one_pairs
from signature_verification_system.src.verification.features import extract_features
from signature_verification_system.src.verification.similarity import (
    compare, ink_density_agreement, pair_evidence_signals, proportion_agreement,
)

PKG = Path(__file__).resolve().parent.parent
OUT = PKG / "src" / "verification" / "evidence_reference.json"
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

BASIS = ("2-fold writer-disjoint CV on {writers} CEDAR writers: cut-offs re-selected on one fold, counted on "
         "the other; production cut-offs use all {writers} writers. Random-forgery counts cover same-fold pairs "
         "only. Indicative, not a guarantee.")


def _band_reliability(thresholds_path: Path) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Held-out (writer-disjoint CV) composition of each decision band, from
    the thresholds JSON written by select_thresholds.py."""
    cv = json.loads(thresholds_path.read_text())
    out: Dict[str, Dict[str, Dict[str, int]]] = {}
    for mode in ("single", "multi"):
        counts = cv[mode]["cv_heldout_counts"]
        out[mode] = {}
        for lab, c in counts.items():
            review = c["n"] - c["accept"] - c["reject"]
            for band, k in (("ACCEPT", c["accept"]), ("REVIEW", review), ("REJECT", c["reject"])):
                out[mode].setdefault(band, {})[lab] = int(k)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(PKG / "data" / "samples"))
    ap.add_argument("--thresholds", default=str(PKG / "benchmark" / "results" / "thresholds.json"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--condition", choices=sorted(CONDITIONS), default="harmonized",
                    help="input condition; harmonized (default since EXP-017) matches the clearance score")
    args = ap.parse_args()
    images = [s for s in load_images(Path(args.data_dir)) if s.domain == "cedar"]
    prepare = CONDITIONS[args.condition]
    feats = {s.image_id: extract_features(prepare(cv2.imread(s.path))) for s in images}
    pairs = [p for p in one_to_one_pairs(images) if p.label in ("genuine", "skilled")]

    measured: Dict[str, Dict[str, list]] = {}
    for p in pairs:
        a, b = feats[p.ref_id], feats[p.query_id]
        pair = compare(a, b)
        values = {
            "shape_similarity": pair.shape,
            "keypoint_similarity": pair.keypoint.similarity,
            "keypoint_inliers": float(pair.keypoint.inliers),
            "proportion_agreement": proportion_agreement(a, b),
            "ink_density_agreement": ink_density_agreement(a, b),
            **pair_evidence_signals(pair),
        }
        for name, value in values.items():
            measured.setdefault(name, {"genuine": [], "skilled": []})[p.label].append(value)

    ref: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, by_label in measured.items():
        ref[name] = {}
        for lab in ("genuine", "skilled"):
            vals = np.array(by_label[lab])
            ref[name][lab] = {f"p{int(q * 100):02d}": round(float(np.quantile(vals, q, method="lower")), 6) for q in QUANTILES}
            ref[name][lab]["n"] = int(len(vals))
        ref[name]["auc_genuine_vs_skilled"] = round(M.auc(np.array(by_label["genuine"]), np.array(by_label["skilled"])), 4)
    writers = len({s.writer for s in images})
    payload = {
        "source": f"CEDAR {writers} writers, 1:1 genuine vs skilled pairs",
        "basis": BASIS.format(writers=writers),
        "signals": ref,
        "band_reliability": _band_reliability(Path(args.thresholds)),
    }
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    for name, d in ref.items():
        print(f"{name:30s} AUC={d['auc_genuine_vs_skilled']:.3f} genuine p25={d['genuine']['p25']:.3f} median={d['genuine']['p50']:.3f} | skilled median={d['skilled']['p50']:.3f} p75={d['skilled']['p75']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
