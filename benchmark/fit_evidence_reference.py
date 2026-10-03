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

import json
from pathlib import Path
from typing import Callable, Dict

import cv2
import numpy as np

from signature_verification_system.benchmark import metrics as M
from signature_verification_system.benchmark.protocol import load_images, one_to_one_pairs
from signature_verification_system.src.verification.features import SignatureFeatures, extract_features
from signature_verification_system.src.verification.similarity import (
    compare, ink_density_agreement, keypoint_similarity, proportion_agreement,
)

PKG = Path(__file__).resolve().parent.parent
OUT = PKG / "src" / "verification" / "evidence_reference.json"
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

SIGNALS: Dict[str, Callable[[SignatureFeatures, SignatureFeatures], float]] = {
    "shape_similarity": lambda a, b: compare(a, b).shape,   # aligned layout, as in production
    "keypoint_similarity": lambda a, b: keypoint_similarity(a, b).similarity,
    "keypoint_inliers": lambda a, b: float(keypoint_similarity(a, b).inliers),
    "proportion_agreement": proportion_agreement,
    "ink_density_agreement": ink_density_agreement,
}


def _band_reliability() -> Dict[str, Dict[str, Dict[str, int]]]:
    """Held-out (writer-disjoint CV) composition of each decision band, from
    benchmark/results/thresholds.json written by select_thresholds.py."""
    cv = json.loads((PKG / "benchmark" / "results" / "thresholds.json").read_text())
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
    images = [s for s in load_images(PKG / "data" / "samples") if s.domain == "cedar"]
    feats = {s.image_id: extract_features(cv2.imread(s.path)) for s in images}
    pairs = [p for p in one_to_one_pairs(images) if p.label in ("genuine", "skilled")]
    ref: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, fn in SIGNALS.items():
        ref[name] = {}
        by_label = {}
        for lab in ("genuine", "skilled"):
            vals = np.array([fn(feats[p.ref_id], feats[p.query_id]) for p in pairs if p.label == lab])
            by_label[lab] = vals
            ref[name][lab] = {f"p{int(q * 100):02d}": round(float(np.quantile(vals, q, method="lower")), 6) for q in QUANTILES}
            ref[name][lab]["n"] = int(len(vals))
        ref[name]["auc_genuine_vs_skilled"] = round(M.auc(by_label["genuine"], by_label["skilled"]), 4)
    payload = {
        "source": "CEDAR 12 writers, 1:1 genuine vs skilled pairs",
        "signals": ref,
        "band_reliability": _band_reliability(),
    }
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    for name, d in ref.items():
        print(f"{name:22s} AUC={d['auc_genuine_vs_skilled']:.3f} genuine p25={d['genuine']['p25']:.3f} median={d['genuine']['p50']:.3f} | skilled median={d['skilled']['p50']:.3f} p75={d['skilled']['p75']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
