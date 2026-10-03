#!/usr/bin/env python3
"""Select production ACCEPT / REVIEW / REJECT logit thresholds (EXP-004).

    python -m signature_verification_system.benchmark.select_thresholds

Rule (validated under writer-disjoint CV, reported below):
  accept      = max impostor logit (skilled + random) + accept_margin
  reject      = genuine logit at the reject_genuine_quantile (order statistic)
  hard_reject = min genuine logit
Production values are selected on ALL writers and printed for
`DecisionThresholds` in src/core/config.py. Deterministic: order statistics only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from signature_verification_system.benchmark import metrics as M
from signature_verification_system.benchmark.protocol import (
    load_images, one_to_one_pairs, writer_dependent_trials, writer_folds,
)
from signature_verification_system.src.core.config import DEFAULT_CONFIG
from signature_verification_system.src.verification.features import extract_features
from signature_verification_system.src.verification.similarity import compare, compare_multi

PKG = Path(__file__).resolve().parent.parent
Row = Tuple[str, float, int]  # label, logit, fold (-1 = cross-fold)


def _select(rows: List[Row]) -> Dict[str, float]:
    d = DEFAULT_CONFIG.decision
    g = [x for l, x, _ in rows if l == "genuine"]
    imp = [x for l, x, _ in rows if l != "genuine"]
    return M.select_zone_thresholds(g, imp, 0.0, d.reject_genuine_quantile, d.accept_margin_logit)


def _cv(rows: List[Row]) -> Dict[str, Dict[str, float]]:
    counts: Dict[str, Dict[str, float]] = {}
    for test in (0, 1):
        z = _select([r for r in rows if r[2] == 1 - test])
        for lab in ("genuine", "skilled", "random"):
            s = np.array([x for l, x, f in rows if f == test and l == lab])
            c = counts.setdefault(lab, {"n": 0, "accept": 0, "reject": 0})
            c["n"] += len(s)
            c["accept"] += int(np.sum(s >= z["accept"]))
            c["reject"] += int(np.sum(s < z["reject"]))
    return counts


def main() -> int:
    images = load_images(PKG / "data" / "samples")
    by_id = {s.image_id: s for s in images}
    folds = writer_folds(images)
    feats = {s.image_id: extract_features(cv2.imread(s.path)) for s in images}

    def fold(a: str, b: str) -> int:
        x, y = folds[by_id[a].writer], folds[by_id[b].writer]
        return x if x == y else -1

    single: List[Row] = [
        (p.label, compare(feats[p.ref_id], feats[p.query_id]).fused_logit, fold(p.ref_id, p.query_id))
        for p in one_to_one_pairs(images)
    ]
    multi: List[Row] = [
        (t.label, compare_multi([feats[r] for r in t.ref_ids], feats[t.query_id]).fused_logit, fold(t.ref_ids[0], t.query_id))
        for t in writer_dependent_trials(images)
    ]
    report = {}
    for name, rows in (("single", single), ("multi", multi)):
        report[name] = {"production": _select(rows), "cv_heldout_counts": _cv(rows)}
        p, cv = report[name]["production"], report[name]["cv_heldout_counts"]
        print(f"{name}: accept={p['accept']:.4f} reject={p['reject']:.4f} hard_reject={p['hard_reject']:.4f}")
        for lab, c in cv.items():
            print(f"   CV {lab:8s} n={c['n']:4d} auto-accept={c['accept']:3d} ({c['accept']/c['n']:.3f})  rejected={c['reject']:3d} ({c['reject']/c['n']:.3f})")
    (PKG / "benchmark" / "results" / "thresholds.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
