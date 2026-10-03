#!/usr/bin/env python3
"""Fit the writer-independent logistic fusion used by the v3 verifier.

    python -m signature_verification_system.benchmark.fit_fusion

1. Computes (shape, keypoint) similarities for every 1:1 protocol pair.
2. Writer-disjoint 2-fold CV: fit on one fold, score the other -> honest
   held-out AUC/EER for the *fusion procedure itself*.
3. Fits the production coefficients on all writers and prints them for
   `FusionModel` in src/core/config.py.

Deterministic: lbfgs on a fixed, sorted design matrix; no randomness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression

from signature_verification_system.benchmark import metrics as M
from signature_verification_system.benchmark.protocol import load_images, one_to_one_pairs, writer_folds
from signature_verification_system.src.verification.features import extract_features
from signature_verification_system.src.verification.similarity import compare

PKG = Path(__file__).resolve().parent.parent
FEATURE_NAMES = ("shape", "keypoint")


def _fit(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """Balanced, standardised L2 logistic regression; returns raw-unit weights."""
    mu, sd = X.mean(0), X.std(0) + 1e-12
    pos = max(1, int(y.sum()))
    w = np.where(y == 1, (len(y) - pos) / pos, 1.0)
    model = LogisticRegression(C=1.0, solver="lbfgs").fit((X - mu) / sd, y, sample_weight=w)
    coef = model.coef_[0] / sd
    bias = float(model.intercept_[0] - np.sum(model.coef_[0] * mu / sd))
    return coef, bias


def main() -> int:
    images = load_images(PKG / "data" / "samples")
    by_id = {s.image_id: s for s in images}
    folds = writer_folds(images)
    feats = {s.image_id: extract_features(cv2.imread(s.path)) for s in images}
    pairs = one_to_one_pairs(images)

    sims = [compare(feats[p.ref_id], feats[p.query_id]) for p in pairs]  # same signals as production
    X = np.array([[s.shape, s.keypoint.similarity] for s in sims])
    labels = np.array([p.label for p in pairs])
    y = (labels == "genuine").astype(int)
    fold = np.array([
        folds[by_id[p.ref_id].writer] if folds[by_id[p.ref_id].writer] == folds[by_id[p.query_id].writer] else -1
        for p in pairs
    ])

    held_out: Dict[str, List[float]] = {"genuine": [], "skilled": [], "random": []}
    per_fold = []
    for test in (0, 1):
        coef, bias = _fit(X[fold == 1 - test], y[fold == 1 - test])
        logit = bias + X[fold == test] @ coef
        for lab in held_out:
            held_out[lab] += logit[labels[fold == test] == lab].tolist()
        per_fold.append({"test_fold": test, "weights": dict(zip(FEATURE_NAMES, coef.tolist())), "bias": bias})

    coef, bias = _fit(X, y)
    report = {
        "features": list(FEATURE_NAMES),
        "cv_heldout": M.summarize_population(held_out["genuine"], held_out["skilled"], held_out["random"]),
        "per_fold_fits": per_fold,
        "production": {"shape_weight": coef[0], "keypoint_weight": coef[1], "bias": bias},
    }
    out = PKG / "benchmark" / "results" / "fusion_fit.json"
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    cv = report["cv_heldout"]
    print(f"CV held-out: skilled AUC={cv['skilled']['auc']:.4f} EER={cv['skilled']['eer_eer']:.4f} | "
          f"random AUC={cv['random']['auc']:.4f} EER={cv['random']['eer_eer']:.4f}")
    print(f"production: shape_weight={coef[0]:.3f} keypoint_weight={coef[1]:.3f} bias={bias:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
