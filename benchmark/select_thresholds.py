#!/usr/bin/env python3
"""Select production ACCEPT / REVIEW / REJECT logit thresholds (EXP-018).

    python -m signature_verification_system.benchmark.select_thresholds

The fusion is fitted on the dev pairs, so dev logits from the production fusion are
in-sample and over-separated; cut-offs read from them do not hold on new writers.
Thresholds are therefore picked from OUT-OF-FOLD logits: the fusion is refitted
(same solver and settings as fit_fusion.py) on one writer fold and scores the other
fold's pairs, and vice versa. Cross-fold random pairs have no out-of-fold model and
are left out. Multi-specimen trials use the same fold fits on max-aggregated signals.

Rule (quantiles of the out-of-fold logits, order statistics only):
  accept      = max(skilled logit with <= max_skilled_match of skilled at or above it,
                    random  logit with <= max_random_match  of random  at or above it)
                + accept_margin   (multi: multi_accept_margin)
  reject      = genuine logit at the reject_genuine_quantile
  hard_reject = lowest out-of-fold genuine logit
The production fusion stays the full-dev fit; its thresholds come from the rule applied
to the out-of-fold logits of all dev writers.

Held-out check (`cv_heldout_counts`, also read by fit_evidence_reference.py): nested.
For each outer test fold, the rule is selected on inner out-of-fold logits of the
other fold (itself split in two writer folds) and counted on the test fold's logits
from the fusion fitted on that other fold. Fusion and cut-offs never see the test
writers. Deterministic: lbfgs on fixed, sorted design matrices and order statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from signature_verification_system.benchmark import metrics as M
from signature_verification_system.benchmark.conditions import CONDITIONS
from signature_verification_system.benchmark.fit_fusion import _fit
from signature_verification_system.benchmark.protocol import (
    cap_random_pairs, load_images, one_to_one_pairs, writer_dependent_trials, writer_folds,
)
from signature_verification_system.src.core.config import DEFAULT_CONFIG, FUSION_SIGNALS
from signature_verification_system.src.verification.features import extract_features
from signature_verification_system.src.verification.similarity import compare

PKG = Path(__file__).resolve().parent.parent
LABELS = ("genuine", "skilled", "random")


class Rows:
    """Signal matrix of one protocol with labels and writer groups."""

    def __init__(self, X: np.ndarray, labels: np.ndarray, ref_writer: np.ndarray, query_writer: np.ndarray):
        self.X, self.labels, self.ref_writer, self.query_writer = X, labels, ref_writer, query_writer

    def within(self, writers: Sequence[str]) -> np.ndarray:
        """Mask of rows whose reference and query writers both lie in `writers`."""
        w = set(writers)
        return np.array([r in w and q in w for r, q in zip(self.ref_writer, self.query_writer)], dtype=bool)


def split_writers(writers: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Deterministic two-way writer split: alternate by sorted order."""
    s = sorted(writers)
    return s[0::2], s[1::2]


def fit_logits(train: Rows, train_mask: np.ndarray, score: Rows, score_mask: np.ndarray) -> np.ndarray:
    """Fit the fusion on `train[train_mask]` (single-specimen pairs) and score `score[score_mask]`."""
    y = (train.labels[train_mask] == "genuine").astype(int)
    coef, bias = _fit(train.X[train_mask], y)
    return bias + score.X[score_mask] @ coef


def out_of_fold(single: Rows, scored: Rows, writers: Sequence[str]) -> Dict[str, np.ndarray]:
    """Two-fold writer-disjoint out-of-fold logits of `scored` rows inside `writers`."""
    a, b = split_writers(writers)
    out: Dict[str, List[float]] = {lab: [] for lab in LABELS}
    for train_w, test_w in ((a, b), (b, a)):
        test_mask = scored.within(test_w)
        logit = fit_logits(single, single.within(train_w), scored, test_mask)
        for lab in LABELS:
            out[lab] += logit[scored.labels[test_mask] == lab].tolist()
    return {lab: np.array(v) for lab, v in out.items()}


def select_rule(oof: Dict[str, np.ndarray], max_skilled_match: float, max_random_match: float,
                accept_margin: float, reject_genuine_quantile: float) -> Dict[str, float]:
    """Quantile rule on out-of-fold logits (see module docstring)."""
    g = oof["genuine"]
    s = M.select_zone_thresholds(g, oof["skilled"], max_skilled_match, reject_genuine_quantile)
    r = M.select_zone_thresholds(g, oof["random"], max_random_match, reject_genuine_quantile)
    accept = max(s["accept"], r["accept"]) + accept_margin
    reject = min(s["reject"], accept)
    return {"accept": accept, "reject": reject, "hard_reject": min(float(g.min()), reject)}


def count(logits: Dict[str, np.ndarray], z: Dict[str, float]) -> Dict[str, Dict[str, int]]:
    return {lab: {"n": int(len(v)), "accept": int(np.sum(v >= z["accept"])), "reject": int(np.sum(v < z["reject"]))}
            for lab, v in logits.items()}


def nested_heldout(single: Rows, scored: Rows, folds: Dict[str, int], rule: Dict[str, float]) -> Dict[str, Dict[str, int]]:
    """Held-out counts where neither the fusion nor the cut-offs saw the test writers."""
    total = {lab: {"n": 0, "accept": 0, "reject": 0} for lab in LABELS}
    for test in (0, 1):
        train_w = [w for w, f in folds.items() if f == 1 - test]
        test_w = [w for w, f in folds.items() if f == test]
        z = select_rule(out_of_fold(single, scored, train_w), **rule)
        test_mask = scored.within(test_w)
        logit = fit_logits(single, single.within(train_w), scored, test_mask)
        c = count({lab: logit[scored.labels[test_mask] == lab] for lab in LABELS}, z)
        for lab in LABELS:
            for k in total[lab]:
                total[lab][k] += c[lab][k]
    return total


def _rule_from_config(mode: str) -> Dict[str, float]:
    d = DEFAULT_CONFIG.decision
    margin = d.multi_accept_margin_logit if mode == "multi" else d.accept_margin_logit
    return {"max_skilled_match": d.max_skilled_match_rate, "max_random_match": d.max_random_match_rate,
            "accept_margin": margin, "reject_genuine_quantile": d.reject_genuine_quantile}


def collect(data_dir: Path, condition: str, max_random: int, max_random_per_query: int) -> Tuple[Rows, Rows, Dict[str, int]]:
    """Signals for the 1:1 pairs (the fusion training set) and the 3-specimen trials."""
    images = load_images(data_dir)
    by_id = {s.image_id: s for s in images}
    prepare = CONDITIONS[condition]
    feats = {s.image_id: extract_features(prepare(cv2.imread(s.path))) for s in images}

    def vec(signals: Dict[str, float]) -> List[float]:
        return [signals[n] for n in FUSION_SIGNALS]

    pairs = cap_random_pairs(one_to_one_pairs(images), max_random)
    single = Rows(np.array([vec(compare(feats[p.ref_id], feats[p.query_id]).signals) for p in pairs]),
                  np.array([p.label for p in pairs]),
                  np.array([by_id[p.ref_id].writer for p in pairs]), np.array([by_id[p.query_id].writer for p in pairs]))
    trials = writer_dependent_trials(images, max_random_per_query=max_random_per_query)
    multi_X = []
    for t in trials:  # signal-wise max over references, exactly as compare_multi
        per_ref = [compare(feats[r], feats[t.query_id]).signals for r in t.ref_ids]
        multi_X.append([max(s[n] for s in per_ref) for n in FUSION_SIGNALS])
    multi = Rows(np.array(multi_X), np.array([t.label for t in trials]),
                 np.array([by_id[t.ref_ids[0]].writer for t in trials]), np.array([by_id[t.query_id].writer for t in trials]))
    return single, multi, writer_folds(images)


def report(single: Rows, multi: Rows, folds: Dict[str, int]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for name, rows in (("single", single), ("multi", multi)):
        rule = _rule_from_config(name)
        oof = _two_fold_oof(single, rows, folds)
        z = select_rule(oof, **rule)
        out[name] = {"rule": rule, "production": z,
                     "oof_counts": count(oof, z),
                     "cv_heldout_counts": nested_heldout(single, rows, folds, rule)}
    return out


def _two_fold_oof(single: Rows, scored: Rows, folds: Dict[str, int]) -> Dict[str, np.ndarray]:
    """Out-of-fold logits over the protocol's own writer folds (writer_folds)."""
    out: Dict[str, List[float]] = {lab: [] for lab in LABELS}
    for test in (0, 1):
        train_w = [w for w, f in folds.items() if f == 1 - test]
        test_w = [w for w, f in folds.items() if f == test]
        test_mask = scored.within(test_w)
        logit = fit_logits(single, single.within(train_w), scored, test_mask)
        for lab in LABELS:
            out[lab] += logit[scored.labels[test_mask] == lab].tolist()
    return {lab: np.array(v) for lab, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(PKG / "data" / "samples"))
    ap.add_argument("--max-random", type=int, default=0,
                    help="cap on random 1:1 pairs (0 = all); use the fit_fusion.py value so the fold fits match")
    ap.add_argument("--max-random-per-query", type=int, default=0, help="cap on other-writer queries per held-out genuine")
    ap.add_argument("--out", default=str(PKG / "benchmark" / "results" / "thresholds.json"))
    ap.add_argument("--condition", choices=sorted(CONDITIONS), default="harmonized",
                    help="input condition; harmonized (default since EXP-017) matches the clearance score")
    args = ap.parse_args()
    single, multi, folds = collect(Path(args.data_dir), args.condition, args.max_random, args.max_random_per_query)
    rep = report(single, multi, folds)
    for name in ("single", "multi"):
        p, cv = rep[name]["production"], rep[name]["cv_heldout_counts"]
        print(f"{name}: accept={p['accept']:.4f} reject={p['reject']:.4f} hard_reject={p['hard_reject']:.4f}")
        for lab, c in cv.items():
            print(f"   nested held-out {lab:8s} n={c['n']:5d} auto-accept={c['accept']:4d} ({c['accept']/c['n']:.4f})"
                  f"  rejected={c['reject']:5d} ({c['reject']/c['n']:.4f})")
    Path(args.out).write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
