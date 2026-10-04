#!/usr/bin/env python3
"""Reproducible signature-verification benchmark.

Run from the workspace parent directory:

    python -m signature_verification_system.benchmark.run_benchmark \
        --system current --condition harmonized --out signature_verification_system/benchmark/results/current.json

Produces two files:
  <out>            deterministic results (no timestamps) — byte-identical across runs
  <out>.runtime.json   wall-clock / memory measurements (naturally vary)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import cv2
import numpy as np

from signature_verification_system.benchmark import metrics as M
from signature_verification_system.benchmark.conditions import CONDITIONS
from signature_verification_system.benchmark.protocol import (
    EnrollmentTrial,
    Pair,
    SampleImage,
    load_images,
    cap_random_pairs,
    one_to_one_pairs,
    writer_dependent_trials,
    writer_folds,
)

PKG_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PKG_DIR / "data" / "samples"

PairScorer = Callable[[np.ndarray, np.ndarray], float]
MultiScorer = Callable[[Sequence[np.ndarray], np.ndarray], float]


# ---------------------------------------------------------------------------
# Scorer registry
# ---------------------------------------------------------------------------

def _legacy_scorers() -> Tuple[PairScorer, Dict[str, MultiScorer]]:
    from signature_verification_system.benchmark.legacy.verifier_v2_0 import DeterministicVerifier

    v = DeterministicVerifier()

    def pair(ref: np.ndarray, que: np.ndarray) -> float:
        return float(v.verify(ref, que).similarity_score)

    return pair, {}


def _current_scorers() -> Tuple[PairScorer, Dict[str, MultiScorer]]:
    """Scores = the verifier's unrounded match logit (the quantity its decision
    bands use). Features are cached per image object so each image is extracted
    once; compare()/compare_multi() are exactly what verify*() call."""
    from signature_verification_system.src.verification.features import extract_features
    from signature_verification_system.src.verification.similarity import compare, compare_multi

    cache: Dict[int, object] = {}

    def feat(img: np.ndarray):
        key = id(img)
        if key not in cache:
            cache[key] = extract_features(img)
        return cache[key]

    def pair(ref: np.ndarray, que: np.ndarray) -> float:
        return round(compare(feat(ref), feat(que)).fused_logit, 6)

    def system(refs: Sequence[np.ndarray], que: np.ndarray) -> float:
        return round(compare_multi([feat(r) for r in refs], feat(que)).fused_logit, 6)

    return pair, {"system": system}


SYSTEMS: Dict[str, Callable[[], Tuple[PairScorer, Dict[str, MultiScorer]]]] = {
    "legacy_v2_0": _legacy_scorers,
    "current": _current_scorers,
}
# ACCEPT safety margin applied above the highest dev impostor score, in each
# system's own score units (legacy: probability-like score; current: logit).
ACCEPT_MARGIN: Dict[str, float] = {"legacy_v2_0": 0.0, "current": 1.0}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _load(images: List[SampleImage], condition: str) -> Dict[str, np.ndarray]:
    transform = CONDITIONS[condition]
    out = {}
    for s in images:
        img = cv2.imread(s.path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(s.path)
        out[s.image_id] = transform(img)
    return out


def score_pairs(
    scorer: PairScorer, pixels: Dict[str, np.ndarray], keys: List[Tuple[str, str]]
) -> Dict[Tuple[str, str], float]:
    return {k: scorer(pixels[k[0]], pixels[k[1]]) for k in keys}


def _digest(values: Sequence[float]) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Cross-validated operating points
# ---------------------------------------------------------------------------

def _split(
    rows: List[Tuple[str, float]], fold_of: List[int]
) -> Dict[int, Dict[str, List[float]]]:
    out: Dict[int, Dict[str, List[float]]] = {0: {}, 1: {}}
    for (label, score), f in zip(rows, fold_of):
        if f < 0:
            continue  # cross-fold trial: excluded from CV to avoid identity leakage
        out[f].setdefault(label, []).append(score)
    return out


def cross_validate(rows: List[Tuple[str, float]], fold_of: List[int], accept_margin: float = 0.0) -> Dict[str, object]:
    """Writer-disjoint 2-fold CV: thresholds chosen on dev fold, applied to test fold."""
    folds = _split(rows, fold_of)
    pooled: Dict[str, Dict[str, List[float]]] = {"eer_thr": {}, "zones": {}}
    per_fold = []
    zone_counts: Dict[str, Dict[str, float]] = {}
    eer_counts = {"g_acc": 0, "g_n": 0, "s_acc": 0, "s_n": 0, "r_acc": 0, "r_n": 0}
    for test_f in (0, 1):
        dev, test = folds[1 - test_f], folds[test_f]
        dg, ds, dr = dev.get("genuine", []), dev.get("skilled", []), dev.get("random", [])
        tg, ts, tr = test.get("genuine", []), test.get("skilled", []), test.get("random", [])
        eer_thr = M.eer(dg, ds)["threshold"]
        zones = M.select_zone_thresholds(dg, ds + dr, max_far=0.0, max_frr_reject=0.05, accept_margin=accept_margin)
        outcome = M.zone_outcomes(tg, {"skilled": ts, "random": tr}, zones)
        per_fold.append({
            "test_fold": test_f,
            "dev_eer_threshold": eer_thr,
            "test_at_eer_threshold": {
                "skilled": M.rates_at(tg, ts, eer_thr),
                "random": M.rates_at(tg, tr, eer_thr),
            },
            "zones": zones,
            "test_zone_outcomes": outcome,
        })
        eer_counts["g_n"] += len(tg)
        eer_counts["g_acc"] += int(np.sum(np.asarray(tg) >= eer_thr))
        eer_counts["s_n"] += len(ts)
        eer_counts["s_acc"] += int(np.sum(np.asarray(ts) >= eer_thr))
        eer_counts["r_n"] += len(tr)
        eer_counts["r_acc"] += int(np.sum(np.asarray(tr) >= eer_thr))
        for pop, o in outcome.items():
            agg = zone_counts.setdefault(pop, {"n": 0, "accept": 0.0, "review": 0.0, "reject": 0.0})
            n = o.get("n", 0)
            agg["n"] += n
            for k in ("accept", "review", "reject"):
                agg[k] += o.get(k, 0.0) * n
    for pop, agg in zone_counts.items():
        if agg["n"]:
            for k in ("accept", "review", "reject"):
                agg[k] = agg[k] / agg["n"]
    c = eer_counts
    pooled_eer = {
        "frr": 1 - c["g_acc"] / c["g_n"] if c["g_n"] else float("nan"),
        "far_skilled": c["s_acc"] / c["s_n"] if c["s_n"] else float("nan"),
        "far_random": c["r_acc"] / c["r_n"] if c["r_n"] else float("nan"),
    }
    return {"per_fold": per_fold, "pooled_test_at_dev_eer_threshold": pooled_eer, "pooled_test_zone_outcomes": zone_counts}


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run(
    system: str, data_dir: Path, repeats: int, condition: str = "harmonized",
    max_random: int = 0, max_random_per_query: int = 0,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    images = load_images(data_dir)
    by_id = {s.image_id: s for s in images}
    folds = writer_folds(images)
    pairs: List[Pair] = cap_random_pairs(one_to_one_pairs(images), max_random)
    trials: List[EnrollmentTrial] = writer_dependent_trials(
        images, max_random_per_query=max_random_per_query)
    pixels = _load(images, condition)

    needed = sorted({(p.ref_id, p.query_id) for p in pairs} | {(r, t.query_id) for t in trials for r in t.ref_ids})

    timings = []
    digests = []
    scores: Dict[Tuple[str, str], float] = {}
    multi_scores: Dict[str, List[float]] = {}
    for _ in range(max(1, repeats)):
        pair_scorer, multi = SYSTEMS[system]()  # fresh instance each repeat
        t0 = time.perf_counter()
        scores = score_pairs(pair_scorer, pixels, needed)
        timings.append(time.perf_counter() - t0)
        multi_scores = {
            name: [fn([pixels[r] for r in t.ref_ids], pixels[t.query_id]) for t in trials]
            for name, fn in sorted(multi.items())
        }
        digests.append(_digest([scores[k] for k in needed] + [v for n in sorted(multi_scores) for v in multi_scores[n]]))

    def fold_of_pair(ref: str, que: str) -> int:
        a, b = folds[by_id[ref].writer], folds[by_id[que].writer]
        return a if a == b else -1

    # ---- 1:1 protocol
    rows = [(p.label, scores[(p.ref_id, p.query_id)]) for p in pairs]
    pop = {lab: [s for l, s in rows if l == lab] for lab in ("genuine", "skilled", "random")}
    one2one = {
        "counts": {k: len(v) for k, v in pop.items()},
        "summary": M.summarize_population(pop["genuine"], pop["skilled"], pop["random"]),
        "cross_validation": cross_validate(rows, [fold_of_pair(p.ref_id, p.query_id) for p in pairs], ACCEPT_MARGIN[system]),
    }

    # ---- writer-dependent protocol (multi-reference aggregation)
    aggregations: Dict[str, List[float]] = {
        "mean": [float(np.mean([scores[(r, t.query_id)] for r in t.ref_ids])) for t in trials],
        "max": [float(np.max([scores[(r, t.query_id)] for r in t.ref_ids])) for t in trials],
    }
    aggregations.update(multi_scores)
    wd_fold = [fold_of_pair(t.ref_ids[0], t.query_id) for t in trials]
    writer_dep = {}
    for name in sorted(aggregations):
        vals = aggregations[name]
        rws = [(t.label, v) for t, v in zip(trials, vals)]
        p2 = {lab: [s for l, s in rws if l == lab] for lab in ("genuine", "skilled", "random")}
        writer_dep[name] = {
            "counts": {k: len(v) for k, v in p2.items()},
            "summary": M.summarize_population(p2["genuine"], p2["skilled"], p2["random"]),
            "cross_validation": cross_validate(rws, wd_fold, ACCEPT_MARGIN[system]),
        }

    # ---- hardest cases (for diagnosis)
    def _name(i: str) -> str:
        return Path(by_id[i].path).name

    hardest = {
        "highest_skilled": sorted(
            [{"ref": _name(p.ref_id), "query": _name(p.query_id), "score": scores[(p.ref_id, p.query_id)]} for p in pairs if p.label == "skilled"],
            key=lambda d: (-d["score"], d["ref"], d["query"]))[:8],
        "lowest_genuine": sorted(
            [{"ref": _name(p.ref_id), "query": _name(p.query_id), "score": scores[(p.ref_id, p.query_id)]} for p in pairs if p.label == "genuine"],
            key=lambda d: (d["score"], d["ref"], d["query"]))[:8],
        "highest_random": sorted(
            [{"ref": _name(p.ref_id), "query": _name(p.query_id), "score": scores[(p.ref_id, p.query_id)]} for p in pairs if p.label == "random"],
            key=lambda d: (-d["score"], d["ref"], d["query"]))[:5],
    }

    results = {
        "system": system,
        "condition": condition,
        "dataset": {
            "distinct_images": len(images),
            "writers": len({s.writer for s in images}),
            "fold_assignment": {w: folds[w] for w in sorted(folds)},
        },
        "determinism": {"repeats": len(digests), "identical": len(set(digests)) == 1, "score_sha256": digests[0]},
        "one_to_one": one2one,
        "writer_dependent_3ref": writer_dep,
        "hardest_cases": hardest,
        "pair_scores": [
            [_name(p.ref_id), _name(p.query_id), p.label, scores[(p.ref_id, p.query_id)]] for p in pairs
        ],
    }
    runtime = {
        "system": system,
        "pair_evaluations": len(needed),
        "seconds_per_repeat": timings,
        "ms_per_pair": [1000 * t / len(needed) for t in timings],
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024),
    }
    return results, runtime


def _headline(res: Dict[str, object]) -> str:
    o = res["one_to_one"]["summary"]
    cv = res["one_to_one"]["cross_validation"]["pooled_test_zone_outcomes"]
    lines = [
        f"system={res['system']}  condition={res['condition']}  deterministic={res['determinism']['identical']}  sha={res['determinism']['score_sha256'][:12]}",
        f"1:1  skilled AUC={o['skilled']['auc']:.4f} EER={o['skilled']['eer_eer']:.4f} | random AUC={o['random']['auc']:.4f} EER={o['random']['eer_eer']:.4f}",
        "1:1  CV zones (test folds): " + "  ".join(
            f"{k}: A={v['accept']:.2f} R={v['review']:.2f} X={v['reject']:.2f}" for k, v in sorted(cv.items())),
    ]
    for name, wd in sorted(res["writer_dependent_3ref"].items()):
        s = wd["summary"]
        z = wd["cross_validation"]["pooled_test_zone_outcomes"]
        lines.append(
            f"WD[{name:>6}] skilled AUC={s['skilled']['auc']:.4f} EER={s['skilled']['eer_eer']:.4f} | random AUC={s['random']['auc']:.4f} EER={s['random']['eer_eer']:.4f}"
            + "  zones: " + "  ".join(f"{k}: A={v['accept']:.2f} X={v['reject']:.2f}" for k, v in sorted(z.items())))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", choices=sorted(SYSTEMS), default="current")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--out", required=True)
    ap.add_argument("--condition", choices=sorted(CONDITIONS), default="harmonized")
    ap.add_argument("--repeats", type=int, default=1, help="in-process determinism repeats")
    ap.add_argument("--max-random", type=int, default=0, help="cap on random 1:1 pairs (0 = all)")
    ap.add_argument("--max-random-per-query", type=int, default=0,
                    help="cap on other-writer queries per held-out genuine in the 3-specimen protocol (0 = all)")
    args = ap.parse_args(argv)

    results, runtime = run(args.system, Path(args.data_dir), args.repeats, args.condition,
                           args.max_random, args.max_random_per_query)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    Path(str(out) + ".runtime.json").write_text(json.dumps(runtime, indent=1, sort_keys=True) + "\n")
    print(_headline(results))
    print(f"runtime: {runtime['ms_per_pair'][0]:.2f} ms/pair, max RSS {runtime['max_rss_mb']:.0f} MB")
    return 0 if results["determinism"]["identical"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
