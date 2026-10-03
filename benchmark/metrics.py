"""Exact, deterministic biometric metrics.

All metrics are computed from the sorted unique score values (no fixed
threshold grid), so they do not depend on grid resolution and are identical
across runs for identical scores.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def _arr(scores: Sequence[float]) -> np.ndarray:
    return np.asarray(scores, dtype=np.float64)


def auc(genuine: Sequence[float], impostor: Sequence[float]) -> float:
    """ROC-AUC via the Mann-Whitney U statistic (ties count 0.5)."""
    g, i = _arr(genuine), _arr(impostor)
    if len(g) == 0 or len(i) == 0:
        return float("nan")
    greater = (g[:, None] > i[None, :]).sum()
    ties = (g[:, None] == i[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(g) * len(i)))


def rates_at(genuine: Sequence[float], impostor: Sequence[float], threshold: float) -> Dict[str, float]:
    """Accept iff score >= threshold."""
    g, i = _arr(genuine), _arr(impostor)
    far = float(np.mean(i >= threshold)) if len(i) else float("nan")
    frr = float(np.mean(g < threshold)) if len(g) else float("nan")
    n = len(g) + len(i)
    correct = float(np.sum(g >= threshold) + np.sum(i < threshold))
    return {"threshold": threshold, "far": far, "frr": frr, "accuracy": correct / n if n else float("nan")}


def eer(genuine: Sequence[float], impostor: Sequence[float]) -> Dict[str, float]:
    """Equal error rate over every candidate threshold (unique scores + inf).

    Returns the threshold minimising |FAR - FRR| (ties -> lowest threshold) and
    the EER as the mean of FAR and FRR at that point.
    """
    g, i = _arr(genuine), _arr(impostor)
    candidates = np.unique(np.concatenate([g, i, [np.inf]]))
    best = None
    for t in candidates:
        far = float(np.mean(i >= t))
        frr = float(np.mean(g < t))
        gap = abs(far - frr)
        if best is None or gap < best[0] - 1e-15:
            best = (gap, float(t), far, frr)
    _, t, far, frr = best
    return {"eer": (far + frr) / 2.0, "threshold": t, "far": far, "frr": frr}


def distribution(scores: Sequence[float]) -> Dict[str, float]:
    s = _arr(scores)
    if len(s) == 0:
        return {"n": 0}
    return {
        "n": int(len(s)),
        "mean": float(np.mean(s)),
        "std": float(np.std(s)),
        "min": float(np.min(s)),
        "p05": float(np.quantile(s, 0.05, method="lower")),
        "p25": float(np.quantile(s, 0.25, method="lower")),
        "median": float(np.quantile(s, 0.50, method="lower")),
        "p75": float(np.quantile(s, 0.75, method="lower")),
        "p95": float(np.quantile(s, 0.95, method="lower")),
        "max": float(np.max(s)),
    }


def separation(genuine: Sequence[float], impostor: Sequence[float]) -> float:
    """Fisher-style d' = |mu_g - mu_i| / sqrt((var_g + var_i)/2)."""
    g, i = _arr(genuine), _arr(impostor)
    denom = np.sqrt((np.var(g) + np.var(i)) / 2.0)
    return float(abs(np.mean(g) - np.mean(i)) / denom) if denom > 0 else float("inf")


def select_zone_thresholds(
    dev_genuine: Sequence[float],
    dev_impostor: Sequence[float],
    max_far: float = 0.0,
    max_frr_reject: float = 0.05,
    accept_margin: float = 0.0,
) -> Dict[str, float]:
    """Pick ACCEPT / REJECT cut-offs on development data only.

    accept: lowest threshold whose dev FAR <= max_far (impostors strictly below).
    reject: highest threshold whose dev FRR <= max_frr_reject, capped at accept.
    Scores in [reject, accept) go to manual REVIEW.
    """
    g = np.sort(_arr(dev_genuine))
    i = np.sort(_arr(dev_impostor))
    # accept: need fraction of impostors >= t to be <= max_far
    k = int(np.floor(max_far * len(i)))           # impostors allowed above
    accept = float(i[len(i) - 1 - k]) + 1e-9 + accept_margin if len(i) > k else float(i.min())
    # reject: allow at most floor(max_frr * n) genuines strictly below t
    m = int(np.floor(max_frr_reject * len(g)))
    reject = float(g[m]) if len(g) > m else float(g.max())
    reject = min(reject, accept)
    return {"accept": accept, "reject": reject, "hard_reject": min(float(g.min()), reject)}


def zone_outcomes(
    genuine: Sequence[float], impostor_by_type: Dict[str, Sequence[float]], zones: Dict[str, float]
) -> Dict[str, Dict[str, float]]:
    """Fraction of each population routed to ACCEPT / REVIEW / REJECT."""

    def route(s: np.ndarray) -> Dict[str, float]:
        n = len(s)
        if n == 0:
            return {"n": 0}
        acc = float(np.sum(s >= zones["accept"])) / n
        rej = float(np.sum(s < zones["reject"])) / n
        return {"n": n, "accept": acc, "review": 1.0 - acc - rej, "reject": rej}

    out = {"genuine": route(_arr(genuine))}
    for name in sorted(impostor_by_type):
        out[name] = route(_arr(impostor_by_type[name]))
    return out


def summarize_population(
    genuine: List[float], skilled: List[float], random: List[float]
) -> Dict[str, object]:
    """Threshold-free summary: AUC, EER, d' for skilled / random / all."""
    allimp = skilled + random
    return {
        "distributions": {
            "genuine": distribution(genuine),
            "skilled": distribution(skilled),
            "random": distribution(random),
        },
        "skilled": {"auc": auc(genuine, skilled), "d_prime": separation(genuine, skilled), **{"eer_" + k: v for k, v in eer(genuine, skilled).items()}},
        "random": {"auc": auc(genuine, random), "d_prime": separation(genuine, random), **{"eer_" + k: v for k, v in eer(genuine, random).items()}},
        "all_impostors": {"auc": auc(genuine, allimp), **{"eer_" + k: v for k, v in eer(genuine, allimp).items()}},
    }
