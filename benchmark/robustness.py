#!/usr/bin/env python3
"""Perturbation robustness benchmark (requirement D).

    python -m signature_verification_system.benchmark.robustness \
        --out signature_verification_system/benchmark/results/robustness.json

For every writer-dependent trial (3 enrolled specimens, production operating
points) the *questioned* image is perturbed and re-scored. A robust verifier
keeps genuine signatures out of REJECT and never lifts forgeries into ACCEPT.

Determinism: every perturbation is a fixed geometric/photometric transform.
The only stochastic one (additive noise) draws from
`src.core.determinism.seeded_rng` (PCG64, GLOBAL_SEED + the image's SHA-256
prefix), so it is reproducible and independent of evaluation order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np

from signature_verification_system.benchmark.protocol import load_images, writer_dependent_trials
from signature_verification_system.src.core.determinism import seeded_rng
from signature_verification_system.src.adjudication.thresholds import ACCEPT, INCONCLUSIVE, REJECT, classify_band
from signature_verification_system.src.preprocessing.quality import assess_signature_quality
from signature_verification_system.src.verification.features import extract_features
from signature_verification_system.src.verification.similarity import compare_multi

PKG = Path(__file__).resolve().parent.parent

Perturb = Callable[[np.ndarray, str], np.ndarray]


def _paper(img: np.ndarray) -> Tuple[int, int, int]:
    return tuple(int(v) for v in np.median(img.reshape(-1, 3), axis=0))


def rotate(deg: float) -> Perturb:
    def f(img: np.ndarray, _: str) -> np.ndarray:
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
        return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=_paper(img))
    return f


def scale(factor: float) -> Perturb:
    def f(img: np.ndarray, _: str) -> np.ndarray:
        h, w = img.shape[:2]
        interp = cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC
        return cv2.resize(img, (max(8, round(w * factor)), max(8, round(h * factor))), interpolation=interp)
    return f


def translate(dx: int, dy: int) -> Perturb:
    def f(img: np.ndarray, _: str) -> np.ndarray:
        h, w = img.shape[:2]
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(img, m, (w + abs(dx), h + abs(dy)), borderValue=_paper(img))
    return f


def blur(sigma: float) -> Perturb:
    return lambda img, _: cv2.GaussianBlur(img, (0, 0), sigma)


def jpeg(quality: int) -> Perturb:
    def f(img: np.ndarray, _: str) -> np.ndarray:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return f


def brightness(delta: int) -> Perturb:
    return lambda img, _: np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)


def contrast(alpha: float) -> Perturb:
    def f(img: np.ndarray, _: str) -> np.ndarray:
        mean = float(img.mean())
        return np.clip((img.astype(np.float64) - mean) * alpha + mean, 0, 255).astype(np.uint8)
    return f


def noise(sigma: float) -> Perturb:
    def f(img: np.ndarray, image_id: str) -> np.ndarray:
        rng = seeded_rng(int(image_id[:8], 16))
        return np.clip(img.astype(np.float64) + rng.normal(0.0, sigma, img.shape), 0, 255).astype(np.uint8)
    return f


def thicken(px: int) -> Perturb:
    """Simulates a broader pen: grey-level erosion widens dark strokes."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return lambda img, _: cv2.erode(img, k)


PERTURBATIONS: Dict[str, Perturb] = {
    "none": lambda img, _: img,
    "rotate_+5": rotate(5), "rotate_-10": rotate(-10), "rotate_+15": rotate(15),
    "scale_0.5": scale(0.5), "scale_0.75": scale(0.75), "scale_1.5": scale(1.5),
    "translate_40_20": translate(40, 20),
    "blur_1": blur(1.0), "blur_2": blur(2.0), "blur_3": blur(3.0),
    "jpeg_50": jpeg(50), "jpeg_20": jpeg(20), "jpeg_10": jpeg(10),
    "brightness_+40": brightness(40), "brightness_-60": brightness(-60),
    "contrast_0.5": contrast(0.5), "contrast_0.3": contrast(0.3),
    "noise_10": noise(10), "noise_25": noise(25),
    "thicker_pen_1px": thicken(1),
}


def key_in_baseline(band: str) -> bool:
    return band != INCONCLUSIVE


def run() -> Dict[str, object]:
    images = load_images(PKG / "data" / "samples")
    pixels = {s.image_id: cv2.imread(s.path, cv2.IMREAD_COLOR) for s in images}
    feats = {i: extract_features(p) for i, p in pixels.items()}
    trials = [t for t in writer_dependent_trials(images) if t.label in ("genuine", "skilled")]
    queries = sorted({t.query_id for t in trials})

    out: Dict[str, object] = {}
    baseline: Dict[Tuple, float] = {}
    for name, fn in PERTURBATIONS.items():
        qfeat = {}
        gated = set()
        failed = 0
        for q in queries:
            img = fn(pixels[q], q)
            if not assess_signature_quality(img).passed:
                gated.add(q)
                continue
            try:
                qfeat[q] = extract_features(img)
            except ValueError:
                failed += 1
        stats: Dict[str, List] = {"genuine": [], "skilled": []}
        for t in trials:
            key = (t.ref_ids, t.query_id)
            if t.query_id in gated:
                stats[t.label].append((INCONCLUSIVE, None))
                continue
            if t.query_id not in qfeat:
                continue
            logit = compare_multi([feats[r] for r in t.ref_ids], qfeat[t.query_id]).fused_logit
            band = classify_band(logit, len(t.ref_ids)).band
            if name == "none":
                baseline[key] = logit
            stats[t.label].append((band, logit - baseline[key]))
        row = {"extraction_failures": failed}
        for lab, rows in stats.items():
            n = len(rows)
            deltas = [d for b, d in rows if d is not None and key_in_baseline(b)]
            row[lab] = {
                "n": n,
                "accept": sum(b == ACCEPT for b, _ in rows) / n,
                "reject": sum(b == REJECT for b, _ in rows) / n,
                "inconclusive": sum(b == INCONCLUSIVE for b, _ in rows) / n,
                "mean_logit_change": float(np.mean(deltas)) if deltas else float("nan"),
            }
        out[name] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PKG / "benchmark" / "results" / "robustness.json"))
    args = ap.parse_args()
    res = run()
    Path(args.out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(f"{'perturbation':18s} | genuine ACCEPT  REJECT  INCONC | skilled ACCEPT  REJECT  INCONC | fail")
    for name, r in res.items():
        g, s = r["genuine"], r["skilled"]
        print(f"{name:18s} |   {g['accept']:5.2f}   {g['reject']:5.2f}   {g['inconclusive']:5.2f} |   "
              f"{s['accept']:5.2f}   {s['reject']:5.2f}   {s['inconclusive']:5.2f} | {r['extraction_failures']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
