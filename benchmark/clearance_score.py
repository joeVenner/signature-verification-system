#!/usr/bin/env python3
"""Frozen 0-10 Clearance Score for the 1:1 problem (one reference vs one questioned image).

    python -m signature_verification_system.benchmark.clearance_score \
        --data-dir /path/to/cedar55_val --out /path/to/score.json [--workers 8]

Definition (fixed in .claude/decisions.md D-001 before any baseline was seen; do not
change it to move the score):

    D = 1 - (0.75 * EER_skilled + 0.25 * EER_random) / 0.25       clean images
    R = mean over CAPTURE_CONDITIONS of (1 - EER_skilled_c / 0.25)  questioned image perturbed
    O = 0.5 * genuine_MATCH + 0.5 * skilled_NOMATCH - genuine_NOMATCH   production thresholds
    Score = 10 * (0.4 D + 0.3 R + 0.3 O), each component clamped to [0, 1]
    Safety gate: skilled MATCH > 1% or random MATCH > 0.5% caps the score at 4.0.

Every image first goes through the `harmonized` benchmark condition (removes the CEDAR
scan-session shortcut, EXP-000). Capture conditions are fixed, label-agnostic transforms
with no randomness. A questioned image whose features cannot be extracted scores
FAILED_LOGIT (a genuine one counts as a false reject, a forgery as a correct reject).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from signature_verification_system.benchmark import metrics as M
from signature_verification_system.benchmark.conditions import harmonize
from signature_verification_system.benchmark.protocol import Pair, cap_random_pairs, load_images, one_to_one_pairs

FAILED_LOGIT = -1.0e6
EER_ZERO_POINT = 0.25
WEIGHTS = {"D": 0.4, "R": 0.3, "O": 0.3}
GATE_SKILLED_MATCH = 0.01
GATE_RANDOM_MATCH = 0.005
GATED_SCORE_CAP = 4.0
DEFAULT_MAX_RANDOM = 3000

Capture = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# Capture conditions (deterministic; applied to the harmonized questioned image)
# ---------------------------------------------------------------------------

def _tinted_paper(img: np.ndarray) -> np.ndarray:
    """Cream / yellow paper: multiply channels (BGR) and lift the ink slightly."""
    tint = np.array([0.72, 0.90, 0.98])
    return np.clip(img.astype(np.float64) * tint + 8.0, 0, 255).astype(np.uint8)


def _shadow(img: np.ndarray) -> np.ndarray:
    """Strong illumination fall-off across the image (phone shadow)."""
    h, w = img.shape[:2]
    xs = np.linspace(1.0, 0.45, w)[None, :, None]
    ys = np.linspace(1.0, 0.85, h)[:, None, None]
    return np.clip(img.astype(np.float64) * xs * ys, 0, 255).astype(np.uint8)


def _ruled_lines(img: np.ndarray) -> np.ndarray:
    """Blue ruled lines every 36 px under the ink (multiplicative, like printed paper)."""
    out = img.astype(np.float64)
    for y in range(18, img.shape[0], 36):
        out[y:y + 2] *= np.array([1.0, 0.80, 0.70])
    return np.clip(out, 0, 255).astype(np.uint8)


def _large_canvas(img: np.ndarray) -> np.ndarray:
    """Signature small and off-centre on a page three times its size."""
    h, w = img.shape[:2]
    canvas = np.full((h * 3, w * 3, 3), 255, np.uint8)
    y0, x0 = int(h * 0.3), int(w * 1.7)
    canvas[y0:y0 + h, x0:x0 + w] = img
    return canvas


def _distractor_print(img: np.ndarray) -> np.ndarray:
    """Printed form text far from the signature (cheque / form context)."""
    h, w = img.shape[:2]
    canvas = np.full((h * 2, w * 2, 3), 255, np.uint8)
    canvas[h:, w // 2:w // 2 + w] = img
    scale = max(0.6, w / 900.0)
    cv2.putText(canvas, "PAY TO THE ORDER OF", (int(0.05 * w), int(0.25 * h)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (40, 40, 40), max(1, int(round(2 * scale))), cv2.LINE_AA)
    cv2.line(canvas, (int(0.05 * w), int(0.55 * h)), (int(1.9 * w), int(0.55 * h)), (60, 60, 60),
             max(1, int(round(2 * scale))))
    return canvas


def _faint_ink(img: np.ndarray) -> np.ndarray:
    """Light pen / faded scan: ink contrast reduced to 30%."""
    return np.clip(255.0 - (255.0 - img.astype(np.float64)) * 0.30, 0, 255).astype(np.uint8)


def _scale(factor: float) -> Capture:
    def f(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        interp = cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC
        return cv2.resize(img, (max(8, round(w * factor)), max(8, round(h * factor))), interpolation=interp)
    return f


def _rotate(deg: float) -> Capture:
    """Rotate on an expanded canvas so no ink is clipped."""
    def f(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
        cos, sin = abs(m[0, 0]), abs(m[0, 1])
        nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
        m[0, 2] += nw / 2 - w / 2
        m[1, 2] += nh / 2 - h / 2
        return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return f


def _phone_photo(img: np.ndarray) -> np.ndarray:
    """Grey-ish paper, mild vignette, defocus, JPEG q40."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2))
    vignette = (1.0 - 0.25 * np.clip(r, 0, 1.5) ** 2)[..., None]
    out = img.astype(np.float64) * 0.82 * vignette + 12.0
    out = cv2.GaussianBlur(np.clip(out, 0, 255).astype(np.uint8), (0, 0), 0.9)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 40])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


CAPTURE_CONDITIONS: Dict[str, Capture] = {
    "tinted_paper": _tinted_paper,
    "shadow_gradient": _shadow,
    "ruled_lines": _ruled_lines,
    "large_canvas_offcentre": _large_canvas,
    "distractor_print": _distractor_print,
    "faint_ink": _faint_ink,
    "scale_0.5": _scale(0.5),
    "scale_2.0": _scale(2.0),
    "rotate_+20": _rotate(20),
    "rotate_-15": _rotate(-15),
    "phone_photo": _phone_photo,
}


# ---------------------------------------------------------------------------
# Feature extraction (process pool; each image is independent and deterministic)
# ---------------------------------------------------------------------------

def _extract(job: Tuple[str, str, str]):
    from signature_verification_system.src.verification.features import extract_features

    image_id, path, condition = job
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = harmonize(img)
    if condition != "clean":
        img = CAPTURE_CONDITIONS[condition](img)
    try:
        return image_id, extract_features(img)
    except ValueError:
        return image_id, None


def _extract_all(jobs: List[Tuple[str, str, str]], workers: int) -> Dict[str, object]:
    if workers <= 1:
        return dict(map(_extract, jobs))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(_extract, jobs, chunksize=4))


def _logit(ref, que) -> float:
    from signature_verification_system.src.verification.similarity import compare

    if ref is None or que is None:
        return FAILED_LOGIT
    return round(float(compare(ref, que).fused_logit), 6)


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

def _clamp(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _scores(pairs: Sequence[Pair], ref_feats: Dict[str, object], que_feats: Dict[str, object]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {"genuine": [], "skilled": [], "random": []}
    for p in pairs:
        out[p.label].append(_logit(ref_feats[p.ref_id], que_feats[p.query_id]))
    return out


def compute(data_dir: Path, workers: int, max_random: int, conditions: Optional[Sequence[str]] = None) -> Dict[str, object]:
    from signature_verification_system.src.core.config import DEFAULT_CONFIG

    images = load_images(data_dir)
    paths = {s.image_id: s.path for s in images}
    pairs = cap_random_pairs(one_to_one_pairs(images), max_random)
    t0 = time.perf_counter()
    clean = _extract_all([(i, paths[i], "clean") for i in sorted(paths)], workers)
    s = _scores(pairs, clean, clean)

    thr = DEFAULT_CONFIG.decision
    acc, rej = thr.single_accept_logit, thr.single_reject_logit
    rate = lambda v, f: float(np.mean([f(x) for x in v])) if v else 0.0  # noqa: E731
    op = {
        lab: {"MATCH": rate(v, lambda x: x >= acc), "NOMATCH": rate(v, lambda x: x < rej),
              "REVIEW": rate(v, lambda x: rej <= x < acc), "n": len(v)}
        for lab, v in s.items()
    }
    eer_s = M.eer(s["genuine"], s["skilled"])["eer"]
    eer_r = M.eer(s["genuine"], s["random"])["eer"]
    D = _clamp(1.0 - (0.75 * eer_s + 0.25 * eer_r) / EER_ZERO_POINT)
    O = _clamp(0.5 * op["genuine"]["MATCH"] + 0.5 * op["skilled"]["NOMATCH"] - op["genuine"]["NOMATCH"])

    # Robustness: only genuine/skilled questioned images are perturbed (references stay clean).
    gs_pairs = [p for p in pairs if p.label in ("genuine", "skilled")]
    queries = sorted({p.query_id for p in gs_pairs})
    per_cond: Dict[str, Dict[str, float]] = {}
    for name in (conditions or list(CAPTURE_CONDITIONS)):
        qf = _extract_all([(q, paths[q], name) for q in queries], workers)
        cs = _scores(gs_pairs, clean, qf)
        e = M.eer(cs["genuine"], cs["skilled"])["eer"]
        per_cond[name] = {
            "eer_skilled": round(e, 6),
            "auc_skilled": round(M.auc(cs["genuine"], cs["skilled"]), 6),
            "extraction_failures": int(sum(v is None for v in qf.values())),
            "genuine_NOMATCH": round(rate(cs["genuine"], lambda x: x < rej), 6),
            "skilled_MATCH": round(rate(cs["skilled"], lambda x: x >= acc), 6),
            "r": round(_clamp(1.0 - e / EER_ZERO_POINT), 6),
        }
    R = float(np.mean([c["r"] for c in per_cond.values()])) if per_cond else 0.0

    score = 10.0 * (WEIGHTS["D"] * D + WEIGHTS["R"] * R + WEIGHTS["O"] * O)
    gated = op["skilled"]["MATCH"] > GATE_SKILLED_MATCH or op["random"]["MATCH"] > GATE_RANDOM_MATCH
    if gated:
        score = min(score, GATED_SCORE_CAP)
    digest = hashlib.sha256(np.asarray(s["genuine"] + s["skilled"] + s["random"], np.float64).tobytes()).hexdigest()
    return {
        "score": round(score, 3),
        "components": {"D": round(D, 4), "R": round(R, 4), "O": round(O, 4)},
        "safety_gate_triggered": gated,
        "clean": {"eer_skilled": round(eer_s, 6), "eer_random": round(eer_r, 6),
                  "auc_skilled": round(M.auc(s["genuine"], s["skilled"]), 6),
                  "auc_random": round(M.auc(s["genuine"], s["random"]), 6),
                  "operating_point": op, "thresholds": {"accept": acc, "reject": rej},
                  "extraction_failures": int(sum(v is None for v in clean.values()))},
        "capture_conditions": per_cond,
        "n_pairs": {k: len(v) for k, v in s.items()},
        "clean_score_sha256": digest,
        "seconds": round(time.perf_counter() - t0, 1),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-random", type=int, default=DEFAULT_MAX_RANDOM)
    ap.add_argument("--conditions", nargs="*", default=None, help="subset of capture conditions (diagnostics only)")
    args = ap.parse_args(argv)
    res = compute(Path(args.data_dir), args.workers, args.max_random, args.conditions)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    c = res["clean"]
    print(f"SCORE {res['score']:.2f}/10  D={res['components']['D']:.3f} R={res['components']['R']:.3f} "
          f"O={res['components']['O']:.3f} gate={res['safety_gate_triggered']}")
    print(f"clean: skilled EER={c['eer_skilled']:.4f} AUC={c['auc_skilled']:.4f} | random EER={c['eer_random']:.4f}")
    for lab, v in c["operating_point"].items():
        print(f"  {lab:8s} MATCH={v['MATCH']:.3f} REVIEW={v['REVIEW']:.3f} NOMATCH={v['NOMATCH']:.3f} n={v['n']}")
    for name, v in res["capture_conditions"].items():
        print(f"  {name:24s} EER={v['eer_skilled']:.4f} AUC={v['auc_skilled']:.4f} genNOMATCH={v['genuine_NOMATCH']:.3f} "
              f"sklMATCH={v['skilled_MATCH']:.3f} fail={v['extraction_failures']}")
    print(f"{res['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
