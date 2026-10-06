#!/usr/bin/env python3
"""Verification on cheque-like backgrounds (CEDAR signatures composited onto cheque paper).

    python -m signature_verification_system.benchmark.cheque_composite \
        --data-dir ../cedar55_val --cache-dir ../cheque/composites_val \
        --bcsd-dir ../cheque/raw/bcsd --out ../results/cheque_composite.json [--both-textured]

Questioned images are composited onto cheque backgrounds and low-res captured; references
stay clean (the real use case: a clean specimen card against a cheque crop). Every
composite is deterministic: its random generator is seeded from SHA-256(image id, variant).

Background variants (all followed by the same low-resolution capture: downscale to
190-230 px wide, Gaussian blur, sensor noise, JPEG q55-80):

* ``guilloche_grey``  grey paper with dense wavy security lines (the failing field case)
* ``guilloche_tint``  tinted paper with two crossing coloured line families
* ``bcsd_real``       a real cheque background (BCSD, signature inpainted out)
* ``clean_lowres``    white paper, capture degradation only (control: resolution vs texture)

``mixed`` assigns each questioned image one of the three textured variants (by hash).

Score (fixed before any number was measured; mirrors the clearance score, D-001):

    D = 1 - (0.75 * EER_skilled + 0.25 * EER_random) / 0.25     on ``mixed``
    R = mean over the three textured variants of (1 - EER_skilled_v / 0.25)
    O = 0.5 * genuine_MATCH + 0.5 * skilled_NOMATCH - genuine_NOMATCH   on ``mixed``
    Score = 10 * (0.4 D + 0.3 R + 0.3 O); safety gate as in clearance_score.

Each questioned image goes through the production path: quality gate, then feature
extraction. A gate or extraction failure is INCONCLUSIVE and scores FAILED_LOGIT (a genuine
one counts as a false reject, a forgery as a correct reject); INCONCLUSIVE rates are
reported separately. ``--both-textured`` also composites the references (``mixed``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from signature_verification_system.benchmark import metrics as M
from signature_verification_system.benchmark.clearance_score import (
    EER_ZERO_POINT, FAILED_LOGIT, GATE_RANDOM_MATCH, GATE_SKILLED_MATCH, GATED_SCORE_CAP, WEIGHTS,
)
from signature_verification_system.benchmark.conditions import harmonize
from signature_verification_system.benchmark.protocol import cap_random_pairs, load_images, one_to_one_pairs

TEXTURED_VARIANTS: Tuple[str, ...] = ("guilloche_grey", "guilloche_tint", "bcsd_real")
VARIANTS: Tuple[str, ...] = TEXTURED_VARIANTS + ("clean_lowres",)
DEFAULT_MAX_RANDOM = 3000
INK_COLOURS_BGR: Tuple[Tuple[float, float, float], ...] = ((35, 30, 30), (120, 45, 25), (90, 40, 40))


def _rng(image_id: str, variant: str) -> np.random.Generator:
    seed = int.from_bytes(hashlib.sha256(f"{image_id}|{variant}".encode()).digest()[:8], "little")
    return np.random.default_rng(seed)


def mixed_variant(image_id: str) -> str:
    """Textured variant assigned to an image in the ``mixed`` condition (deterministic)."""
    return TEXTURED_VARIANTS[hashlib.sha256(image_id.encode()).digest()[0] % len(TEXTURED_VARIANTS)]


# ---------------------------------------------------------------------------
# Backgrounds (float64 BGR in [0, 255], shape (h, w, 3))
# ---------------------------------------------------------------------------

def _line_family(h: int, w: int, period: float, rng: np.random.Generator) -> np.ndarray:
    """Darkness in [0, 1] of one family of parallel wavy lines (guilloche)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    theta = rng.uniform(-0.6, 0.6) + (np.pi / 2 if rng.random() < 0.3 else 0.0)
    u = xx * np.cos(theta) + yy * np.sin(theta)
    v = -xx * np.sin(theta) + yy * np.cos(theta)
    wave_len = rng.uniform(6.0, 14.0) * period
    amp = rng.uniform(0.6, 1.8) * period
    phase = 2 * np.pi * (v + amp * np.sin(2 * np.pi * u / wave_len + rng.uniform(0, 2 * np.pi))) / period
    duty = rng.uniform(2.5, 8.0)   # higher = thinner lines
    return np.clip(np.cos(phase), 0.0, 1.0) ** duty


def guilloche_grey(h: int, w: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    # Field case (screenshot): paper ~240, lines dip to 165-190, period ~4 px at 208 px wide.
    paper = rng.uniform(222.0, 245.0)
    period = rng.uniform(3.5, 6.0) * scale
    dark = _line_family(h, w, period, rng) * rng.uniform(0.25, 0.45)
    if rng.random() < 0.4:
        dark = np.maximum(dark, _line_family(h, w, period * rng.uniform(0.9, 1.3), rng) * rng.uniform(0.12, 0.25))
    return np.repeat((paper * (1.0 - dark))[..., None], 3, axis=2)


def guilloche_tint(h: int, w: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    tints = ((215, 240, 220), (225, 225, 248), (245, 230, 215), (205, 238, 245))   # green, pink, blue, yellow
    paper = np.array(tints[int(rng.integers(len(tints)))], np.float64) * rng.uniform(0.96, 1.02)
    line_colour = paper * rng.uniform(0.55, 0.75)
    period = rng.uniform(4.0, 8.0) * scale
    a = np.maximum(_line_family(h, w, period, rng), _line_family(h, w, period * rng.uniform(0.8, 1.25), rng))
    a = (a * rng.uniform(0.45, 0.9))[..., None]
    return np.clip(paper * (1.0 - a) + line_colour * a, 0, 255)


def bcsd_mask_path(image_path: Path) -> Path:
    """BCSD pairs <split>/X/X_nnn.jpeg with <split>/y/y_nnn.jpeg."""
    return image_path.parent.parent / "y" / image_path.name.replace("X_", "y_", 1)


@lru_cache(maxsize=1)
def _bcsd_backgrounds(bcsd_dir: str) -> Tuple[np.ndarray, ...]:
    """Signature regions of every BCSD cheque with the signature inpainted out (sorted)."""
    out = []
    for xp in sorted(Path(bcsd_dir).glob("*/X/*.jpeg")):
        img = cv2.imread(str(xp))
        mask = cv2.imread(str(bcsd_mask_path(xp)), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        m = (mask > 127).astype(np.uint8)
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        y0, y1 = max(0, ys.min() - 20), min(m.shape[0], ys.max() + 20)
        x0, x1 = max(0, xs.min() - 20), min(m.shape[1], xs.max() + 20)
        hole = cv2.dilate(m, np.ones((9, 9), np.uint8))
        clean = cv2.inpaint(img, hole, 7, cv2.INPAINT_TELEA)
        out.append(clean[y0:y1, x0:x1])
    if not out:
        raise FileNotFoundError(f"No BCSD images under {bcsd_dir}")
    return tuple(out)


def bcsd_real(h: int, w: int, scale: float, rng: np.random.Generator, bcsd_dir: str) -> np.ndarray:
    bgs = _bcsd_backgrounds(bcsd_dir)
    bg = bgs[int(rng.integers(len(bgs)))]
    return cv2.resize(bg, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float64)


def white_paper(h: int, w: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    return np.full((h, w, 3), rng.uniform(235.0, 252.0))


# ---------------------------------------------------------------------------
# Composite + capture
# ---------------------------------------------------------------------------

def composite(signature_bgr: np.ndarray, image_id: str, variant: str, bcsd_dir: str) -> np.ndarray:
    """Harmonize, put the ink on a cheque background and simulate a low-res capture."""
    rng = _rng(image_id, variant)
    gray = cv2.cvtColor(harmonize(signature_bgr), cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = gray.shape
    target_w = int(rng.integers(190, 231))
    scale = w / float(target_w)   # background periods are specified in output pixels
    if variant == "guilloche_grey":
        bg = guilloche_grey(h, w, scale, rng)
    elif variant == "guilloche_tint":
        bg = guilloche_tint(h, w, scale, rng)
    elif variant == "bcsd_real":
        bg = bcsd_real(h, w, scale, rng, bcsd_dir)
    elif variant == "clean_lowres":
        bg = white_paper(h, w, scale, rng)
    else:
        raise ValueError(f"Unknown variant {variant}")
    alpha = np.clip((255.0 - gray) / (255.0 * 0.8), 0.0, 1.0)[..., None] * rng.uniform(0.85, 1.0)
    ink = np.array(INK_COLOURS_BGR[int(rng.integers(len(INK_COLOURS_BGR)))], np.float64)
    out = bg * (1.0 - alpha) + ink * alpha
    target_h = max(8, int(round(h * target_w / float(w))))
    out = cv2.resize(np.clip(out, 0, 255).astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_AREA)
    out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 0.7)).astype(np.float64)
    out += rng.normal(0.0, rng.uniform(2.0, 5.0), out.shape)
    ok, buf = cv2.imencode(".jpg", np.clip(out, 0, 255).astype(np.uint8),
                           [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(55, 81))])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _composite_path(cache_dir: Path, variant: str, image_id: str) -> Path:
    """Composite location; refuses any path that would resolve outside `cache_dir`."""
    path = (cache_dir / variant / f"{image_id}.png").resolve()
    if not path.is_relative_to(cache_dir.resolve()):
        raise ValueError(f"Composite path escapes the cache dir: {path}")
    return path


def build(data_dir: Path, cache_dir: Path, bcsd_dir: str) -> int:
    """Write every composite (all images x all variants) under `cache_dir`."""
    n = 0
    images = load_images(data_dir)
    if not images:
        raise FileNotFoundError(f"No images under {data_dir}")
    for s in images:
        img = cv2.imread(s.path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(s.path)
        for v in VARIANTS:
            p = _composite_path(cache_dir, v, s.image_id)
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), composite(img, s.image_id, v, bcsd_dir))
            n += 1
    return n


# ---------------------------------------------------------------------------
# Production-path extraction
# ---------------------------------------------------------------------------

def _extract(job: Tuple[str, str]):
    """(image_id, path) -> (image_id, features or None, gate_passed)."""
    from signature_verification_system.src.preprocessing.quality import assess_signature_quality
    from signature_verification_system.src.verification.features import extract_features

    image_id, path = job
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    if not assess_signature_quality(img).passed:
        return image_id, None, False
    try:
        return image_id, extract_features(img), True
    except ValueError:
        return image_id, None, True


def _extract_clean(job: Tuple[str, str]):
    from signature_verification_system.src.verification.features import extract_features

    image_id, path = job
    img = harmonize(cv2.imread(path, cv2.IMREAD_COLOR))
    try:
        return image_id, extract_features(img), True
    except ValueError:
        return image_id, None, True


def _run(fn: Callable, jobs: List[Tuple[str, str]], workers: int) -> Dict[str, Tuple[object, bool]]:
    if workers <= 1:
        res = list(map(fn, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            res = list(pool.map(fn, jobs, chunksize=4))
    return {i: (f, ok) for i, f, ok in res}


def _logit(ref, que) -> float:
    from signature_verification_system.src.verification.similarity import compare

    if ref is None or que is None:
        return FAILED_LOGIT
    return round(float(compare(ref, que).fused_logit), 6)


def _ref_grouped_logits(evaluations, ref_feats) -> Dict[Tuple[str, int], float]:
    """Logit of every (reference, query features) pair of all evaluations, each computed once.

    Pairs are scored grouped by reference so the capture-scale re-extraction cache
    (features._coarser: one entry per reference and factor) is reused across the
    variants instead of being cycled out between evaluation passes. `compare` is
    deterministic and the cache is content-keyed, so the order changes no logit.
    """
    jobs: Dict[Tuple[str, int], Tuple[object, object]] = {}
    for pairs, que_feats in evaluations:
        for p in pairs:
            f = que_feats[p.query_id][0]
            jobs.setdefault((p.ref_id, id(f)), (ref_feats[p.ref_id][0], f))
    return {key: _logit(*jobs[key]) for key in sorted(jobs, key=lambda k: k[0])}


def _clamp(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _evaluate(pairs, que_feats, logits, acc: float, rej: float) -> Dict[str, object]:
    s: Dict[str, List[float]] = {"genuine": [], "skilled": [], "random": []}
    inconclusive: Dict[str, List[bool]] = {"genuine": [], "skilled": [], "random": []}
    for p in pairs:
        f, _ = que_feats[p.query_id]
        s[p.label].append(logits[(p.ref_id, id(f))])
        inconclusive[p.label].append(f is None)
    rate = lambda v, fn: float(np.mean([fn(x) for x in v])) if v else 0.0  # noqa: E731
    out: Dict[str, object] = {
        "eer_skilled": round(M.eer(s["genuine"], s["skilled"])["eer"], 6),
        "auc_skilled": round(M.auc(s["genuine"], s["skilled"]), 6),
        "operating_point": {
            lab: {"MATCH": round(rate(v, lambda x: x >= acc), 6), "NOMATCH": round(rate(v, lambda x: x < rej), 6),
                  "INCONCLUSIVE": round(float(np.mean(inconclusive[lab])) if inconclusive[lab] else 0.0, 6),
                  "n": len(v)}
            for lab, v in s.items() if v
        },
    }
    if s["random"]:
        out["eer_random"] = round(M.eer(s["genuine"], s["random"])["eer"], 6)
    return out


def score(data_dir: Path, cache_dir: Path, workers: int, max_random: int, both_textured: bool) -> Dict[str, object]:
    from signature_verification_system.src.core.config import DEFAULT_CONFIG

    t0 = time.perf_counter()
    images = load_images(data_dir)
    ids = sorted(s.image_id for s in images)
    paths = {s.image_id: s.path for s in images}
    pairs = cap_random_pairs(one_to_one_pairs(images), max_random)
    acc, rej = DEFAULT_CONFIG.decision.single_accept_logit, DEFAULT_CONFIG.decision.single_reject_logit

    per_variant = {v: _run(_extract, [(i, str(_composite_path(cache_dir, v, i))) for i in ids], workers) for v in VARIANTS}
    mixed = {i: per_variant[mixed_variant(i)][i] for i in ids}
    clean = _run(_extract_clean, [(i, paths[i]) for i in ids], workers)
    refs = mixed if both_textured else clean

    gs_pairs = [p for p in pairs if p.label in ("genuine", "skilled")]
    logits = _ref_grouped_logits([(pairs, mixed)] + [(gs_pairs, per_variant[v]) for v in VARIANTS], refs)
    res_mixed = _evaluate(pairs, mixed, logits, acc, rej)
    res_var = {v: _evaluate(gs_pairs, per_variant[v], logits, acc, rej) for v in VARIANTS}

    op = res_mixed["operating_point"]
    D = _clamp(1.0 - (0.75 * res_mixed["eer_skilled"] + 0.25 * res_mixed["eer_random"]) / EER_ZERO_POINT)
    R = float(np.mean([_clamp(1.0 - res_var[v]["eer_skilled"] / EER_ZERO_POINT) for v in TEXTURED_VARIANTS]))
    O = _clamp(0.5 * op["genuine"]["MATCH"] + 0.5 * op["skilled"]["NOMATCH"] - op["genuine"]["NOMATCH"])
    total = 10.0 * (WEIGHTS["D"] * D + WEIGHTS["R"] * R + WEIGHTS["O"] * O)
    gated = op["skilled"]["MATCH"] > GATE_SKILLED_MATCH or op["random"]["MATCH"] > GATE_RANDOM_MATCH
    if gated:
        total = min(total, GATED_SCORE_CAP)
    query_ids = sorted({p.query_id for p in pairs})
    return {
        "score": round(total, 3),
        "components": {"D": round(D, 4), "R": round(R, 4), "O": round(O, 4)},
        "safety_gate_triggered": gated,
        "both_textured": both_textured,
        "mixed": res_mixed,
        "mixed_inconclusive_rate": round(float(np.mean([mixed[i][0] is None for i in query_ids])), 6),
        "variants": res_var,
        "thresholds": {"accept": acc, "reject": rej},
        "n_pairs": {k: v["n"] for k, v in op.items()},
        "seconds": round(time.perf_counter() - t0, 1),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache-dir", required=True, help="where composites are written / read")
    ap.add_argument("--bcsd-dir", required=True, help="BCSD root (TrainSet/, TestSet/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-random", type=int, default=DEFAULT_MAX_RANDOM)
    ap.add_argument("--both-textured", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="regenerate composites even if cached")
    args = ap.parse_args(argv)
    cache = Path(args.cache_dir)
    if args.rebuild or not (cache / VARIANTS[-1]).is_dir():
        print(f"built {build(Path(args.data_dir), cache, args.bcsd_dir)} composites")
    res = score(Path(args.data_dir), cache, args.workers, args.max_random, args.both_textured)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    m = res["mixed"]
    print(f"CHEQUE SCORE {res['score']:.2f}/10  D={res['components']['D']:.3f} R={res['components']['R']:.3f} "
          f"O={res['components']['O']:.3f} gate={res['safety_gate_triggered']} both_textured={res['both_textured']}")
    print(f"mixed: skilled EER={m['eer_skilled']:.4f} random EER={m['eer_random']:.4f} "
          f"INCONCLUSIVE(queries)={res['mixed_inconclusive_rate']:.3f}")
    for lab, v in m["operating_point"].items():
        print(f"  {lab:8s} MATCH={v['MATCH']:.3f} NOMATCH={v['NOMATCH']:.3f} INCONCL={v['INCONCLUSIVE']:.3f} n={v['n']}")
    for name, v in res["variants"].items():
        g, s = v["operating_point"]["genuine"], v["operating_point"]["skilled"]
        print(f"  {name:16s} EER={v['eer_skilled']:.4f} genMATCH={g['MATCH']:.3f} genNOMATCH={g['NOMATCH']:.3f} "
              f"genINC={g['INCONCLUSIVE']:.3f} sklMATCH={s['MATCH']:.3f} sklINC={s['INCONCLUSIVE']:.3f}")
    print(f"{res['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
