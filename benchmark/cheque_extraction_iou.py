#!/usr/bin/env python3
"""Ink-extraction quality on real cheques with signature masks (BCSD).

    python -m signature_verification_system.benchmark.cheque_extraction_iou \
        --bcsd-dir ../cheque/raw/bcsd --out ../results/cheque_iou.json [--qa-dir ../cheque/qa]

BCSD (Khan 2021, arXiv:2104.12203): 158 bank cheques with manual pixel masks of
the signatures (TrainSet 129, TestSet 29). Each signature is cropped like a
detector would: the mask's bounding box plus 20% / 10% vertical / horizontal
margin (the crop therefore uses ground truth; localisation is not evaluated).

For every crop the ink mask the verifier actually uses (normalisation darkness
above INK_MASK_LEVEL, after rule / text isolation) is compared with the ground
truth: ``before`` without and ``after`` with background routing
(background.prepare_signature). Reported at native resolution and downscaled to
220 px wide (field crops are ~200 px). Deterministic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from signature_verification_system.benchmark.cheque_composite import bcsd_mask_path
from signature_verification_system.src.preprocessing.background import prepare_signature
from signature_verification_system.src.preprocessing.isolation import isolate_signature_ink
from signature_verification_system.src.preprocessing.normalization import (
    INK_MASK_LEVEL, ensure_dark_ink, harmonize_photometric, to_gray,
)

LOWRES_WIDTH = 220


def bcsd_crops(bcsd_dir: Path, width: Optional[int]) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
    """(name, BGR crop, bool GT mask) per cheque, sorted by path."""
    for xp in sorted(bcsd_dir.glob("*/X/*.jpeg")):
        img = cv2.imread(str(xp))
        gt = cv2.imread(str(bcsd_mask_path(xp)), cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            continue
        m = gt > 127
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        mh, mw = int(ys.max() - ys.min()), int(xs.max() - xs.min())
        y0, y1 = max(0, int(ys.min()) - mh // 5), int(ys.max()) + mh // 5
        x0, x1 = max(0, int(xs.min()) - mw // 10), int(xs.max()) + mw // 10
        crop, mask = img[y0:y1, x0:x1], m[y0:y1, x0:x1]
        if width:
            h = max(8, round(crop.shape[0] * width / crop.shape[1]))
            crop = cv2.resize(crop, (width, h), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask.astype(np.uint8), (width, h), interpolation=cv2.INTER_AREA) > 0
        yield f"{xp.parent.parent.name}/{xp.stem}", crop, mask


def ink_mask(image: np.ndarray) -> np.ndarray:
    """The verifier's ink mask (normalisation without routing), input resolution."""
    harmonized = harmonize_photometric(image)
    darkness = isolate_signature_ink((255.0 - harmonized.astype(np.float64)) / 255.0, INK_MASK_LEVEL)
    return darkness > INK_MASK_LEVEL


def routed_ink_mask(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    """(mask at input resolution, prepared work image, background removed)."""
    prepared = prepare_signature(image, ensure_dark_ink(to_gray(image))[0])
    mask = ink_mask(prepared.work_image)
    h, w = image.shape[:2]
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA) > 0
    return mask, prepared.work_image, prepared.background_removed


def iou_f1(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    tp = float(np.count_nonzero(pred & gt))
    fp = float(np.count_nonzero(pred & ~gt))
    fn = float(np.count_nonzero(~pred & gt))
    return tp / max(tp + fp + fn, 1.0), 2 * tp / max(2 * tp + fp + fn, 1.0)


def _summary(rows: List[Dict[str, object]]) -> Dict[str, object]:
    def mean(key: str, sel: List[Dict[str, object]]) -> Optional[float]:
        return round(float(np.mean([r[key] for r in sel])), 4) if sel else None

    routed = [r for r in rows if r["routed"]]
    return {
        "n": len(rows), "routed": len(routed),
        "all": {k: mean(k, rows) for k in ("iou_before", "iou_after", "f1_before", "f1_after")},
        "routed_only": {k: mean(k, routed) for k in ("iou_before", "iou_after", "f1_before", "f1_after")},
    }


def _qa_panel(crop: np.ndarray, gt: np.ndarray, before: np.ndarray, after: np.ndarray, work: np.ndarray) -> np.ndarray:
    width = 360
    height = max(8, round(crop.shape[0] * width / crop.shape[1]))

    def fit(im: np.ndarray) -> np.ndarray:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if im.ndim == 2 else im
        return cv2.resize(im, (width, height), interpolation=cv2.INTER_AREA)

    as_img = lambda m: np.where(m, 0, 255).astype(np.uint8)  # noqa: E731
    return np.hstack([fit(crop), fit(as_img(gt)), fit(as_img(before)), fit(work), fit(as_img(after))])


def run(bcsd_dir: Path, qa_dir: Optional[Path]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for label, width in (("native", None), (f"w{LOWRES_WIDTH}", LOWRES_WIDTH)):
        rows: List[Dict[str, object]] = []
        panels: List[np.ndarray] = []
        for name, crop, gt in bcsd_crops(bcsd_dir, width):
            before = ink_mask(crop)
            after, work, routed = routed_ink_mask(crop)
            ib, fb = iou_f1(before, gt)
            ia, fa = iou_f1(after, gt)
            rows.append({"name": name, "routed": routed, "iou_before": ib, "iou_after": ia, "f1_before": fb, "f1_after": fa})
            if qa_dir is not None and routed and len(panels) < 8:
                panels.append(_qa_panel(crop, gt, before, after, work))
        if not rows:
            raise FileNotFoundError(f"No BCSD image / mask pairs under {bcsd_dir}")
        out[label] = {split: _summary([r for r in rows if r["name"].startswith(split)]) for split in ("TrainSet", "TestSet")}
        out[label]["per_image"] = rows
        if qa_dir is not None and panels:
            qa_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(qa_dir / f"bcsd_extraction_{label}.png"), np.vstack(panels))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bcsd-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--qa-dir", default=None, help="write before/after panels of routed crops here")
    args = ap.parse_args(argv)
    res = run(Path(args.bcsd_dir), Path(args.qa_dir) if args.qa_dir else None)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    for label, splits in res.items():
        for split in ("TrainSet", "TestSet"):
            s = splits[split]
            print(f"{label:7s} {split:8s} n={s['n']:3d} routed={s['routed']:3d} all={s['all']} routed_only={s['routed_only']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
