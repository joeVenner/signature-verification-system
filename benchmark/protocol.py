"""Leak-free evaluation protocol built from the sample manifests.

The three sample folders re-use the same physical images across pairs (every
CEDAR reference appears in up to three pairs). Evaluating the 25 listed pairs
per category therefore under-uses the data and double-counts references. This
module de-duplicates images by SHA-256 and rebuilds every valid comparison:

* 1:1 genuine pairs  : all unordered pairs of genuine images of the same writer
* 1:1 skilled pairs  : every skilled forgery vs every genuine of its target writer
* 1:1 random pairs   : every genuine vs every genuine of a different writer
                       (within the same source domain, CEDAR or synthetic)
* writer-dependent   : enroll N-1 genuines, query the held-out genuine,
                       the writer's forgeries, and other writers' genuines

Writers are split into two deterministic, disjoint folds so thresholds can be
selected on one fold and evaluated on the other.

Everything is sorted explicitly; no filesystem or dict ordering leaks in.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

CATEGORIES: Tuple[str, ...] = ("genuine_pairs", "skilled_forgeries", "random_forgeries")


@dataclass(frozen=True)
class SampleImage:
    """One distinct physical signature image."""

    image_id: str          # short sha256 prefix, stable identity
    path: str              # first path (sorted) that holds these bytes
    writer: str            # writer whose name is signed
    is_genuine: bool       # False => skilled forgery of `writer`
    domain: str            # "cedar" | "synthetic"
    turn: int              # signing turn / forgery index (0 if unknown)


@dataclass(frozen=True)
class Pair:
    """Ordered (reference, questioned) comparison with ground truth."""

    ref_id: str
    query_id: str
    label: str             # "genuine" | "skilled" | "random"
    writer: str            # claimed identity (reference writer)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _domain(writer: str) -> str:
    return "synthetic" if writer.startswith("synthetic") else "cedar"


def load_images(data_dir: Path) -> List[SampleImage]:
    """Index every distinct image referenced by the three pair manifests."""
    records: Dict[str, dict] = {}

    def add(path: Path, writer: str, is_genuine: bool, turn: int) -> None:
        image_id = _sha(path)
        existing = records.get(image_id)
        if existing is not None:
            if existing["writer"] != writer or existing["is_genuine"] != is_genuine:
                raise ValueError(f"Conflicting labels for identical image {path}")
            existing["paths"].append(str(path))
            return
        records[image_id] = {
            "writer": writer, "is_genuine": is_genuine, "turn": turn, "paths": [str(path)],
        }

    for cat in CATEGORIES:
        manifest = json.loads((data_dir / cat / "manifest.json").read_text())
        for entry in manifest:
            ref = data_dir / cat / entry["ref_image"]
            que = data_dir / cat / entry["questioned_image"]
            if cat == "genuine_pairs":
                writer = entry["author"]
                add(ref, writer, True, int(entry.get("ref_turn", 1)))
                add(que, writer, True, int(entry.get("questioned_turn", 2)))
            elif cat == "skilled_forgeries":
                writer = entry["target_author"]
                add(ref, writer, True, int(entry.get("ref_turn", 1)))
                add(que, writer, False, int(entry.get("forgery_turn", 1)))
            else:
                add(ref, entry["ref_author"] if "ref_author" in entry else _writer_from_name(entry["ref_image"]), True, 0)
                add(que, entry["questioned_author"] if "questioned_author" in entry else _writer_from_name(entry["questioned_image"]), True, 0)

    images = [
        SampleImage(
            image_id=iid,
            path=sorted(rec["paths"])[0],
            writer=rec["writer"],
            is_genuine=rec["is_genuine"],
            domain=_domain(rec["writer"]),
            turn=rec["turn"],
        )
        for iid, rec in records.items()
    ]
    return sorted(images, key=lambda s: (s.writer, not s.is_genuine, s.turn, s.image_id))


def _writer_from_name(filename: str) -> str:
    # pair_01_random_w02.png / pair_01_ref_w01.png -> cedar_writer_02
    stem = Path(filename).stem
    return f"cedar_writer_{stem.rsplit('_w', 1)[1]}"


def writer_folds(images: List[SampleImage]) -> Dict[str, int]:
    """Deterministic writer-disjoint 2-fold split (alternate CEDAR writers by
    sorted order so both folds see a mix; synthetic writers go to fold 0)."""
    cedar = sorted({s.writer for s in images if s.domain == "cedar"})
    folds = {w: i % 2 for i, w in enumerate(cedar)}
    for s in images:
        if s.domain == "synthetic":
            folds[s.writer] = 0
    return folds


def one_to_one_pairs(images: List[SampleImage]) -> List[Pair]:
    """All valid 1:1 comparisons. Reference is always a genuine image."""
    by_writer: Dict[str, List[SampleImage]] = {}
    for s in images:
        by_writer.setdefault(s.writer, []).append(s)
    pairs: List[Pair] = []
    for writer in sorted(by_writer):
        gen = [s for s in by_writer[writer] if s.is_genuine]
        forg = [s for s in by_writer[writer] if not s.is_genuine]
        for a, b in itertools.combinations(gen, 2):
            pairs.append(Pair(a.image_id, b.image_id, "genuine", writer))
        for g in gen:
            for f in forg:
                pairs.append(Pair(g.image_id, f.image_id, "skilled", writer))
    genuines = [s for s in images if s.is_genuine]
    for a, b in itertools.combinations(genuines, 2):
        if a.writer != b.writer and a.domain == b.domain:
            pairs.append(Pair(a.image_id, b.image_id, "random", a.writer))
    return pairs


@dataclass(frozen=True)
class EnrollmentTrial:
    """Writer-dependent trial: questioned image vs a set of enrolled references."""

    ref_ids: Tuple[str, ...]
    query_id: str
    label: str
    writer: str


def writer_dependent_trials(images: List[SampleImage], n_refs: int = 3) -> List[EnrollmentTrial]:
    """Leave-one-genuine-out enrollment for writers with > n_refs genuines."""
    by_writer: Dict[str, List[SampleImage]] = {}
    for s in images:
        by_writer.setdefault(s.writer, []).append(s)
    trials: List[EnrollmentTrial] = []
    for writer in sorted(by_writer):
        gen = [s for s in by_writer[writer] if s.is_genuine]
        if len(gen) <= n_refs:
            continue
        forg = [s for s in by_writer[writer] if not s.is_genuine]
        others = [
            s for s in images
            if s.is_genuine and s.writer != writer and s.domain == gen[0].domain
        ]
        for held_out in gen:
            refs = tuple(s.image_id for s in gen if s.image_id != held_out.image_id)[:n_refs]
            trials.append(EnrollmentTrial(refs, held_out.image_id, "genuine", writer))
            for f in forg:
                trials.append(EnrollmentTrial(refs, f.image_id, "skilled", writer))
            for o in others:
                trials.append(EnrollmentTrial(refs, o.image_id, "random", writer))
    return trials
