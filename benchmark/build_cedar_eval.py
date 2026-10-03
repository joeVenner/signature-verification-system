#!/usr/bin/env python3
"""Build a manifest-compatible evaluation folder from the full CEDAR download.

The benchmark protocol (benchmark/protocol.py) reads three manifests. This script
writes them for a chosen slice of full CEDAR (55 writers x 24 genuine + 24 skilled
forgeries) and links the images instead of copying them.

    # download once (academic, non-commercial): https://cedar.buffalo.edu/NIJ/data/signatures.rar
    python -m signature_verification_system.benchmark.build_cedar_eval \
        --cedar-dir /path/to/signatures --out-dir /path/to/cedar55_dev  --genuine 6 --forgeries 6 --offset 0
    python -m signature_verification_system.benchmark.build_cedar_eval \
        --cedar-dir /path/to/signatures --out-dir /path/to/cedar55_test --genuine 6 --forgeries 6 --offset 12

EXP-015 used offset 0 (images 1-6) for development and offset 12 (images 13-18) as an
untouched test set. Slices with different offsets share no image.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

WRITERS = 55
CATEGORIES = ("genuine_pairs", "skilled_forgeries", "random_forgeries")


def _link(category_dir: Path, name: str, target: Path) -> None:
    if not target.is_file():
        raise FileNotFoundError(f"CEDAR image not found: {target}")
    link = category_dir / name
    if not link.exists():
        os.symlink(target.resolve(), link)


def build(cedar_dir: Path, out_dir: Path, genuine: int, forgeries: int, offset: int) -> Dict[str, int]:
    """Write manifests + image links; returns the number of manifest entries per category."""
    if genuine < 2 or genuine % 2:
        raise ValueError("--genuine must be an even number >= 2 (images are paired)")
    if forgeries < 1:
        raise ValueError("--forgeries must be >= 1")
    for category in CATEGORIES:
        (out_dir / category).mkdir(parents=True, exist_ok=True)

    genuine_entries: List[dict] = []
    skilled_entries: List[dict] = []
    for writer in range(1, WRITERS + 1):
        author = f"cedar_writer_{writer:02d}"
        for first in range(offset + 1, offset + genuine + 1, 2):
            names = [f"original_{writer}_{first}.png", f"original_{writer}_{first + 1}.png"]
            for name in names:
                _link(out_dir / "genuine_pairs", name, cedar_dir / "full_org" / name)
            genuine_entries.append({
                "author": author, "ref_image": names[0], "questioned_image": names[1],
                "ref_turn": first, "questioned_turn": first + 1,
            })
        reference = f"original_{writer}_{offset + 1}.png"
        _link(out_dir / "skilled_forgeries", reference, cedar_dir / "full_org" / reference)
        for turn in range(offset + 1, offset + forgeries + 1):
            forged = f"forgeries_{writer}_{turn}.png"
            _link(out_dir / "skilled_forgeries", forged, cedar_dir / "full_forg" / forged)
            skilled_entries.append({
                "target_author": author, "ref_image": reference, "questioned_image": forged,
                "ref_turn": offset + 1, "forgery_turn": turn,
            })

    # Random-forgery pairs are re-derived from the genuine images by the protocol.
    manifests = {"genuine_pairs": genuine_entries, "skilled_forgeries": skilled_entries, "random_forgeries": []}
    for category, entries in manifests.items():
        (out_dir / category / "manifest.json").write_text(json.dumps(entries, indent=1) + "\n")
    return {category: len(entries) for category, entries in manifests.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cedar-dir", required=True, type=Path, help="folder holding full_org/ and full_forg/")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--genuine", type=int, default=6, help="genuine images per writer (even)")
    parser.add_argument("--forgeries", type=int, default=6, help="skilled forgeries per writer")
    parser.add_argument("--offset", type=int, default=0, help="skip this many images per writer (0 = start at image 1)")
    args = parser.parse_args()
    counts = build(args.cedar_dir, args.out_dir, args.genuine, args.forgeries, args.offset)
    print(counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
