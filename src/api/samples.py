"""Read-only gallery of labelled sample pairs for manual testing of the console.

Enabled only when the operator points `SIGVERIFY_SAMPLES_DIR` at a dataset
directory laid out as `<category>/manifest.json` + images (see CATEGORIES).
Manifests are treated as untrusted: every referenced file must be a plain file
name that resolves inside its own category directory. Clients only ever see
server-generated integer IDs, never paths.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from pydantic import BaseModel

LOGGER = logging.getLogger(__name__)

SAMPLES_DIR_ENV = "SIGVERIFY_SAMPLES_DIR"
# (sub-directory, dataset label shown to testers)
CATEGORIES: Tuple[Tuple[str, str], ...] = (
    ("genuine_pairs", "Genuine"),
    ("skilled_forgeries", "Skilled forgery"),
    ("random_forgeries", "Random forgery"),
)
IMAGE_SUFFIXES: Mapping[str, str] = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".tif": "image/tiff", ".tiff": "image/tiff", ".bmp": "image/bmp",
}
MAX_MANIFEST_BYTES = 5 * 1024 * 1024
MAX_PAIRS = 5000
ID_PATTERN = re.compile(r"^[0-9]{1,9}$")


class SamplePair(BaseModel):
    id: int
    dataset_label: str
    category: str
    writer: Optional[str] = None
    reference_image_id: int
    questioned_image_id: int
    reference_name: str
    questioned_name: str


class SampleList(BaseModel):
    samples: List[SamplePair]


@dataclass(frozen=True)
class SampleCatalog:
    pairs: Tuple[SamplePair, ...]
    images: Tuple[Path, ...]   # index = image id

    def image(self, raw_id: str) -> Optional[Path]:
        """Resolve a client-supplied image id; None for anything that is not a known id."""
        if not ID_PATTERN.match(raw_id):
            return None
        index = int(raw_id)
        return self.images[index] if index < len(self.images) else None


def _safe_file(directory: Path, name: object) -> Optional[Path]:
    """`directory / name` if `name` is a plain image file name directly inside `directory`, else None.

    Manifest names may not contain separators, so a manifest can never point
    outside its directory. Symbolic links that the operator placed *in* the
    directory are followed (dataset slices are commonly built that way): the
    directory's contents are operator-controlled, manifest strings are not.
    """
    if not isinstance(name, str) or not name or name in (".", ".."):
        return None
    if "/" in name or "\\" in name or "\x00" in name or Path(name).name != name:
        return None
    if Path(name).suffix.lower() not in IMAGE_SUFFIXES:
        return None
    candidate = directory / name
    return candidate if candidate.is_file() else None


def _read_manifest(path: Path) -> List[dict]:
    if not path.is_file() or path.stat().st_size > MAX_MANIFEST_BYTES:
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        LOGGER.warning("Ignoring unreadable sample manifest %s: %s", path.name, exc)
        return []
    return [e for e in data if isinstance(e, dict)] if isinstance(data, list) else []


def load_catalog(root: Optional[str]) -> Optional[SampleCatalog]:
    """Build the gallery from `root`; None (gallery disabled) when unset or not a directory."""
    if not root:
        return None
    base = Path(root).resolve()
    if not base.is_dir():
        LOGGER.warning("%s does not point to a directory; sample gallery disabled", SAMPLES_DIR_ENV)
        return None
    image_ids: Dict[Path, int] = {}
    raw_pairs: List[Tuple[str, str, Optional[str], Path, Path]] = []
    for sub, label in CATEGORIES:
        directory = (base / sub).resolve()
        if not directory.is_relative_to(base) or not directory.is_dir():
            continue
        for entry in _read_manifest(directory / "manifest.json"):
            ref = _safe_file(directory, entry.get("ref_image"))
            que = _safe_file(directory, entry.get("questioned_image"))
            if ref is None or que is None or len(raw_pairs) >= MAX_PAIRS:
                continue
            writer = entry.get("author") or entry.get("target_author")
            raw_pairs.append((sub, label, writer if isinstance(writer, str) else None, ref, que))
    for path in sorted({p for _, _, _, r, q in raw_pairs for p in (r, q)}):
        image_ids[path] = len(image_ids)
    pairs = tuple(
        SamplePair(id=i, dataset_label=label, category=sub, writer=writer,
                   reference_image_id=image_ids[ref], questioned_image_id=image_ids[que],
                   reference_name=ref.name, questioned_name=que.name)
        for i, (sub, label, writer, ref, que) in enumerate(raw_pairs)
    )
    return SampleCatalog(pairs=pairs, images=tuple(image_ids))


def catalog_from_env() -> Optional[SampleCatalog]:
    return load_catalog(os.environ.get(SAMPLES_DIR_ENV))
