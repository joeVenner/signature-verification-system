"""Load runtime settings from a `.env` file into the process environment.

Standard library only, on purpose: this module must run BEFORE numpy / OpenCV are
imported, because BLAS / OpenMP thread counts and OpenCV's decoder pixel limit are
read once when those libraries load (see docs/DETERMINISM.md).

Format: one `KEY=VALUE` per line; blank lines and `#` comments are ignored; a ` # note`
after an unquoted value is a comment; values may be wrapped in single or double quotes.
An empty value (`KEY=`) means "use the built-in default". Variables already present in the environment
always win, so a deployment can override any file value without editing it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

ENV_FILE_VAR = "SIGVERIFY_ENV_FILE"
DEFAULT_ENV_FILE = Path(__file__).resolve().parent / ".env"


def parse_env_file(path: Path) -> Dict[str, str]:
    """Parse `path` into a dict.

    Raises:
        ValueError: a non-comment line has no `=` or an empty key.
    """
    values: Dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key:
            raise ValueError(f"{path}:{number}: expected KEY=VALUE")
        values[key] = _clean_value(value, path, number)
    return values


def _clean_value(value: str, path: Path, number: int) -> str:
    """Unquote a value; for unquoted values drop a trailing ` # comment`.

    Raises:
        ValueError: an opening quote is never closed.
    """
    if value[:1] in ("'", '"'):
        quote = value[0]
        end = value.find(quote, 1)
        if end < 0:
            raise ValueError(f"{path}:{number}: unterminated {quote} quote")
        return value[1:end]
    comment = value.find(" #")
    return value[:comment].rstrip() if comment >= 0 else value


def load_env_file(path: Optional[Path] = None) -> Dict[str, str]:
    """Copy file values into `os.environ` without overriding existing variables.

    The file is `path`, else `$SIGVERIFY_ENV_FILE`, else `.env` next to this module.
    A missing file is not an error (defaults apply). Returns the values applied.
    """
    target = path or Path(os.environ.get(ENV_FILE_VAR) or DEFAULT_ENV_FILE)
    if not target.is_file():
        return {}
    applied: Dict[str, str] = {}
    for key, value in parse_env_file(target).items():
        if key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied
