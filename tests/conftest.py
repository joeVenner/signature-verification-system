"""Skip data-dependent test modules when the (non-redistributed) sample images are absent.

The public repository ships without the CEDAR / cheque images (see README).
Synthetic-image tests always run; tests that need the samples are skipped
instead of failing.
"""

import os
from pathlib import Path

# Hermetic tests: never pick up a developer's local `.env` (see envfile.py).
os.environ["SIGVERIFY_ENV_FILE"] = str(Path(__file__).resolve().parent / "no-such.env")

_SAMPLE = Path(__file__).resolve().parent.parent / "data" / "samples" / "genuine_pairs" / "pair_01_cedar_w01_ref.png"
_TEMPLATE = Path(__file__).resolve().parent.parent / "data" / "samples" / "cheques" / "cheque_procedural_pantograph_1001.png"

collect_ignore = []
if not (_SAMPLE.exists() and _TEMPLATE.exists()):
    collect_ignore = [
        "test_api_and_cli.py",
        "test_multi_reference.py",
        "test_pipeline.py",
        "test_signature_compare.py",
    ]
