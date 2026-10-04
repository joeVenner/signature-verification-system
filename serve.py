"""Start the verification API + console with settings taken from `.env`.

    python -m signature_verification_system.serve      (from the directory that contains the package)

The `.env` file is loaded before anything imports numpy / OpenCV so the thread and
pixel-limit settings take effect. Copy `.env.example` to `.env` and edit it.
"""

from __future__ import annotations

import os

from signature_verification_system.envfile import load_env_file

DEFAULT_HOST = "127.0.0.1"   # local console by default; set SIGVERIFY_HOST to expose it
DEFAULT_PORT = 8765


def main() -> None:
    load_env_file()
    import uvicorn  # imported after the environment is final

    uvicorn.run(
        "signature_verification_system.src.api.app:app",
        host=os.environ.get("SIGVERIFY_HOST", DEFAULT_HOST),
        port=int(os.environ.get("SIGVERIFY_PORT", DEFAULT_PORT)),
        workers=1,
    )


if __name__ == "__main__":
    main()
