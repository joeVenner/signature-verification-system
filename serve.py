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
    try:
        load_env_file()
    except ValueError as exc:
        raise SystemExit(f"Invalid .env file: {exc}") from None
    port_text = os.environ.get("SIGVERIFY_PORT") or str(DEFAULT_PORT)
    if not port_text.isdigit() or not 0 < int(port_text) < 65536:
        raise SystemExit(f"SIGVERIFY_PORT must be a port number, got {port_text!r}")
    import uvicorn  # imported after the environment is final

    uvicorn.run(
        "signature_verification_system.src.api.app:app",
        host=os.environ.get("SIGVERIFY_HOST") or DEFAULT_HOST,
        port=int(port_text),
        workers=1,
    )


if __name__ == "__main__":
    main()
