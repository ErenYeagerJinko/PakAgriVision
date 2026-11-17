"""Local Flask launcher with preflight checks and optional browser opening."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import socket
import sys
import threading
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--check", action="store_true", help="Validate startup without serving")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    modules = ["flask", "torch", "torchvision", "PIL", "numpy", "pandas", "sklearn",
               "matplotlib", "jsonschema"]
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        print("Missing dependencies: " + ", ".join(missing), file=sys.stderr)
        print(f'Install with: "{sys.executable}" -m pip install -r "{ROOT / "requirements.txt"}"', file=sys.stderr)
        return 1
    if not (ROOT / "runs/baseline/best.pt").is_file():
        print("Missing trained weights. Run: python scripts/download_weights.py", file=sys.stderr)
        return 1
    from app.app import app, ENGINE, TREATMENT_KB, KB_ERROR
    if not ENGINE.ready or TREATMENT_KB is None:
        print("Startup failed: " + (ENGINE.error or KB_ERROR), file=sys.stderr)
        return 1
    if args.check:
        print(f"Startup checks passed: trained ResNet-50 on {ENGINE.device}; {len(TREATMENT_KB)} treatment records.")
        return 0
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", args.port))
        except OSError:
            print(f"Port {args.port} is unavailable. Choose another with --port NUMBER.", file=sys.stderr)
            return 1
    from werkzeug.serving import make_server
    # Bind before opening the browser; never open a different process's occupied port.
    try:
        server = make_server("127.0.0.1", args.port, app, threaded=True)
    except (OSError, SystemExit):
        print(f"Could not start local server on port {args.port}", file=sys.stderr)
        return 1
    url = f"http://127.0.0.1:{args.port}"
    if not args.no_browser:
        timer = threading.Timer(0.8, webbrowser.open, args=(url,))
        timer.daemon = True
        timer.start()
    print(f"PakAgriVision: {url}\nPress Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
