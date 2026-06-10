#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
exec .venv/bin/python main.py --listen 127.0.0.1 --port 8188 "$@"
