#!/usr/bin/env bash
set -e

unset VIRTUAL_ENV

if ! command -v uv >/dev/null 2>&1; then
  pip3 install --user uv --break-system-packages
  export PATH="$HOME/.local/bin:$PATH"
fi

uv sync

nohup uv run uvicorn youarebot.main:app --host 0.0.0.0 --port 8000 > service.log 2>&1 &

