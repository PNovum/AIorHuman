#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <ssh-alias> [LOCAL_PORT] [REMOTE_PORT]"
  exit 1
fi

ALIAS="$1"
LOCAL_PORT="${2:-8000}"
REMOTE_PORT="${3:-8000}"
REMOTE_DIR="\$HOME/n.polishchuk"
REPO_URL="$(git config --get remote.origin.url)"

echo "[local] Deploy to $ALIAS (remote:$REMOTE_PORT -> local:$LOCAL_PORT)"
echo "[local] Repo: $REPO_URL"

ssh "$ALIAS" "
  set -e
  sudo apt-get update -y
  sudo apt-get install -y python3-pip python3-venv curl git lsof >/dev/null

  mkdir -p $REMOTE_DIR

  if [ -d $REMOTE_DIR/.git ]; then
    git -C $REMOTE_DIR remote set-url origin '$REPO_URL'
    git -C $REMOTE_DIR fetch origin -q
    git -C $REMOTE_DIR checkout -B main origin/main -q
  else
    git clone '$REPO_URL' $REMOTE_DIR -q
  fi

  P=\$(lsof -ti tcp:$REMOTE_PORT || true)
  [ -n \"\$P\" ] && kill -9 \$P || true

  if ! command -v uv >/dev/null 2>&1; then
    pip3 install --user uv --break-system-packages -q || true
    export PATH=\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH
    if ! command -v uv >/dev/null 2>&1; then
      curl -LsSf https://astral.sh/uv/install.sh | sh
      export PATH=\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH
    fi
  fi

  cd $REMOTE_DIR
  unset VIRTUAL_ENV
  uv sync -q
  nohup uv run uvicorn youarebot.main:app --host 0.0.0.0 --port $REMOTE_PORT > service.log 2>&1 &
  sleep 1
  curl -s -I http://127.0.0.1:$REMOTE_PORT/docs | head -n1 || true
"

if lsof -ti tcp:"$LOCAL_PORT" >/dev/null 2>&1; then
  kill -9 "$(lsof -ti tcp:"$LOCAL_PORT")" || true
fi
ssh -fN -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "$ALIAS"

echo "[local] Tunnel ready → http://127.0.0.1:${LOCAL_PORT}/docs"
