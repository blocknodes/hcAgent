#!/usr/bin/env bash
# hcAgent 启动脚本：FastAPI mock（唯一对外接口 POST /slowAgent/poc_<id>）
set -euo pipefail
cd "$(dirname "$0")"

PORT="${HC_PORT:-8082}"
HOST="${HC_HOST:-0.0.0.0}"

# 先杀掉占用同一端口的旧进程（往往是上次没退干净的 uvicorn）
if command -v lsof >/dev/null 2>&1; then
  pids="$(lsof -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    echo "[run.sh] 端口 $PORT 被以下进程占用，先结束：$pids"
    kill $pids 2>/dev/null || true
    sleep 1
  fi
fi

exec uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
