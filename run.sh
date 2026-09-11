#!/usr/bin/env bash
# hcAgent 启动脚本：FastAPI mock（唯一对外接口 POST /slowAgent/poc_<id>）
set -euo pipefail
cd "$(dirname "$0")"

PORT="${HC_PORT:-8082}"
HOST="${HC_HOST:-0.0.0.0}"

exec uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
