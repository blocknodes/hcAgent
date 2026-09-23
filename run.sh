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

# 日志按次落盘：默认 logs/hcAgent_<启动时间戳>.log，每次启动一个新文件，永不覆盖。
#   - HC_LOG_FILE 显式指定时原样使用（保持兼容，追加）。
#   - logs/latest.log 软链指向最新一次启动，tail -f logs/latest.log 即可。
LOG_DIR="logs"
if [ -n "${HC_LOG_FILE:-}" ]; then
  LOG_FILE="$HC_LOG_FILE"          # 显式覆盖：沿用旧语义（追加）
  mkdir -p "$(dirname "$LOG_FILE")"
else
  mkdir -p "$LOG_DIR"
  TS="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="${LOG_DIR}/hcAgent_${TS}.log"
  # 同秒重启兜底：避免覆盖
  n=1
  while [ -e "$LOG_FILE" ]; do
    LOG_FILE="${LOG_DIR}/hcAgent_${TS}_${n}.log"
    n=$((n+1))
  done
fi
touch "$LOG_FILE"
ln -sfn "$(cd "$(dirname "$LOG_FILE")" && pwd)/$(basename "$LOG_FILE")" "${LOG_DIR}/latest.log"
echo "[run.sh] 日志 -> ${LOG_FILE}"

# 保留：历史单体 run.log 若存在则只追加不删除（兼容旧文件）
exec uvicorn app.main:app --host "$HOST" --port "$PORT" --reload 2>&1 | tee -a "$LOG_FILE"
