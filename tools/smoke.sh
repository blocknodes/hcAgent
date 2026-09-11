#!/usr/bin/env bash
# hcAgent 纯 LLM 主干 smoke 测试：用 curl 打本地 poc 接口，覆盖四个关键用例。
#
#   · 单意图           —— LLM 收敛成一条自包含意图
#   · 多意图并行       —— 一次 T0 拆出多条独立意图(无 dep)
#   · 依赖：首轮        —— 依赖链只放前置步骤, stop=False
#   · 依赖：续跑        —— 带 s1 真实结果, 补发依赖步, stop=True
#
# 用法:
#   bash ./tools/smoke.sh                      # 默认 http://localhost:8082
#   HOST=127.0.0.1 PORT=8082 bash ./tools/smoke.sh
set -euo pipefail

HOST="${HOST:-localhost}"
PORT="${PORT:-8082}"
BASE="http://${HOST}:${PORT}"
POC="$BASE/slowAgent/poc_smoke"

step() { printf '\n\033[36m== %s ==\033[0m\n' "$1"; }

step "健康检查"
curl -s -m 5 "$BASE/health" && echo

step "1) 单意图"
curl -s -m 40 "$POC" -X POST -H 'content-type: application/json' \
  -d '{"traceId":"t1","deviceId":"d1","data":{"query":"我要看流浪地球","tvMode":"0"}}' \
  | python3 -c "import sys,json;b=json.load(sys.stdin);print('code',b['code'],'stop',b['stop']);[print(' ',s['id'],s['toolName'],'|',s['parameters'].get('query'),'| deps',s['dependsOn']) for s in b['data']['steps']]"

step "多意图并行"
curl -s -m 30 "$POC" -X POST -H 'content-type: application/json' \
  -d '{"traceId":"t2","deviceId":"d1","data":{"query":"点播战狼2同时调大音量","tvMode":"0"}}' \
  | python3 -c "import sys,json;b=json.load(sys.stdin);print('stop',b['stop']);[print(' ',s['id'],s['toolName'],'|',s['parameters'].get('query'),'| deps',s['dependsOn']) for s in b['data']['steps']]"

step "依赖：首轮(只发前置步, stop=False)"
curl -s -m 30 "$POC" -X POST -H 'content-type: application/json' \
  -d '{"traceId":"t3","deviceId":"d1","data":{"query":"搜索刘德华的电影，第三个的导演是谁","tvMode":"0"}}' \
  | python3 -c "import sys,json;b=json.load(sys.stdin);print('stop',b['stop']);[print(' ',s['id'],s['toolName'],'|',s['parameters'].get('query'),'| deps',s['dependsOn']) for s in b['data']['steps']]"

step "依赖：续跑(带 s1 结果, 补发依赖步, stop=True)"
curl -s -m 40 "$POC" -X POST -H 'content-type: application/json' \
  -d '{"traceId":"t3","deviceId":"d1","data":{"query":"搜索刘德华的电影，第三个的导演是谁","tvMode":"0","toolHistory":[{"id":"s1","result":{"data":[{"mediaTitle":"无间道"},{"mediaTitle":"暗战"},{"mediaTitle":"黑金"}]}}]}}' \
  | python3 -c "import sys,json;b=json.load(sys.stdin);print('stop',b['stop']);[print(' ',s['id'],s['toolName'],'|',s['parameters'].get('query'),'| deps',s['dependsOn']) for s in b['data']['steps']]"

printf '\n\033[32m完成. 服务=%s\n\033[0m' "$BASE"