#!/usr/bin/env python3
"""0924 指标表 E 列「平均耗时(1并发)」实测器。

直接打本地 hcAgent(mock 8082) POST /slowAgent/poc_<id>，1 并发逐条发起，记录
每次请求往返耗时，按场景输出平均/中位/P90/P95。只读，不改表、不改任何评测脚本。

场景 → 用例集（与 0924 表 C 列一一对应，数据量以 C 列为准）：
  单业务:    cases/sheet0901_{vod,children,device,audio,education,sports,music}.csv
             影视445/少儿327/设备1140/有声369/教育216/体育207/音乐208 = 2912
  息屏亮屏:  cases/sheet0821_brightoff.csv        (99 条, 亮屏列)
  串行:      cases/sheet0821_serial.csv          (100 条)
  并行:      cases/e80803_multiintent.csv        (200 条)
  多轮:      cases/sheet0821_multiturn.csv       (150 条 30会话×5轮)
  多tab:     cases/e80803_multiintent.csv 里 multi_tab 分支 (60)

口径：并发数由 --workers 指定(默认 1)；延迟 = HTTP POST 到拿到 steps 的墙钟时间。

用法:
  python3 measure_latency.py --smoke 5          # 每场景只跑前 5 条(冒烟)
  python3 measure_latency.py --samples 20       # 每场景最多 20 条(快)
  python3 measure_latency.py                    # 全量(慢, 建议后台)
  python3 measure_latency.py --samples 10 -w 8  # 每场景 10 条, 8 并发
  python3 measure_latency.py --scenario 单业务-影视   # 只测指定场景
输出: output/latency_measure.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time as _time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

BENCH = Path(__file__).resolve().parent
AGENT = BENCH.parent
sys.path.insert(0, str(AGENT))
sys.path.insert(0, str(BENCH))
from runtime_execute import _run_runtime, DEFAULT_FEATURE_CODE  # noqa: E402
from run_sse_multiturn import device_id_for  # noqa: E402
CASES = BENCH / "cases"
OUT = BENCH / "output"
HOST = "http://localhost:8082"
TIMEOUT = 120

DOMAINS = ["vod", "children", "device", "audio", "education", "sports", "music"]
DOMAIN_CN = {"vod": "影视", "children": "少儿", "device": "设备", "audio": "有声",
             "education": "教育", "sports": "体育", "music": "音乐"}

_dev_seed = 0
_REMOTE = False
RUNTIME_DEVICE = device_id_for(f"latavgs0")


def _device() -> str:
    global _dev_seed
    _dev_seed += 1
    return f"861003009000014000000712lat{_dev_seed % 100:02d}0001"


POC_ID = "poc_measure"      # /slowAgent/{poc_id} 通配；任意 id 都走同一条 handler


def call(query: str, cid: str = "", tv: str = "0") -> tuple[float, str]:
    """返回 (耗时ms, tool)。_REMOTE=True 直连远端 runtime SSE；否则打本地 8082 mock。
    1 并发由调用方保证(串行 for)。tv 亮屏"0"/息屏"6"。路径走 /slowAgent/{poc_id} 通配。"""
    if _REMOTE:
        t0 = _time.perf_counter()
        r = _run_runtime(query, feature_code=DEFAULT_FEATURE_CODE,
                         device_id=RUNTIME_DEVICE, client_sid=cid or None,
                         tv_mode=tv, debug=True)
        ms = (_time.perf_counter() - t0) * 1000.0
        steps = [s for p in (r.get("plans") or []) for s in (p.get("steps") or [])]
        return ms, (steps[0].get("tool", "") if steps else "")
    body = {
        "traceId": f"lat-{int(_time.time()*1000)}-{_dev_seed}",
        "deviceId": _device(),
        "data": {"query": query, "tvMode": tv, "debug": True},
    }
    if cid:
        body["data"]["memory"] = {"shortMemory": [], "longMemory": {}}
        body["data"]["toolHistory"] = []
        body["data"]["clientSid"] = cid
    req = urllib.request.Request(f"{HOST}/slowAgent/{POC_ID}",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = _time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            raw = resp.read().decode()
        ms = (_time.perf_counter() - t0) * 1000.0
        try:
            tool = json.loads(raw)["data"]["steps"][0]["toolName"]
        except Exception:  # noqa: BLE001
            tool = ""
        return ms, tool
    except Exception as exc:  # noqa: BLE001
        return -1.0, f"ERR:{type(exc).__name__}"


def read_csv(rel: str, col: int = 1) -> list[str]:
    """读用例 csv 的第 col 列(默认 query)。跳表头、空、跨行 JSON。"""
    qs = []
    with open(CASES / rel, encoding="utf-8-sig") as f:
        for r in csv.reader(f):
            if len(r) > col:
                v = (r[col] or "").strip()
                if v and not v.startswith(("业务域", "query", "用户请求", "场景")):
                    qs.append(v)
    return qs


def build_plan(scenario_filter: str | None) -> list[tuple[str, str, str]]:
    """(场景, query, 桶key)。不并发、只收集。"""
    plan: list[tuple[str, str, str]] = []

    # 单业务 7 域
    for dom in DOMAINS:
        cn = DOMAIN_CN[dom]
        scn = f"单业务-{cn}"
        for q in read_csv(f"sheet0901_{dom}.csv", 1):
            plan.append((scn, q, cn))

    # 息屏亮屏 → brightoff 用亮屏列(=查询列)
    for q in read_csv("sheet0821_brightoff.csv", 1):
        plan.append(("息屏亮屏·亮", q, "亮"))

    # 串行
    for q in read_csv("sheet0821_serial.csv", 1):
        plan.append(("多意图-串行", q, "串行"))

    # 并行
    for q in read_csv("e80803_multiintent.csv", 1):
        plan.append(("多意图-并行", q, "并行"))

    # 多轮 150 → 按轮次分桶（08 21 多轮表: 会话 30, 每会 5 轮）
    for q in read_csv("sheet0821_multiturn.csv", 1):
        plan.append(("多轮", q, "多轮"))

    # 多tab 60 —— parallelem... 用 e80803 里含 "动画/动漫/卡通" 的查
    multitab = [q for q in read_csv("e80803_multiintent.csv", 1)
                if any(k in q for k in ("动画", "动漫", "卡通"))]
    for q in multitab:
        plan.append(("多tab", q, "多tab"))

    if scenario_filter:
        want = {s.strip() for s in scenario_filter.split(",") if s.strip()}
        plan = [p for p in plan if p[0] in want]
    return plan


def main():
    global _REMOTE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", type=int, default=0, help="每场景前 N 条(冒烟)")
    ap.add_argument("--samples", type=int, default=0, help="每场景最多取 N 条")
    ap.add_argument("--scenario", default=None, help="只测指定场景(逗号分隔)")
    ap.add_argument("--remote", action="store_true",
                    help="走远端 runtime SSE(10.18.210.7:31392)，非本地 8082 mock")
    ap.add_argument("-w", "--workers", type=int, default=1, help="并发数(默认 1)")
    args = ap.parse_args()
    _REMOTE = args.remote

    plan = build_plan(args.scenario)
    cap = args.smoke or args.samples or 0
    if cap:
        # 按场景截断，只取每场景前 cap 条
        seen: dict[str, int] = {}
        plan = [p for p in plan
                if (seen.__setitem__(p[0], seen.get(p[0], 0) + 1) or seen[p[0]]) <= cap]

    workers = max(1, args.workers)
    print(f"场景 {len({p[0] for p in plan})} 个, 计划 {len(plan)} 次调用 ({workers} 并发) ...", flush=True)

    samples: dict[str, list[float]] = {}
    n_err = 0
    done = 0
    t_start = _time.perf_counter()
    if workers == 1:
        for i, (scn, q, _bucket) in enumerate(plan):
            ms, _tool = call(q)
            if ms < 0:
                n_err += 1
            else:
                samples.setdefault(scn, []).append(ms)
            if i % 20 == 0 or i == len(plan) - 1:
                el = _time.perf_counter() - t_start
                print(f"\r[{i+1}/{len(plan)}] {el:.0f}s err={n_err}", end="", flush=True)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed as _ac
        import threading as _th
        lock = _th.Lock()
        futs = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for idx, (scn, q, _bucket) in enumerate(plan):
                futs[pool.submit(call, q)] = (idx, scn)
            for fut in _ac(futs):
                idx, scn = futs[fut]
                ms, _tool = fut.result()
                if ms < 0:
                    n_err += 1
                else:
                    samples.setdefault(scn, []).append(ms)
                with lock:
                    done += 1
                if done % 20 == 0 or done == len(plan):
                    el = _time.perf_counter() - t_start
                    print(f"\r[{done}/{len(plan)}] {el:.0f}s err={n_err}", end="", flush=True)
    print()

    OUT.mkdir(parents=True, exist_ok=True)
    rows = [["场景", "样本", "平均ms", "中位ms", "P90ms", "P95ms"]]
    for scn in sorted(samples):
        v = sorted(samples[scn])
        if not v:
            continue
        rows.append([scn, len(v), round(statistics.mean(v), 1), round(statistics.median(v), 1),
                     round(v[int(len(v)*0.9)-1], 1), round(v[int(len(v)*0.95)-1], 1)])
    for r in rows:
        print(f"{r[0]:16} {r[1]:>5} {r[2]:>9} {r[3]:>9} {r[4]:>9} {r[5]:>9}")
    path = OUT / ("latency_measure.csv" if workers == 1 else f"latency_measure_c{workers}.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()