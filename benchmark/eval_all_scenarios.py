#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 eval_query.py 口径全场景评测：逐条 POST 本地 hcAgent mock(8082)，采集
每请求 latency + hitSource(primary/all steps)，按场景/桶输出分布。

场景桶(与 measure_latency.py 一致，息屏亮屏额外按 tvMode 拆 亮/息)：
  单业务:    sheet0901_{vod,children,device,audio,education,sports,music}.csv（tv=0）
  息屏亮屏:  sheet0821_brightoff.csv  每行跑 2 条：tv=0(亮) 与 tv=6(息)
  串行:      sheet0821_serial.csv   并行: e80803_multiintent.csv
  多轮:      sheet0821_multiturn.csv（不分轮次，一桶统计）
  多tab:     e80803 里含 动画/动漫/卡通 的查询

口径同 eval_query：延迟 = HTTP POST 到拿全 steps 的墙钟时间；hitSource 取每一步返回值。

用法:
  python3 benchmark/eval_all_scenarios.py --smoke 5
  python3 benchmark/eval_all_scenarios.py --samples 20 -w 8
  python3 benchmark/eval_all_scenarios.py --scenario 息屏亮屏.bright
  python3 benchmark/eval_all_scenarios.py           # 全量（~1.7 万条，慢）
输出: output/eval_all_scenarios.csv + output/eval_all_scenarios_raw.json
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time as _time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))
try:
    from measure_latency import POC_ID, HOST, TIMEOUT, _device  # noqa: E402
except Exception:  # noqa: BLE001
    POC_ID, HOST, TIMEOUT = "poc_measure", "http://localhost:8082", 120
    _device = lambda: "eval-all-00000"  # noqa: E731

CASES = BENCH / "cases"
OUT = BENCH / "output"

DOMAINS = ["vod", "children", "device", "audio", "education", "sports", "music"]
DOMAIN_CN = {"vod": "影视", "children": "少儿", "device": "设备", "audio": "有声",
             "education": "教育", "sports": "体育", "music": "音乐"}
TAB_KEYS = ("动画", "动漫", "卡通")


def read_csv(rel: str, col: int = 1) -> list[str]:
    qs = []
    with open(CASES / rel, encoding="utf-8-sig") as f:
        for r in csv.reader(f):
            if len(r) > col:
                v = (r[col] or "").strip()
                if v and not v.startswith(("业务域", "query", "用户请求", "场景")):
                    qs.append(v)
    return qs


def build_plan(scenario_filter: str | None) -> list[tuple[str, str, str]]:
    """(桶, query, tvMode)。tvMode: 亮屏"0" / 息屏"6"。"""
    plan: list[tuple[str, str, str]] = []

    for dom in DOMAINS:
        for q in read_csv(f"sheet0901_{dom}.csv", 1):
            plan.append((f"单业务-{DOMAIN_CN[dom]}", q, "0"))

    for q in read_csv("sheet0821_brightoff.csv", 1):
        plan.append(("息屏亮屏.bright", q, "0"))
        plan.append(("息屏亮屏.dark", q, "6"))

    for q in read_csv("sheet0821_serial.csv", 1):
        plan.append(("串行", q, "0"))

    for q in read_csv("e80803_multiintent.csv", 1):
        plan.append(("并行", q, "0"))

    for q in read_csv("sheet0821_multiturn.csv", 1):
        plan.append(("多轮", q, "0"))

    # 多tab：手写用例文件，仿亮息屏拆 tv=0/tv=6 两拍
    for q in read_csv("sheet0821_multitab.csv", 1):
        plan.append(("多tab.bright", q, "0"))
        plan.append(("多tab.dark", q, "6"))

    if scenario_filter:
        want = {s.strip() for s in scenario_filter.split(",") if s.strip()}
        plan = [p for p in plan if any(p[0].startswith(w) for w in want)]
    return plan


def call(query: str, tv: str) -> dict:
    body = {
        "traceId": f"eall-{int(_time.time()*1000)}-{abs(hash(query)) & 0xffff}",
        "deviceId": _device(),
        "data": {"query": query, "tvMode": tv, "enableMultiIntent": False, "debug": True},
    }
    req = urllib.request.Request(f"{HOST}/slowAgent/{POC_ID}",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = _time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            raw = resp.read().decode()
        ms = (_time.perf_counter() - t0) * 1000.0
        j = json.loads(raw)
        steps = (j.get("data") or {}).get("steps") or []
        hs = [s.get("hitSource") for s in steps] or [None]
        return {"query": query, "tv": tv, "cost_ms": ms,
                "ok": resp.status == 200 and j.get("code") == 200,
                "hitSources": hs, "primary": hs[0],
                "tools": [s.get("toolName") for s in steps], "nSteps": len(steps),
                "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"query": query, "tv": tv, "cost_ms": None, "ok": False,
                "hitSources": [], "primary": None, "tools": [], "nSteps": 0,
                "error": f"{type(exc).__name__}: {exc}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", type=int, default=0, help="每桶前 N 条(冒烟)")
    ap.add_argument("--samples", type=int, default=0, help="每桶最多 N 条")
    ap.add_argument("--scenario", default=None, help="只跑指定场景(逗号分隔，前缀匹配)")
    ap.add_argument("-w", "--workers", type=int, default=4, help="并发数")
    ap.add_argument("--out", default=None, help="csv 输出名，默认 eval_all_scenarios.csv")
    args = ap.parse_args()

    plan = build_plan(args.scenario)
    cap = args.smoke or args.samples or 0
    if cap:
        seen: dict[str, int] = {}
        plan = [p for p in plan
                if (seen.__setitem__(p[0], seen.get(p[0], 0) + 1) or seen[p[0]]) <= cap]

    print(f"桶 {len({p[0] for p in plan})} 个, 计划 {len(plan)} 条, 并发 {args.workers}", flush=True)

    results: dict[str, list[dict]] = {}
    fail = 0
    done = 0
    t_start = _time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        fmap = {pool.submit(call, q, tv): scn for scn, q, tv in plan}
        for fut in as_completed(fmap):
            scn = fmap[fut]
            r = fut.result()
            results.setdefault(scn, []).append(r)
            fail += 0 if r["ok"] else 1
            done += 1
            if done % 100 == 0 or done == len(plan):
                el = _time.perf_counter() - t_start
                print(f"\r  [{done}/{len(plan)}] {el:.0f}s fail={fail}", end="", flush=True)
    print()

    OUT.mkdir(parents=True, exist_ok=True)
    rows = [["场景", "样本", "失败", "平均ms", "中位ms", "P90ms", "P95ms", "P99ms"]]
    for scn in sorted(results):
        rs = results[scn]
        costs = [r["cost_ms"] for r in rs if r["ok"] and r["cost_ms"] is not None]
        nfail = sum(1 for r in rs if not r["ok"])
        if not costs:
            rows.append([scn, len(rs), nfail, "-", "-", "-", "-", "-"])
            continue
        c = sorted(costs)
        p = lambda q: c[min(len(c) - 1, int(len(c) * q))]
        rows.append([scn, len(rs), nfail,
                     round(statistics.mean(costs), 1), round(statistics.median(costs), 1),
                     round(p(0.90), 1), round(p(0.95), 1), round(p(0.99), 1)])
    for r in rows:
        print(f"{r[0]:20} 样本{r[1]:>5} 失败{r[2]:>3} mean{r[3]:>8} med{r[4]:>8} "
              f"p90{r[5]:>8} p95{r[6]:>8} p99{r[7]:>8}")
    path = OUT / (args.out or "eval_all_scenarios.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)

    # hitSource 分布（primary + all steps），按场景与总体
    hs_primary = Counter()
    hs_all = Counter()
    per_scn: dict[str, dict] = {}
    for scn, rs in sorted(results.items()):
        hp, ha = Counter(), Counter()
        for r in rs:
            hp[r["primary"]] += 1
            for s in r["hitSources"]:
                ha[s] += 1
        per_scn[scn] = {"primary": dict(hp), "all": dict(ha)}
        hs_primary.update(hp)
        hs_all.update(ha)
    print("\n----- hitSource 分布(全部场景合并, primary) -----")
    for k, v in hs_primary.most_common():
        print(f"  {k}: {v}  ({v / len(plan) * 100:.1f}%)")
    print("\n----- hitSource 分布(全部场景合并, all steps) -----")
    for k, v in hs_all.most_common():
        print(f"  {k}: {v}  ({v / sum(hs_all.values()) * 100:.1f}%)")
    for scn in sorted(per_scn):
        hp = per_scn[scn]["primary"]
        if hp:
            top = ", ".join(f"{k}={v}" for k, v in sorted(hp.items(), key=lambda x: -x[1])[:4])
            print(f"  [{scn}] {top}")

    rawpath = OUT / "eval_all_scenarios_raw.json"
    with open(rawpath, "w", encoding="utf-8") as f:
        json.dump({"per_scenario": per_scn,
                   "rows": rows,
                   "raw": results}, f, ensure_ascii=False, indent=1)
    print(f"\n-> {path}\n-> {rawpath}")


if __name__ == "__main__":
    main()