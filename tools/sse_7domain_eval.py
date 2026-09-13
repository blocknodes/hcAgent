"""7 域(设备/体育/音乐/有声/影视/少儿/教育)SSE 真实链路评测。

读聚合后的 sheet0827_caseset.json，对每条 query 直连远端 runtime(SSE, enable_slow)
真跑，从 DEBUG_MAP 计划帧取"数据集命名"真工具名 + 参数，与 golden(期望工具+期望参数)比对。

指标(全域 & 按域分组)：
  tool        tool 命中率
  param       param 命中率(golden 无参数则该条不计 param、由 tool 决定)
  tool+param  joint 命中率 = tool 且 param 都对   ← 主指标，目标 ≥90%

并行：ThreadPoolExecutor 并发直连 SSE。
用法:
  python tools/sse_7domain_eval.py                 # 全量 2912 条
  python tools/sse_7domain_eval.py -n 10           # 前 10 条(冒烟)
  python tools/sse_7domain_eval.py -d vod,music    # 只跑指定域
  python tools/sse_7domain_eval.py --json          # JSON 输出
  python tools/sse_7domain_eval.py --workers 16    # 并发数(默认 8)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

TOOLS = Path(__file__).resolve().parent                # hcAgent/tools
AGENT = TOOLS.parent                                  # hcAgent
CASESET = AGENT / "data" / "sheet0827_caseset.json"
DOMAIN_ORDER = ["device", "vod", "audio", "music", "sports", "children", "education"]

sys.path.insert(0, str(AGENT))
from runtime_execute import (                         # noqa: E402
    DEFAULT_FEATURE_CODE,
    _run_runtime,
    random_device_id,
)


def _empty(v):
    if v is None:
        return True
    if isinstance(v, str):
        return not v.strip()
    if isinstance(v, (list, dict)):
        return len(v) == 0
    return False


def canonical(x):
    if isinstance(x, dict):
        return {k: canonical(v) for k, v in x.items() if not _empty(v)}
    if isinstance(x, list):
        return sorted(
            (canonical(i) for i in x if not _empty(i)),
            key=lambda e: json.dumps(e, ensure_ascii=False, sort_keys=True),
        )
    if isinstance(x, str):
        return x.strip()
    return x


def params_equal(a, b):
    return canonical(a) == canonical(b)


def normalize_golden_params(v):
    """unify golden params. 已由 build 脚本规范化为 dict 或 {_raw:..}。"""
    if isinstance(v, dict) and set(v.keys()) == {"_raw"}:
        return None           # 不可比 → 只比 tool
    return v or {}


def run_one(case, tv_mode="0"):
    q = case["query"]
    started = time.perf_counter()
    r = _run_runtime(
        q, feature_code=DEFAULT_FEATURE_CODE, client_sid=None,
        device_id=random_device_id(), tv_mode=tv_mode, debug=True,
    )
    latency = round((time.perf_counter() - started) * 1000.0, 1)
    steps = [
        {"tool": s.get("toolName") or "", "retext": s.get("retext", ""),
         "params": s.get("parameters") or {},
         "hit_source": s.get("hitSource") or s.get("hit_source") or ""}
        for p in r["plans"] for s in (p.get("steps") or [])
        if isinstance(s, dict) and s.get("toolName")
    ]
    if not steps:
        steps = [{"tool": t.get("tool") or "", "retext": t.get("retext", ""),
                  "params": t.get("params") or {},
                  "hit_source": t.get("hitSource") or t.get("hit_source") or ""}
                 for t in r["tools"]]
    return {
        "row": case["row"], "domain": case["domain"], "domain_cn": case.get("domain_cn", ""),
        "query": q, "latency_ms": latency, "steps": steps, "stop": r["stop"],
        "n_tools": len(r["tools"]), "n_plan": len(r["plans"]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=None, help="只跑前 N 条")
    ap.add_argument("-d", "--domains", default=None, help="逗号分隔域: device,vod,music")
    ap.add_argument("--caseset", default=str(CASESET), help="caseset json 路径(覆盖默认)")
    ap.add_argument("--w", "--workers", dest="workers", type=int, default=8)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--rules", action="store_true",
                    help="打印每条 query 命中 hcTools 的哪个 rule(steps 里的 hitSource)")
    args = ap.parse_args()

    caseset_path = Path(args.caseset)
    if not caseset_path.exists():
        raise SystemExit(f"评测集不存在: {caseset_path}")
    cases = json.loads(caseset_path.read_text(encoding="utf-8"))["records"]
    if args.domains:
        want = {x.strip() for x in args.domains.split(",") if x.strip()}
        cases = [c for c in cases if c["domain"] in want]
    if args.n:
        cases = cases[: args.n]

    def _one(c):
        try:
            r = run_one(c)
            r["status"] = "ok"
        except Exception as exc:
            r = {"row": c["row"], "domain": c["domain"], "domain_cn": c.get("domain_cn", ""),
                 "query": c["query"], "status": "err",
                 "error": f"{type(exc).__name__}: {exc}"}
        return r

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for r in as_completed([pool.submit(_one, c) for c in cases]):
            results.append(r.result())

    # 聚合后每条还真名归到 english 域(与 raw json 一致)
    cn_of = {c["domain"]: c.get("domain_cn") for c in cases}
    by_row = {c["row"]: c for c in cases}
    agg: dict[str, dict] = {d: {"ok_t": 0, "ok_p": 0, "ok_both": 0, "score": 0,
                                "t_miss": 0, "rows": []} for d in DOMAIN_ORDER}
    diffs = []
    errs = []
    for r in results:
        dom = r.get("domain")
        a = agg.get(dom)
        if a is None:
            continue
        if r["status"] != "ok":
            errs.append(r)
            a["rows"].append(r)
            continue
        gt = by_row.get(r["row"])
        if gt is None:
            continue
        et = gt["expected_tool"]
        gp = normalize_golden_params(gt.get("expected_params"))
        pred_tool = ""
        pred_params = {}
        if r.get("steps"):
            s = r["steps"][0]
            pred_tool = s.get("tool", "")
            pred_params = s.get("params") or {}
        if not et or not pred_tool:
            continue
        ok_t = pred_tool == et
        ok_p = (not gp) or params_equal(pred_params, gp)
        a["score"] += 1
        a["ok_t"] += ok_t
        a["ok_p"] += ok_p
        a["ok_both"] += ok_t and ok_p
        if not ok_t:
            a["t_miss"] += 1
            diffs.append({"row": r["row"], "domain": cn_of.get(dom, dom), "query": r["query"],
                          "gold": et, "pred": pred_tool,
                          "gold_params": gp, "pred_params": pred_params})
        if not (ok_t and ok_p):
            a["rows"].append(r)

    tot = {"ok_t": 0, "ok_p": 0, "ok_both": 0, "score": 0}
    for d in DOMAIN_ORDER:
        for k in tot:
            tot[k] += agg[d][k]
    tot["t_miss"] = sum(agg[d]["t_miss"] for d in DOMAIN_ORDER)

    if args.json:
        mc = Counter((d["gold"], d["pred"]) for d in diffs).most_common()
        out = {
            "total": len(cases), "scored": tot["score"], "err": len(errs),
            "tool_acc": round(tot["ok_t"] / tot["score"] * 100, 4) if tot["score"] else 0,
            "param_acc": round(tot["ok_p"] / tot["score"] * 100, 4) if tot["score"] else 0,
            "joint_acc": round(tot["ok_both"] / tot["score"] * 100, 4) if tot["score"] else 0,
            "by_domain": {cn_of.get(d, d): {**{k: agg[d][k] for k in ("ok_t", "ok_p", "ok_both", "score")},
                              "tool_acc": round(agg[d]["ok_t"] / agg[d]["score"] * 100, 4) if agg[d]["score"] else 0,
                              "joint_acc": round(agg[d]["ok_both"] / agg[d]["score"] * 100, 4) if agg[d]["score"] else 0}
                          for d in DOMAIN_ORDER},
            "tool_miss": [{"gold": k[0], "pred": k[1], "n": v} for k, v in mc],
        }
        # 明细写独立文件,避免污染 stdout 统计
        diff_path = TOOLS.parent / "data" / "sse0827_diffs.json"
        diff_path.write_text(json.dumps({"count": len(diffs), "diffs": diffs},
                                        ensure_ascii=False, indent=2), encoding="utf-8")
        out["diff_file"] = str(diff_path)
        # 汇总 JSON 也落盘一份(避免 stdout 被 diff 明细截断,便于机器读取)
        res_path = TOOLS.parent / "data" / "sse0827_result.json"
        res_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    total = tot["score"]
    def pct(a, b):
        return f"{a}/{b} = {a/b*100:.2f}%" if b else "0/0"
    print(f"=== SSE 7域评测 {len(cases)} 条  scored {total}  err {len(errs)} ===")
    print(f"tool        {pct(tot['ok_t'], total)}")
    print(f"param       {pct(tot['ok_p'], total)}")
    print(f"tool+param  {pct(tot['ok_both'], total)}   (主指标,目标≥90%)")
    print("\n=== 按域 ===")
    for dom in DOMAIN_ORDER:
        cn = cn_of.get(dom, dom)
        a = agg[dom]
        print(f"  {cn:6s} tool {pct(a['ok_t'], a['score']):>18s}   "
              f"param {pct(a['ok_p'], a['score']):>18s}   "
              f"joint {pct(a['ok_both'], a['score']):>20s}   tool_miss {a['t_miss']}")
    if diffs:
        mc = Counter((d["gold"], d["pred"]) for d in diffs).most_common(30)
        print("\n=== tool 失配 (gold工具 -> pred工具) ===")
        for (gt, pt), n in mc:
            print(f"  {gt:26s} -> {pt:26s} x{n}")
        print("\n=== tool+param diff 示例 ===")
        for d in diffs[:25]:
            print(f"  [{d['row']}] ({d['domain']}) {d['query']!r}")
            print(f"     gold=({d['gold']}, {json.dumps(d['gold_params'], ensure_ascii=False)})")
            print(f"     pred=({d['pred']}, {json.dumps(d['pred_params'], ensure_ascii=False)})")
    if errs:
        print("\n=== errors ===")
        for e in errs[:20]:
            print(f"  [{e.get('row')}] ({e.get('domain')}) {e.get('query')!r}: {e.get('error')}")

    if args.rules:
        print("\n=== 每条 query 命中规则 (hitSource) ===")
        for r in sorted(results, key=lambda x: (cn_of.get(x.get('domain'), x.get('domain')), x.get('row', 0))):
            if r["status"] != "ok" or not r.get("steps"):
                continue
            src = r["steps"][0].get("hit_source") or ""
            print(f"  [{r.get('row')}] ({cn_of.get(r.get('domain'), r.get('domain'))}) "
                  f"{r.get('query')!r}\n      tool={r['steps'][0].get('tool')}  hitSource={src or '(空:非规则命中,走LLM/兜底)'}")


if __name__ == "__main__":
    main()