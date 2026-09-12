"""SSE 真实链路评测：跑飞书《0821多意图&多业务》sheet 的用例。

对每条用例(query)直连远端 runtime(SSE, enable_slow) 真跑，从 DEBUG_MAP 计划
帧取"数据集命名"的真工具名 + 参数，与 sheet golden(默认 亮屏 列)比对。

device_id 约定：
  - 单条用例(每条 sheet 行独立) → device_id=random（避免会话/多样性复用）
  - 同组多步 → 同 device_id（本 sheet 每行即一条独立用例，暂全单步）

指标：tool 准确率、tool+param(joint, order-insensitive canonical)。

用法：
  python tools/sse_caseset_eval.py                 # 全量
  python tools/sse_caseset_eval.py -n 5            # 前 5 条
  python tools/sse_caseset_eval.py --row 11,30,52  # 指定行
  python tools/sse_caseset_eval.py --json          # JSON 输出
  python tools/sse_caseset_eval.py --col off       # 用息屏列比对
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

TOOLS = Path(__file__).resolve().parent                 # hcAgent/tools
AGENT = TOOLS.parent                                  # hcAgent
CASESET = AGENT / "data" / "sheet0821_caseset.json"

sys.path.insert(0, str(AGENT))
from runtime_execute import (                         # noqa: E402
    DEFAULT_FEATURE_CODE,
    _run_runtime,
    random_device_id,
)


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


def _empty(v):
    if v is None:
        return True
    if isinstance(v, str):
        return not v.strip()
    if isinstance(v, (list, dict)):
        return len(v) == 0
    return False


def params_equal(a, b):
    return canonical(a) == canonical(b)


def load_golden_params(raw):
    raw = (raw or "").strip()
    if not raw:
        return {}
    # 去掉 golden 单元格里可能带的前导说明文案（如 "(当前息屏进入影视智能体，如果进入有声参数如下)\n{...}"），
    # 只保留其中的 JSON 对象；这类注解会让整体 JSON 解析失败而误判为参数不匹配。
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        raw = m.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = re.sub(r"[\r\n]", " ", raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"_raw": raw}


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
         "params": s.get("parameters") or {}}
        for p in r["plans"] for s in (p.get("steps") or [])
        if isinstance(s, dict) and s.get("toolName")
    ]
    if not steps:
        steps = [{"tool": t.get("tool") or "", "retext": t.get("retext", ""),
                  "params": t.get("params") or {}} for t in r["tools"]]
    return {
        "row": case["row"], "domain": case["domain"], "query": q,
        "latency_ms": latency, "steps": steps, "stop": r["stop"],
        "n_tools": len(r["tools"]), "n_plan": len(r["plans"]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=None)
    ap.add_argument("--row", default=None)
    ap.add_argument("--col", choices=["bright", "off"], default="bright")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not CASESET.exists():
        raise SystemExit(f"评测集不存在：{CASESET}")
    cases = json.loads(CASESET.read_text(encoding="utf-8"))
    if args.row:
        want = {str(x).strip() for x in args.row.split(",") if x.strip()}
        cases = [c for c in cases if str(c["row"]).strip() in want]
    if args.n:
        cases = cases[: args.n]

    # golden 列选择 + 对应真实屏态：亮屏 tvMode=0，息屏 tvMode=6（见 docs/runtime_execute_upstream.md）
    bright = args.col == "bright"
    tk, pk = ("bright_tool", "bright_params") if bright else ("off_tool", "off_params")
    tv_mode = "0" if bright else "6"

    def _one(c, mode):
        try:
            r = run_one(c, tv_mode=mode)
            r["status"] = "ok"
        except Exception as exc:
            r = {"row": c["row"], "domain": c["domain"], "query": c["query"],
                 "status": "err", "error": f"{type(exc).__name__}: {exc}"}
        return r

    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for r in as_completed(
            [pool.submit(_one, c, tv_mode) for c in cases]
        ):
            results.append(r.result())

    by_row = {c["row"]: c for c in cases}
    tool_hit = param_hit = joint_hit = scorable = 0
    diff_table = []
    errs = []
    tool_miss = Counter()
    for r in results:
        if r["status"] != "ok":
            errs.append(r)
            continue
        case = by_row.get(r["row"])
        if case is None:
            continue
        gt = (case.get(tk) or "").strip()
        gp = load_golden_params(case.get(pk))
        pred_tool = ""
        pred_params = {}
        if r.get("steps"):
            s = r["steps"][0]
            pred_tool = s.get("tool", "")
            pred_params = s.get("params") or {}
        if not gt or not pred_tool:
            r["status"] = "unscored"
            continue
        ok_t = pred_tool == gt
        # golden 参数为空(预期无参数)时，参数不参与比对，只要工具对即 pass
        ok_p = (not gp) or params_equal(pred_params, gp)
        scorable += 1
        tool_hit += ok_t
        param_hit += ok_p
        joint_hit += ok_t and ok_p
        if not (ok_t and ok_p):
            tool_miss[(case.get("domain"), gt, pred_tool)] += 1
            diff_table.append({
                "row": r["row"], "domain": case.get("domain"), "query": case.get("query"),
                "gold_tool": gt, "pred_tool": pred_tool,
                "gold_params": gp, "pred_params": pred_params,
                "tool_ok": ok_t, "param_ok": ok_p,
            })

    tool_acc = tool_hit / scorable * 100 if scorable else 0
    param_acc = param_hit / scorable * 100 if scorable else 0
    joint_acc = joint_hit / scorable * 100 if scorable else 0

    if args.json:
        out = {
            "col": args.col, "total": len(cases), "scorable": scorable,
            "tool_hit": tool_hit, "param_hit": param_hit, "joint_hit": joint_hit,
            "tool_acc": round(tool_acc, 4), "param_acc": round(param_acc, 4),
            "joint_acc": round(joint_acc, 4),
            "err_count": len(errs), "diffs": diff_table,
            "tool_miss": [{"domain": k[0], "gold": k[1], "pred": k[2], "n": v}
                          for k, v in tool_miss.most_common()],
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    print(f"=== {len(cases)} 条用例  col={args.col} ===")
    print(f"scorable {scorable}/{len(cases)}  err {len(errs)}")
    print(f"tool        {tool_hit}/{scorable} = {tool_acc:.2f}%")
    print(f"param       {param_hit}/{scorable} = {param_acc:.2f}%")
    print(f"tool+param  {joint_hit}/{scorable} = {joint_acc:.2f}%  (tool 且 param 正确)")
    if tool_miss:
        print("\n=== tool 失配 (gold域 → golden工具 vs pred工具) ===")
        for (d, gt, pt), n in tool_miss.most_common(20):
            print(f"  {d:8s} {gt:22s} -> {pt:22s} x{n}")
    if diff_table:
        print("\n=== tool+param diff ===")
        for d in diff_table[:40]:
            print(f"- [row{d['row']}] ({d['domain']}) {d['query']!r}")
            print(f"    gold=({d['gold_tool']}, {json.dumps(d['gold_params'], ensure_ascii=False)})")
            print(f"    pred=({d['pred_tool']}, {json.dumps(d['pred_params'], ensure_ascii=False)})")
    if errs:
        print("\n=== errors ===")
        for e in errs:
            print(f"  [row{e['row']}] {e.get('query')!r}: {e.get('error')}")


if __name__ == "__main__":
    main()