"""SSE 亮屏/息屏双形态评测：读飞书「0821多意图&多业务」息亮屏子表整理后的 csv。

每条 query 有**两套期望**(亮屏期望 + 息屏期望)，链路分别以 tvMode=0(亮屏)/tvMode=6(息屏)
真跑两条 SSE，各取计划帧真工具名 + 参数，与对应 gold 比对。指标：
  bright_tool / bright_both    亮屏 工具 与 工具+参数
  off_tool / off_both          息屏 工具 与 工具+参数

参数比对口径与 run_sse_multiintent 一致：**retext 字段完全不看**、figures 冗余忽略、
query 检索词子串宽容、其余结构化字段严格对齐。

输入: benchmark/cases/sheet0821_brightoff.csv
列:   业务域,query,意图,亮屏期望工具,亮屏期望参数,息屏期望工具,息屏期望参数,源行号
用法:
  python benchmark/run_sse_brightoff.py                # 亮+息屏全量
  python benchmark/run_sse_brightoff.py -w 16         # 并发(默认 8)
  python benchmark/run_sse_brightoff.py -n 10         # 只跑前 10 条(冒烟)
输出:
  benchmark/output/detail_brightoff.csv   # 逐条明细(亮/息并列)
  benchmark/output/summary_brightoff.csv  # 按业务 + 全局汇总
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BENCH = Path(__file__).resolve().parent
AGENT = BENCH.parent
OUTDIR = BENCH / "output"
_CSV = BENCH / "cases" / "sheet0821_brightoff.csv"

sys.path.insert(0, str(AGENT))
from tools.sse_7domain_eval import run_one, canonical, params_equal  # noqa: E402

DOMAIN_ORDER = ["影视", "少儿", "教育", "有声", "音乐", "设备控制"]


def _params(v):
    if not v or not v.strip():
        return {}
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return {"_raw": str(v)}


def load_cases():
    with open(_CSV, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    out = []
    for i, r in enumerate(rows[1:], start=1):
        if len(r) < 7:
            continue
        biz = (r[0] or "").strip()
        q = (r[1] or "").strip()
        intent = (r[2] or "").strip()
        bt = (r[3] or "").strip()
        bp = r[4] if len(r) > 4 else ""
        ot = (r[5] or "").strip()
        offp = r[6] if len(r) > 6 else ""
        src = r[7] if len(r) > 7 else str(i + 1)
        if not q:
            continue
        out.append({
            "row": src, "biz": biz, "intent": intent, "query": q,
            "bright_tool": bt, "bright_params": _params(bp),
            "off_tool": ot, "off_params": _params(offp),
        })
    return out


# ---- 参数比对口径：retext 不算标准，figures 冗余忽略，query 子串宽容 ----
_REDUNDANT_FIELDS = {"figures"}


def canonical_copy(x):
    if isinstance(x, dict):
        return {k: canonical_copy(v) for k, v in x.items()
                if not (v is None or (isinstance(v, str) and not v.strip()))}
    if isinstance(x, list):
        return sorted((canonical_copy(i) for i in x), key=lambda e: json.dumps(e, ensure_ascii=False, sort_keys=True))
    return x


def _drop_redundant_list(items):
    out = []
    for it in items:
        if isinstance(it, dict):
            f = it.get("field") or it.get("k") or it.get("field_name")
            if f in _REDUNDANT_FIELDS or f == "retext":
                continue
            if any(k in _REDUNDANT_FIELDS or k == "retext" for k in it.keys()):
                if (set(it.keys()) & (_REDUNDANT_FIELDS | {"retext"})) == set(it.keys()) \
                   or it.get("field") in _REDUNDANT_FIELDS:
                    continue
        if isinstance(it, list):
            it = _drop_redundant_list(it)
        out.append(it)
    return out


# fuzzy 检索工具：只有 query 自然语言重写参数（LLM 生成、随轮次变化），
# 不作为判定标准 —— 只有 tool 对不对，param 一律不作数。
_PARAM_WAIVED_TOOLS = {"vod_fuzzy_search", "educ_fuzzy_search", "edu_fuzzy_search", "edu_slow_search_data_search"}

# 工具等价别名：新名 fuzzy_search 与旧名 slow_search_data_search / slow_data_search 是同一工具（仅改名）。
# 归一化到新名，避免 gold 用旧名、runtime 回退新名（或反之）时 tool 误判不匹配。
_TOOL_ALIAS = {
    "educ_fuzzy_search": "educ_fuzzy_search",
    "educ_slow_search_data_search": "educ_fuzzy_search",
    "educ_slow_data_search": "educ_fuzzy_search",
    "edu_fuzzy_search": "edu_fuzzy_search",
    "edu_slow_search_data_search": "edu_fuzzy_search",
    "edu_slow_data_search": "edu_fuzzy_search",
}


def _norm_tool(name):
    """把工具名归一到规范名：命中别名表则用新名，否则原样返回。"""
    n = (name or "").strip()
    return _TOOL_ALIAS.get(n, n)


def _params_ok(gold, pred, depth=0):
    """参数校验：retext 半分不差；figures 冗余字段忽略；query 字符串子串宽容；其余严格。"""
    if isinstance(gold, dict) and isinstance(pred, dict):
        g = {k: v for k, v in canonical_copy(gold).items()
             if k not in _REDUNDANT_FIELDS and k != "retext"}
        p = {k: v for k, v in canonical_copy(pred).items()
             if k not in _REDUNDANT_FIELDS and k != "retext"}
        for k in set(g) | set(p):
            gv = g.get(k)
            pv = p.get(k)
            if gv is None or pv is None:
                continue
            if k == "query" and isinstance(gv, str) and isinstance(pv, str):
                gs, ps = (gv or "").strip(), (pv or "").strip()
                if gs and ps and (gs in ps or ps in gs):
                    continue
                return False
            if isinstance(gv, list) and isinstance(pv, list):
                gv2 = _drop_redundant_list(gv)
                pv2 = _drop_redundant_list(pv)
                if len(gv2) != len(pv2):
                    return False
                for a, b in zip(gv2, pv2):
                    if not _params_ok(a, b, depth + 1):
                        return False
                continue
            if isinstance(gv, dict) and isinstance(pv, dict):
                if not _params_ok(gv, pv, depth + 1):
                    return False
                continue
            if params_equal(gv, pv) is False:
                return False
        return True
    return params_equal(gold, pred)


def _param_repr(p):
    try:
        return json.dumps(p, ensure_ascii=False, sort_keys=True) if p else ""
    except Exception:
        return str(p)


def _param_diff(gt, gp, pt, pp):
    if not gp:
        return ""
    gpc, ppc = canonical(gp), canonical(pp)
    if params_equal(gp, pp):
        return ""
    if isinstance(gpc, dict) and isinstance(ppc, dict):
        parts = [f"{k}: gold={_param_repr(gpc.get(k))} pred={_param_repr(ppc.get(k))}"
                 for k in sorted(set(gpc) | set(ppc)) if gpc.get(k) != ppc.get(k)]
        return "; ".join(parts)
    if gt != pt:
        return f"tool: gold={gt} pred={pt}"
    return f"gold={_param_repr(gp)} pred={_param_repr(pp)}"


def _run_case(case, tv_mode):
    started = time.perf_counter()
    r = run_one({"query": case["query"], "row": case["row"], "domain": case["biz"]}, tv_mode=tv_mode)
    steps = r.get("steps") or []
    s = steps[0] if steps else {}
    return {"ok": True, "tool": s.get("tool", ""), "params": s.get("params") or {},
            "latency": round((time.perf_counter() - started) * 1000, 1)}


def _work(case, cols):
    out = {"case": case}
    for col in cols:
        tv = "0" if col == "bright" else "6"
        try:
            out[col] = _run_case(case, tv)
        except Exception as exc:  # noqa: BLE001
            out[col] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return out


DETAIL_HEADER = ["源行", "业务域", "query", "意图",
                 "亮gold", "亮gold参", "亮pred", "亮pred参", "亮tool", "亮param", "亮both", "亮diff",
                 "息gold", "息gold参", "息pred", "息pred参", "息tool", "息param", "息both", "息diff", "err"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-w", "--workers", type=int, default=8)
    ap.add_argument("-n", type=int, default=0, help="只跑前 N 条(冒烟)")
    ap.add_argument("--col", choices=["bright", "off", "both"], default="both")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cases = load_cases()
    if args.n:
        cases = cases[: args.n]
    cols = ["bright", "off"] if args.col == "both" else [args.col]
    active = set(cols)

    print(f"SSE 亮/息屏评测 {len(cases)} 条, cols={cols}, 并发 {args.workers}", flush=True)

    details = []
    fp_err = 0
    done = 0
    agg_bright = defaultdict(lambda: [0, 0, 0])  # biz -> [n, tool, both]
    agg_off = defaultdict(lambda: [0, 0, 0])

    def fold(res):
        nonlocal fp_err, done
        c = res["case"]
        biz = c["biz"]
        row = [c["row"], biz, c["query"], c["intent"]]
        blocks = []
        for col in ("bright", "off"):
            if col not in active:
                blocks.append(["", "", "", "", "", "", "", ""])
                continue
            r = res.get(col) or {}
            if not r.get("ok"):
                fp_err += 1
                blocks.append(["E", "", "", "", "E", "E", "E", r.get("error", "")])
                continue
            if col == "bright":
                gt, gp = (c.get("bright_tool") or ""), c.get("bright_params") or {}
                agg = agg_bright
            else:
                gt, gp = (c.get("off_tool") or ""), c.get("off_params") or {}
                agg = agg_off
            if isinstance(gp, dict) and set(gp.keys()) == {"_raw"}:
                gp = None
            pt, pp = r["tool"], r["params"] or {}
            ok_t = bool(gt) and bool(pt) and _norm_tool(pt) == _norm_tool(gt)
            # fuzzy 检索类工具只看 tool 是否命中；参数(retext 等 LLM 重写)不作为评判标准。
            ok_p = True if (_norm_tool(gt) in _PARAM_WAIVED_TOOLS) else ((not gp) or _params_ok(pp, gp))
            ob = ok_t and ok_p
            agg[biz][0] += 1
            agg[biz][1] += int(ok_t)
            agg[biz][2] += int(ob)
            diff = "" if ok_p else _param_diff(gt, gp, pt, pp)
            blocks.append([gt, _param_repr(gp), pt, _param_repr(pp),
                           "Y" if ok_t else "N", "Y" if ok_p else "N", "Y" if ob else "N", diff])
        # 对齐列位: bright block(8) + off block(8) + err(1) => 4+16+1=21
        row += blocks[0] + blocks[1] + [""]
        details.append(row)
        done += 1
        print(f"\r[{done}/{len(cases)}] err={fp_err}", end="", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for fut in as_completed([pool.submit(_work, c, cols) for c in cases]):
            fold(fut.result())
    print()

    bN = sum(v[0] for v in agg_bright.values())
    bT = sum(v[1] for v in agg_bright.values())
    bB = sum(v[2] for v in agg_bright.values())
    oN = sum(v[0] for v in agg_off.values())
    oT = sum(v[1] for v in agg_off.values())
    oB = sum(v[2] for v in agg_off.values())

    def pct(x, n):
        return f"{x/n*100:.1f}" if n else "0"

    summary_rows = [["域", "n", "亮tool%", "亮both%", "息tool%", "息both%"]]
    for dom in DOMAIN_ORDER:
        bn, bt, bb = agg_bright.get(dom, [0, 0, 0])
        on_, ot_, ob_ = agg_off.get(dom, [0, 0, 0])
        if not (bn or on_):
            continue
        summary_rows.append([dom, max(bn, on_),
                             pct(bt, bn), pct(bb, bn), pct(ot_, on_), pct(ob_, on_)])
    summary_rows.append(["全局", len(cases), pct(bT, bN), pct(bB, bN), pct(oT, oN), pct(oB, oN)])

    OUTDIR.mkdir(parents=True, exist_ok=True)
    with open(OUTDIR / "detail_brightoff.csv", "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow(DETAIL_HEADER)
        csv.writer(f).writerows(details)
    with open(OUTDIR / "summary_brightoff.csv", "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(summary_rows)

    if args.json:
        print(json.dumps({
            "turns": len(cases), "errors": fp_err,
            "bright": {"n": bN, "tool_ok": bT, "both_ok": bB,
                       "tool_acc": round(bT / bN * 100, 1) if bN else 0,
                       "both_acc": round(bB / bN * 100, 1) if bN else 0},
            "off": {"n": oN, "tool_ok": oT, "both_ok": oB,
                    "tool_acc": round(oT / oN * 100, 1) if oN else 0,
                    "both_acc": round(oB / oN * 100, 1) if oN else 0},
            "by_biz": {d: {"n": max(agg_bright.get(d, [0, 0, 0])[0], agg_off.get(d, [0, 0, 0])[0]),
                           "bright_both": agg_bright.get(d, [0, 0, 0])[2],
                           "off_both": agg_off.get(d, [0, 0, 0])[2]} for d in DOMAIN_ORDER},
        }, ensure_ascii=False, indent=2))
        return

    print(f"[亮屏] joint {pct(bB, bN)}%  (tool {pct(bT, bN)}%)")
    print(f"[息屏] joint {pct(oB, oN)}%  (tool {pct(oT, oN)}%)")
    print(f"\n逐条明细 -> {OUTDIR}/detail_brightoff.csv")
    print(f"汇总     -> {OUTDIR}/summary_brightoff.csv")


if __name__ == "__main__":
    main()