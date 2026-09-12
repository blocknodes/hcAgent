"""SSE 真实链路按域输出评测明细 CSV。

对 sheet0827_caseset.json 全量 query 直连远端 runtime(SSE, enable_slow)真跑，
每个域写一个 CSV，每行一条 query，含：
  row, domain, intent, query,
  gold_tool, gold_params, pred_tool, pred_params,
  tool_ok, param_ok, both_ok,
  param_diff   (param 不一致差异摘要；一致/未比为空)

用法:
  python tools/sse_domain_output.py                # 全 7 域
  python tools/sse_domain_output.py -d vod,music   # 指定域
  python tools/sse_domain_output.py --workers 16   # 并发
输出: data/sse_out_<domain>.csv (每域一个) + data/sse_out_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
AGENT = TOOLS.parent
CASESET = AGENT / "data" / "sheet0827_caseset.json"
OUTDIR = AGENT / "data"

DOMAIN_ORDER = ["device", "vod", "audio", "music", "sports", "children", "education"]
CN = {"device": "设备", "vod": "影视", "audio": "有声", "music": "音乐",
      "sports": "体育", "children": "少儿", "education": "教育"}

import sys  # noqa: E402
sys.path.insert(0, str(AGENT))
from tools.sse_7domain_eval import (  # noqa: E402
    run_one, canonical, params_equal, normalize_golden_params,
)


def _param_repr(p):
    return json.dumps(p, ensure_ascii=False, sort_keys=True) if p else ""


def _param_diff(gp, pp):
    """参数差异的人类可读摘要。gp/pp 为规范化前的原参(可能 _raw→None)。"""
    if gp is None:
        return ""  # 不可比 only-tool
    gp_c = canonical(gp)
    pp_c = canonical(pp)
    if params_equal(gp, pp):
        return ""
    if isinstance(gp_c, dict) and isinstance(pp_c, dict):
        keys = sorted(set(list(gp_c.keys()) + list(pp_c.keys())))
        parts = []
        for k in keys:
            gv = gp_c.get(k)
            pv = pp_c.get(k)
            if gv != pv:
                parts.append(f"{k}: gold={_param_repr(gv)} pred={_param_repr(pv)}")
        return "; ".join(parts)
    return f"gold={_param_repr(gp)} pred={_param_repr(pp)}"


def _run(case):
    try:
        r = run_one(case)
        s = r["steps"][0] if r.get("steps") else {}
        return {
            "ok": True, "query": case["query"], "row": case["row"],
            "domain": case.get("domain"),
            "pred_tool": s.get("tool", ""), "pred_params": s.get("params") or {},
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "query": case["query"], "row": case["row"],
                "domain": case.get("domain"), "error": f"{type(exc).__name__}: {exc}"}


HEADER = ["row", "intent", "query", "gold_tool", "gold_params",
          "pred_tool", "pred_params", "tool_ok", "param_ok", "both_ok", "param_diff"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-d", "--domains", default=None)
    ap.add_argument("-w", "--workers", dest="workers", type=int, default=8)
    ap.add_argument("-n", type=int, default=0, help="只跑前 N 条每域(冒烟)")
    args = ap.parse_args()

    records = json.loads(CASESET.read_text(encoding="utf-8"))["records"]
    if args.domains:
        want = {d.strip() for d in args.domains.split(",") if d.strip()}
        records = [c for c in records if c["domain"] in want]
    if args.n:
        seen = {}
        out = []
        for c in records:
            if seen.get(c["domain"], 0) < args.n:
                out.append(c)
                seen[c["domain"]] = seen.get(c["domain"], 0) + 1
        records = out
    by_row = {c["row"]: c for c in records}

    print(f"SSE 评测 {len(records)} 条 (并发 {args.workers}) ...", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(_run, c) for c in records]):
            rows.append(fut.result())

    per_domain: dict[str, list[list]] = defaultdict(list)
    err_count = 0
    for res in rows:
        dom = res.get("domain")
        if dom not in CN:
            continue
        gt = by_row.get(res["row"])
        if not res["ok"]:
            err_count += 1
            per_domain[dom].append([res["row"], gt.get("intent", "") if gt else "", res["query"],
                                    "", "", "", "", "ERR", "ERR", "ERR", res.get("error", "")])
            continue
        et = gt.get("expected_tool") or ""
        gp = normalize_golden_params(gt.get("expected_params"))
        pt = res.get("pred_tool") or ""
        pp = res.get("pred_params") or {}

        ok_t = bool(et) and pt == et
        ok_p = (not gp) or params_equal(pp, gp)
        both = ok_t and ok_p
        diff = "" if ok_p else _param_diff(gp, pp)
        per_domain[dom].append([res["row"], gt.get("intent", ""), res["query"],
                                et, _param_repr(gp), pt, _param_repr(pp),
                                "Y" if ok_t else "N", "Y" if ok_p else "N",
                                "Y" if both else "N", diff])

    print("\n=== SSE 按域 joint ===")
    summary_rows = []
    for dom in DOMAIN_ORDER:
        d = per_domain[dom]
        total = len(d)
        ok_t = sum(1 for r in d if r[7] == "Y")
        ok_p = sum(1 for r in d if r[8] == "Y")
        ok_both = sum(1 for r in d if r[9] == "Y")

        def pct(a):
            return f"{a}/{total} = {a/total*100:.2f}%" if total else "0/0"

        print(f"  {CN.get(dom, dom):6s} tool {pct(ok_t):>20}  param {pct(ok_p):>20}  "
              f"joint {pct(ok_both):>22}")
        summary_rows.append([dom, total, ok_t, ok_p, ok_both,
                             round(ok_t / total * 100, 2) if total else 0,
                             round(ok_p / total * 100, 2) if total else 0,
                             round(ok_both / total * 100, 2) if total else 0])
        path = OUTDIR / f"sse_out_{dom}.csv"
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["域", "row", "intent", "query", "gold_tool",
                        "gold_params", "pred_tool", "pred_params",
                        "tool_ok", "param_ok", "both_ok", "param_diff"])
            w.writerows([[CN.get(dom, dom)] + r for r in d])

    if err_count:
        print(f"\n注意: {err_count} 条 SSE 调用报错, 已标 ERR 写入对应域 CSV。")

    sp = OUTDIR / "sse_out_summary.csv"
    with open(sp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["domain_cn", "total", "tool_ok", "param_ok", "both_ok",
                    "tool_acc%", "param_acc%", "joint_acc%"])
        w.writerows(summary_rows)
    print(f"\n每域明细 → {OUTDIR}/sse_out_<domain>.csv")
    print(f"汇总    → {sp}")


if __name__ == "__main__":
    main()