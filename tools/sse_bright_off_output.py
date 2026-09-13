"""SSE 亮屏/息屏场景按域评测明细 CSV。

读飞书《息亮屏》tab 生成的 sheet0821_caseset.json(99 条,x: query + bright_tool/params + off_tool/params),
对每条 query 直连远端 runtime(SSE) 分别以 tvMode=0(亮屏) / tvMode=6(息屏) 真跑,
每域写一个 CSV,亮屏/息屏两列并列,每行含:
  row, domain, intent, query,
  gold_bright, gold_bright_params, pred_bright, pred_bright_params, bright_tool_ok, bright_param_ok, bright_both_ok,
  gold_off, gold_off_params, pred_off, pred_off_params, off_tool_ok, off_param_ok, off_both_ok,
  bright_param_diff, off_param_diff, err

用法:
  python tools/sse_bright_off_output.py                 # 亮+息屏全量
  python tools/sse_bright_off_output.py --workers 16    # 并发
  python tools/sse_bright_off_output.py --col bright    # 只亮屏
输出: data/sse_bright_off_<domain>.csv + data/sse_bright_off_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
AGENT = TOOLS.parent
CASESET = AGENT / "data" / "sheet0821_caseset.json"
OUTDIR = AGENT / "data"

DOMAIN_ORDER = ["device", "vod", "audio", "music", "children", "education", "sports"]
CN = {"device": "设备", "vod": "影视", "audio": "有声", "music": "音乐",
      "sports": "体育", "children": "少儿", "education": "教育"}

sys.path.insert(0, str(AGENT))
from tools.sse_7domain_eval import canonical, params_equal  # noqa: E402
from tools.sse_caseset_eval import run_one, load_golden_params  # noqa: E402


def _param_repr(p):
    try:
        return json.dumps(p, ensure_ascii=False, sort_keys=True) if p else ""
    except Exception:
        return str(p)


def _param_diff(gp, pp):
    if not gp:
        return ""
    gp_c = canonical(gp)
    pp_c = canonical(pp)
    if params_equal(gp, pp):
        return ""
    if isinstance(gp_c, dict) and isinstance(pp_c, dict):
        keys = sorted(set(list(gp_c.keys()) + list(pp_c.keys())))
        parts = []
        for k in keys:
            if gp_c.get(k) != pp_c.get(k):
                parts.append(f"{k}: gold={_param_repr(gp_c.get(k))} pred={_param_repr(pp_c.get(k))}")
        return "; ".join(parts)
    return f"gold={_param_repr(gp)} pred={_param_repr(pp)}"


def _run(case, col):
    tv = "0" if col == "bright" else "6"
    try:
        r = run_one(case, tv_mode=tv)
        s = r["steps"][0] if r.get("steps") else {}
        return {"ok": True, "tool": s.get("tool", ""), "params": s.get("params") or {}}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--col", choices=["bright", "off", "both"], default="both")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cases = json.loads(CASESET.read_text(encoding="utf-8"))
    cols = ["bright", "off"] if args.col == "both" else [args.col]

    print(f"SSE 亮/息屏评测 {len(cases)} 条, cols={cols}, 并发 {args.workers}", flush=True)

    # run each case under needed cols
    def _work(case):
        out = {"case": case}
        for col in cols:
            out[col] = _run(case, col)
        return out

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(_work, c) for c in cases]):
            results.append(fut.result())

    per_domain = defaultdict(list)
    summary = {}
    for res in results:
        c = res["case"]
        dom = c.get("domain")
        if dom not in CN:
            dom = c.get("domain")
        if dom not in CN:
            continue
        row = [c["row"], dom, c.get("intent", ""), c["query"]]
        for col in cols:
            tk, pk = ("bright_tool", "bright_params") if col == "bright" else ("off_tool", "off_params")
            gt = (c.get(tk) or "").strip()
            gp = load_golden_params(c.get(pk))
            r = res[col]
            # each block is exactly 9 fields: [gold_tool, gold_params, pred_tool,
            # pred_params, tool_ok, param_ok, both_ok, param_diff, err]
            if not r["ok"]:
                row += [gt, _param_repr(gp) if gp else "", "ERR", "ERR", "ERR", "ERR", "ERR", "", r.get("error", "")]
                continue
            pt = r["tool"]
            pp = r["params"] or {}
            ok_t = bool(gt) and pt == gt
            ok_p = (not gp) or params_equal(pp, gp)
            both = ok_t and ok_p
            row += [gt, _param_repr(gp) if gp else "", pt, _param_repr(pp),
                    "Y" if ok_t else "N", "Y" if ok_p else "N", "Y" if both else "N",
                    _param_diff(gp, pp) if not ok_p else "", ""]
        per_domain[dom].append(row)

    # metrics (both rows: bright idx 8,9,10 ; off idx 17,18,19)
    summary = defaultdict(lambda: {"n": 0, "b_t": 0, "b_p": 0, "b_b": 0, "o_t": 0, "o_p": 0, "o_b": 0})
    for dom, rows in per_domain.items():
        for r in rows:
            if len(r) < 12:
                continue
            m = summary[dom]
            m["n"] += 1
            if r[8] == "Y": m["b_t"] += 1
            if r[9] == "Y": m["b_p"] += 1
            if r[10] == "Y": m["b_b"] += 1
            if len(r) >= 20:
                if r[17] == "Y": m["o_t"] += 1
                if r[18] == "Y": m["o_p"] += 1
                if r[19] == "Y": m["o_b"] += 1

    def pct(a, b):
        return f"{a}/{b} = {a/b*100:.2f}%" if b else "0/0"

    print("\n=== SSE 亮息屏 按域 joint ===")
    for dom in DOMAIN_ORDER:
        m = summary[dom]
        if not m["n"]:
            continue
        print(f"  {CN.get(dom,dom):6s} 亮屏 tool {pct(m['b_t'],m['n']):>22} joint {pct(m['b_b'],m['n']):>22}"
              f" | 息屏 tool {pct(m['o_t'],m['n']):>22} joint {pct(m['o_b'],m['n']):>22}")

    # write per-domain CSV
    if args.col == "bright":
        header = ["row", "domain", "intent", "query", "gold_tool", "gold_params",
                  "pred_tool", "pred_params", "tool_ok", "param_ok", "both_ok", "param_diff", "err"]
    elif args.col == "off":
        header = ["row", "domain", "intent", "query", "gold_tool", "gold_params",
                  "pred_tool", "pred_params", "tool_ok", "param_ok", "both_ok", "param_diff", "err"]
    else:
        header = ["row", "domain", "intent", "query",
                  "gold_bright_tool", "gold_bright_params", "pred_bright_tool", "pred_bright_params",
                  "bright_tool_ok", "bright_param_ok", "bright_both_ok", "bright_param_diff",
                  "b_err",
                  "gold_off_tool", "gold_off_params", "pred_off_tool", "pred_off_params",
                  "off_tool_ok", "off_param_ok", "off_both_ok", "off_param_diff", "o_err"]

    for dom in DOMAIN_ORDER:
        rows = per_domain.get(dom, [])
        if not rows:
            continue
        path = OUTDIR / f"sse_bright_off_{dom}.csv"
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    # summary csv
    sp = OUTDIR / "sse_bright_off_summary.csv"
    with open(sp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["domain", "n", "bright_tool_acc%", "bright_joint%", "off_tool_acc%", "off_joint%"])
        for dom in DOMAIN_ORDER:
            m = summary[dom]
            if m["n"]:
                w.writerow([CN.get(dom, dom), m["n"],
                            round(m["b_t"]/m["n"]*100, 2), round(m["b_b"]/m["n"]*100, 2),
                            round(m["o_t"]/m["n"]*100, 2), round(m["o_b"]/m["n"]*100, 2)])
    print(f"\n每域明细 → {OUTDIR}/sse_bright_off_<domain>.csv")
    print(f"汇总    → {sp}")


if __name__ == "__main__":
    main()