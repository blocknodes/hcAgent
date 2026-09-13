"""SSE 真实链路评测：输入 7 域 CSV → 逐条跑远端 runtime → 输出逐条明细 + 汇总。

与 tools/sse_domain_output.py 的区别：
  - 输入是 benchmark/cases/sheet0901_<domain>.csv 这 7 个域的 7 个 csv(保留“是否合理”列)，
    而非聚合 caseset json；
  - 输出固定落在 benchmark/output/，明细逐域 + 汇总 csv + 汇总 json；
  - 支持 --ok-only 只评测标注为“合理/空”的用例，并在明细里保留“是否合理”供核对。

SSE 链路、工具/参数比对口径与 tools/sse_7domain_eval.py 完全一致(复用其 run_one 等)。

用法:
  python benchmark/run_sse_bench.py                 # 跑全部 7 域，全部标注用例
  python benchmark/run_sse_bench.py --ok-only       # 只跑合理/未标注用例
  python benchmark/run_sse_bench.py -d vod,music    # 只跑指定域
  python benchmark/run_sse_bench.py -n 10           # 每域swer前 N 条(冒烟)
  python benchmark/run_sse_bench.py --workers 16    # 并发数(默认 8)
输出:
  benchmark/output/detail_<domain>.csv   # 每域逐条明细(含 是否合理/tool_ok/param_ok/both_ok/param_diff)
  benchmark/output/summary.csv           # 按域 + 总体 tool/param/joint 汇总
  benchmark/output/summary.json          # 机器可读汇总
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BENCH = Path(__file__).resolve().parent          # hcAgent/benchmark
AGENT = BENCH.parent                              # hcAgent
TOOLS = AGENT / "tools"
CASEDIR = BENCH / "cases"
OUTDIR = BENCH / "output"

DOMAIN_ORDER = ["device", "vod", "audio", "music", "sports", "children", "education"]
CN = {"device": "设备", "vod": "影视", "audio": "有声", "music": "音乐",
      "sports": "体育", "children": "少儿", "education": "教育"}

sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(AGENT))
from build_sheet0901_cases import parse_params                       # noqa: E402
from tools.sse_7domain_eval import (                                 # noqa: E402
    run_one, canonical, params_equal, normalize_golden_params,
)


def read_csv_records(domain: str):
    """读 benchmark/cases/sheet0901_<domain>.csv → records。

    与 build_sheet0901_cases.build 同口径，但额外保留“是否合理”(第6列,索引5)。
    各域前 6 列位置一致: 业务域, query, 意图(线上意图), 期望工具, 期望参数, 是否合理。
    空 query/空期望工具 直接跳过。
    """
    path = CASEDIR / f"sheet0901_{domain}.csv"
    records = []
    if not path.exists():
        print(f"!! 缺测csv: {path}")
        return records
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    for i, r in enumerate(rows[1:], start=2):
        if len(r) < 4:
            continue
        q = (r[1] or "").strip()
        et = (r[3] or "").strip()
        if not q or not et:
            continue
        reason = ""
        if len(r) > 5:
            reason = (r[5] or "").strip()
        records.append({
            "row": i, "domain": domain, "domain_cn": CN.get(domain, domain),
            "query": q,
            "intent": (r[2] or "").strip() if len(r) > 2 else "",
            "expected_tool": et,
            "expected_params": parse_params(r[4] if len(r) > 4 else ""),
            "reasonable": reason,      # 空=合理未标; 不合理/存疑/合理 = 人工标注
        })
    return records


def _param_repr(p):
    return json.dumps(p, ensure_ascii=False, sort_keys=True) if p else ""


def _param_diff(gp, pp):
    """参数差异的人类可读摘要；gp=None 表示 only-tool 不可比 → “”。"""
    if gp is None:
        return ""
    if params_equal(gp, pp):
        return ""
    gp_c = canonical(gp)
    pp_c = canonical(pp)
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
        return {"ok": True, "case": case, "pred_tool": s.get("tool", ""),
                "pred_params": s.get("params") or {}}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "case": case, "error": f"{type(exc).__name__}: {exc}"}


# 逐条明细列
DETAIL_HEADER = ["域", "row", "query", "intent", "是否合理",
                 "gold_tool", "gold_params", "pred_tool", "pred_params",
                 "tool_ok", "param_ok", "both_ok", "param_diff"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-d", "--domains", default=None, help="逗号分隔域: device,vod,music")
    ap.add_argument("-w", "--workers", dest="workers", type=int, default=8)
    ap.add_argument("-n", type=int, default=0, help="每域只跑前 N 条(冒烟)")
    ap.add_argument("--ok-only", action="store_true",
                    help="只评测“是否合理”列为空或=合理 的用例，剔除 不合理/存疑")
    args = ap.parse_args()

    wants = {d.strip() for d in args.domains.split(",") if d.strip()} if args.domains \
        else set(DOMAIN_ORDER)
    domains = [d for d in DOMAIN_ORDER if d in wants]

    all_records: list[dict] = []
    for dom in domains:
        recs = read_csv_records(dom)
        if args.ok_only:
            before = len(recs)
            recs = [c for c in recs if c["reasonable"] in ("", "合理")]
            dropped = before - len(recs)
            if dropped:
                print(f"  {dom}: 剔除 {dropped} 条标记非合理({before}->{len(recs)})")
        if args.n:
            recs = recs[: args.n]
        all_records.extend(recs)

    print(f"SSE 评测 {len(all_records)} 条 (域={'/'.join(domains)}, 并发 {args.workers}) ...",
          flush=True)

    per_domain: dict[str, list[list]] = defaultdict(list)
    err_count = 0
    done = 0
    ok_t = 0
    ok_both = 0

    def _fold(res):
        """单条结果归一：计入 per_domain，返回 (ok_t, ok_both)；错误返回 None。"""
        case = res["case"]
        dom = case["domain"]
        if not res["ok"]:
            per_domain[dom].append([CN.get(dom, dom), case["row"], case["query"],
                                    case["intent"], case["reasonable"],
                                    "", "", "", "", "ERR", "ERR", "ERR", res.get("error", "")])
            return None
        et = case.get("expected_tool") or ""
        gp = normalize_golden_params(case.get("expected_params"))
        pt = res.get("pred_tool") or ""
        pp = res.get("pred_params") or {}
        ot = bool(et) and pt == et
        op = (not gp) or params_equal(pp, gp)
        ob = ot and op
        diff = "" if op else _param_diff(gp, pp)
        per_domain[dom].append([CN.get(dom, dom), case["row"], case["query"],
                                case["intent"], case["reasonable"],
                                et, _param_repr(gp), pt, _param_repr(pp),
                                "Y" if ot else "N", "Y" if op else "N",
                                "Y" if ob else "N", diff])
        return (ot, ob)

    total = len(all_records)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_run, c) for c in all_records]
        for fut in as_completed(futs):
            res = fut.result()
            done += 1
            r = _fold(res)
            if r is None:
                err_count += 1
            else:
                ok_t += r[0]
                ok_both += r[1]
            # 实时刷新：每完成一条重算 tool / tool+param 准确率
            print(f"\r[{done}/{total}] tool_acc={ok_t/done*100:5.1f}%"
                  f"   tool+param_acc={ok_both/done*100:5.1f}%   err={err_count}",
                  end="", flush=True)
    print()

    # 写每域明细
    for dom in DOMAIN_ORDER:
        if dom not in per_domain or not per_domain[dom]:
            continue
        path = OUTDIR / f"detail_{dom}.csv"
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(DETAIL_HEADER)
            w.writerows(per_domain[dom])

    # 汇总
    print("\n=== SSE 按域 joint ===")
    summary_rows = []
    totals = {"total": 0, "ok_t": 0, "ok_p": 0, "ok_both": 0}
    for dom in DOMAIN_ORDER:
        d = per_domain.get(dom)
        if not d:
            continue

        def _n(idx):
            return sum(1 for r in d if r[idx] == "Y")
        total = len(d)
        ok_t, ok_p, ok_both = _n(9), _n(10), _n(11)

        def pct(a):
            return f"{a}/{total} = {a/total*100:.2f}%" if total else "0/0"
        print(f"  {CN.get(dom, dom):6s} tool {pct(ok_t):>20}  param {pct(ok_p):>20}  "
              f"joint {pct(ok_both):>22}")
        summary_rows.append([CN.get(dom, dom), total, ok_t, ok_p, ok_both,
                             round(ok_t / total * 100, 2) if total else 0,
                             round(ok_p / total * 100, 2) if total else 0,
                             round(ok_both / total * 100, 2) if total else 0])
        for k, v in (("total", total), ("ok_t", ok_t), ("ok_p", ok_p), ("ok_both", ok_both)):
            totals[k] += v

    # 总体(只对出现过的域求和)
    summary_rows.append(["总体(实际域)", totals["total"], totals["ok_t"], totals["ok_p"],
                         totals["ok_both"],
                         round(totals["ok_t"] / totals["total"] * 100, 2) if totals["total"] else 0,
                         round(totals["ok_p"] / totals["total"] * 100, 2) if totals["total"] else 0,
                         round(totals["ok_both"] / totals["total"] * 100, 2) if totals["total"] else 0])

    st = OUTDIR / "summary.csv"
    with open(st, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["domain", "total", "tool_ok", "param_ok", "both_ok",
                    "tool_acc%", "param_acc%", "joint_acc%"])
        w.writerows(summary_rows)

    # 机器可读 json
    total = totals["total"]
    tool_acc = round(totals["ok_t"] / total * 100, 4) if total else 0
    param_acc = round(totals["ok_p"] / total * 100, 4) if total else 0
    joint_acc = round(totals["ok_both"] / total * 100, 4) if total else 0
    js = {
        "total": total, "err": err_count, "ok_only": args.ok_only,
        "tool_acc": tool_acc, "param_acc": param_acc, "joint_acc": joint_acc,
        "by_domain": {r[0]: {"total": r[1], "tool_ok": r[2], "param_ok": r[3],
                             "both_ok": r[4], "tool_acc": r[5], "param_acc": r[6],
                             "joint_acc": r[7]}
                      for r in summary_rows if r[0] != "总体(实际域)"},
    }
    sp = OUTDIR / "summary.json"
    sp.write_text(json.dumps(js, ensure_ascii=False, indent=2), encoding="utf-8")

    if err_count:
        print(f"\n注意: {err_count} 条 SSE 调用报错, 已标 ERR 写入对应域明细。")
    print(f"\n每域明细 → {OUTDIR}/detail_<domain>.csv")
    print(f"汇总    → {st}\n         → {sp}")


if __name__ == "__main__":
    main()