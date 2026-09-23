"""SSE 真实链路评测 train_2k.csv：输入 0910 同分布训练集 → 逐条跑远端 runtime → 明细 + 汇总。

与 run_sse_bench.py 完全同口径（复用 tools/sse_7domain_eval 的 run_one/canonical/
params_equal_no_retext），仅输入源不同：bc/eval0910/train_2k.csv（业务域,query,意图,期望工具,期望参数）。

用法:
  python benchmark/run_sse_bench_train2k.py                # 全 6 域 2000 条
  python benchmark/run_sse_bench_train2k.py -d vod,music   # 指定域
  python benchmark/run_sse_bench_train2k.py -n 10          # 每域前 N 条(冒烟)
  python benchmark/run_sse_bench_train2k.py -w 16          # 并发数(默认 8)
输出:
  benchmark/output/detail_train2k_<domain>.csv
  benchmark/output/summary_train2k.csv / .json
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
TRAIN2K = BENCH.parent.parent / "bc/eval0910/train_2k.csv"
OUTDIR = BENCH / "output"

DOMAIN_ORDER = ["vod", "children", "audio", "music", "sports", "education"]
CN = {"vod": "影视", "audio": "有声", "music": "音乐",
      "sports": "体育", "children": "少儿", "education": "教育"}
DOM_MAP = {"影视": "vod", "少儿\\多tab": "children", "音乐": "music",
           "体育": "sports", "教育": "education", "有声": "audio"}

sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(AGENT))
from build_sheet0901_cases import parse_params                       # noqa: E402
from tools.sse_7domain_eval import (                                 # noqa: E402
    run_one, params_equal, normalize_golden_params,
)
from run_sse_bench import (                                          # noqa: E402
    _norm_query_shape, _norm_fee_tag, params_equal_no_retext, _param_diff, _param_repr,
)


def read_train2k(domains=None):
    """读 bc/eval0910/train_2k.csv → records，列: 业务域,query,意图,期望工具,期望参数。"""
    records = []
    with open(TRAIN2K, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    for i, r in enumerate(rows[1:], start=2):
        if len(r) < 5:
            continue
        dom = DOM_MAP.get((r[0] or "").strip())
        q = (r[1] or "").strip()
        et = (r[3] or "").strip()
        if not q or not et:
            continue
        if domains and dom not in domains:
            continue
        records.append({
            "row": i, "domain": dom, "domain_cn": (r[0] or "").strip(),
            "query": q, "intent": (r[2] or "").strip(),
            "expected_tool": et, "expected_params": parse_params(r[4] or ""),
            "reasonable": "",
        })
    return records


DETAIL_HEADER = ["域", "row", "query", "intent",
                 "gold_tool", "gold_params", "pred_tool", "pred_params",
                 "tool_ok", "param_ok", "both_ok", "param_diff", "hit_source", "latency_ms"]


def _run(case):
    try:
        r = run_one(case)
        s = r["steps"][0] if r.get("steps") else {}
        return {"ok": True, "case": case, "pred_tool": s.get("tool", ""),
                "pred_params": s.get("params") or {},
                "hit_source": s.get("hit_source", ""), "latency": r.get("latency_ms", 0)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "case": case, "error": f"{type(exc).__name__}: {exc}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-d", "--domains", default=None, help="逗号分隔域: vod,music")
    ap.add_argument("-w", "--workers", dest="workers", type=int, default=8)
    ap.add_argument("-n", type=int, default=0, help="每域只跑前 N 条(冒烟)")
    args = ap.parse_args()

    wants = {d.strip() for d in args.domains.split(",") if d.strip()} if args.domains else None
    all_records = read_train2k(wants)
    if args.n:
        by = defaultdict(list)
        for c in all_records:
            by[c["domain"]].append(c)
        all_records = [c for d in DOMAIN_ORDER for c in by.get(d, [])[: args.n]]

    print(f"SSE 评测 train_2k {len(all_records)} 条 (域={wants or 'all'}, 并发 {args.workers}) ...",
          flush=True)

    per_domain: dict[str, list[list]] = defaultdict(list)
    err_count = done = ok_t = ok_both = 0

    def _fold(res):
        case = res["case"]
        dom = case["domain"]
        if not res["ok"]:
            per_domain[dom].append([case["domain_cn"], case["row"], case["query"],
                                    case["intent"], "", "", "", "", "ERR", "ERR", "ERR",
                                    res.get("error", ""), "", ""])
            return None
        et = case.get("expected_tool") or ""
        gp = normalize_golden_params(case.get("expected_params"))
        pt = res.get("pred_tool") or ""
        pp = res.get("pred_params") or {}
        ot = bool(et) and pt == et
        op = (not gp) or params_equal_no_retext(pp, gp)
        ob = ot and op
        diff = "" if op else _param_diff(gp, pp)
        per_domain[dom].append([case["domain_cn"], case["row"], case["query"],
                                case["intent"], et, json.dumps(gp, ensure_ascii=False),
                                pt, json.dumps(pp, ensure_ascii=False),
                                "Y" if ot else "N", "Y" if op else "N",
                                "Y" if ob else "N", diff,
                                res.get("hit_source", ""), res.get("latency", "")])
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
            print(f"\r[{done}/{total}] tool_acc={ok_t/done*100:5.1f}%"
                  f"   tool+param_acc={ok_both/done*100:5.1f}%   err={err_count}",
                  end="", flush=True)
    print()

    for dom in DOMAIN_ORDER:
        if dom not in per_domain or not per_domain[dom]:
            continue
        path = OUTDIR / f"detail_train2k_{dom}.csv"
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(DETAIL_HEADER)
            w.writerows(per_domain[dom])

    print("\n=== SSE 按域 joint (train_2k) ===")
    summary_rows = []
    totals = {"total": 0, "ok_t": 0, "ok_p": 0, "ok_both": 0}
    for dom in DOMAIN_ORDER:
        d = per_domain.get(dom)
        if not d:
            continue

        def _n(idx):
            return sum(1 for r in d if r[idx] == "Y")
        total_d = len(d)
        ok_t, ok_p, ok_both = _n(8), _n(9), _n(10)

        def pct(a):
            return f"{a}/{total_d} = {a/total_d*100:.2f}%" if total_d else "0/0"
        print(f"  {CN.get(dom, dom):6s} tool {pct(ok_t):>20}  param {pct(ok_p):>20}  "
              f"joint {pct(ok_both):>22}")
        summary_rows.append([CN.get(dom, dom), total_d, ok_t, ok_p, ok_both,
                             round(ok_t / total_d * 100, 2) if total_d else 0,
                             round(ok_p / total_d * 100, 2) if total_d else 0,
                             round(ok_both / total_d * 100, 2) if total_d else 0])
        for k, v in (("total", total_d), ("ok_t", ok_t), ("ok_p", ok_p), ("ok_both", ok_both)):
            totals[k] += v

    summary_rows.append(["总体", totals["total"], totals["ok_t"], totals["ok_p"],
                         totals["ok_both"],
                         round(totals["ok_t"] / totals["total"] * 100, 2) if totals["total"] else 0,
                         round(totals["ok_p"] / totals["total"] * 100, 2) if totals["total"] else 0,
                         round(totals["ok_both"] / totals["total"] * 100, 2) if totals["total"] else 0])

    st = OUTDIR / "summary_train2k.csv"
    with open(st, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["domain", "total", "tool_ok", "param_ok", "both_ok",
                    "tool_acc%", "param_acc%", "joint_acc%"])
        w.writerows(summary_rows)

    total = totals["total"]
    js = {
        "set": "train_2k.csv", "total": total, "err": err_count,
        "tool_acc": round(totals["ok_t"] / total * 100, 4) if total else 0,
        "param_acc": round(totals["ok_p"] / total * 100, 4) if total else 0,
        "joint_acc": round(totals["ok_both"] / total * 100, 4) if total else 0,
        "by_domain": {r[0]: {"total": r[1], "tool_ok": r[2], "param_ok": r[3],
                             "both_ok": r[4], "tool_acc": r[5], "param_acc": r[6],
                             "joint_acc": r[7]}
                      for r in summary_rows if r[0] != "总体"},
    }
    sp = OUTDIR / "summary_train2k.json"
    sp.write_text(json.dumps(js, ensure_ascii=False, indent=2), encoding="utf-8")

    if err_count:
        print(f"\n注意: {err_count} 条 SSE 调用报错, 已标 ERR 写入对应域明细。")
    print(f"\n每域明细 → {OUTDIR}/detail_train2k_<domain>.csv")
    print(f"汇总    → {st}\n         → {sp}")


if __name__ == "__main__":
    main()
