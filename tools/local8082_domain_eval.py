"""对本地 8082 编排接口跑单个域的评测（复用 canonical/params_equal 比对）。

用法:
   python tools/local8082_domain_eval.py -d vod                # 本地 8082 跑 vod 域
   python tools/local8082_domain_eval.py -d vod -n 20          # 只跑前 N 条(冒烟)
   python tools/local8082_domain_eval.py -d music,children     # 多域
指标: tool / param / tool+param(joint)  与 sse_7domain_eval 相同口径,主指标 joint ≥90%。
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from local8082_eval import call_local
from sse_7domain_eval import canonical, params_equal, normalize_golden_params  # noqa: E402

TOOLS = Path(__file__).resolve().parent
AGENT = TOOLS.parent
CASESET = AGENT / "data" / "sheet0827_caseset.json"

DOMAIN_ORDER = ["device", "vod", "audio", "music", "sports", "children", "education"]
CN = {"device": "设备", "vod": "影视", "audio": "有声", "music": "音乐",
      "sports": "体育", "children": "少儿", "education": "教育"}


def run_one(case):
    q = case["query"]
    started = time.perf_counter()
    try:
        out = call_local(q)
    except Exception as exc:
        return {"case": case, "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "error": f"{type(exc).__name__}: {exc}"}
    steps = [{"tool": s.get("tool", ""), "params": s.get("params") or {}} for s in out["steps"]]
    return {"case": case, "steps": steps, "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "error": None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-d", "--domains", default="vod", help="逗号分隔域")
    ap.add_argument("-n", type=int, default=0, help="只跑前 N 条(默认全部)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    cs = json.loads(CASESET.read_text("utf-8"))["records"]
    cases = [c for c in cs if c.get("domain") in domains]
    if args.n:
        cases = cases[: args.n]
    print(f"本地 8082 评测域 {domains}  {len(cases)} 条")

    args_ = args
    if args_.workers == 8 and len(cases) < 8:
        workers = len(cases) or 1
    else:
        workers = args_.workers

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda c: run_one(c), cases))

    # 汇总
    agg = {d: {"score": 0, "ok_t": 0, "ok_p": 0, "ok_both": 0, "n_err": 0} for d in DOMAIN_ORDER}
    diffs = []
    for r in results:
        case = r["case"]
        dom = case.get("domain")
        if r["error"]:
            agg[dom]["n_err"] += 1
            continue
        et = case.get("expected_tool") or ""
        gp = normalize_golden_params(case.get("expected_params"))
        s = r["steps"][0] if r["steps"] else {}
        pred_tool = s.get("tool", "") or ""
        pred_params = s.get("params") or {}
        if not et or not pred_tool:
            continue
        ok_t = pred_tool == et
        ok_p = (not gp) or params_equal(pred_params, gp)
        a = agg[dom]
        a["score"] += 1
        a["ok_t"] += ok_t
        a["ok_p"] += ok_p
        a["ok_both"] += ok_t and ok_p
        if not ok_t:
            diffs.append({"row": case["row"], "domain": CN.get(dom, dom), "query": case["query"],
                          "gold": et, "pred": pred_tool,
                          "gold_params": gp, "pred_params": pred_params})

    tot = {k: sum(agg[d][k] for d in DOMAIN_ORDER) for k in ("score", "ok_t", "ok_p", "ok_both")}
    tot["n_err"] = sum(agg[d]["n_err"] for d in DOMAIN_ORDER)
    n = tot["score"]
    def pct(a, b):
        return f"{a}/{b} = {a/b*100:.2f}%" if b else "0/0"
    print(f"=== 本地 8082 单域评测 分={len(results)}  scored={n}  err={tot['n_err']} ===")
    for dom in DOMAIN_ORDER:
        a = agg[dom]
        if a["score"]:
            print(f"  {CN.get(dom,dom)}({dom:8}) tool {pct(a['ok_t'],a['score']):>22}  "
                  f"param {pct(a['ok_p'],a['score']):>22}  joint {pct(a['ok_both'],a['score']):>22}"
                  + (f"  err {a['n_err']}" if a["n_err"] else ""))
    print(f"\n全量 tool       {pct(tot['ok_t'], n)}   (主指标 joint≥90%)")
    print(f"       param      {pct(tot['ok_p'], n)}")
    print(f"       tool+param {pct(tot['ok_both'], n)}")

    if diffs:
        print(f"\n换 mis t_miss {len(diffs)}:")
        mc = Counter((x["gold"], x["pred"]) for x in diffs).most_common(15)
        for (g, p), m in mc:
            print(f"    {g:28}->{p:26} n={m}")

    out = AGENT / "data" / f"local8082_{'_'.join(domains)}_eval.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        import csv
        wr = csv.writer(f)
        wr.writerow(["row", "domain", "query", "gold", "pred"])
        for x in diffs:
            wr.writerow([x["row"], x["domain"], x["query"], x["gold"], x["pred"]])
    if diffs:
        print(f"\nmiss 明细 CSV → {out}")


if __name__ == "__main__":
    main()