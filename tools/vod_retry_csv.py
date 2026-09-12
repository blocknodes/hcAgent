"""重跑 vod 失败用例并输出 CSV。

对 data/sse0827_diffs.json 里的失败用例(或 --row 指定)用 SSE 重新跑,带重试(规避
远端 10.19.96.219:4006 慢端规划服务波动的假失败),输出每条的判定明细 CSV。

用法:
  python tools/sse_retry_diff_csv.py                    # 重跑 diffs.json 全部失败用例
  python tools/sse_retry_diff_csv.py -o retry.csv --retries 3
  python tools/sse_retry_diff_csv.py --row 影视28,影视110
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
AGENT = TOOLS.parent
DIFFS = AGENT / "data" / "sse0827_diffs.json"
CASESET = AGENT / "data" / "sheet0827_caseset.json"

sys.path.insert(0, str(AGENT))
from runtime_execute import (_run_runtime, DEFAULT_FEATURE_CODE, random_device_id)


def canonical(x):
    if isinstance(x, dict):
        return {k: canonical(v) for k, v in x.items() if not _empty(v)}
    if isinstance(x, list):
        return sorted((canonical(i) for i in x if not _empty(i)),
                      key=lambda e: json.dumps(e, ensure_ascii=False, sort_keys=True))
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


def run_one(q, tv_mode="0", retries=3):
    for attempt in range(retries + 1):
        t0 = time.perf_counter()
        r = _run_runtime(q, feature_code=DEFAULT_FEATURE_CODE, client_sid=None,
                         device_id=random_device_id(), tv_mode=tv_mode, debug=True)
        lat = round((time.perf_counter() - t0) * 1000, 1)
        steps = [{"tool": s.get("toolName") or "", "params": s.get("parameters") or {}}
                 for p in r["plans"] for s in (p.get("steps") or [])
                 if isinstance(s, dict) and s.get("toolName")]
        if not steps:
            steps = [{"tool": t.get("tool") or "", "params": t.get("params") or {}}
                     for t in r["tools"]]
        if steps:
            return steps, len(r["plans"]), lat, attempt
    return [], -1, round((time.perf_counter() - t0) * 1000, 1), attempt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--row", default=None)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("-o", "--out", default="sse0827_retry.csv", help="输出文件名(data/[name])")
    ap.add_argument("-w", "--workers", type=int, default=8)
    ap.add_argument("--col", choices=["bright", "off"], default="bright")
    args = ap.parse_args()

    if not CASESET.exists():
        raise SystemExit(f"caseset 不存在: {CASESET}")
    cases = {c["row"]: c for c in json.loads(CASESET.read_text("utf-8"))["records"]}

    if args.row:
        want_rows = [x.strip() for x in args.row.split(",") if x.strip()]
    else:
        want_rows = [x["row"] for x in json.loads(DIFFS.read_text("utf-8"))["diffs"]]

    tv_mode = "0" if args.col == "bright" else "6"
    out_path = AGENT / "data" / args.out

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _one(row):
        c = cases.get(row)
        if not c:
            return None, "skip", "row not in caseset"
        q = c["query"]
        et = c["expected_tool"]
        ep = c.get("expected_params") or {}
        gp = json.dumps(ep, ensure_ascii=False)
        steps, n_plan, lat, attempts = run_one(q, tv_mode=tv_mode, retries=args.retries)
        if steps:
            pred_tool = steps[0]["tool"]
            pred_params = json.dumps(steps[0]["params"], ensure_ascii=False)
            ok_t = pred_tool == et
            ok_p = params_equal(steps[0]["params"], ep) if ep else ok_t
            joint = ok_t and ok_p
        else:
            pred_tool, pred_params, ok_t, ok_p, joint = "NO_PLAN", "{}", False, False, False
        rec = {
            "row": row, "domain_cn": c.get("domain_cn", ""), "query": q,
            "gold_tool": et, "gold_params": gp,
            "pred_tool": pred_tool, "pred_params": pred_params,
            "tool_ok": ok_t, "param_ok": ok_p, "joint_ok": joint,
            "n_plan": n_plan, "retries_used": 1, "latency_ms": lat,
        }
        return rec, c.get("query", ""), f"gold={et} pred={pred_tool} t={ok_t} p={ok_p} plans={n_plan}"

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_one, row): row for row in want_rows}
        for fut in as_completed(futs):
            rec, _q, msg = fut.result()
            if rec is None:
                print(f"[skip] {msg}", file=sys.stderr)
                continue
            rows.append(rec)
            print(f"[{rec['row']}] {rec['query'][:24]:24s} gold={rec['gold_tool']:22s} "
                  f"pred={rec['pred_tool']:22s} t={rec['tool_ok']} p={rec['param_ok']}")

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\n已写 {len(rows)} 条 → {out_path}")
    if rows:
        ok_t = sum(1 for r in rows if r["tool_ok"])
        ok_j = sum(1 for r in rows if r["joint_ok"])
        np = sum(1 for r in rows if r["pred_tool"] == "NO_PLAN")
        print(f"tool {ok_t}/{len(rows)}   joint {ok_j}/{len(rows)}   NO_PLAN {np}")


if __name__ == "__main__":
    main()