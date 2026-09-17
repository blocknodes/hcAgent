"""vod_fuzzy_search query 打 8082 编排接口的时延/并发压测。

并发梯度逐档打同一批 query, 每档输出 p50/p90/p95/p99/max、吞吐 RPS、成功率。

  python tools/perf8082_vodfuzzy.py                       # 默认梯度 1,2,4,8,16,32
  python tools/perf8082_vodfuzzy.py -c 1,8,64 -n 100      # 自定义并发 / 每档请求数
  python tools/perf8082_vodfuzzy.py -o perf.csv           # 落 CSV 到 data/perf.csv
  python tools/perf8082_vodfuzzy.py --query-mode cycle     # 循环复用 query(否则不重复抽)
"""
from __future__ import annotations
import argparse, csv, json, statistics, sys, time, urllib.error, urllib.request, uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
AGENT = TOOLS.parent
HC_ROOT = AGENT.parent
URL = "http://127.0.0.1:8082/slowAgent/poc_local8082"

# 优先用 vod 域测试集里金标准为 vod_fuzzy_search 的 query(现网真实说法, 长度/口语度分布真实)
DEFAULT_QUERY_FILES = [
    HC_ROOT / "hcTools" / "domains" / "vod" / "testset.json",
]


def load_queries(paths, only_tool: str) -> list[str]:
    for p in paths:
        if not p.exists():
            continue
        data = json.loads(p.read_text("utf-8"))
        recs = data["records"] if isinstance(data, dict) else data
        qs = [r["query"] for r in recs if r.get("expected_tool") == only_tool and r.get("query")]
        if qs:
            print(f"[queries] {p} 取到 {len(qs)} 条 {only_tool}")
            return qs
    sys.exit(f"[fatal] 没有找到含 {only_tool} 的测试集: {[str(p) for p in paths]}")


def call_once(query: str, base: str, timeout: float) -> dict:
    """单次请求, 返回 {lat_ms, ok, err, hit_source}。lat_ms 含建连+读全响应。"""
    trace_id = f"perf_{uuid.uuid4().hex[:12]}"
    body = {
        "traceId": trace_id, "deviceId": "d1",
        "data": {"query": query, "tvMode": "0"},
    }
    req = urllib.request.Request(
        base, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        dt = (time.perf_counter() - t0) * 1000
        code = None
        hit = ""
        try:
            r = json.loads(raw.decode("utf-8"))
            code = r.get("code")
            steps = (r.get("data") or {}).get("steps") or []
            hit = steps[0].get("hitSource", "") if steps else ""
        except Exception:
            pass
        ok = code == 200
        return {"lat_ms": dt, "ok": ok, "err": "" if ok else f"code={code}",
                "hit_source": hit, "query": query}
    except Exception as exc:
        dt = (time.perf_counter() - t0) * 1000
        return {"lat_ms": dt, "ok": False, "err": f"{type(exc).__name__}: {exc}",
                "hit_source": "", "query": query}


def pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def run_level(queries: list[str], conc: int, base: str, timeout: float) -> dict:
    """conc 个线程各自领一段 query 跑完。用 barrier 让所有线程同时起跑, 避免斜坡。"""
    import threading
    n = len(queries)
    chunks = [queries[i::conc] for i in range(conc)]   # 轮询切分, 长度均衡
    results: list[dict] = []
    lock = threading.Lock()
    start = threading.Barrier(conc, timeout=30)

    def worker(chunk):
        start.wait()
        out = [call_once(q, base, timeout) for q in chunk]
        with lock:
            results.extend(out)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as pool:
        list(pool.map(worker, chunks))
    wall = time.perf_counter() - t0

    lats = sorted(r["lat_ms"] for r in results)
    oks = [r for r in results if r["ok"]]
    oks_lat = sorted(r["lat_ms"] for r in oks)
    # 时延按 hitSource 分层: badcase/rule 是纯规则命中(亚毫秒), llm 走大模型(百毫秒级)。
    # 混合 p50 会掩盖这个双峰, 所以逐层各报一份。
    by_src: dict[str, list[float]] = {}
    for r in oks:
        by_src.setdefault(r["hit_source"] or "(none)", []).append(r["lat_ms"])
    src_stats = {
        s: {"n": len(v), "p50": pct(sorted(v), 0.50), "p90": pct(sorted(v), 0.90)}
        for s, v in sorted(by_src.items(), key=lambda kv: -len(kv[1]))
    }
    return {
        "by_source": src_stats,
        "concurrency": conc,
        "requests": n,
        "ok": len(oks),
        "fail": n - len(oks),
        "wall_s": wall,
        "rps": n / wall if wall > 0 else 0.0,
        "mean": statistics.fmean(lats) if lats else float("nan"),
        "p50": pct(oks_lat, 0.50), "p90": pct(oks_lat, 0.90),
        "p95": pct(oks_lat, 0.95), "p99": pct(oks_lat, 0.99),
        "max": max(oks_lat) if oks_lat else float("nan"),
        "err_sample": next((r["err"] for r in results if not r["ok"]), ""),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-u", "--url", default=URL)
    ap.add_argument("-c", "--concurrency", default="1,2,4,8,16,32",
                    help="并发梯度, 逗号分隔 (默认 1,2,4,8,16,32)")
    ap.add_argument("-n", "--requests", type=int, default=60,
                    help="每档总请求数 (默认 60)")
    ap.add_argument("--query-mode", choices=["cycle", "unique"], default="cycle",
                    help="cycle=不足则循环复用(默认); unique=从测试集多抽 n 条不重复")
    ap.add_argument("-q", "--queries-file", action="append", default=None)
    ap.add_argument("--tool", default="vod_fuzzy_search")
    ap.add_argument("--warmup", type=int, default=3, help="每档前热身请求数")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("-o", "--out", default="", help="结果 CSV 写到 data/[name]")
    args = ap.parse_args()

    concs = [int(x) for x in args.concurrency.split(",") if x.strip()]
    if max(concs) > args.requests:
        print(f"[warn] 最大并发 {max(concs)} > 每档请求数 {args.requests}, 部分线程空转")

    pool = [Path(p) for p in args.queries_file] if args.queries_file else DEFAULT_QUERY_FILES
    all_q = load_queries(pool, args.tool)

    def pick(n: int, seed: int) -> list[str]:
        if args.query_mode == "cycle":
            return [all_q[(seed + i) % len(all_q)] for i in range(n)]
        if n > len(all_q):
            sys.exit(f"[fatal] unique 模式需要 {n} 条, 测试集只有 {len(all_q)} 条")
        return [all_q[(seed * n + i) % len(all_q)] for i in range(n)]

    print(f"[target] {args.url}")
    print(f"[mode]   {args.query_mode}, 每档 {args.requests} 请求, 热身 {args.warmup}\n")

    rows, offset = [], 0
    for conc in concs:
        for i in range(args.warmup):        # 热身不复用 urllib 连接池, 只为把规则层/缓存捂热
            call_once(all_q[(offset + i) % len(all_q)], args.url, args.timeout)
        qs = pick(args.requests, offset)
        offset += args.requests
        row = run_level(qs, conc, args.url, args.timeout)
        rows.append(row)
        print(f"conc={conc:3d}  n={row['requests']:4d}  ok={row['ok']:4d} fail={row['fail']:3d}  "
              f"wall={row['wall_s']:6.2f}s  rps={row['rps']:7.2f}  "
              f"p50={row['p50']:7.1f}  p90={row['p90']:7.1f}  p95={row['p95']:7.1f}  "
              f"p99={row['p99']:7.1f}  max={row['max']:7.1f} ms"
              + (f"  ERR={row['err_sample']}" if row["fail"] else ""))
        if row["by_source"]:
            seg = "  ".join(f"{s}:n={v['n']},p50={v['p50']:.0f}" for s, v in row["by_source"].items())
            print(f"           hitSource 分层 → {seg}")

    base_row = next((r for r in rows if r["concurrency"] == 1), None)
    print("\n[单位 ms] 并发 → 时延/吞吐:")

    if args.out:
        out = AGENT / "data" / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        flat = [{k: (json.dumps(v, ensure_ascii=False) if k == "by_source" else v)
                 for k, v in r.items()} for r in rows]
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
            wr.writeheader()
            wr.writerows(flat)
        print(f"[out] {out}")
    if base_row:
        print(f"[基线] 单并发 p50={base_row['p50']:.1f}ms rps={base_row['rps']:.2f}")


if __name__ == "__main__":
    main()
