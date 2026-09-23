"""多轮(会话)SSE 真实链路评测：query-only 表(multiturn_100groups_query.csv)。

表结构(4 列, 无 gold): 组数,业务,轮数,query —— 100 个会话 × 5 轮 = 500 条。
会话 = (组数, 业务)：同会话各轮共享同一 device_id + client_sid，顺序喂给远端
runtime 累积上下文；不同会话用不同 device_id。产出逐轮预测(工具+参数)，
供后续回填 gold 后评分。

入口:
  python benchmark/run_sse_multiturn_100.py                # 全量 100 会话 500 轮
  python benchmark/run_sse_multiturn_100.py -n 4           # 只跑前 4 个会话(冒烟)
  python benchmark/run_sse_multiturn_100.py -b 影视        # 只跑指定业务
  python benchmark/run_sse_multiturn_100.py -c <path.csv>  # 换数据集路径
  python benchmark/run_sse_multiturn_100.py --seed <s>     # 显式种子派生 device_id（缺省=每次随机）
说明:
  同会话(组数,业务) 5 轮共享同一 device_id; 不同会话 device_id 不同。
  device_id = sha256("<seed>::<组>_<业务>") 前缀化 → 省略 --seed 时每次运行全变,
  同 --seed 复现同一批 device_id（同会话仍共享）。
输出:
  benchmark/output/detail_multiturn_100.csv   # 逐轮预测
  benchmark/output/summary_multiturn_100.csv  # 会话/业务级完成度
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import secrets
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BENCH = Path(__file__).resolve().parent
AGENT = BENCH.parent
OUTDIR = BENCH / "output"
_CSV = AGENT.parent / "multiturn_100groups_query.csv"

sys.path.insert(0, str(AGENT))
sys.path.insert(0, str(BENCH))
from runtime_execute import _run_runtime, DEFAULT_FEATURE_CODE  # noqa: E402
from run_sse_multiturn import device_id_for, _param_repr  # noqa: E402


def _round_no(rnd: str) -> int:
    m = re.search(r"\d+", rnd or "")
    return int(m.group()) if m else 0


def device_id_for_seed(session_id: str, seed: str) -> str:
    """由 session_id + 种子派生 device_id：同 seed 同会话恒同、异会话/异 seed 恒异。

    每次运行带不同的 seed → device_id 整体换一批（远端上下文缓存不复用上次），
    但同位内同一会话的 5 轮仍共享同一 device_id（多轮累积语义不破）。
    无 seed 时退化为原生 device_id_for(session_id) 的行为（固定哈希，不依赖 seed）。
    """
    if not seed:
        return device_id_for(session_id)
    h = hashlib.sha256(f"{seed}::{session_id}".encode("utf-8")).hexdigest()
    return "861003009000014000000712" + h[:16]


def load_sessions(*, seed: str = "") -> list[dict]:
    """读 query-only CSV -> [{"session_id",组数,业务,device_id,client_sid,turns:[...]}]。

    seed 非空：device_id = device_id_for_seed(f"{grp}_{biz}", seed)，同会话共享、异会话隔离
    且随 seed 每次运行变化；seed 为空：沿用原确定性 device_id_for（行为不变）。
    """
    with open(_CSV, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    sessions: dict[tuple, dict] = {}
    for r in rows[1:]:
        if len(r) < 4:
            continue
        grp, biz, rnd, q = (x.strip() for x in r[:4])
        if not q:
            continue
        key = (grp, biz)
        sid = f"{grp}_{biz}"
        s = sessions.setdefault(key, {
            "session_id": sid, "grp": grp, "biz": biz,
            "device_id": device_id_for_seed(sid, seed),
            "client_sid": f"mt_{sid}",
            # 用 seed 参与可变的 client_sid：同会话 5 轮一致、异会话/异次不同
            "client_sid_raw": f"mt_{grp}_{biz}",
            "turns": [],
        })
        s["client_sid"] = s["client_sid_raw"] + (f"_{seed}" if seed else "")
        # 补偿 pepper：同 seed 下 client_sid 也跟 seed 走，避免远端按 sid 复用上一轮上下文
        s["turns"].append({"round": rnd, "round_no": _round_no(rnd), "query": q})
    for s in sessions.values():
        s["turns"].sort(key=lambda t: t["round_no"])
    return list(sessions.values())


def _seed_from_args(text: str | None) -> str:
    """归一 CLI seed：留空 → 当前时间戳随机种子（每次运行不同）；否则原样下传。"""
    if not text:
        return secrets.token_hex(4)
    return text.strip()


def run_turn(turn: dict, device_id: str, client_sid: str | None) -> dict:
    started = time.perf_counter()
    try:
        r = _run_runtime(turn["query"], feature_code=DEFAULT_FEATURE_CODE,
                         client_sid=client_sid, device_id=device_id,
                         tv_mode="0", debug=True)
        steps = [
            {"tool": s.get("toolName") or "", "retext": s.get("retext", ""),
             "params": s.get("parameters") or {},
             "hit_source": s.get("hitSource") or s.get("hit_source") or ""}
            for p in (r.get("plans") or []) for s in (p.get("steps") or [])
            if isinstance(s, dict) and s.get("toolName")
        ]
        if not steps:
            steps = [{"tool": t.get("tool") or "", "retext": t.get("retext", ""),
                      "params": t.get("params") or {},
                      "hit_source": t.get("hitSource") or t.get("hit_source") or ""}
                     for t in (r.get("tools") or [])]
        return {"ok": True, "turn": turn,
                "pred_tool": (steps[0]["tool"] if steps else ""),
                "pred_params": (steps[0]["params"] or {} if steps else {}),
                "retext": (steps[0]["retext"] if steps else ""),
                "hit_source": (steps[0]["hit_source"] if steps else ""),
                "latency_ms": round((time.perf_counter() - started) * 1000, 1)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "turn": turn, "error": f"{type(exc).__name__}: {exc}"}


def run_session(sess: dict) -> dict:
    return {"session": sess,
            "turns": [run_turn(t, sess["device_id"], sess["client_sid"])
                      for t in sess["turns"]]}


DETAIL_HEADER = ["组数", "业务", "轮次", "query", "pred_tool", "pred_params",
                 "retext", "hit_source", "latency_ms", "status", "error"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-b", "--biz", default=None, help="逗号分隔业务: 影视,少儿,设备控制,教育,音乐,体育")
    ap.add_argument("-w", "--workers", type=int, default=4, help="并发会话数")
    ap.add_argument("-n", type=int, default=0, help="只跑前 N 个会话(冒烟)")
    ap.add_argument("-c", "--config", default=None, help="覆盖数据集路径")
    ap.add_argument("--seed", default=None,
                    help="device_id 派生种子。缺省=每次随机(运行间 device_id 全变)；"
                         "显式传 同 seed 则同会话同 device_id 可复现")
    args = ap.parse_args()

    global _CSV
    if args.config:
        _CSV = Path(args.config)

    seed = _seed_from_args(args.seed)
    sessions = load_sessions(seed=seed)
    if args.biz:
        want = {x.strip() for x in args.biz.split(",") if x.strip()}
        sessions = [s for s in sessions if s["biz"] in want]
    sessions.sort(key=lambda s: (s["biz"], int(s["grp"]) if s["grp"].isdigit() else s["grp"]))
    if args.n:
        sessions = sessions[: args.n]

    n_turns = sum(len(s["turns"]) for s in sessions)
    print(f"SSE 多轮评测(query-only): {len(sessions)} 会话, {n_turns} 轮 "
          f"(并发 {args.workers}) ...", flush=True)

    detail: list[list] = []
    sess_stat: dict[str, tuple] = {}
    done = 0

    def fold(res):
        nonlocal done
        s = res["session"]
        ok = 0
        for t in res["turns"]:
            tt = t["turn"]
            if not t["ok"]:
                detail.append([s["grp"], s["biz"], tt["round"], tt["query"],
                               "", "", "", "", "", "ERR", t.get("error", "")])
                continue
            ok += 1
            detail.append([s["grp"], s["biz"], tt["round"], tt["query"],
                           t["pred_tool"], _param_repr(t["pred_params"]),
                           t.get("retext", ""), t.get("hit_source", ""),
                           t.get("latency_ms", ""), "OK", ""])
        done += 1
        sess_stat[s["session_id"]] = (ok, len(res["turns"]))
        print(f"\r[{done}/{len(sessions)} 会话] 完成 {sum(v[0] for v in sess_stat.values())}/{n_turns} 轮",
              end="", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for fut in as_completed([pool.submit(run_session, s) for s in sessions]):
            try:
                fold(fut.result())
            except Exception as exc:  # noqa: BLE001
                print(f"\n[会话异常] {exc}")
    print()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUTDIR / "detail_multiturn_100.csv"
    with open(detail_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(DETAIL_HEADER)
        # 按组数+轮次排序，方便回填 gold
        detail.sort(key=lambda r: (r[0], r[1], _round_no(r[2])))
        w.writerows(detail)

    # 会话/业务级完成度
    biz_tot: Counter = Counter()
    biz_ok: Counter = Counter()
    for s in sessions:
        ok, n = sess_stat.get(s["session_id"], (0, len(s["turns"])))
        biz_tot[s["biz"]] += n
        biz_ok[s["biz"]] += ok
    rows = [["业务", "轮数", "成功轮数", "成功率"]]
    for b in sorted(biz_tot):
        rows.append([b, biz_tot[b], biz_ok[b], f"{biz_ok[b]/biz_tot[b]*100:.1f}%"])
    rows.append(["全局", n_turns, sum(v[0] for v in sess_stat.values()),
                 f"{sum(v[0] for v in sess_stat.values())/n_turns*100:.1f}%" if n_turns else "0"])
    summary_path = OUTDIR / "summary_multiturn_100.csv"
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)

    print(f"逐轮预测 -> {detail_path}")
    print(f"完成度汇总 -> {summary_path}")
    for r in rows:
        print("  ", r)


if __name__ == "__main__":
    main()
