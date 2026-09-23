#!/usr/bin/env python3
"""补齐 0924 指标表 F 列「平均耗时(1并发)」缺的口径。

复用 measure_latency.call()（本地 8082，1 并发），只测缺的几类，每类 10 条：
  息屏(tv=6) / 多tab / 多轮按轮次 1-5
已测过的 7 单业务 + 亮屏 + 串行 + 并行 直接用 output/latency_measure.csv（1 并发）的值。
输出: output/latency_fill_0924.csv
"""
from __future__ import annotations
import csv
import statistics
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))
import measure_latency as M  # noqa: E402


def run_batch(name: str, qs: list[str], tv: str = "0", cap: int = 10) -> list[float]:
    ms_list = []
    n_err = 0
    for q in qs[:cap]:
        ms, _ = M.call(q, tv=tv)
        if ms < 0:
            n_err += 1
        else:
            ms_list.append(ms)
    mean = statistics.mean(ms_list) if ms_list else 0
    print(f"{name}: n={len(ms_list)} err={n_err} mean={mean:.0f}ms", flush=True)
    return ms_list


def main() -> None:
    # 息屏: brightoff 99 条 tv=6, 取 10
    bright = M.read_csv("sheet0821_brightoff.csv", 1)
    off = run_batch("息屏", bright, tv="6", cap=10)

    # 多tab: e80803 里 动画/动漫/卡通 动画 的查询(col2), 取 10
    e8 = M.read_csv("e80803_multiintent.csv", 2)
    multitab = [q for q in e8 if any(k in q for k in ("动画", "动漫", "卡通"))]
    tab = run_batch("多tab", multitab, "0", cap=10)

    # 多轮: multiturn csv 按「轮次」列(下标3, 第N轮) 分桶, 每轮取 10
    rounds: dict[str, list[str]] = {}
    with open(BENCH / "cases" / "sheet0821_multiturn.csv", encoding="utf-8-sig") as f:
        for r in csv.reader(f):
            if len(r) < 5 or not r[3].startswith("第"):
                continue
            rounds.setdefault(r[3], []).append(r[4])
    round_ms: dict[str, float] = {}
    for rnd in sorted(rounds, key=lambda s: int(s[1:-1])):
        v = run_batch(rnd, rounds[rnd], "0", cap=10)
        round_ms[rnd] = statistics.mean(v) if v else 0

    (M.OUT).mkdir(parents=True, exist_ok=True)
    rows = [["行", "样本", "平均ms"]]
    rows.append(["息屏", len(off), round(statistics.mean(off), 1) if off else 0])
    rows.append(["多tab", len(tab), round(statistics.mean(tab), 1) if tab else 0])
    for rnd in sorted(round_ms):
        rows.append([rnd, 0, round(round_ms[rnd], 1)])
    path = M.OUT / "latency_fill_0924.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)
    print("->", path)


if __name__ == "__main__":
    main()