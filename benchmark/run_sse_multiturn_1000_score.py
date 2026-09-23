#!/usr/bin/env python3
"""1000 组多轮表评分: gold(cases/sheet_multiturn_1000groups_gold.csv) × 预测(output/detail_multiturn_1000groups.csv)。

口径与 run_sse_multiturn.py 完全一致:
  - tool: 归一化后相等(_TOOL_ALIAS)
  - both: tool 对 且 _params_ok(retext/figures 忽略、query 子串宽容)
  - fuzzy 工具参数免判(_PARAM_WAIVED_TOOLS)

用法:
  python3 benchmark/run_sse_multiturn_1000_score.py            # 全量
  python3 benchmark/run_sse_multiturn_1000_score.py -b 影视     # 指定业务
输出:
  output/score_multiturn_1000groups.csv  (业务×轮次 汇总)
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH.parent))

from run_sse_multiturn import _params_ok, _norm_tool, _params, _PARAM_WAIVED_TOOLS  # noqa: E402

GOLD = BENCH / "cases" / "sheet_multiturn_1000groups_gold.csv"
DET = BENCH / "output" / "detail_multiturn_1000groups.csv"
OUT = BENCH / "output" / "score_multiturn_1000groups.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-b", default="", help="只统计指定业务(逗号分隔)")
    args = ap.parse_args()
    want = {x.strip() for x in args.b.split(",") if x.strip()}

    gold = {}
    for r in list(csv.reader(open(GOLD, encoding="utf-8-sig")))[1:]:
        gold[(r[2], r[3])] = r  # (组号, 轮次) -> row

    biz_stat = defaultdict(lambda: [0, 0, 0])
    turn_stat = defaultdict(lambda: [0, 0, 0])
    biz_turn = defaultdict(lambda: [0, 0, 0])
    for r in list(csv.reader(open(DET, encoding="utf-8-sig")))[1:]:
        grp, biz, rnd, q, pt, pp, _rt, status, _err = r[:9]
        if want and biz not in want:
            continue
        g = gold.get((grp, rnd))
        if not g:
            continue
        et = g[5]
        tool_ok = _norm_tool(pt) == _norm_tool(et)
        param_ok = True
        if tool_ok and et not in _PARAM_WAIVED_TOOLS:
            try:
                ppd = json.loads(pp) if pp else {}
            except json.JSONDecodeError:
                ppd = {"_raw": pp}
            param_ok = _params_ok(_params(g[6]), ppd)
        for st in (biz_stat[biz], turn_stat[rnd], biz_turn[(biz, rnd)]):
            st[0] += 1
            st[1] += tool_ok
            st[2] += tool_ok and param_ok

    lines = ["维度,键,轮数,tool_ok,both_ok,tool%,both%"]

    def fmt(dim, key, st):
        n, t, b = st
        return f"{dim},{key},{n},{t},{b},{t / n * 100:.2f}%,{b / n * 100:.2f}%"

    tot = [0, 0, 0]
    for b in sorted({k[0] for k in biz_turn}):
        for rnd in ["第1轮", "第2轮", "第3轮", "第4轮", "第5轮"]:
            st = biz_turn.get((b, rnd))
            if st and st[0]:
                lines.append(fmt("业务×轮次", f"{b}|{rnd}", st))
    for b, st in sorted(biz_stat.items()):
        lines.append(fmt("业务", b, st))
    for rnd, st in sorted(turn_stat.items()):
        lines.append(fmt("轮次", rnd, st))
    for st in biz_stat.values():
        tot[0] += st[0]; tot[1] += st[1]; tot[2] += st[2]
    lines.append(fmt("全局", "ALL", tot))

    text = "\n".join(lines)
    print(text)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
