"""把 hcTools 7 域规则层 testset.json → benchmark/cases/sheet_rules_<domain>.csv。

testset 列: query, expected_tool, expected_params(dict / 空)。
CSV 前 6 列对齐 run_sse_bench.read_csv_records 口径:
  业务域, query, 意图, 期望工具, 期望参数(JSON), 是否合理(空)
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
CASES = BENCH / "cases"
DOMAINS = ["device", "vod", "audio", "music", "sports", "children", "education"]
CN = {"device": "设备", "vod": "影视", "audio": "有声", "music": "音乐",
      "sports": "体育", "children": "少儿", "education": "教育"}
HCTOOLS = (Path(__file__).resolve().parents[3]) / "hcTools"   # parents: benchmark->hcAgent->hc->workspace? no:
# __file__=hc/hcAgent/benchmark/build_ruleset_cases.py
# parents[0]=benchmark parents[1]=hcAgent parents[2]=hc parents[3]=workspace/hc? let's be safe:
_HC = Path(__file__).resolve().parents[2]                       # hc
HCTOOLS = _HC / "hcTools"

def main() -> int:
    total = 0
    for dom in DOMAINS:
        ts = HCTOOLS / "domains" / dom / "testset.json"
        if not ts.exists():
            print(f"!! 缺 testset: {ts}")
            continue
        data = json.loads(ts.read_text(encoding="utf-8"))
        rows = [["业务域", "query", "意图", "期望工具", "期望参数", "是否合理"]]
        for r in data.get("records", []):
            q = (r.get("query") or "").strip()
            t = (r.get("expected_tool") or "").strip()
            if not q or not t:
                continue
            p = r.get("expected_params")
            ptext = json.dumps(p, ensure_ascii=False) if p else ""
            rows.append([CN[dom], q, "", t, ptext, ""])
        out = CASES / f"sheet_rules_{dom}.csv"
        with open(out, "w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"{dom:10s} {len(rows)-1} -> {out}")
        total += len(rows) - 1
    print(f"共 {total} 条")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
