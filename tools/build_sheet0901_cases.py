"""把飞书 0901测试集 各子表 csv → SSE eval 用的 caseset json。

目标：hcAgent/data/sheet0901_<domain>.json（records，字段与 sheet0827_caseset.json 对齐：
  row, domain, domain_cn, query, intent, expected_tool, expected_params）

CSV 列统一为: 业务域, query, 意图, 期望工具, 期望参数(JSON文本)。music 的列名略有不同
('线上意图'/'模型参数')但位置相同。工具名校验：与 hcTools 域 tools_by_name 对齐，
不匹配的工具名告警并 strip 空格。

用法:
  python tools/build_sheet0901_cases.py [--dom vod,music]
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

AGENT = Path(__file__).resolve().parent.parent          # hcAgent
DATA = AGENT / "data"
HCTOOLS = AGENT.parent / "hcTools"
sys.path.insert(0, str(HCTOOLS.parent))                 # hc/
sys.path.insert(0, str(HCTOOLS))

from hcTools.domains import (audio, children, device, education, music, sports, vod)  # noqa: E402

DOMAIN_OBJ = {
    "audio": audio.domain, "children": children.domain, "device": device.domain,
    "education": education.domain, "music": music.domain, "sports": sports.domain,
    "vod": vod.domain,
}
DOMAIN_CN = {k: v.name for k, v in DOMAIN_OBJ.items()}


def parse_params(txt):
    txt = (txt or "").strip()
    if not txt:
        return None
    if txt.startswith(("{", "[")):
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            return {"_raw": txt}
    return {"_raw": txt}


def build(domain: str) -> tuple[int, list, list]:
    csv_path = DATA / f"sheet0901_{domain}.csv"
    if not csv_path.exists():
        raise SystemExit(f"csv 不存在: {csv_path}")
    obj = DOMAIN_OBJ[domain]
    valid = set(obj.tools_by_name)
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    records, bad_tools = [], []
    for i, r in enumerate(rows[1:], start=2):
        if len(r) < 4:
            continue
        dom_cn = (r[0] or "").strip()
        q = (r[1] or "").strip()
        intent = (r[2] or "").strip()
        et = (r[3] or "").strip()
        ep = r[4] if len(r) > 4 else ""
        if not q or not et:
            continue
        if et not in valid:
            bad_tools.append((i, et))
        records.append({
            "row": i, "domain": domain, "domain_cn": dom_cn or DOMAIN_CN[domain],
            "query": q, "intent": intent, "expected_tool": et,
            "expected_params": parse_params(ep),
        })
    return len(rows) - 1, records, bad_tools


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-d", "--domains", default=None,
                    help="逗号分隔域名单，默认全部 7 域")
    args = ap.parse_args()
    want = {x.strip() for x in args.domains.split(",") if x.strip()} if args.domains \
        else set(DOMAIN_OBJ)

    total = 0
    for domain in sorted(want):
        if domain not in DOMAIN_OBJ:
            print(f"!! 未知域 {domain}, 跳过")
            continue
        n_rows, records, bad = build(domain)
        # 仅当该域存在 csv 才写出
        if not records:
            print(f"!! {domain}: 无记录(csv 缺失?) 跳过")
            continue
        out = DATA / f"sheet0901_{domain}_caseset.json"
        out.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        total += len(records)
        print(f"{domain:<10} {len(records):>5} 条 (csv {n_rows}行)  → {out.name}"
              + (f"  ⚠️ 期望工具不在域 tools: {sorted(set(bad))}" if bad else ""))
    print(f"\n合计 {total} 条")

if __name__ == "__main__":
    main()