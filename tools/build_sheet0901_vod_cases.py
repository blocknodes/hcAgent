"""把 sheet0901_yingshi.csv(0901测试集/影视) 转成 SSE caseset json。

CSV 列: 业务域, query, 意图, 期望工具, 期望参数(JSON文本), ...
输出字段与 sheet0827_caseset.json 对齐: {"records":[{row,domain,domain_cn,query,intent,expected_tool,expected_params}]}
"""
from __future__ import annotations
import csv, json
from pathlib import Path

AGENT = Path(__file__).resolve().parent.parent
CSV = AGENT / "data" / "sheet0901_yingshi.csv"
OUT = AGENT / "data" / "sheet0901_vod_caseset.json"
DOMAIN_MAP = {"影视": "vod"}


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


records = []
with open(CSV, encoding="utf-8-sig") as f:
    rows = list(csv.reader(f))
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
    records.append({
        "row": i, "domain": DOMAIN_MAP.get(dom_cn, dom_cn), "domain_cn": dom_cn,
        "query": q, "intent": intent, "expected_tool": et,
        "expected_params": parse_params(ep),
    })
OUT.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {len(records)} records -> {OUT}")