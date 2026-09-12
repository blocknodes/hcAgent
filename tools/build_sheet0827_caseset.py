"""把飞书《0827 优化单域》7 张子表(设备/体育/音乐/有声/影视/少儿/教育)聚合成统一评测集。

每张子表列: 业务域, query, 意图, 期望工具, 期望参数(JSON 字符串), 是否合理, ...
聚合字段: row, domain(英文key), domain_cn, query, intent, expected_tool, expected_params(dict/空)。

用法:
  python tools/build_aggregate_caseset.py          # 输出 data/sheet0827_caseset.json
"""
from __future__ import annotations
import csv, io, json, re, sys
from pathlib import Path

TOOLSDIR = Path(__file__).resolve().parent          # hcAgent/tools
DATA = TOOLSDIR.parent / "data"

# 子表名 -> (英文域key, 中文名)
SHEETS = {
    "PFA1bv": ("device",    "设备"),
    "z8W4rk": ("sports",    "体育"),
    "Y9D74h": ("music",     "音乐"),
    "P9Zabq": ("audio",     "有声"),
    "j1Tcfm": ("vod",       "影视"),
    "acwSry": ("children",  "少儿"),
    "a852JA": ("education", "教育"),
}

# 飞书 sheet 原始 json 文件(同目录)
_RAW = {
    "device":    "sheet0827_device.json",
    "sports":    "sheet0827_sports.json",
    "music":     "sheet0827_music.json",
    "audio":     "sheet0827_audio.json",
    "vod":       "sheet0827_vod.json",
    "children":  "sheet0827_children.json",
    "education": "sheet0827_education.json",
}


def load_golden_params(raw: str):
    """从 golden 参数列提取目标 JSON；失败返回 {} (golden 不可比/预期空参数)。"""
    raw = (raw or "").strip()
    if not raw:
        return {}
    # 去掉前导说明文案(如 "(当前息屏…)\n{...}")，只留首个完整 JSON 对象
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {}
    txt = m.group(0)
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        # 修复常见瑕疵: 值串尾部多引号、行间因内嵌引号残留
        cleaned = re.sub(r'([^\\])""', r'\1"', txt)          # "value": "x"" -> "value": "x"
        cleaned = re.sub(r',\s*}', '}', cleaned)
        cleaned = re.sub(r',\s*\]', ']', cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"_raw": raw}   # 无法解析则标记，评测时该条只比 tool


def _read_sheet(name: str) -> list[list[str]]:
    d = json.loads((DATA / _RAW[name]).read_text(encoding="utf-8"))
    return [r for r in csv.reader(io.StringIO(d["annotated_csv"])) if r]


def main() -> int:
    out: list[dict] = []
    seen: dict[tuple, int] = {}
    for sheet_id, (dom, cn) in SHEETS.items():
        rows = _read_sheet(dom)
        for i, r in enumerate(rows[1:], start=2):        # row 号 = 表内行号(含表头)
            if not r or (len(r) < 2) or not r[1].strip():
                continue
            query = r[1].strip()
            intent = r[2].strip() if len(r) > 2 else ""
            tool = r[3].strip() if len(r) > 3 else ""
            pstr = r[4].strip() if len(r) > 4 else ""
            rec = {
                "row": f"{cn}{i}",
                "domain": dom,
                "domain_cn": cn,
                "query": query,
                "intent": intent,
                "expected_tool": tool,
                "expected_params": load_golden_params(pstr),
            }
            key = (dom, query)
            seen.setdefault(key, 0)
            seen[key] += 1
            out.append(rec)
    payload = {"count": len(out), "records": out}
    (DATA / "sheet0827_caseset.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    from collections import Counter
    by_dom = Counter(r["domain"] for r in out)
    print(f"聚合完成 {len(out)} 条 → {DATA/'sheet0827_caseset.json'}")
    for k, v in by_dom.items():
        print(f"  {k:10s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())