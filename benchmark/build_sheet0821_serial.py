"""飞书 0821「多意图串行」(wiki KqtDwk775iDDrukzIm5cG2mDnFc / sheet pQYZry) → benchmark case CSV。

把源表有效列（query, 是否合理, tool1, param1, tool2, param2 及源行号）落成本地 case 文件
cases/sheet0821_serial.csv，供串行多意图评测复用（对齐 existing sheet0821_* 命名）。

依赖 lark-cli（已认证，需 sheets:spreadsheet:read）。也可用 --csv <本地csv> 离线回放
（数据直接来自 lark-cli sheets +csv-get 的 annotated_csv，先存 csv 再转 case）。

用法:
  python benchmark/build_sheet0821_serial.py                 # 在线拉取飞书表
  python benchmark/build_sheet0821_serial.py --csv raw_serial.csv   # 用本地已拉取的 csv
  python benchmark/build_sheet0821_serial.py --out cases/sheet0821_serial.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import subprocess
from pathlib import Path

BENCH = Path(__file__).resolve().parent
CASES = BENCH / "cases"

# 源表定位（飞书 wiki URL + 目标子表 sheet_id）
WIKI_URL = "https://hisense.feishu.cn/wiki/KqtDwk775iDDrukzIm5cG2mDnFc"
SHEET_ID = "pQYZry"
FULL_RANGE = "A1:X102"          # 表头 + 101 条数据，覆盖至 X 列

# 目标 case 列（与源表字段一一对应）
HEADER = ["业务域", "query", "是否合理", "tool1", "param1", "tool2", "param2", "源行号"]


def fetch_csv(url: str, sheet_id: str, rng: str) -> str:
    """调用 lark-cli sheets +csv-get 拉取 annotated_csv 文本。"""
    cmd = [
        "lark-cli", "sheets", "+csv-get",
        "--url", url, "--sheet-id", sheet_id, "--range", rng, "--as", "user",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"lark-cli 失败 (rc={proc.returncode}):\n{proc.stderr or proc.stdout}")
    import json
    payload = json.loads(proc.stdout)
    if not payload.get("ok"):
        raise SystemExit(f"lark-cli 返回异常:\n{proc.stdout}")
    return payload["data"]["annotated_csv"]


def rows_from_csv(annotated: str) -> list[list[str]]:
    """annotated_csv 每行带 [row=N] 前缀且含 JSON 换行；剥前缀后走 csv 解析（正确处理跨行引号）。"""
    lines = annotated.split("\n")
    out: list[list[str]] = []
    for ln in lines:
        ln = ln.rstrip("\r\n")
        if not ln.strip():
            continue
        # 剥 "[row=N] " 前缀
        if ln.startswith("[row="):
            ln = ln[ln.index("]") + 1:].lstrip()
        out.append(ln)
    return list(csv.reader(io.StringIO("\n".join(out))))


def build(annotated: str) -> tuple[list[list], int]:
    rows = rows_from_csv(annotated)
    rec, skipped = [], 0
    for i, r in enumerate(rows):
        if i == 0:  # 表头
            continue
        if len(r) < 6:
            skipped += 1
            continue
        q = (r[0] or "").strip()
        t1 = (r[2] or "").strip()
        t2 = (r[8] or "").strip()
        if not q or not t1:
            skipped += 1
            continue
        row_no = 1 + i
        rec.append([
            "串行",                    # 业务域（全部为 vod/fuzzy/relate 首步 + fan_knowledge 尾步）
            q,                         # query
            (r[1] or "").strip(),      # 是否合理
            t1,                        # tool1
            (r[3] or "").strip(),      # param1
            t2,                        # tool2
            (r[9] or "").strip(),      # param2
            str(row_no),               # 源行号
        ])
    return rec, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=None, help="直接读已保存的 annotated_csv(离线复用)")
    ap.add_argument("--out", default=str(CASES / "sheet0821_serial.csv"), help="输出 case csv")
    args = ap.parse_args()

    if args.csv:
        annotated = Path(args.csv).read_text(encoding="utf-8")
        print(f"[build] 复用本地 csv: {args.csv}")
    else:
        print(f"[build] 拉取飞书 {WIKI_URL} sheet={SHEET_ID} range={FULL_RANGE} ...")
        annotated = fetch_csv(WIKI_URL, SHEET_ID, FULL_RANGE)

    recs, skipped = build(annotated)
    CASES.mkdir(exist_ok=True)
    out = Path(args.out)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(recs)
    print(f"[build] 写入 {out}  条数={len(recs)}  skipped={skipped}")


if __name__ == "__main__":
    main()