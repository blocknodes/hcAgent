"""飞书多轮评测表(wiki GWfvwwX59ifXy8kaHZAclNXGn4b / sheet f75dc2) → benchmark case CSV。

把源表 50 组 × 5 轮 = 250 条多轮 query（组数/业务/轮数/query/期望工具/期望参数）落成本地
case 文件 cases/sheet_multiturn_250.csv，供 run_sse_multiturn.py -c 复用。

会话id 规则：`<业务>-组<组号>`（源表组号在各业务间重复，直接用组数会跨域撞车）。

依赖 lark-cli（已认证）。也可用 --csv <本地json> 离线回放（+csv-get 落盘的 JSON payload）。

用法:
  python benchmark/build_sheet_multiturn_250.py                # 在线拉取飞书表
  python benchmark/build_sheet_multiturn_250.py --csv /tmp/sheetf75dc2.json
  python benchmark/build_sheet_multiturn_250.py --out cases/sheet_multiturn_250.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
from pathlib import Path

BENCH = Path(__file__).resolve().parent
CASES = BENCH / "cases"

WIKI_URL = "https://hisense.feishu.cn/wiki/GWfvwwX59ifXy8kaHZAclNXGn4b"
SHEET_ID = "f75dc2"
FULL_RANGE = "A1:T260"          # 250 条数据，留余量

# 目标 case 列（对齐 run_sse_multiturn.py load_sessions 期望: 前7列 + 源行号）
HEADER = ["会话id", "业务", "组号", "轮数", "query", "期望工具", "期望参数", "源行号"]


def fetch_csv(url: str, sheet_id: str, rng: str) -> str:
    """调用 lark-cli sheets +csv-get 拉取 annotated_csv。"""
    cmd = [
        "lark-cli", "sheets", "+csv-get",
        "--url", url, "--sheet-id", sheet_id, "--range", rng, "--as", "user",
        "--output-path", "/tmp/_build_multiturn_250.json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"lark-cli 失败 (rc={proc.returncode}):\n{proc.stderr or proc.stdout}")
    payload = json.loads(Path("/tmp/_build_multiturn_250.json").read_text(encoding="utf-8"))
    return payload.get("annotated_csv") or ""


def rows_from_csv(annotated: str) -> list[list[str]]:
    """annotated_csv 每行带 [row=N] 前缀且含 JSON 换行；剥前缀后走 csv 解析。"""
    out = []
    for ln in annotated.split("\n"):
        ln = ln.rstrip("\r\n")
        if not ln.strip():
            continue
        if ln.startswith("[row="):
            ln = ln[ln.index("]") + 1:].lstrip()
        out.append(ln)
    return list(csv.reader(io.StringIO("\n".join(out))))


def build(annotated: str) -> tuple[list[list], int]:
    rows = rows_from_csv(annotated)
    rec, skipped = [], 0
    cur_group, cur_biz = "", ""
    for i, r in enumerate(rows[1:], start=2):
        if len(r) < 4:
            skipped += 1
            continue
        if r[0].strip():
            cur_group = r[0].strip()
        if r[1].strip():
            cur_biz = r[1].strip()
        rnd = r[2].strip()
        q = r[3].strip()
        if not q or not rnd:
            skipped += 1
            continue
        sess_id = f"{cur_biz}-组{cur_group}"
        rec.append([
            sess_id,                   # 会话id（业务+组号，避免跨域撞车）
            cur_biz,                   # 业务
            cur_group,                 # 组号
            rnd,                       # 轮数（第N轮）
            q,                         # query
            r[4].strip() if len(r) > 4 else "",   # 期望工具（源表当前为空，待填）
            r[5].strip() if len(r) > 5 else "",   # 期望参数（源表当前为空，待填）
            str(i),                    # 源行号
        ])
    return rec, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=None, help="直接读已保存的 +csv-get JSON(离线复用)")
    ap.add_argument("--out", default=str(CASES / "sheet_multiturn_250.csv"), help="输出 case csv")
    args = ap.parse_args()

    if args.csv:
        annotated = json.loads(Path(args.csv).read_text(encoding="utf-8")).get("annotated_csv") or ""
        print(f"[build] 复用本地 json: {args.csv}")
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
    n_gold = sum(1 for r in recs if r[5])
    n_sess = len({r[0] for r in recs})
    print(f"[build] 写入 {out}  条数={len(recs)}  会话={n_sess}  有gold={n_gold}  skipped={skipped}")


if __name__ == "__main__":
    main()
