"""离线评估设备判域信号规则候选（不改生产代码）。

输入：7 域 sheet_rules CSV + 当前失败明细（local_device_fails.json）。
候选规则为 (domain, regex) 列表；对每条候选：
  - coverage：在 179 条 device 失败查询中命中的数量（目标越高越好）
  - leak：
      a) 其余 6 域测试集里任何一条命中 → 判技误伤
      b) device 测试集里当前已经 PASS 的 1520 条中命中 → 可能回归（但要再判断是不是无害）
输出每个候选的 coverage/leak。用法：
  cd hcAgent && python3 tools/offline_rule_score.py
"""
from __future__ import annotations
import csv, glob, json, re, sys
from pathlib import Path

CASE = Path('benchmark/cases')
FAILS = json.load(open('benchmark/output/local_device_fails.json'))


def load_sets():
    sets = {}
    for f in sorted(glob.glob(str(CASE / 'sheet_rules_*.csv'))):
        dom = f.split('sheet_rules_')[1].replace('.csv', '')
        sets[dom] = list(csv.DictReader(open(f, encoding='utf-8-sig')))
    return sets


def main():
    sets = load_sets()
    dev_rows = sets['device']
    # 失败查询集
    fail_qs = [x['query'] for x in FAILS if not x['tool_ok']]
    dev_ok_qs = [r['query'].strip() for r in dev_rows
                 if r['query'].strip() not in fail_qs]
    # 其它 6 域全部 query
    other_qs = []
    for dom in [d for d in sets if d != 'device']:
        for r in sets[dom]:
            other_qs.append((dom, r['期望工具'], r['query'].strip()))

    print(f"device tool-fail queries: {len(fail_qs)}, device-ok: {len(dev_ok_qs)}, other: {len(other_qs)}\n")

    cands = [
        ("A1 音乐/歌+停关退接续",
         r"(?:音乐|歌)[^。！？]{0,8}(?:停|暂停|关了?|关掉|退|接着|继续|放出来了?|放着|别放|不放了|退出)|(?:把|帮|请|麻烦|帮我|先)?\s*.{0,6}(?:音乐|歌)[^。！？]{0,3}(?:停了|停一下|停掉|退出|关了?|关掉|接着|继续|先停)"),
        ("A2 音乐/歌+接着放/继续",
         r"(?:音乐|歌)[^。！]{0,10}(?:接着|继续|放)|(?:接着|继续)[^。！]{0,6}(?:音乐|歌)"),
        ("B1 裸停止/下一/上一/换",
         r"^(?:开始播放|播放|停止播放|停止|暂停|下一个|上一个|重播|顺序播放|列表播放|随机播放|播放顺序(?:吧|了)?)$"),
        ("B2 不想听/别放/不放了/别唱",
         r"(?:不想听|不想看了|别(?:放|放吧|放了|唱|唱了|播|播吧|停了|停吧)|不要放了?|别再放|不要再放|不听了|不要听|不播了|放下一个)"),
        ("B3 换一个/听别的/换部",
         r"(?:换一个|换一批|听个别的|听别的|来个别的|换个别的|帮我换一个|给我换一个|换另一个|换一部|切一部|随机换一部|换一个看看|换下一个|换一首别的)和(?:换了|换首别的|换另一个|这个不好换)"),
    ]

    for lab, pat in cands:
        rx = re.compile(pat)
        cov_n = len(set(q for q in fail_qs if rx.search(q)))
        leak_other = [(dom, t, q) for (dom, t, q) in other_qs if rx.search(q)]
        # 在 device ok 中命中（可能无害/需要二次确认）
        dev_ok_hit = [q for q in dev_ok_qs if rx.search(q)]
        print(f"== {lab}\n   regex: {pat[:120]}")
        print(f"   coverage: {cov_n}/{len(fail_qs)}")
        print(f"   leak-other: {len(leak_other)}")
        for x in leak_other[:8]:
            print(f"      OTHER {x}")
        print(f"   hit-in-dev-ok: {len(dev_ok_hit)}")
        for x in dev_ok_hit[:8]:
            print(f"      DEVOK {x}")
        print()


if __name__ == "__main__":
    main()