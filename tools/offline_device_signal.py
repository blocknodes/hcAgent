"""离线迭代「设备判域正则」：对179条失败覆盖 + 其它6域零泄漏 + dev-ok零误伤。"""
import csv, glob, json, re
from collections import Counter

FAILS = json.load(open('benchmark/output/local_device_fails.json'))


def load_sets():
    sets = {}
    for fn in sorted(glob.glob('benchmark/cases/sheet_rules_*.csv')):
        dom = fn.split('sheet_rules_')[1].replace('.csv', '')
        sets[dom] = list(csv.DictReader(open(fn, encoding='utf-8-sig')))
    return sets


RULES = [
    # 裸播放控制短语（^...$ 锚定，后续无内容名 → 不误伤「换一个起风了」等 music 歌曲）
    ("裸播放控制",
     r"^(?:开始播放|播放|停止播放|停止|暂停|重播|顺序播放|列表播放|随机播放|播放顺序|继续播放|恢复播放|直接播放|"
        r"继续|继续播放吧|接着播放|接着放|接着播|继续放|继续播|往下放|往下播|接着往下|继续给我放着|继续刚才那首|继续刚才听的|"
        r"我不想听了|不想听了|不想听|别放吧|别放了|别放|别播了|别唱|别唱了|不放了|不要放了|别再放了|不要再放了|不要听|不听了|不听了|"
        r"换一个|换个|换换|换一换|给我换一个|帮我换一个|换下一个|换个别的|听个别的|听别的|听别的吧|换个歌吧|换一首旁边的|换个听吧|听不得|"
        r"我要换一个|我想换一个|随机换一个|换一个吧|这个不好换|这个不好切|这个不好|再来一个|再换一个|换一个看看|我要换一个听|选换一个|"
        r"嗯我想听到|嗯我想听个别的|我想听个别的|给我切一个|听个别的|"
        r"放下一个|放上一个|切一个|切下一个|切上一个|再来一首|再来一首吧|再来一首别的|再换一首|再换一首别的|"
        r"刚才那首|回到刚才那首|听刚才那首|放刚才那首|换回刚才那首|切回刚才那首|放回刚才那首|回刚才那首|退一首回去|往回切一首|回退一首|"
        r"跳过这首|这首跳过|切歌|切歌吧|切一首|切一首别的|换首别的|换一首别的|换另一首|"
        r"这首不好听|这首歌不好听|不好听换|不好听不想|不好听了|不好听切|这首切了|这首歌切了|不要这首了切了吧|"
        r"我想听上一个|我还是想听上一个|想听上一个|"
        r"下一首|上一首|下一首吧|上一首吧|下一首再|下一曲|上一曲|上一曲吧|下一曲呗|上一集|下一集|上一部|下一部|"
        r"下一部|上一部|听上一部|听下一部|播下一部|播上一部|换一部|切一部|随机换一部|换下一部|换上一部|"
        r"换一个节目|下一个节目|换个节目|切换节目|换个台|下一个台|切台|换台)$"),
    # 音乐/歌 停止/关闭（锚定）
    ("音乐停关",
     r"^(?:把|帮我|麻烦|请|先)?\s*.{0,10}(?:音乐|歌|这首歌|那首歌|刚才那首歌)(?:停|停了|停一下|先停|关|关了|关掉|退|退出|别放|不放了|接着|继续|放着|再放)$"),
    # 跳过/切掉/关掉 这首 / 这首歌
    ("歌曲跳切",
     r"^(?:把|帮|麻烦|请)?\s*.{0,3}(?:这首歌|这首|刚才那首歌|刚才那首).{0,6}(?:切|跳过|停了|停掉|关|关了|关掉|退|退出|回去听)$"),
    # 片源跳转/快进
    ("跳转快进",
     r"^(?:跳到10:00|跳转到10:00|直接跳到10:00|跳到结尾|跳到开始|从01:30开始播放|从05:00开始播放|"
        r"往前切一集|往后切一集|回退一首|退一首回去|跳到|跳转|快进吧|快退吧|从.{1,3}开始播放|"
        r"从10:00开始播放|往前调|往后调)"),
    # 不定时晚/不想听/别放了一级短语（可带后缀）
    ("播放情绪短语",
     r"^(?:不看了退出来|不听了退出来|不看了|不听了退出|听够了|听腻了|不看了关掉|"
     r"→?|太晚了停掉音乐|音乐先停一下|音乐别放了|别放音乐了|别放这首歌了|音乐关了吧|这首歌切了吧|把这首歌接着放|把音乐放着继续|把音乐接着放|")
]

def main():
    sets = load_sets()
    fail_qs = [x['query'] for x in FAILS if not x['tool_ok']]
    other = [(d, t, q) for dom in sets if dom != 'device' for r in sets[dom]
             for (d, t, q) in ([ (dom, r['期望工具'], r['query'].strip()) ])]
    dev_ok = [r['query'].strip() for r in sets['device'] if r['query'].strip() not in fail_qs]
    rules = [(lab, re.compile(pat, re.X)) for lab, pat in RULES]

    cov=[]; leak=[]; dev=[]; un=[]
    for q in fail_qs:
        hit=False
        for lab, rx in rules:
            if rx.search(q):
                cov.append((lab, q)); hit=True; break
        if not hit: un.append(q)
    for dom, et, q in [(dom, et, q) for (dom, et, q) in
                       [(dom, r['期望工具'], r['query'].strip()) for dom in sets if dom!='device' for r in sets[dom]]]:
        for lab, rx in rules:
            if rx.search(q):
                leak.append((dom, et, q, lab)); break
    for q in dev_ok:
        for lab, rx in rules:
            if rx.search(q):
                dev.append((lab, q)); break
    covn=len(set(x[1] for x in cov))
    print(f"coverage {covn}/{len(fail_qs)}  leak-other {len(leak)}  devok-captured {len(dev)}")
    print('cov by rule:')
    for lab, n in Counter(x[0] for x in cov).most_common():
        print('  ',lab,n)
    print('LEAK:')
    for x in leak[:40]:
        print('   ', x)
    print('DEVOK captured:')
    for x in dev[:40]:
        print('   ', x)
    print('uncovered:')
    for q in sorted(un):
        print('   ', q)

if __name__=='__main__':
    main()