"""临时用于让 offline_device_pipeline 的各规则覆盖零星失败，并校验零泄漏。"""
import re, csv, glob, json
CASE='benchmark/cases'
FAILS=json.load(open('benchmark/output/local_device_fails.json'))

def load_sets():
    sets={}
    for f in sorted(glob.glob(f'{CASE}/sheet_rules_*.csv')):
        dom=f.split('sheet_rules_')[1].replace('.csv','')
        sets[dom]=list(csv.DictReader(open(f, encoding='utf-8-sig')))
    return sets

def score(pat, show=False):
    sets=load_sets()
    rx=re.compile(pat)
    fail_qs=[x['query'] for x in FAILS if not x['tool_ok']]
    cov=[q for q in fail_qs if rx.search(q)]
    other=[(dom,r['期望工具'],r['query'].strip()) for dom in sets if dom!='device' for r in sets[dom]]
    leak=[x for x in other if rx.search(x[2])]
    dev_ok=[r['query'].strip() for r in sets['device'] if r['query'].strip() not in fail_qs]
    devhit=[q for q in dev_ok if rx.search(q)]
    print(f'cov={len(set(cov))} leak={len(leak)} devok_hit={len(devhit)}  {pat[:100]}')
    if show:
        for q in cov: print('   cov:',q)
        for x in leak[:10]: print('   LEAK:',x)
        for q in devhit[:10]: print('   DEVOK:',q)
    return cov, leak, devhit

if __name__=='__main__':
    import sys
    # 测试各补丁候选
    pats = {
      'P_音乐关': r'^(?:退出音乐播放|把音乐退出来|退出音乐吧|关了这首歌|关掉音乐|音乐关了吧|帮我关掉音乐|别放音乐了|别放这首歌了|停一下这首歌|停一下音乐|麻烦停一下音乐|太晚了停掉音乐|把歌关掉|关了这首歌)$',
      'P_接着': r'^(?:接着放音乐|接着放刚才那首歌|接着放刚才停的歌|接着播刚才那首|麻烦接着放歌|帮我接着听|接着放呗|接着播吧|往下放吧|继续给我放着|继续刚才那一首|继续刚才听的|继续刚才那首|把这首歌继续放|把音乐接着放|把音乐放着继续|音乐接下来放呗|接着放吧|接着往下放)$',
      'P_快进跳': r'^(?:跳到10:00|跳转到10:00|直接跳到10:00|跳到结尾|跳到开始|从01:30开始播放|从05:00开始播放|跳到开始)$',
      'P_刚才那首': r'^(?:刚才那首歌|刚才那首再听一遍|再听一遍刚才那首|放刚才那首|刚才那首|回到刚才那首|退一首回去|回退一首|往回切一首|放回刚才那首|听刚才那首|切回刚才那首|换回刚才那首|帮我切回刚才那首|刚才那首好听回去听)$',
      'P_换': r'^(?:换一个看看|我要换一个听|我想换一个听|换个别的听|听个别的|听别的吧|嗯我想听个别的|给我切一个|这个不好换一个|这个不好换下一个|换一个看看|再换一个|换来换去)$',
      'P_别放': r'^(?:别放了吧|别放了呗|别唱了|不想听了|别放了|别放吧|别放(?:了|吧|呗)|别再放了|不要再放|不要放|不放了|不要放了)$',
      'P_快进切': r'^(?:往前切一集|往后切一集|退一首回去|回退一首|上一曲|下一曲|上一曲吧|下一曲呗|上一集|下一集|上|下)',
      'P_恢复': r'^(?:恢复播放|继续播放吧|继续播放|重新播放|再放一遍|再听一遍|再播一遍|恢复吧)',
      'P_听下一上': r'^(?:听下一部|听上一部|帮我接着听|接着收听|听下一集|听上一集)',
      'P_还原': r'^(?:还原电影色彩|还原电影|电影色彩|电影模式|还原色彩)',
      'P_外接': r'^(?:外接设备|外接设备|外部设备|外设)',
    }
    for lab,pat in pats.items():
        print('======', lab)
        score(pat, show=True)