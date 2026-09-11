"""跨域判域层：对 LLM 输出的 domain 做后处理修正。

hcAgent 的纯 LLM 主干在多域名模糊（sports/children/audio vs vod、device 预设）上
会判错 domain。这里在 T0 之后追加两层确定性判域修正，把 LLM 明显判错的 domain 改掉；
domain 定了，tool+params 仍交给 hcTools 各域流水线出（保 97-100%）。

黄金原则：**只在高置信时覆盖，绝不用易误伤的通用词单独判域。**
不用 `第N集`、`回放`、`卫视`、`全屏`、`动画`这类横跨多域的词。
层次：L2 精确句 badcase 优先；L1 高置信信号规则其次；未命中保留 LLM 判定。
确定性判域只改 domain，不代做语言拆分/选工具。"""
from __future__ import annotations

import json
import re
from pathlib import Path


def _load_badcases() -> dict[str, str]:
    p = Path(__file__).resolve().parent / "badcases_domain.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {b.get("query", ""): b.get("domain", "") for b in data.get("badcases", [])}
    except Exception:  # noqa: BLE001
        return {}


_BAD: dict[str, str] = {}


def _badcases() -> dict[str, str]:
    global _BAD
    if not _BAD:
        _BAD = _load_badcases()
    return _BAD


# 高置信信号规则：顺序即优先级，命中即返回该域。全为正则子串匹配。
_RULES: list[tuple[str, str]] = [
    # device：设备设置类强词（模式/氛围/屏保/外设）——命中必为 device，放最前
    ("device", r"(音乐模式|声音模式|音频模式|音频效果|音频增强|氛围墙|dlna|DLNA|[uU]盘|ai动画|关机动画|动画定制|运动画面|音乐功能|屏保|待机|熄屏|护眼模式)"),
    # device：播放控制 / 画质音质 / 演示
    ("device", r"(快进|快退|暂停|停止播放|倍速|静音|音量|调大声|调小声|音质|清晰度|分辨率|对比度|护眼|倒看|倒进)"),
    ("device", r"(演示|杜比|全景声|云健身|云关爱|AI映画|控光|声学设计|数字艺术|疾速|暗夜精灵|火箭炮|超广视角|画面增强)"),

    # audio 强信号：收听 / 广播 / 有声 / 评书 —— 命中即 audio（含儿童IP场景），放最前
    ("audio", r"(广播剧|广播|有声书|听书|评书|收听|音频|听录音|有声|听音频|有声故事|有声剧|的FM|电台|广播)"),

    # audio 睡前/助眠/听单 —— 强信号（不含「适合..听」，避免误伤 music「适合听的歌」）
    ("audio", r"(睡前|晚安|助眠|听单|我昨晚听)"),

    # music 歌曲词（.的歌/歌单/主题曲…）——全 music 专属，放 children IP 前
    ("music", r"(的歌|歌版|全歌|主题曲|片头曲|插曲|歌单|K歌|k歌|点歌|唱歌|要唱|想唱|歌手|歌词|MV|mv|曲库|音乐家|播放列表)"),
    ("music", r"(历史播放|听歌历史|历史的歌|历史里的|播放历史)"),

    # children 专属 IP / 剧名 —— 明确角色，最不易误伤
    ("children", r"(汪汪队|海绵宝宝|小羊肖恩|佩[琪奇]|光头强|熊出没|熊大|熊二|开心锤锤|喜羊羊|灰太狼|彼得兔|海底小纵队|小马宝[莉丽]|爆笑虫子|罗小黑|奶龙|小公主|聪明一休|白雪公主|丑小鸭|大耳朵图图|超级宝贝|JOJO|jojo|汽车世界|米小圈|萌鸡|挖掘机|消防车|托马斯|螺丝钉|长征先锋|鬼灭之刃|忍者|迷你特工队|熊猫姐姐|沃福|细胞|动物神探队|拼搭|恐龙世界|植物大战僵尸|愤怒的小鸟|胡巴|超级土豆|万达坏蛋联盟|超人兽战|宇宙护卫队)"),

    # sports：竞技 / 赛事 / 队伍
    ("sports", r"(比赛|赛事|联赛|锦标赛|世界杯|奥运会|世锦赛|亚洲杯|欧洲杯|亚冠|欧冠|NBA|CBA|英超|中超|决赛|预选赛|半决赛|积分|赛程|球队|夺冠|金牌|奖牌|比分|对阵|球赛|胜负)"),

    # vod：电影 / 戏剧 / 乐团（剧场演出）——「音乐剧/歌剧/X演」必为影视，放 music 前
    ("vod", r"(音乐剧|歌剧|晚会|电影院|观影)"),

    # music：歌曲 / 唱 / K歌 / 点歌 / MV / 主题曲 —— 「X的歌/曲」必为音乐
    ("music", r"(K歌|k歌|点歌|唱歌|要唱|想唱|歌手|歌词|歌单|歌曲|MV|mv|主题曲|片头曲|插曲|曲子|音乐|的?歌|打动人)"),

    # education 课程/技能词（教程/教学/训练/书法/绘画…）——education 专属标志，放 children 前
    ("education", r"(教程|教学|训练|书法|绘画|彩铅|画画|手账|预习|辅导|培训)"),

    # audio 听力语料：想听/要听 + 内容名词
    ("audio", r"((?:我想|我要|想|要)听.{0,18}(?:直播|小说|书|故事|节目|评书|音频|历史|篮球|课|英语|童话|评书))"),

    # children 内容型（成长/动画/启蒙）——避免"英语卡通/少儿英语"被反成 education
    ("children", r"(孩子|儿童|宝宝|幼儿|亲子|绘本|启蒙|少儿|卡通|动漫|动画|儿歌|小朋友|早教|幼儿园|益智|成长|寓言|公主故事|睡前故事)"),

    # education：学科 / 教材 / 课程（放最后）
    ("education", r"(语文|数学|英语|物理|化学|生物|地理|历史|政治|道法|人教版|沪教版|湘教版|北师大|苏教版|同步课|教材|习题|试卷|期末|中考|高考|一年级|二年级|三年级|四年级|五年级|六年级|初一|初二|初三|高一|高二|高三|小学|初中|高中|年级|课程|奥数)"),
]


def _match(q: str) -> tuple[str, str]:
    for dom, pat in _RULES:
        if re.search(pat, q):
            return dom, pat
    return "", ""


def detect_domain(query: str, llm_domain: str) -> str:
    """后修正 LLM 判的 domain。badcase 优先，其次高置信规则；都不命中保留 LLM。"""
    q = (query or "").strip()
    if not q:
        return llm_domain
    exact = _badcases().get(q)
    if exact:
        return exact
    dom, _p = _match(q)
    return dom if dom else llm_domain