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

try:
    from .detect_rulebase import DetectRuleSet, Rule
except ImportError:  # 被 tools 差异测试独立加载时无包上下文
    from detect_rulebase import DetectRuleSet, Rule


def _load_badcases() -> dict[str, str]:
    return _load_badcases_for("badcases_domain.json")


_BAD: dict[str, str] = {}


def _badcases() -> dict[str, str]:
    global _BAD
    if not _BAD:
        _BAD = _load_badcases()
    return _BAD


def _load_badcases_for(fname: str) -> dict[str, str]:
    p = Path(__file__).resolve().parent / fname
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {b.get("query", ""): b.get("domain", "") for b in data.get("badcases", [])}
    except Exception:  # noqa: BLE001
        return {}


# 高置信信号规则：顺序即优先级，命中即返回该域。全为正则子串匹配。
_RULES: list[tuple[str, str]] = [
    # device：设备设置类强词（模式/氛围/屏保/外设）——命中必为 device，放最前
    ("device", r"(音乐模式|声音模式|音频模式|音频效果|音频增强|氛围墙|dlna|DLNA|[uU]盘|ai动画|关机动画|动画定制|运动画面|音乐功能|屏保|待机|熄屏|护眼模式)"),
    # device：播放控制（上下首/集、单曲循环、起播暂停快退音量…）/ 画质音质 / 演示
    # 注意：单曲已移交 music（裸「单曲」= 单首歌）；「单曲循环」仍由末尾「循环」兜底 device。
    ("device", r"(快进|快退|往后退|倒回|退回|暂停|停止播放|倍速|静音|音量|调大声|调小声|音质|清晰度|分辨率|对比度|护眼|倒看|倒进|下一首|上一首|下一集|上一集|循环|切换)"),
    ("device", r"(演示|杜比|全景声|云健身|云关爱|AI映画|控光|声学设计|数字艺术|疾速|暗夜精灵|火箭炮|超广视角|画面增强)"),
    
    # audio 强信号：收听 / 广播 / 有声 / 评书 —— 命中即 audio（含儿童IP场景），放最前
    ("audio", r"(广播剧|广播|有声书|小说|听书|评书|收听|音频|听录音|有声|听音频|有声故事|有声剧|的FM|电台|广播)"),
    # audio 听剧/小说分集：我要听X第N集/趁X第二部 —— 「我要听」明确 audio，放 vod 具名前
    ("audio", r"(?:我要|我想|想|要|请帮我)?听.{0,4}(?:三体|庆余年|偷偷藏不住|风起洛阳|斗罗大陆).{0,6}第[0-9一二三四五六七八九十]+(?:[集部季]|部分)"),
    # audio 通用「听 + 有声剧名 + 第N集/部/回」：听力动词显式标记听书意图，任何剧名都归 audio，
    # 不依赖具名剧名言表。占比 0 误伤其它域；放 audio 具体剧名之后、vod 具名播放之前。
    ("audio", r"(收听|想听|要听|听一下|给我听|请听|想去听|继续听|听听)\D{0,8}(?:第|部|回)?[0-9一二三四五六七八九十百]+(?:集|部|回)"),
    ("audio", r"^\s*听\D{0,8}第[0-9一二三四五六七八九十百]+(?:集|部|回)"),

    # children 儿歌 —— 儿歌是少儿内容，golden 统一 edu search；放 music 的「.的歌」前
    ("children", r"儿歌"),
    # children 幼龄听的歌（1岁半宝宝…）—— 宝宝/岁 幼儿语境 优先于 music 歌曲
    ("children", r"(?:\d+岁(?:半)?|宝宝|幼儿|小孩子).{0,4}(?:听|看)的.{0,2}(?:歌|音乐)"),

    # children 动漫/动画明确标签：具名IP+动漫 = 少儿内容（educ_search），放 vod 具名前
    ("children", r"(?:斗罗大陆|汪汪队|三体).{0,2}(?:动漫|动画|卡通)(?:版|片|影视剧)?"),

    # vod 影视具名播放：播放 golden 统一 vod_search。命中已知剧/片名 + 播放动词即 vod。
    # 名词覆盖 sheet065 亮屏 vod 播放（面具52-59/音乐68-70），避免误伤 music 真歌名。
    ("vod", r"(?:播放|直接播|给我放|帮我播放|放|看|点播|打开|播).{0,6}(?:庆余年|三体|偷偷藏不住|风起洛阳|斗罗大陆|西游记|觉醒年代|三国演义|雪中悍刀行|虫儿飞|送你一朵小红花|情深深雨濛濛)"),
    ("vod", r"(?:庆余年|三体|偷偷藏不住|风起洛阳|斗罗大陆|西游记|觉醒年代|三国演义|雪中悍刀行|虫儿飞|送你一朵小红花|情深深雨濛濛).{0,6}(?:第[0-9一二三四五六七八九十]+[集部]|大结局)"),
    # vod 影视具名片名（无播放动词，亮屏 golden 也视作 vod_search）：裸剧/片名词
    ("vod", r"^(?:播放)?(?:庆余年|三体|偷偷藏不住|风起洛阳|斗罗大陆|觉醒年代|三国演义|雪中悍刀行|送你一朵小红花)$"),

    # audio 睡前/助眠/听单 —— 强信号（不含「适合..听」，避免误伤 music「适合听的歌」）
    ("audio", r"(睡前|晚安|助眠|听单|我昨晚听)"),

    # music 歌曲词（.的歌/歌单/主题曲…）——全 music 专属，放 children IP 前
    ("music", r"(的歌|歌版|全歌|主题曲|片头曲|插曲|歌单|K歌|k歌|点歌|唱歌|要唱|想唱|歌手|歌词|MV|mv|曲库|音乐家|播放列表)"),
    ("music", r"(历史播放|听歌历史|历史的歌|历史里的|播放历史)"),

    # children 专属 IP / 剧名 —— 明确角色，最不易误伤
    ("children", r"(汪汪队|海绵宝宝|小羊肖恩|佩[琪奇]|光头强|熊出没|熊大|熊二|开心锤锤|喜羊羊|灰太狼|彼得兔|海底小纵队|小马宝[莉丽]|爆笑虫子|罗小黑|奶龙|小公主|聪明一休|白雪公主|丑小鸭|大耳朵图图|超级宝贝|JOJO|jojo|汽车世界|米小圈|萌鸡|挖掘机|消防车|托马斯|螺丝钉|鬼灭之刃|忍者|迷你特工队|熊猫姐姐|沃克|细胞|动物神探队|恐龙世界|植物大战僵尸|愤怒的小鸟|胡巴|超级土豆|万达坏蛋联盟|坏蛋联盟|猪屁登|超人兽战|宇宙护卫队)"),
    # children：卡通/动画主角或特摄 IP（儿童向，非车企/球队歧义场景）。承上层顺序：
    # 道奇=汪汪队角色、尼克狐尼克=疯狂动物城主角、奥特曼/迪迦=特摄、芭比=玩偶 IP → children。
    # 注意只加儿童向「角色词」，勿加 影视剧名/音乐剧 等宽泛词（如冰雪奇缘会误伤 vod 检索）。
    ("children", r"(道奇|尼克狐尼克|奥特曼|迪迦|奈克瑟斯|芭比)"),

    # sports：竞技 / 赛事 / 队伍
    ("sports", r"(比赛|赛事|联赛|锦标赛|世界杯|奥运会|世锦赛|亚洲杯|欧洲杯|亚冠|欧冠|NBA|CBA|英超|中超|决赛|预选赛|半决赛|积分|赛程|球队|夺冠|金牌|奖牌|比分|对阵|球赛|胜负)"),

    # vod：电影 / 戏剧 / 乐团（剧场演出）——「音乐剧/歌剧/X演」必为影视，放 music 前
    ("vod", r"(音乐剧|歌剧|晚会|电影院|观影)"),
    # children 动画/动漫名词（动画电影/动漫系列）—— 少儿内容 educ_search，放 vod「X的电影」前
    ("children", r"(动画电影|动漫电影|动画片|卡通片|动画类(?:的)?电影|动漫类)"),
    #       避免「胡歌的电影」被 music 的「歌手/主题曲」误收（胡歌有歌、歌字）。
    #       排除影视原声带/OST 等音乐语境（立对人不算影视）。
    ("vod", r"(?:\S+的(?:电影|电视剧|片子|剧集)|(?:\S+)?主演的(?:电影|电视剧)|饰演.{0,4}(?:电影|电视剧))(?![^ ]{0,6}(?:原声带|OST|配乐|插曲))"),

    # device 播放控制会话型：继续/换/重放 播放层状态 → playback_control。
    # 置于内容域规则之后、music「音乐/歌」裸词之前：内容域优先（白雪公主/有声书/三体），
    # 无内容锚时「继续放音乐/换一首」归 device，避免被 music 裸「音乐」抢判。
    ("device", r"(继续放|继续播放|换一首|重新放|重新播放|不好听|没听完|重放一遍|往后放)"),

    # music：歌曲 / 唱 / K歌 / 点歌 / MV / 主题曲 —— 「X的歌/曲」必为音乐
    ("music", r"(K歌|k歌|点歌|唱歌|要唱|想唱|歌曲|歌词|歌单|歌曲|MV|mv|主题曲|片头曲|插曲|曲子|音乐|的?歌|打动人)"),

    # education 课程/技能词（教程/教学/训练/书法/绘画…）——education 专属标志，放 children 前
    ("education", r"(教程|教学|训练|书法|绘画|彩铅|画画|手账|预习|辅导|培训)"),

    # audio 听力语料：想听/要听 + 内容名词
    ("audio", r"((?:我想|我要|想|要)听.{0,18}(?:直播|小说|书|故事|节目|评书|音频|历史|篮球|课|英语|童话|评书))"),

    # children 内容型（成长/动画/启蒙）——避免"英语卡通/少儿英语"被反成 education
    ("children", r"(孩子|儿童|宝宝|幼儿|亲子|绘本|启蒙|少儿|卡通|动漫|动画|儿歌|小朋友|早教|幼儿园|益智|成长|寓言|公主故事|睡前故事)"),

    # education：学科 / 教材 / 课程（放最后）
    ("education", r"(语文|数学|英语|物理|化学|生物|地理|历史|政治|道法|人教版|沪教版|湘教版|北师大|苏教版|同步课|教材|习题|试卷|期末|中考|高考|一年级|二年级|三年级|四年级|五年级|六年级|初一|初二|初三|高一|高二|高三|小学|初中|高中|年级|课程|奥数)"),

    ]


# 可审计规则集：由上方 _RULES 派生，保证与改造前逐条正则完全一致（零漂移）。
# 每条 (domain, regex) 转成一个 Rule，decide 命中模式即返回域名，未匹配返回 None，
# 交由下一优先级的规则继续评。供 tools/sheet0901_rule_regress.py 等审计工具枚举命中。
# 业务的判定顺序仍由 detect_domain 的显式步骤决定，_SIGNALS 只作为「高置信信号」可枚举层。
_SIGNALS = DetectRuleSet([
    Rule(
        id=f"signal_{i:02d}_{dom}",
        priority=100 + i,
        title=f"{dom} 高置信信号 #{i + 1}",
        explain=f"正则子串命中 {dom}：{pat[:60]}",
        decide=(lambda q, llm, tv, dom=dom, pat=pat:
                dom if re.search(pat, q) else None),
        scope="both",
    )
    for i, (dom, pat) in enumerate(_RULES)
])


def _match(q: str) -> tuple[str, str]:
    for dom, pat in _RULES:
        if re.search(pat, q):
            return dom, pat
    return "", ""


def _match_rule(q: str) -> str | None:
    """信号层命中明细：返回命中的 Rule id（审计用），保持 _RULES 为单一事实源。"""
    for i, (dom, pat) in enumerate(_RULES):
        if re.search(pat, q):
            return f"signal_{i:02d}_{dom}"
    return None


# 泛知识开放域问答信号：影视/少儿/音乐/教育 的"信息求助"问句 → fan_knowledge_agent(qa 域)。
# 必须在明确 播放/片段/设备/学科检索 之外才生效，避免误伤。
_QA_HIT = re.compile(
    r"是谁|是不是一台|谁是|谁在|谁唱|谁写|扮演|主演|演员表|简介|短评|票房|获奖|金鸡奖|奖项|"
    r"哪年|哪一年|哪部|哪一|哪些|哪几|哪个|哪只|什么时候|何时|"
    r"多少|几只|几个|几岁|多大了|多久|多高|"
    r"为什么|为何|为什么呀|"
    r"改编|写了|写的|写作|翻唱|作曲|作词|写词|填词|收录|所属|所在|"
    r"总部|在哪里|在哪|上映|了吗|写的歌|唱的歌|唱的|"
    r"讲了什么|讲了|讲述|讲述了|关于|"
    r"治愈系|治愈|好听|更好听|好吧听|比较好听|最好|最火|适合.{0,3}听.{0,3}歌|睡前听歌"
)

# 影视媒资载体词：query 命中它 → 用户在"按维度检索媒资/定位片段"，而非开放知识问答。
# 用途：LLM 已判 vod 时，防止 _QA_HIT 的通用问词(哪些/哪部/改编/上映/出自)把媒资检索误伤成 qa。
# 只作为"保留 vod"的保护，不做正向判域，故是 wordlist 而非会误判他域的强信号。
# 刻意不含单字"剧/片/番"等过宽词（会误伤 music/audio 的"唱段/广播剧"语境）；"唱段"留在此域。
_MEDIA_TYPE = re.compile(
    r"(电影|电视剧|电视连续剧|连续剧|影片|影院|影视|动漫|动画片|纪录片|电视节目|综艺|"
    r"节目|话剧|戏曲|评书|相声|小品|春晚|MV|mv|花絮|片段|番剧|演唱会|舞台剧|"
    r"科幻片|剧集|影视剧|片子里|片子|电影版|剧作|"
    r"唱段|花鼓戏|豫剧|京剧|黄梅戏|越剧|评剧|秦腔)"
)


def _is_media_query(query: str) -> bool:
    """query 是否带着明确的影视媒资载体（含 检索/定位 意向常伴随 LLM 已判 vod）。"""
    q = (query or "").strip()
    return bool(q and _MEDIA_TYPE.search(q))


# 影视「内容定位/出处」信号：不是开放知识问答，而是把某句台词/某片段/某集/某场面
# 归位到具体影视作品。特征是"引一段内容 + 问它是哪部/出自哪/第几集/有哪些场面"。
# 单独成信号而非并入 _MEDIA_TYPE：这些文案常缺"电影/剧"字面载体（用剧名+台词/出宾语），
# 但 LLM 已判 vod 时必是媒资检索。刻意不含单字"剧"：children 儿童台词(动画片/绘本/动漫)
# 走 educ_fuzzy，不宜被 vod 强拉。只在 llm_domain=="vod" 保护位生效。
_MEDIA_LOCATOR = re.compile(
    r"(台词|名场面|片段|那段|桥段|第几集|哪几集|多少集|哪集|第几|哪部影视剧|什么影视剧|"
    r"打斗|高光|逆袭那段|爆燃|试唱|经典场面|名场面|名句|经典台词|出哪部|出自|来自哪|"
    r"是哪个(?:电视剧|电影|影片|视频|片段|电视)|是哪部|是来自哪个|出自什么|是什么剧[里的]*|"
    r"哪部[剧电影影片电视剧探案]|女主男主|主角是谁|名场)"
)


def _is_media_locator(query: str) -> bool:
    """query 是否在「定位某段影视内容」：台词/片段/集数/名场面 -> 归 vod（配合 LLM vod）。"""
    q = (query or "").strip()
    if not q:
        return False
    # 明确儿童向：绘本/动漫/动画 → 不属 vod 媒资定位（应 educ/children）
    if re.search(r"绘本|动画片|动漫", q):
        return False
    return bool(_MEDIA_LOCATOR.search(q))


def _is_qa(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    # 明确非 QA：片段/名场面 → vod_fuzzy；学科/课程检索；点播设备词
    if re.search(
        r"片段|cut|那段|桥段|名场面|那场戏|儿歌|"
        r"课程|辅导|考点|刷题|真题|知识点|讲义|备课|作业|期末|测试|质检|精讲|推荐课程|"
        r"切换|快进|音量|关机|静音|暂停|循环", q
    ):
        return False
    # 含播放/放 命令词：除非明确知识问（简介/扮演/…/是谁/何时），否则按起播/换歌
    if re.search(r"播放|放|我要|我给你|给我|打开|继续|换一首|再来|重新", q):
        if not re.search(r"简介|讲了什么|扮演|主演|演员表|票房|奖项|总部|专辑|是谁|哪些|何时|"
                         r"多久|谁唱|谁写|哪年|什么时候|讲述|关于|歌曲名单|治愈系|最好|最火", q):
            return False
    return bool(_QA_HIT.search(q))


# 教育「事实知识问答」判据：无 学段/教材/课程 锚点 → 泛知识 qa（非课程检索）。
# 如 游泳安全知识 / 直角三角形的面积；有锚点（如 小学数学内容/免费初中生物课程）则是教育课程搜索。
_EDU_NO_ANCHOR_QA = re.compile(
    r"(安全知识|面积|定律|原理|常识|是怎么回事|是什么|为什么|关于.{0,6}(?:的)?(?:知识|作用|联系))"
)
_EDU_ANCHOR = re.compile(
    r"(小学|初中|高中|一年级|二年级|三年级|四年级|五年级|六年级|初一|初二|初三|高一|高二|高三|"
    r"年级|教材|同步|辅导|考点|冲刺|课程|课后|课外|期中|期末|"
    r"语文|数学|英语|物理|化学|生物|地理|历史|政治|道法|奥数|刷题|真题|讲义|习题|知识点)"
)


def _is_edu_no_anchor_qa(query: str, llm_domain: str) -> bool:
    """教育域事实知识问答：query 带教育知识信号但无学段/学科/课程锚点 → qa。"""
    if llm_domain != "education":
        return False
    q = (query or "").strip()
    if not q:
        return False
    if _EDU_ANCHOR.search(q):
        return False      # 有学段/教材/课程锚点 → 教育课程检索
    return bool(_EDU_NO_ANCHOR_QA.search(q))


# ---------------------------------------------------------------------------
# 息屏(tvMode=6)分诊链 —— 与 query_pipeline.md 上游优先级对齐，不命中交 LLM 兜底。
# 关键点：同一 query 在息屏走"听/点/问"而非亮屏的"播片/搜课程"，故判别词独立于亮屏。
# 优先级：设备播控 → 有声剧 → 点歌/儿歌 → 泛知识问答 → LLM 兜底。
# ---------------------------------------------------------------------------

# 设备播控：暂停/下一首/快进/继续/音量/关机/静音/循环/快退/定时…… 息屏不变，仍 device
_OFF_DEVICE = re.compile(
    r"(暂停|播放下一|下一首|下一集|上一首|上一集|快进|快退|倒|退|"
    r"音(量|大|小|高|低)|静音|关机|开机|循环|单曲|洗掉|定个|定时|分钟后|h后|小时后又|"
    r"分钟|秒|不好听|没听完|换一|继续|重放|接着|放一遍|往后)"
)
# 有声剧：息屏放片子 → 听有声。片名词表复用亮屏 vod 具名剧（有声52-61）
_OFF_AUDIO_VERB = re.compile(r"(播放|直接播|给我放|帮我播放|放|听|打开|播)")
_OFF_AUDIO_TITLE = re.compile(
    r"(庆余年|三体|偷偷藏不住|风起洛阳|斗罗大陆|西游记|觉醒年代|三国演义|雪中悍刀行|"
    r"流浪地球|长津湖|流浪地球|爱情公寓|武林外传|甄嬛传|亮剑|琅琊榜|隐秘的角落|漫长的季节|三生三世|楚乔传)"
)
# 点歌/儿歌：玩∥儿歌|X的歌|听X+ 歌名歌单 → music。歌名言表兜歌曲同名
_OFF_MUSIC_VERB = re.compile(r"(播放|放|听|要听|来一首|点一首|来首歌|唱|点歌)")
_OFF_MUSIC_HINT = re.compile(r"(儿歌|主题曲|片头曲|插曲|歌单|曲库|听歌|点唱)")
_OFF_SONG = re.compile(
    r"(虫儿飞|送你一朵小红花|告白气球|大鱼|情深深雨濛濛|New[ ]?jeans|七里香|1989|"
    r"宝贝去哪儿|小苹果|绿光|童话|白月光|天黑黑|遇见|指纹|晴天|倒带|奶酪陷阱|两只老虎|"
    r"小兔子乖乖|拔萝卜|白龙马|巴士|蚕之羽|梁祝|送别|小跳蛙|葫芦娃|采蘑菇|数鸭子|折纸飞机|"
    r"一二三|小星星|好朋友|不敢听|贝乐虎|小兔子)"
)
# 知识问答(息屏把 片段/学科 也归 qa)：疑问/信息求助
_OFF_QUESTION_QA = re.compile(
    r"(是谁|谁是|谁在|谁唱|谁写|扮演|主演|演员表|简介|短评|票房|获奖|金鸡奖|奖项|"
    r"哪|哪一|哪部|哪年|何时|什么时候|哪几|哪些|哪个|哪只|"
    r"多少|几只|几个|几岁|为什么|为何|"
    r"改编|写了|改编|翻唱|作曲|作词|填词|收录|所属|讲|讲述了|关于|"
    r"总部|在哪里|有歌|的好听|好听|治愈|适合.{0,3}听.{0,3}歌|推荐|最好|最火|炒|比较|唱的歌|"
    r"知识点|考点|辅导|课程|内容|课外|习题|面积|定律|作用|专项|小学生|初中生|高中生|"
    r"那?个片段|片段|cut|那段|这段|桥段|名场面|那场戏|片尾|彩蛋|高能|"
    r"多大了|是什么|为什么|在哪|上映|所在|属于|怎么样|讲了什么|推荐什么|有什么)"
)


# 息屏 children：儿童向 IP/角色词分诊（亮屏 children 角色 + 新增长尾词上下背诵）。
# 置在 点歌/儿歌 之后、泛知识问答 之前：儿童内容告知播放（educ_search）优先于泛问答。
_OFF_CHILDREN = re.compile(
    r"(汪汪队|海绵宝宝|小羊肖恩|佩[琪奇]|光头强|熊出没|喜羊羊|灰太狼|彼得兔|海底小纵队|"
    r"罗小黑|道奇|尼克狐尼克|奥特曼|迪迦|奈克瑟斯|芭比|大耳兔|超级宝贝|JOJO|jojo|"
    r"汽车世界|米小圈|萌鸡|挖掘机|消防车|托马斯|螺丝钉|忍者|迷你特工队|植物大战僵尸|"
    r"愤怒的小鸟|胡巴|宇宙护卫队|孩子|儿童|宝宝|幼儿|绘本|启蒙|儿歌|动画|卡通|动漫|"
    r"小朋友|早教|幼儿园|童话|睡前故事|小公主|小王子|白雪公主|大耳朵图图|罗小黑|奶龙)"
)


def _off_routing(q: str, llm_domain: str) -> str:
    """息屏分诊链（优先级）：设备播控 → 有声剧 → 点歌/儿歌 → 泛知识问答 → 兜底 LLM。

    规则只在**高置信信号**上落地，歧义一律交 LLM 兜底（用户明确要求"没命中走 LLM 兜底"）。
    点歌/儿歌优先于泛知识：因为"儿童歌/告白气球/New-jeans"这类具体歌名点唱意图明确，不能被
    疑问词抢走；反之真疑问句（哪些/谁/怎么/适合…但没有点名歌）走 qa。
    """
    if not q:
        _det_note("off:empty->llm")
        return llm_domain or "qa"
    # 1. 设备播控（暂停/环绕/快退/音量/定时/关机）——息屏第一优先级，避免"播放"误入
    if _OFF_DEVICE.search(q) and not _OFF_AUDIO_TITLE.search(q):
        _det_note("off:device")
        return "device"
    # 2. 有声剧：播放动词 + 明确剧名 → audio（息屏不播片，改听有声）
    if _OFF_AUDIO_VERB.search(q) and _OFF_AUDIO_TITLE.search(q):
        _det_note("off:audio_drama")
        return "audio"
    # 3. 点歌/儿歌（强信号）——
    #    a) 儿歌/歌单/主题曲 歌单意图 → music
    #    b) 播放动词 + 歌名/听的歌 → music（播放命令说话）
    #    c) 裸歌名且无误问词 → music（具体点名播放）
    if _OFF_MUSIC_HINT.search(q):
        _det_note("off:music_hint")
        return "music"
    if _OFF_MUSIC_VERB.search(q) and _OFF_SONG.search(q):
        _det_note("off:music_verb_song")
        return "music"
    if re.search(r"(播放|播放|放|点|来|唱|直接放)", q) and re.search(r"(听歌|听的?(?:.{0,4})?歌|歌曲|儿歌|宝宝听)", q):
        _det_note("off:music_play_listen")
        return "music"
    if _OFF_SONG.search(q) and not _OFF_QUESTION_QA.search(q):
        _det_note("off:music_bare_song")
        return "music"
    # 3b. children 儿童向 IP/角色 → children（息屏不播片，也归儿童内容 educ_search）。
    #     置于点歌/儿歌之后，避免"儿童角色"的歌被 music 抢（真正的点唱已在上面 music 返回）。
    #     放 qa 之前：儿童内容播放意图明确，优先于泛知识问答。
    if _OFF_CHILDREN.search(q):
        _det_note("off:children")
        return "children"
    # 4. 泛知识问答：强问词/推荐/信息求助 → qa（含息屏把 片段/学科/歌曲信息 归 qa）
    if _OFF_QUESTION_QA.search(q):
        _det_note("off:qa")
        return "qa"
    # 5. 教育域学科查询在息屏归 qa（off golden 全教育行为 fan_knowledge_agent）。
    #    无播放/点歌信号，纯"内容/知识/科目"查询 → 泛知识问答。
    if llm_domain == "education" and not re.search(r"(播放|放|听|点)", q):
        _det_note("off:edu_qa")
        return "qa"
    # 6. 兜底：交 LLM 判定（息屏 LLM 已按屏幕态给对 domain）
    _det_note("off:llm_fallback")
    return llm_domain or "qa"


_TRACE: dict = {}


def _det_note(x):
    """审计：记录本 query 命中的判定路径。默认空操作，审计工具注入后收集，不改判定。"""
    _TRACE["step"] = x


def _det_note_ctx(x, rule_id):
    """审计：记录路径 + 命中的信号规则 id。"""
    _TRACE["step"] = x
    if rule_id:
        _TRACE["rule"] = rule_id


def detect_domain(query: str, llm_domain: str, tv_mode: str | int = "0") -> str:
    """后处理 LLM 判的 domain。真主链：detect 判定由 _DETECT_RULES(DetectRuleSet) 按 priority 驱动。

    每个 return 分支都是一个带 id 的 Rule（badcase/息屏/各保护节点/高置信信号/LLM 兜底）。
    DetectRuleSet.select() 是唯一判定入口 —— 审计工具打印的 id 即真正决定 domain 的规则。
    """
    q = (query or "").strip()
    if not q:
        _det_note("empty->llm_domain")
        return llm_domain
    domain, rule = _DETECT_RULES.select(q, llm_domain, str(tv_mode))
    if rule is not None:
        _det_note_ctx(rule.id, rule.id if rule.id.startswith("signal_") else None)
    return domain if domain is not None else (llm_domain if llm_domain else "qa")


# ---------------------------------------------------------------------------
# _DETECT_RULES：判域主链（真 RuleSet）。priority 顺序严格等价原 if-栈，
# decide 逐个复刻原分支判断（复用同一组谓词），选命中即返回 ⇒ 判定语义零变化。
# ---------------------------------------------------------------------------
try:
    from .detect_rulebase import DetectRuleSet as _DetectRuleSet, Rule as _DRule
except ImportError:  # 被 tools 独立加载时无包上下文
    from detect_rulebase import DetectRuleSet as _DetectRuleSet, Rule as _DRule


def _media_query_decide(q, llm, tv):
    if llm == "vod" and _is_media_query(q):
        dom_kn, _ = _match(q)
        if dom_kn in ("children", "audio", "music"):
            return dom_kn
        return "vod"
    return None


def _audio_discovery(q):
    return bool(re.search(r"(有声书|有声剧|广播剧|音频|评书|听书|有声读物|故事|电台|节目)", q) and
                re.search(r"(哪些|有哪些|是什么|有什么|推荐|推荐。|哪本|哪些本|最火|口碑|在吗|在哪|有没有|搜索|查|找|唱下|听一下)", q))


def _music_discovery(q):
    return bool(re.search(r"(唱|唱的|唱歌|作词|作曲|填词|创作|演唱|演奏|乐曲|歌曲|民谣|粤语)", q)
                and not re.search(r"(榜单|排行榜|热歌榜|热搜榜|最新歌曲|榜)", q)
                and re.search(r"(哪些|有哪些|推荐|有没有|来一首|唱的歌|填词|作词|作曲|演唱|唱一下|听一下|创作)", q))


def _sports_prediction(q):
    return bool(re.search(r"(队|vs|VS|比赛|联赛|欧冠|世界杯|冬奥|冠军|晋级|小组赛|队决赛|决赛|篮球队|足球队|女排|乒乓球|亚运|国家队)", q)
                and re.search(r"(谁能赢|谁能胜|谁能获胜|会夺冠|能否夺冠|能赢|会不会|谁能捧杯|谁赢|大获胜|谁会赢|能取胜|能否出线|进入决赛|拿到冠军|能取得冠军)", q))


def _signal_table_match(q):
    dom, _ = _match(q)
    return dom or None


def _make_detect_rules():
    rules = []

    def _add(rid, prio, title, dec):
        rules.append(_DRule(id=rid, priority=prio, title=title, decide=dec, scope="both"))

    # 依序复刻原 if-栈（priority 越小越先评估，等价原先后顺序）
    _add("off_routing", 1, "息屏分诊链", lambda q, llm, tv: _off_routing(q, llm) if str(tv) == "6" else None)
    _add("badcase", 100, "精确句 badcase", lambda q, llm, tv: _badcases().get(q) or None)
    _add("media_query", 200, "vod 媒资载体保护", lambda q, llm, tv: _media_query_decide(q, llm, tv))
    _add("media_locator", 300, "台词/片段/集数定位", lambda q, llm, tv: "vod" if (llm in ("vod", "qa") and _is_media_locator(q)) else None)
    _add("children_locator", 400, "children 内容定位",
         lambda q, llm, tv: "children" if (_match(q)[0] == "children" and re.search(r"绘本|动画|动漫|卡通|台词|哪部动画|哪个动画", q)) else None)
    _add("audio_discovery", 500, "audio 内容 Discovery", lambda q, llm, tv: "audio" if _audio_discovery(q) else None)
    _add("music_discovery", 600, "music 内容 Discovery", lambda q, llm, tv: "music" if _music_discovery(q) else None)
    _add("sports_prediction", 700, "sports 赛事预测", lambda q, llm, tv: "sports" if _sports_prediction(q) else None)
    _add("qa_open_knowledge", 800, "泛知识开放问答", lambda q, llm, tv: "qa" if _is_qa(q) else None)
    _add("edu_no_anchor_qa", 900, "教育无锚问答→qa", lambda q, llm, tv: "qa" if _is_edu_no_anchor_qa(q, llm) else None)
    _add("signal_match", 1100, "高置信信号", lambda q, llm, tv: _signal_table_match(q))
    _add("llm_domain_keep", 1200, "保 LLM 兜底", lambda q, llm, tv: llm)
    return _DetectRuleSet(rules)


_DETECT_RULES = _make_detect_rules()
