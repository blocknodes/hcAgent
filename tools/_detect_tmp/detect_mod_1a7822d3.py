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

    # children 内容型 + 有声/朗读 载体 —— 两者齐备时是"儿童有声内容"，归 children 而非 audio。
    # 置于 audio 强信号前，避免「有声朗读绘本/想看有声音朗读的双语绘本/睡前听有声故事」被 audio 劫持。
    ("children", r"(?:绘本|儿歌|动画|动漫|卡通|亲子|启蒙|童话|睡前故事).{0,10}(?:有声|朗读|音频|广播|收听)|(?:有声|朗读|有声音).{0,8}(?:绘本|儿歌|动画|动漫|卡通|亲子|启蒙|童话|睡前故事)"),
    # audio 强信号：收听 / 广播 / 有声 / 评书 —— 命中即 audio（含儿童IP场景），放最前
    ("audio", r"(广播剧|广播|有声书|小说|听书|评书|收听|音频|听录音|有声|听音频|有声故事|有声剧|的FM|电台|广播)"),
    # audio 听剧/小说分集：我要听X第N集/趁X第二部 —— 「我要听」明确 audio，放 vod 具名前
    ("audio", r"(?:我要|我想|想|要|请帮我)?听.{0,4}(?:三体|庆余年|偷偷藏不住|风起洛阳|斗罗大陆).{0,6}第[0-9一二三四五六七八九十]+(?:[集部季]|部分)"),
    # audio 通用「听 + 有声剧名 + 第N集/部/回」：听力动词显式标记听书意图，任何剧名都归 audio，
    # 不依赖具名剧名言表。占比 0 误伤其它域；放 audio 具体剧名之后、vod 具名播放之前。
    ("audio", r"(收听|想听|要听|听一下|给我听|请听|想去听|继续听|听听)\D{0,8}(?:第|部|回)?[0-9一二三四五六七八九十百]+(?:集|部|回)"),
    ("audio", r"^\s*听\D{0,8}第[0-9一二三四五六七八九十百]+(?:集|部|回)"),

    # children 儿歌 —— 带少儿语境词（宝宝/幼儿/小孩/岁/亲子/启蒙…）才归 children；
    # 裸「儿歌」（如「有简单的英文儿歌吗」无年龄/人称语境）为点歌曲场景 → music，见下方 music 规则。
    # 注意：裸推断的「儿歌」若不带身份词，不得在此劫持 music 歌曲场景。
    ("children", r"(?:宝宝|幼儿|儿童|小孩子|小孩|小朋友|亲子|启蒙|早教|幼儿园|\d+\s*岁|几岁|我家|哄|摇篮曲).{0,15}?(?:跟唱|唱|听|中文版|英文版)?儿歌|儿歌.{0,6}(?:孩子|宝宝|幼儿|儿童|小朋友|幼儿园|早教)"),
    # children 幼龄听的歌（1岁半宝宝…）—— 宝宝/岁 幼儿语境 优先于 music 歌曲
    ("children", r"(?:\d+岁(?:半)?|宝宝|幼儿|小孩子).{0,4}(?:听|看)的.{0,2}(?:歌|音乐)"),
    # music 裸儿歌（无少儿年龄词，判歌曲播放/点歌曲目）→ music_song_search；置于 children 儿歌之后、
    # 更靠前的 children 泛词 之前，避免「英文儿歌」被泛 children 词表劫持。
    ("music", r"^\s*(?:有简单的|有没有|来|唱|听|放|点|找|来点|来一|来首|有)?\s*(?:英文|中文|中英文|语|歌)?(?:儿歌|童谣)(?:吗|\?|的)?\s*$"),

    # children 动漫/动画明确标签：具名IP+动漫 = 少儿内容（educ_search），放 vod 具名前
    ("children", r"(?:斗罗大陆|汪汪队|三体).{0,2}(?:动漫|动画|卡通)(?:版|片|影视剧)?"),

    # vod 影视具名播放：播放 golden 统一 vod_search。命中已知剧/片名 + 播放动词即 vod。
    # 名词覆盖 sheet065 亮屏 vod 播放（面具52-59/音乐68-70），避免误伤 music 真歌名。
    ("vod", r"(?:播放|直接播|给我放|帮我播放|放|看|点播|打开|播).{0,6}(?:庆余年|三体|偷偷藏不住|风起洛阳|斗罗大陆|西游记|觉醒年代|三国演义|雪中悍刀行|虫儿飞|送你一朵小红花|情深深雨濛濛)"),
    ("vod", r"(?:庆余年|三体|偷偷藏不住|风起洛阳|斗罗大陆|西游记|觉醒年代|三国演义|雪中悍刀行|虫儿飞|送你一朵小红花|情深深雨濛濛).{0,6}(?:第[0-9一二三四五六七八九十]+[集部]|大结局)"),
    # vod 影视具名片名（无播放动词，亮屏 golden 也视作 vod_search）：裸剧/片名词
    ("vod", r"^(?:播放)?(?:庆余年|三体|偷偷藏不住|风起洛阳|斗罗大陆|觉醒年代|三国演义|雪中悍刀行|送你一朵小红花)$"),

    # children 睡前内容 —— 睡前+儿童内容词（绘本/儿歌/动画/亲子/启蒙/故事）→ 少儿，
    # 须在裸「睡前」audio 强信号之前：纯「睡前听书」仍归 audio，带儿童内容词则归 children。
    ("children", r"(睡前|晚安|助眠).{0,12}(绘本|儿歌|动画|动漫|卡通|亲子|启蒙|童话|故事|幼儿|宝宝|孩子|儿童|小朋[友]|婴幼儿|托儿)"),
    # audio 睡前/助眠/听单 —— 弱信号（不含「适合..听」，避免误伤 music「适合听的歌」）
    ("audio", r"(睡前|晚安|助眠|听单|我昨晚听)"),

    # music 歌曲词（.的歌/歌单/主题曲…）——全 music 专属，放 children IP 前
    ("music", r"(的歌|歌版|全歌|主题曲|片头曲|插曲|歌单|K歌|k歌|点歌|唱歌|要唱|想唱|歌手|歌词|MV|mv|曲库|音乐家|播放列表)"),
    ("music", r"(历史播放|听歌历史|历史的歌|历史里的|播放历史)"),

    # children 专属 IP / 剧名 —— 明确角色，最不易误伤
    ("children", r"(汪汪队|海绵宝宝|小羊肖恩|佩[琪奇]|光头强|熊出没|熊大|熊二|开心锤锤|喜羊羊|灰太狼|彼得兔|海底小纵队|小马宝[莉丽]|爆笑虫子|罗小黑|龙猫|小公主|聪明一休|白雪公主|丑小鸭|大耳朵图图|超级宝贝|JOJO|jojo|汽车世界|米小圈|萌鸡|挖掘机|消防车|托马斯|螺丝钉|鬼灭之刃|忍者|迷你特工队|熊猫姐姐|沃克|细胞|动物神探队|恐龙世界|植物大战僵尸|愤怒的小鸟|胡巴|超级土豆|万达坏蛋联盟|坏蛋联盟|猪屁登|超人兽战|宇宙护卫队|奶龙|快乐星猫|小魔仙|钶龙战记|爆裂飞车|弹珠轨道|猪猪侠|超级飞侠|汽车星球|巴巴爸爸|开心飞车|飞天小女警|娜斯佳|妈妈咪鸭|波妞|磁力拼搭|阿拉丁|小猴子|长征先锋|趣趣知知鸟|鲨鱼一家|非人哉|苏菲亚|吞噬星空|兔子警官|土豆逗|星卡梦少女|大神侦探|神奇宝物|小砾工程队|磁力小车|布鲁伊|阿巳|小铃铛|沃福|依娜|恰恰|小伶玩具|监狱兔|小品一家)"),
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
    # education 课程/技能词（教程/教学/训练/书法/绘画…）——education 专属标志，放 children 前
    # 含 编程/编程课/少儿编程/口语/零基础/试听/实战：课程型用语优先于 children 的泛「少儿」，
    # 否则「少儿编程课」会被 detect.py:125 的 children 泛词抢判成 children（golden 实为 education）。
    ("education", r"(教程|教学|训练|书法|绘画|彩铅|画画|手账|预习|辅导|培训|编程|编程课|口语训练|英语口语|零基础|试听课|少儿编程)"),

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


# 多意图内容子句域路由：规则优先于 LLM。这些格式信号可靠（裸实体 detect 判空的兜底），
# 用于 engine 把"内容+设备"并行拆出的内容子句，在 detect 对裸实体（队名/歌名/剧名/书名）
# 判空时的确定性域兜底；未命中再交 LLM 从用户原话选域。
# 单一事实源在此 —— engine.py 只引用 content_domain_rule()，不再各自内嵌正则。
_CONTENT_DOMAIN_RULES = [
    (r"动画|卡通|动漫|少儿|儿童|宝宝|幼儿|佩奇|汪汪队|熊出没|奥特曼|海绵宝宝|小恐龙|弹珠轨道|大货车|机器人", "children"),
    (r"有声|广播剧|评书|听书|音频|收听|(?:听|想听|要听|听听).{0,10}(?:书|剧|集|故事)", "audio"),
    (r"歌曲|专辑|单曲|MV|唱歌|点歌|歌单|音乐", "music"),
    (r"台词|片段|片单|剧|电影|电视剧|影片|纪录片|影视|影院|哪部|那个片段|这部|看剧", "vod"),
    (r"比赛|赛事|比分|球队|队|联赛|对战|世界杯|球|预约.{0,6}(比赛|球队)", "sports"),
    (r"年级|课本|上册|下册|语文|数学|英语|物理|化学|生物|作文|课文|课程|启蒙", "education"),
]


def content_domain_rule(query: str) -> str:
    """裸实体内容子句的确定性域路由：命中即域，未命中返回 "". """
    for pat, dom in _CONTENT_DOMAIN_RULES:
        if re.search(pat, query):
            return dom
    return ""


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
    # 明确非 QA：片段/名场面 → vod_fuzzy；课程检索（含 古诗/课文 等学科内容）；点播设备词
    if re.search(
        r"片段|cut|那段|桥段|名场面|那场戏|儿歌|"
        r"课程|辅导|考点|刷题|真题|知识点|讲义|备课|作业|期末|测试|质检|精讲|推荐课程|"
        r"古诗|唐诗|宋词|诗词|课文|朗读|按年级|"
        r"切换|快进|音量|关机|静音|暂停|循环", q
    ):
        return False
    # 含播放/放 命令词：除非明确知识问（简介/扮演/…/是谁/何时），否则按起播/换歌
    if re.search(r"播放|放|我要|我给你|给我|打开|继续|换一首|再来|重新", q):
        if not re.search(r"简介|讲了什么|扮演|主演|演员表|票房|奖项|总部|专辑|是谁|哪些|何时|"
                         r"多久|谁唱|谁写|哪年|什么时候|讲述|关于|歌曲名单|治愈系|最好|最火", q):
            return False
    return bool(_QA_HIT.search(q))


# 纯问候/闲聊：只命中礼貌招呼语，且句子不含任何内容/播放/检索意向。
# 用作 qa 兜底前的最后一道空域收口 —— 前面所有内容型规则都没命中时，"hello/你好/在吗"
# 这类无实质内容的话才落到泛知识 qa（而非随 LLM 抖动返回空域）。带具体内容则自然被上方
# 各内容规则拦截，不会误伤。刻意不含"谢谢/再见"等结束语（非咨询问句）。
_GREETING = re.compile(
    r"^(?:\s*)(?:(?:hello|hi|hey|嗨|哈喽|哈啰|哈鲁|你好呀|您好呀|你好|您好|"
    r"早上好|中午好|下午好|晚上好|早安|在吗|在不|在么|在嘛|在不在|"
    r"你好啊|您好啊|在吗啊|在不在啊|"
    r"喂|哥们|朋友|请问|请问一下|打扰一下|有在吗|在不在啊)[\s,，!?。~～]*){1,3}"
    r"[\s!?。~～]*$",
    re.IGNORECASE,
)


def _is_greeting(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    # 带实质内容一律不当作纯问候
    if re.search(r"播放|点|听|唱|搜|查|介绍|推荐|哪个|哪些|是什|怎么样|来自|出自|看|讲|读",
                 q) and not re.fullmatch(_GREETING, q):
        return False
    return bool(_GREETING.fullmatch(q))


# 教育「事实定义」判据：无 学段/教材/课程 锚点 → 泛知识 qa（非课程检索）。
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
    # 3b．儿童IP的**知识问答**（谁是队长/多大/多少只/讲了什么）归 qa，而非 children 看片。
    #     先于普通 children(3c)：children 是"播放/看"意图，而这里有明确疑问词 → 只答不播。
    if _OFF_CHILDREN.search(q) and _OFF_QUESTION_QA.search(q):
        _det_note("off:children_qa")
        return "qa"
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


def detect_domain(query: str, llm_domain: str, tv_mode: str | int = "0",
                  segment: dict | None = None) -> str:
    """后处理 LLM 判的 domain。真主链：detect 判定由 _DETECT_RULES(DetectRuleSet) 按 priority 驱动。

    segment：ASR 分词元数据（data.segment）。高权重标 tvPlay（点播视频）时，
    对「X歌/儿的歌」这类含歌字但实为剧名/节目的弱 music 信号做**分词裁决**——
    ASR 明确它是点播视频却因裸「歌」字误判 music，由分词证据改回 vod。
    尽量在链尾做，不破坏已有高置信 music 判断（真点歌仍走 music）。

    仅 return 分支是带 id 的 Rule（skill/息屏/各保护层/高置信信号/LLM 兜底）。
    DetectRuleSet.select() 是唯一判定入口 —— 审计工具打印的 rule 即真正决定 domain 的规则。
    """
    q = (query or "").strip()
    if not q:
        _det_note("empty->llm_domain")
        return llm_domain
    domain, rule = _DETECT_RULES.select(q, llm_domain, str(tv_mode))
    if rule is not None:
        _det_note_ctx(rule.id, rule.id if rule.id.startswith("signal_") else None)
        # 分词裁决：当规则链被判 music/audio 且 ASR 强标 tvPlay(点播视频) 时改判 vod。
        # 场景：query 含歧义子串（歌/小说/音乐）被 music/audio 抢判，但 ASR 分词明确
        # 它是【点播视频/电影/剧】（tvPlay 权重 >=0.5）。真点歌(唱/听)/真听书(朗读/评书/
        # 广播剧)已在 _segment_vod_ruling 内屏蔽，不会误伤。
        if domain in ("music", "audio") and _segment_vod_ruling(segment, q):
            _det_note_ctx("segment_tvplay->vod", "segment_tvplay_vod")
            return "vod"
    return domain if domain is not None else (llm_domain if llm_domain else "qa")


def _segment_vod_ruling(segment: dict | None, q: str) -> bool:
    """ASR 强标「点播视频」(tvPlay attr >=0.5) + 弱「歌/小说/音乐」内容词 → 改判 vod。

    场景：query 里含歧义子串（歌/音乐/小说/听），可能被 music/audio 规则误判，
    但 ASR 分词明确它是【视频/电视/电影】时，应回到 vod。
    - 命中 tvPlay 权重 >=0.5 才裁决（低权重不可靠）。
    - 触发条件= query 含「歌」或含「小说/说/播报/广播/电台」之一（歧义子串）。
    - 屏蔽：query 带“听” + 音频载体（听书/听说/朗读/评书/广播剧/有声…）→ 真 audio，
      不裁决。query 带“歌”的真点歌动词也屏蔽（见下）。

    安全阀：
      - 点歌/唱歌/K歌/歌词/歌单/MV/主题曲/… → 明确 music，不裁决。
      - “听/朗读/广播剧/评书/有声书/听写” → 真 audio，不裁决。
      - 其余（含歧义子串但无强听/唱意图）→ 裁决 vod。
    """
    if not segment or not isinstance(segment, dict):
        return False
    wl = segment.get("wordList") or []
    if not wl:
        return False
    has_tv = False
    for w in wl:
        if not isinstance(w, dict):
            continue
        at = w.get("wordAttrs") or []
        wts = w.get("attrWeights") or {}
        for attr in at:
            if attr != "tvPlay":
                continue
            wt = wts.get(attr) if isinstance(wts, dict) else None
            try:
                wt = float(wt) if wt is not None else 0.0
            except (TypeError, ValueError):
                wt = 0.0
            if wt < 0.5:
                continue
            has_tv = True
            break
    if not has_tv:
        return False
    # 三段判决策：
    # 级联裁决（顺序即优先级）：
    # 1) 明确「影视动作」：看电影/看剧/追剧/刷剧/看影视/点播 → vod（比载体词优先，
    #    即使片名带 小说/歌/曲 也反转；用户 bug："嗯我要看电影刺杀小说家" ASR tvPlay → vod）。
    if re.search(
        r"(看电影|看电视剧|看剧|刷剧|追剧|看影视|看片|看大片|看影院|点播)",
        q):
        return True
    # 2) 音频/音乐动作 + 音频载体：听/播放/放/点/唱/读/循环/学习/跳… + 书/小说/节目/歌/曲/第N集/历史/大结局。
    #    → 真 听书/听歌/收音，不反转（"收听斗罗大陆大结局那一集" 是听书）。
    if re.search(
        r"(听|收听|放|播放|点|唱|朗读|跟读|学习|循环|读|跳|切到|切|来一首)",
        q) and re.search(
        r"(听书|有声书|朗读|评书|广播剧|有声|有声音|广播|电台|FM|fm|"
        r"书|小说|故事|相声|节目|专辑|音频|演唱会|曲|歌|歌曲|音乐|片尾曲|主题曲|原声带|"
        r"回|章节|连载|历史|播放记录|听过的|上次|循环过|记录|原声|大结局|"
        r"第[一二三四五六七八九十\d]+[回集章节部])",
        q):
        return False
    # 3) 影视名词（电影/电视剧/剧集/影视/影片/电影版/大结局/续集/全集/完整版）→ vod。
    if re.search(
        r"(电影|电视剧|剧集|连续剧|影片|影院|大片|剧场版|电影版|完整版|"
        r"上映|上线|影视|片名|彩蛋|预告|花絮|正片|大结局|续集|全集)",
        q):
        return True
    # 4) 默认不动（tv 弱信号不足）。
    return False
    # 3) 其余无强意图 → 默认不动（tv 弱信号不足）。
    return False
    # 2) 明确 vod 意图（电影/电视剧/剧集/续集/点播/追剧/影院/影视片…）：
    #    ASR 强标 tvPlay + 明确影视动词 → 必是视频，即使片名带 歌/曲 字也翻转。
    #    （用户原 bug："嗯我要看电影刺杀小说家" ASR tvPlay, llm 误判 audio —— 看电影 是 vod 锚。）
    if re.search(
        r"(看电影|看电视剧|看剧|刷剧|追剧|看影视|电影|电视剧|剧集|连续剧|"
        r"影片|影院|点播|大片|剧场版|电影版|大结局|续集|完整版|"
        r"上映|上线|影视|片名|看片|彩蛋|预告|花絮|正片)",
        q):
        return True
    # 3) 其余无强意图 → 默认不翻转（tvPlay 弱信号不足改判）。
    return False


# ---------------------------------------------------------------------------
# _DETECT_RULES：判域主链（真 RuleSet）。priority 顺序严格等价原 if-栈，
# decide 逐个复刻原分支判断（复用同一组谓词），选命中即返回 ⇒ 判定语义零变化。
# ---------------------------------------------------------------------------
try:
    from .detect_rulebase import DetectRuleSet as _DetectRuleSet, Rule as _DRule
except ImportError:  # 被 tools 独立加载时无包上下文
    from detect_rulebase import DetectRuleSet as _DetectRuleSet, Rule as _DRule


def _media_query_decide(q, llm, tv, segment=None):
    if llm == "vod" and _is_media_query(q):
        dom_kn, _ = _match(q)
        # 电影名含"小说/听"等字但 ASR 强标 tvPlay(点播视频) → 是影视剧，不是听书
        if dom_kn in ("audio", "music") and _segment_vod_ruling(segment, q):
            return "vod"
        if dom_kn in ("children", "audio", "music"):
            return dom_kn
        return "vod"
    return None


def _audio_discovery(q):
    return bool(re.search(r"(有声书|有声剧|广播剧|音频|评书|听书|有声读物|故事|电台|音频|听书|故事)", q) and
                re.search(r"(哪些|有哪些|是什么|有什么|推荐|推荐。|哪本|哪些本|最火|口碑|在吗|在哪里|有没有存在|搜索|查|找|唱下|听一下)", q))


# 具名影视 + 「讲的是什么故事/剧情」 → 剧情知识问答，归 qa(fan_knowledge_agent)。
# 串行 step2 常是"对检索结果提问剧情"（如《螺疯狂的车》讲的是什么故事）；
# `_audio_discovery` 会因「讲…故事」把这类误判 audio 有声内容。此谓词锚定「具名片名/影视词
# ＋ 讲…故事/剧情」的剧情问答，在 audio_discovery 之前强判 qa，不误伤真的"推荐有声音故事"。
# 是否"具名影视"：有《》书名号、或片名紧跟“讲的是”前导字，或带 电影/电视剧/影片 载体词。
_PLOT_QA_BLOCK = re.compile(r"(?<!析)播放(?!量|的最|的一部)|放(?!量)|点播|听书|尤其是|想听|要听")
# 串行 step2 剧情问答的「某部指代」：评分最高/最经典/人气最高/最新/这一部/第一部… 的一部/影片。
# 这类无具名载体(无《》/电影)，但对上一步检索结果的某一部提问剧情，同样归 qa 而非 audio。
_PLOT_QA_ORD = re.compile(r"评分最高|最经典|人气最高|播放量最高|最新|最热|这一(部|影片)|那(一?部|影片)|\d+部|第一部|其中一部|上映的|拍的|相关的")


def _is_plot_qa(query: str) -> bool:
    """具名有 + 剧情问句 → qa（覆盖 audio_discovery 的「故事」误判）。"""
    q = (query or "").strip()
    if not q:
        return False
    if _PLOT_QA_BLOCK.search(q):
        return False
    if not re.search(r"(?:讲的是|讲什么|讲的啥|什么故事|什么样的故事|什么剧情|剧情)", q):
        return False
    # 具名影视载体：书名号、或影视载体词 + “X讲…”的（电影/电视剧/影片等），
    # 或「第一/最高…一部」式指代（串行 step2 对上一步检索结果提问剧情）。
    if not (re.search(r"《[^》]{1,20}》|电影|电视剧|影片|番剧|纪录片|动画片|影视", q)):
        if not _PLOT_QA_ORD.search(q):
            return False
    return True


# audio 听书补丁：用户「听/播放 + 有声内容载体」→ audio。优先级高于 children。
# 解决 golden 把儿童有声(睡前故事/童话/国学启蒙/绘本音频/广播剧)归 audio，而 prompt/children
# 规则误归 children/education 的漏判。guard：必须同时有「收听/播放」动作 + 有声载体词，
# 且出现「看视频/剧」语境仍回 vod，避免误伤真 vod 播放。
_AUDIO_LISTEN_VERB = re.compile(r"听|收听|播放|放|点|播|读")
_AUDIO_CARRY = re.compile(
    r"有声书|有声剧|有声故事|有声读物|有声小说|有声版|广播剧|评书|听书|睡前故事|睡前|听故事|讲故事|音频|朗读|录音|原著|连载|完结|追到"
)
_AUDIO_VIDEO_BLOCK = re.compile(r"电影|电视剧|剧场|一部.{0,6}影片|看视频|影院|剧(?:集|场|片|版|影|乐)|影视")
def _audio_listen_carry_patch(q, llm, tv):
    """听/播放 + 有声载体 → audio（覆盖 children 误判听书）。"""
    if str(tv) == "6":
        return None                      # 息屏分诊链已处理
    if not _AUDIO_LISTEN_VERB.search(q):
        return None
    if not _AUDIO_CARRY.search(q):
        return None
    if _AUDIO_VIDEO_BLOCK.search(q):
        return None
    return "audio"


_MUSIC_SKIP_CHILDREN = re.compile(
    r"(?:岁|宝宝|幼儿|小孩子|小孩|小朋友|亲子|启蒙|早教|幼儿园|儿童)"
    r"(?:.{0,15}?)?(?:跟唱|唱|听的)?儿歌"
)


def _music_discovery(q):
    # 排除「年龄/少儿语境 + 儿歌」的儿童场景（golden 归 education/children），
    # 防止「跟唱/演唱」+「儿歌」被误判成 music 歌曲点播。
    if _MUSIC_SKIP_CHILDREN.search(q):
        return False
    return bool(re.search(r"(唱|唱的|唱歌|作词|作曲|填词|创作|演唱|演奏|乐曲|歌曲|民谣|粤语)", q)
                and not re.search(r"(榜单|排行榜|热歌榜|热搜榜|最新歌曲|榜)", q)
                and re.search(r"(哪些|有哪些|推荐|有没有|来一首|唱的歌|填词|作词|作曲|演唱|唱一下|听一下|创作)", q))


# ---------------------------------------------------------------------------
# music_override：高置信歌曲/音乐频道强信号 → music（亮屏）。
# 解决 LLM 对具体歌曲/音乐频道/歌名引用判错域（vod/qa/audio/children）导致 tool 误路由。
# 只在高置信时覆盖：歌名/歌手/播放切歌/音乐频道/CCTV-15 等强锚 + 排除词（少儿IP/影视/有声）。
# 位置在 badcase(100) 之后、media_knowledge_qa(150) 之前插值评估。
# ---------------------------------------------------------------------------
_MUSIC_OVER_SONG_STR = (
    "500年桑田沧海 I Knew You Were Trouble. super star 一路向北 一闪一闪亮晶晶 七里香 上山岗 "
    "世上只有妈妈好 今生最爱 你的眼神 光年之外 公主请开心 兰亭序麒麟 再遇梨花颂 凤凰花开的路口 "
    "十送红军 南山南 卷席筒 原谅我年轻不懂爱 吻别 告白气球 土坡上狗尾巴草 土坡上的狗尾草 大花轿 "
    "大风吹 大香蕉 大鱼 天地龙鳞 女儿殿下 好汉歌 好运来 如愿 孤勇者 富士山下 小手拍拍 小跳蛙 "
    "小鸡小鸡 平凡之路 年轮 情人慢慢 慢慢 战火燃烧 拔萝卜 新不了情 新年快乐 晴天 最真的梦 "
    "唱不来都等你 梁祝 梅花三弄 欧若拉 江南 江南STYLE 沂蒙山小调 没有共产党就没有新中国 消愁 "
    "潮湿的心 爱我你就抱抱我 猪猪侠 画你 留什么给你 疼爱妈妈 白毛女 相思 离别开出花 稻香 "
    "穆桂英下山 童心向党 红山果 背对背拥抱 野狼disco 自由飞翔 花园种花 萱草花 落花 讲不出再见 "
    "赤伶 起风了 追光者 梦里 雪龙吟 青花瓷 谁 梦里水乡 荷塘月色 小苹果 鲁冰花 "
    "蓝莲花 夜空中最亮的星"
)
_MUSIC_OVER_SONGS = sorted(
    {s.strip().lower() for s in _MUSIC_OVER_SONG_STR.split(" ") if len(s.strip()) >= 2},
    key=len, reverse=True,
)
_MUSIC_OVER_BLOCK = re.compile(
    r"广播剧|如有声书|听书|小说|广播|演讲稿|绘本|故事会|安徒生|格林童话|"
    r"动画片|动漫|卡通|动画电影|电影|电视剧|剧集|纪录片|纪录片|大电影|"
    r"宝宝|幼儿|儿童|小朋友|幼儿园|亲子|睡前故事|小猪|佩奇|汪汪|变身|萌宠|"
    r"普通话|粤语|中文字幕|国语|原声|第\d+[集部季]|续集|全集|第[一二三四五六七八九十]+[集部季]|本体"
)
_MUSIC_OVER_VERB = re.compile(
    r"播放|放|听|唱|点|来一|再来一|来首|换|切|再换|下一首|上一首|首歌|给我|想|要唱|唱|点歌|K歌|k歌|循环|白噪音|MV|mv"
)
_MUSIC_OVER_TV = re.compile(
    r"[Cc][Cc][Tt][Vv]|央视\s*15|中央\s?\d+台|中央卫视|音乐台|音乐频道|音乐广播|音乐电台|梨园春|戏曲频道|"
    r"哪个台|哪个频道|1[56]台|六台"
)
_MUSIC_OVER_TV_MUSIC = re.compile(
    r"二胡|民谣|轻音乐|古典乐|乐曲|戏曲|梨园|华语|流行|榜单|流行榜|名曲|舞曲|老歌|"
    r"经典|观天|草原|奥斯卡|电影原声|原声带|OST|音乐|曲目|好歌|天"
)
_MUSIC_OVER_TV_BLOCK = re.compile(r"宝宝|儿童|少儿|幼|卡通|动画|动漫|乐乐|棒|跑男|同同|尼克|童话")
_MUSIC_OVER_HIST = re.compile(
    r"播放记录|历史|听过的|循环过|喜欢听|点过|听过|上次播放|上一次|播放列表|没听完|历史播放|上一首听"
)
_MUSIC_OVER_AUDIO = re.compile(
    r"音频|宽带兖|收听|广播剧|有声|评书|电台|小说|书|节目|故事|演讲|讲座|历史故事|听书|朗读|绘本|评书"
)
_MUSIC_OVER_FAV = re.compile(r"收藏")
_MUSIC_OVER_QQ = re.compile(r"[Qq][Qq]音乐|qq音乐")


def _music_over_has_song(q: str) -> str:
    ql = q.lower()
    for s in _MUSIC_OVER_SONGS:
        if s and s in ql:
            return s
    return ""


def _music_override(q) -> str | None:
    """高置信 music 强信号。命中返回 'music'，否则 None。"""
    q = (q or "").strip()
    if not q:
        return None
    # 裸「单曲」
    if re.fullmatch(r"单曲", q):
        return "music"
    # 白噪音
    if "白噪音" in q:
        return "music"
    # 音乐电视频道：CCTV-15 / 央视15 / 梨园春 / 哪个台+老歌/二胡/名曲/草原天放在了
    if _MUSIC_OVER_TV.search(q) and _MUSIC_OVER_TV_MUSIC.search(q):
        if _MUSIC_OVER_TV_BLOCK.search(q) and not re.search(
            r"老歌|歌曲|音乐|二胡|名曲|戏曲|民谣|天建于|观舞", q
        ):
            return None
        return "music"
    # 歌名 + 播放/切歌动词（排除影视/未成年人语境）
    s = _music_over_has_song(q)
    if s and len(s) >= 2 and _MUSIC_OVER_VERB.search(q) and not _MUSIC_OVER_BLOCK.search(q):
        if re.search(r"动画|动画片|动漫|卡通|电影|电视剧|纪录片|剧场版|剧集", q):
            return None
        return "music"
    # 历史播放/听歌记录（音乐语境）
    if _MUSIC_OVER_HIST.search(q) and not _MUSIC_OVER_AUDIO.search(q)        and not re.search(r"历史|故事|书|广播|音频|朗读|讲解|收听|听书|评书", q):
        return "music"
    # 收藏列表 + 歌/音乐
    if _MUSIC_OVER_FAV.search(q) and re.search(r"歌|音乐|单曲|专辑|华语|经典|列表|曲|收藏", q)        and not _MUSIC_OVER_AUDIO.search(q):
        if re.search(r"动画|幼儿|亲子|绘本|广播|书|故事", q):
            return None
        return "music"
    # QQ 音乐
    if _MUSIC_OVER_QQ.search(q) and not re.search(r"电影|电视剧|动画|纪录片", q):
        return "music"
    # 视频版 / MV / 歌曲视频
    if re.search(r"视频版|MV|mv|歌曲视频|音乐视频|这首歌的视频", q):
        if re.search(r"动画|卡通|动漫|电影|电视剧|纪录片", q):
            return None
        return "music"
    # 裸歌名 + 视频
    sv = _music_over_has_song(q)
    if "视频" in q and sv and len(sv) >= 2 and not _MUSIC_OVER_BLOCK.search(q):
        if re.search(r"动画|卡通|动漫|动画电影|电影|电视剧|纪录片", q):
            return None
        if re.search(r"猪|汪|超级|海底|佩奇|小天才|光头|喜羊|悠悠|多多", q):
            return None
        return "music"
    return None


def _sports_prediction(q):
    return bool(re.search(r"(队|vs|VS|比赛|联赛|欧冠|世界杯|冬奥|冠军|晋级|小组赛|队决赛|决赛|篮球队|足球队|女排|乒乓球|亚运|国家队)", q)
                and re.search(r"(谁能赢|谁能胜|谁能获胜|会夺冠|能否夺冠|能赢|会不会|谁能捧杯|谁赢|大获胜|谁会赢|能取胜|能否出线|进入决赛|拿到冠军|能取得冠军)", q))


# 运动项目名 + 强运动语境词 → sports（纯文本确定性信号，不依赖 ASR 分词）。
# 背景：LLM planner 常把「山地自行车视频/跳水比赛集锦/羽毛球复赛视频」判成 vod，
# detect 原先无运动项目正信号纠正（_RULES 的 sports 词表只含 比赛/球队，不含项目名），
# 这些查询全部落到 llm_domain_keep 保 LLM 的 vod 误判。本信号要求【项目名 + 强语境词】
# 共现才判 sports，实测 7 域 testset：命中 68 条 sports，vod/children/music/education/
# audio/device 零误伤（项目名裸词如「搜索羽毛球」无语境词，不命中，仍交 LLM）。
_SPORT_PROJECT = re.compile(
    r"足球|篮球|网球|排球|乒乓球|乒乓|羽毛球|田径|赛车|拳击|格斗|搏击|台球|高尔夫|"
    r"冰球|电竞|山地自行车|自行车|跳水|跳台滑雪|滑雪|自由体操|体操|击剑|攀岩|马拉松|"
    r"速度轮滑|轮滑|摔跤|钓鱼|射击|举重|冲浪|滑板|斯诺克|飞镖"
)
_SPORT_MEDIA_CTX = re.compile(
    r"视频|录像|集锦|回放|回播|直播|比赛|赛事|联赛|锦标赛|决赛|半决赛|预选赛|复赛|"
    r"精彩|慢动作|瞬间|世界杯|奥运会|奥运|亚运会|亚运|冠军赛|公开赛|大奖赛"
)


def _sports_project_media(q):
    q = (q or "").strip()
    if not q:
        return False
    return bool(_SPORT_PROJECT.search(q) and _SPORT_MEDIA_CTX.search(q))


def _signal_table_match(q):
    dom, _ = _match(q)
    return dom or None


# 亮屏「媒体/明星信息咨询」：pointe 在消息里查"影视/音乐的人物/作品信息"(获奖/主演/改编/简介/翻唱/
# 作曲/票房)，而非请求**播放/点唱某一段**。当 LLM 倾向 vod/music 时，这类纯信息问句应归 qa，
# 否则会被 media_query/music_discovery 抢先判成媒资检索（亮屏判错域，topic 知识问答）。
_MEDIA_KNOW_VERB = re.compile(
    # 只认「明确问信息/索取事实」的问答词，排除"主演/改编/作词/唱…"这类常作媒姿检索框的宽词，
    # 避免把 7 域 testset 里本应按 vod/music 取材的查询误拨成 qa。
    r"演员表|扮演者|是谁|谁演|谁唱|谁写|谁配|配音的是谁|简介|短评|哪部|哪些|哪几|哪一|哪年|哪首|"
    r"什么时候|什么时间|何时|讲了什么|讲了啥|做了什么|多少|几个|几岁|多大|为什么|为何|"
    r"影帝|影后|最佳男主|最佳女主|票房最高|奖项|获奖名单|得了什么奖"
)
_MEDIA_KNOW_BLOCK = re.compile(
    # 媒资定位语境（台词/片段/出处→vod 定位），不是纯知识问答
    r"播放|放|要|点|来一首|唱一下|听一下|切|快进|音量|关机|循环|暂停|给我|帮我|"
    r"出自|来自哪|来自|台词|这句话|这段[话演]|桥段|名场面|片段|镜头|剧情|哪集|哪场|"
    r"哪部(?:剧|片|电影|电视剧|纪录片)(?:里|中)?|打斗|高光|爆燃|小视频"
)


def _media_knowledge_qa(query: str) -> bool:
    """是否为媒体/明星信息咨询(亮屏时段，llm 倾向 vod/music 保护)。"""
    q = (query or "").strip()
    if not q:
        return False
    if _MEDIA_KNOW_BLOCK.search(q):
        return False
    if not _is_qa(q):
        return False
    return bool(_MEDIA_KNOW_VERB.search(q))


# 亮屏「教育事实类强信号」：query 自带明确的学科/定理内容词（如 面积/周长/定律/公式/
# 安全知识），且不因 llm 判成 qa 而丢失教育域。这与 _EDU_NO_ANCHOR_QA 不同——那条以
# llm_domain==”education” 为前提；这里独立于 LLM，把这类裸知识问句在亮屏拉回 education
# （→ education 域，最终工具由 hcTools 教育域规则解析）。只匹配强教育内容词，不碰泛”是什么/为什么”以免误伤。
_EDU_STRONG_TERM = re.compile(r"(直角|三角形|面积|周长|圆周|定律|定理|公式|原理|安全知识|化合价|光合|方程式|化合|分解反应)")
_EDU_STRONG_BLOCK = re.compile(r"播放|放|听|点|切|音量|关机|暂停|有声|听书|广播|录音")


def _edu_bright_strong(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    if _EDU_STRONG_BLOCK.search(q):
        return False
    return bool(_EDU_STRONG_TERM.search(q))


_VOD_AUTEUR_ANIM = re.compile(
    r"宫崎骏|新海诚|大友克洋|细田守|今敏|庵野秀明|押井守|汤浅政明|新川洋司|"
    r"吉卜力|鸟山明|尾田荣一郎|千与千寻|龙猫|哈尔的移动城堡|天空之城|风之谷|红猪|青春猪头"
)


_VOD_DESC_SEARCH = re.compile(
    r"^(?:帮我找一下|帮我找找|帮我搜一下|帮我找|找一下|找找|找一找|找|搜一下|搜|搜索|查一下|查|"
    r"我想找|想找一下|想找|哪个是|哪部是)"
    r"(?:那个|那部|这部|一部|几个|：)?.{0,26}?"
    r"(?:的电影|的动画片|的影片|的电视剧|的那部动画|的动漫电影|的动画电影|动画片|动画|"
    r"电影|影片|电视剧|动漫|剧集|片|作品|系列|短剧|纪录片)"
)


def _is_vod_desc_search(query):
    q = (query or "").strip()
    if not q:
        return False
    if re.search(r"宝宝|幼儿|儿歌|亲子|启蒙|早教|睡前|绘本", q):
        return False
    if "播放" in q or "点播" in q or "跟唱" in q:
        return False
    return bool(_VOD_DESC_SEARCH.search(q))



# 地区/国家 + 拍 + 动画电影/片 → vod 媒资浏览（非 children 少儿内容）。
# children 的「动画片」全是"播放好看的动画片/宝宝动画片/XX动画片"这类无地域维度；
# 一旦带国家/地区参照（美国/日本/法国拍的动画片），属 vod 全库按 area/category 筛选，
# 应归 vod 媒资浏览而非 children 少教内容。独立于 LLM 判域强制→vod。
_VOD_DONGHUA_REGION = re.compile(
    r"(?:美国|日本|中国|法国|英国|德国|韩国|意大利|俄罗斯|西班牙|加拿大|澳大利亚|"
    r"中国香港|香港|中国台湾|台湾|内地|国产|好莱坞)"  # 地区词
    r".{0,5}?"  # 地区后可选间隔词(拍的/的/片 等)
    r"(?:动画|动漫|卡通)"
)
_REGION_WORD = re.compile(r"美国|日本|中国|法国|英国|德国|韩国|意大利|俄罗斯|西班牙|加拿大|澳大利亚|香港|台湾|国产|好莱坞")


def _is_vod_donghua_media(query: str) -> bool:
    """地区/国家 拍的动画片 → vod 媒资浏览。"""
    q = (query or "").strip()
    if not q:
        return False
    # 明确少儿语境（宝宝/幼儿/亲子/启蒙）仍归 children，不受此条覆盖。
    if re.search(r"宝宝|幼儿|亲子|启蒙|早教|幼儿园|绘本", q):
        return False
    return bool(_REGION_WORD.search(q)
                and re.search(r"动画|动漫|卡通", q)
                and _VOD_DONGHUA_REGION.search(q))


def _is_vod_auteur_anim(query: str) -> bool:
    """署名动画作者/吉卜力作品名 → vod（作者电影，非少儿 content）。

    宫崎骏/新海诚/千与千寻 等是 vod 媒资，children「幼儿 IP/绘本」才归 children；
    该白名单明确无误义，故独立于 LLM 判域强制→ vod，挡住「推荐和千与千寻类似的动画片」
    这类 relate 检索被 LLM/children 「动画片」信号劫持到 education/children。
    该 guard 在 children_locator 之前得先评估，避免「宫崎骏的动画」被 children 的「动画」信号抢判。
    """
    return bool(_VOD_AUTEUR_ANIM.search(query))


# 推荐/想看 + 类型片 → vod。多轮 follow-up 弱句("推荐几部动作片""想看美食纪录片")
# 常被 LLM 判成 qa/空；带明确类型片(片/电影/纪录片)即媒资检索，独立于 LLM 判域。
_VOD_REC_GENRE = re.compile(
    r"(推荐|推荐.{0,4}(几部|点|些|看)|想看|想找|找.{0,3}(点|些|几部)?)"
    r".{0,6}(动作|恐怖|科幻|悬疑|搞笑|美食|纪录|古装|爱情|喜剧|文艺|动画|战争|犯罪|惊悚|灾难|剧情|奇幻|武侠|反特/刑侦|谍战)"
    r".{0,4}(片|电影|纪录片|剧|影片|片子)"
)


def _vod_recommend_genre(query: str) -> bool:
    """推荐/想看 + 类型词 + 片/电影 → vod（稳定媒资检索，不被 qa 抢判）。"""
    return bool(_VOD_REC_GENRE.search(query))


# 亮屏「具名/播放儿歌」→ children（educ_search 少儿内容检索）。
# 判据控得很窄，避免误伤 music 点歌：
#   1) 播放/查看类动词 + 儿歌（"播放不花钱的儿歌"）→ children（亮屏播/看少儿内容）
#   2) 具名儿歌（具体儿歌名 + 儿歌，如 巴士儿歌/小兔子乖乖儿歌）→ children
# 若 query 是"来一首/点一首/有没有/适合…的" 等歌曲推荐/点歌/描述，不判（交 music/LLM）。
# 息屏(tv=6) 由 _off_routing 先行处理(点歌听 → music)，不经过这里。
# 只匹配带 播放/放/看/找/搜 动词 + 儿歌，或非描述性里的具名儿歌。
_ERGE_PLAY_VERB = re.compile(r"(?:播放|放|看|搜|给我|直接|来点|点播).{0,10}儿歌")
# 具名儿歌：儿歌前 2~6 字为具体歌名前缀（如"巴士儿歌""小兔子乖乖儿歌"）。
# 用正向前瞻，避免可变宽度 lookbehind 报错。
_ERGE_NAMED = re.compile(r"[一-龥A-Za-z]{2,6}(?=儿歌)")
# 以这些前缀开头的 query 视为 music 点歌/歌曲推荐，亮屏不判 children（gold 反验）。
_ERGE_EXCLUDE_PREFIX = ("有简单的", "有没有", "哪些", "适合", "推荐", "来一首",
                        "来首", "来点", "有的", "有适合", "律动", "歌词", "带",
                        "想听", "能听", "好听", "英语儿歌", "英文儿歌", "循环")


def _is_bright_children_erge(q: str) -> bool:
    q = (q or "").strip()
    if not q:
        return False
    if any(q.startswith(p) for p in _ERGE_EXCLUDE_PREFIX):
        return False
    # 具名儿歌：儿歌前是具体歌名（如"巴士儿歌""小兔子乖乖儿歌"）
    if _ERGE_NAMED.search(q):
        return True
    return bool(_ERGE_PLAY_VERB.search(q))


# 亮屏「影视体裁 / 综艺 / 片段定位 / 观看历史 / 短视频」强信号 → vod。
# 这些是可泛化的影视载体词，常被其它域的弱信号（歌/历史/循环/小说/孩子）误判到
# music/education/children/audio/qa。置于 badcase(100) 之后、music_override(125) 之前，
# 把明确影视载体优先拉回 vod。各类带排除卫士，避免误伤 golden：
#   - CCTV/综艺频道 的歌舞节目 → music（频道点播）；MV → music
#   - 动画片/绘本/动漫 台词 → children；亚运会/奥运会/比赛 片段 → sports
#   - 竖屏/横屏/全屏 短视频 → device
_VOD_MS_GENRE = re.compile(
    r"歌舞剧|歌舞片|音乐剧|歌剧|舞台剧|话剧|宣传片|时间循环剧|循环剧|穿越剧|影视剧|参演|"
    r"演员.{0,6}(?:演技|排行|排名)|拍摄?.{0,4}素材")
_VOD_MS_VARIETY = re.compile(
    r"综艺|选秀|真人秀|音乐节目|音乐综艺|歌唱综艺|晚会直播|发布会直播")
_VOD_MS_LOCATOR = re.compile(
    r"片段|名场面|经典台词|台词|桥段|高光片段|高光时刻|那段|这段|那一段|"
    r"第几集|哪几集|多少集|哪集|什么剧|哪部剧|哪部影视|出自|来自哪|是哪部")
_VOD_MS_HIST = re.compile(r"观看历史|观看记录|看过的剧|看剧记录|观看第")
_VOD_MS_SHORTVID = re.compile(r"短视频")
# 戏曲剧种 → vod（戏曲是视频演出）。卫士：带「唱/K歌/点歌/金曲」是音乐点唱(music_ksong)，
# 带「卫视+频道」是音乐频道点播(tvchannel_music)，不判 vod。
_VOD_MS_OPERA = re.compile(r"评剧|豫剧|黄梅戏|京剧|越剧|秦腔|曲剧|花鼓戏|昆曲|晋剧|沪剧|粤剧|吕剧")
_VOD_MS_OPERA_BLOCK = re.compile(r"唱|K歌|k歌|点歌|金曲|卫视.{0,4}频道|频道.{0,4}金曲")
# 历史/上次看过 + 影视载体 → vod（看影视记录）。卫士：有声/广播剧/评书/听=audio；歌/音乐=music。
_VOD_MS_HIST_CARRIER = re.compile(
    r"(?:历史|上次|上一次|看过的?|之前看|刚才看|播放记录|观看|收藏).{0,10}"
    r"(?:纪录片|电视剧|电影|综艺|影视剧|电视节目)|"
    r"(?:历史|上次|上一次)(?:播放|看)(?:的)?(?:纪录片|电视剧|电影|综艺|影视剧|电视节目)")
_VOD_MS_HIST_BLOCK = re.compile(r"广播剧|有声|听书|评书|电台|FM|fm|的?歌|音乐|唱")
# 泛指纯播放视频/看电视（无内容载体）→ vod。意图无歧义但无剧名，常死成空 execute。
# 紧锚定：动词(播放/放/看/打开) + 视频/电视 后即收尾，排除「视频版X/看电视剧X」等带内容。
_VOD_MS_PLAY_GENERIC = re.compile(
    r"^(?:请|帮我|给我|我要|我想|开始|我现在要)?(?:播放|放|看|打开)"
    r"(?:点|个|一点|一个)?(?:视频|电视)(?:播放|给我看|给我|吧|呢|啊)?[\s，。！？~、]*$")
# 观看历史(时间窗) → vod。根因：裸「历史」被 education 学科词抢判；但「历史记录(成词)/观看
# /我之前看的」+ 时间窗 是看影视记录。卫士：少儿/音乐/有声/歌/播放记录(music_song_history)。
_VOD_MS_WATCH_TIMEWIN = r"今天|昨天|本周|这周|一周|这一周|这几天|最近|近期|前几天|这段时间"
_VOD_MS_WATCH_HIST = re.compile(
    r"(?:" + _VOD_MS_WATCH_TIMEWIN + r").{0,6}(?:历史记录|观看)|"
    r"历史记录.{0,6}(?:" + _VOD_MS_WATCH_TIMEWIN + r")|"
    r"(?:" + _VOD_MS_WATCH_TIMEWIN + r").{0,4}历史记录|"
    r"历史.{0,4}观看|观看.{0,4}历史|观看历史|"
    r"(?:要看|想看)?我之前看的|之前看的")
_VOD_MS_WATCH_BLOCK = re.compile(
    r"少儿|儿童|宝宝|幼儿|音乐|有声|音频|收听|听歌|歌|DJ|dj|评书|广播剧|播放记录")
# relate/推荐/追剧 + 影视载体词（剧/电影/美剧/故事片/影视佳作）→ vod。
# 剧名只是上下文，载体词定域；常被 LLM/education/children 判错。卫士：MV/OST/原声→music，
# 广播剧/有声/评书→audio，动画片/卡通/绘本/少儿→children，裸歌/音乐/唱→music。
_VOD_MS_RELATE_INTENT = (
    r"推荐|类似|想找|想看|找.{0,3}(?:类似|和|跟)|追过|追完|看过|其他的?|别的|换.{0,2}部|"
    r"给我播放|安静的?看|看个")
_VOD_MS_RELATE_CARRIER = (
    r"美剧|韩剧|日剧|英剧|泰剧|古装剧|战争剧|历史剧|政治剧|谍战剧|悬疑剧|都市剧|年代剧|"
    r"武侠剧|仙侠剧|家庭剧|言情剧|电视剧|电影|故事片|影视剧|影视佳作|影视作品|这部剧|"
    r"这个剧|一些剧|几部剧|女性成长的剧|题材的.剧")
_VOD_MS_RELATE = re.compile(
    r"(?:" + _VOD_MS_RELATE_INTENT + r").{0,12}(?:" + _VOD_MS_RELATE_CARRIER + r")|"
    r"(?:" + _VOD_MS_RELATE_CARRIER + r").{0,8}(?:" + _VOD_MS_RELATE_INTENT + r")")
_VOD_MS_RELATE_BLOCK = re.compile(
    r"广播剧|有声|听书|评书|动画片|卡通|动漫|儿歌|歌舞剧|音乐剧|歌剧|唱|的?歌|音乐|绘本|"
    r"少儿|儿童|宝宝|MV|mv|OST|ost|原声")
# 裸个性化推荐（无任何内容载体）→ vod。电视默认内容域是视频，「推荐我喜欢看的/个性化推荐/
# 按我口味推荐」无域锚时判 vod。卫士：带 动漫/歌/有声/少儿/课程/球/综艺 等任意内容域词即不判。
_VOD_MS_PERSONAL = re.compile(
    r"^推荐一个[\s，。！？~]*$|^播放相关推荐[\s，。！？~]*$|"
    r"(?:推荐|找点|来点|给我).{0,6}(?:我喜欢|我偏爱|我的口味|个性化|我可能喜欢|大概率.{0,2}喜欢|偏爱的类型)|"
    r"(?:根据|按).{0,2}我(?:的)?(?:口味|喜好|偏好|兴趣).{0,2}推荐|"
    r"还有什么我可能喜欢看的|我喜欢看的[\s，。！？~]*$")
_VOD_MS_PERSONAL_BLOCK = re.compile(
    r"动漫|动画|卡通|歌|音乐|有声|广播剧|评书|少儿|儿童|宝宝|绘本|课程|教材|年级|"
    r"球|比赛|赛事|联赛|新闻|戏曲|相声|听|唱|专辑|单曲|纪录片|综艺")
# 演员/导演 + 影视行业动作(资料/简介/拍过/演过/转型/作品/创作动机) → vod（影视行业词定域，
# 不靠认人名）。常被 music 的「歌/歌手」裸词抢判。卫士：歌曲/唱/专辑/MV/演唱会/作词作曲→music。
_VOD_MS_FILMPEOPLE = re.compile(
    r"(?:演员|导演|男主角|女主角|主演|男演员|女演员).{0,8}"
    r"(?:个人资料|基本信息|资料|简介|拍过|演过|参演|主演|转型|作品|创作动机|个人简介)|"
    r"(?:导演|演员).{0,2}(?:拍摄|拍).{0,6}(?:的)?(?:创作|动机|作品)|"
    r"(?:导演)(?:陈凯歌|.{2,5}拍过)")
_VOD_MS_FILMPEOPLE_BLOCK = re.compile(r"的?歌曲|唱|专辑|单曲|MV|mv|演唱会|歌单|作词|作曲")
# 春晚/晚会/奇妙游/脱口秀 + 节目/小视频/集锦/段子 → vod（晚会综艺是视频节目）。
# 常被 music 的「歌/唱/歌词」裸词抢判。卫士：「卫视/频道…的歌」结尾是频道点歌→music。
_VOD_MS_GALA = re.compile(
    r"(?:春晚|晚会|奇妙游|脱口秀|跨年晚会|元宵晚会|中秋晚会).{0,20}(?:节目|小视频|集锦|段子|表演环节)|"
    r"(?:春晚|晚会|奇妙游).{0,12}(?:脱口秀|小品)")
_VOD_MS_GALA_BLOCK = re.compile(r"卫视.{0,4}春晚的歌$|频道.{0,4}的歌$")
# 综艺节目档期查询（更新时间/首播/下期直播/最新一期出了没）→ vod。歌曲/单曲不会「更新/下期
# 直播/最新一期」，故这类"节目几号更新/首播定在几月"必是综艺节目档期。卫士：歌曲更新/新歌→music。
_VOD_MS_SCHEDULE = re.compile(
    r"(?:几号|什么时候|何时|几点).{0,4}更新|更新时间|下期.{0,4}(?:直播|播出|更新)|"
    r"首播.{0,6}(?:时间|几月|几号|定在)|最新一期.{0,4}(?:出了没|播了没|更新|是什么)|最新一期出")
_VOD_MS_SCHEDULE_BLOCK = re.compile(r"歌曲更新|单曲|专辑.{0,2}更新|新歌")
# 纯播放「看片/看个片」（无 title）→ vod。
_VOD_MS_WATCH_MOVIE = re.compile(
    r"看个?片[\s，。！？~]*$|^(?:我)?想看个?片$")
# 裸「推荐的电视」→ vod。
_VOD_MS_REC_TV = re.compile(
    r"^(?:有没有|有|给我|来点|找).{0,4}推荐的?电视[\s，。！？~]*$|推荐的电视$")
_VOD_MS_CHANNEL_MUSIC = re.compile(r"频道|CCTV|cctv|央视")
_VOD_MS_MUSIC_CARRIER = re.compile(r"歌|音乐|戏曲")


# 少儿/卡通/动画/儿童频道 → children（频道即少儿内容，golden 为 educ_search_all）。
# 独立成规则并置于 music_override(125) 之前：否则「少儿频道的猪猪侠」会被 music_override
# 的具名歌名(猪猪侠)抢判成 music。
_VOD_MS_KIDS_CHANNEL = re.compile(r"少儿频道|卡通频道|动画频道|儿童频道")


def _is_kids_channel(query: str) -> bool:
    return bool(_VOD_MS_KIDS_CHANNEL.search(query or ""))


def _is_vod_media_strong(query: str) -> bool:
    """影视体裁/综艺/片段定位/观看历史/短视频 → vod（亮屏高置信）。"""
    q = (query or "").strip()
    if not q:
        return False
    # 频道点播（CCTV/综艺频道 的歌舞/音乐节目）/ MV → music，不判 vod
    if _VOD_MS_CHANNEL_MUSIC.search(q) and _VOD_MS_MUSIC_CARRIER.search(q):
        return False
    if re.search(r"MV|mv", q):
        return False
    # 1) 影视体裁词（歌舞剧/歌剧/宣传片/循环剧/参演…）
    if _VOD_MS_GENRE.search(q):
        return True
    # 2) 综艺 / 选秀 / 音乐综艺 / 节目直播
    if _VOD_MS_VARIETY.search(q):
        return True
    # 3) 片段 / 台词 / 集数定位（排除少儿内容与赛事）
    if _VOD_MS_LOCATOR.search(q):
        if re.search(r"动画片|绘本|动漫|动画|卡通", q):
            return False
        if re.search(r"亚运会|奥运会|世界杯|欧洲杯|欧冠|联赛|锦标赛|半决赛|预选赛|球赛", q):
            return False
        return True
    # 4) 影视观看历史
    if _VOD_MS_HIST.search(q):
        return True
    # 5) 短视频（排除竖屏/横屏/全屏 设备布局词）
    if _VOD_MS_SHORTVID.search(q):
        if re.search(r"竖屏|横屏|全屏|分屏", q):
            return False
        return True
    # 6) 戏曲剧种（视频演出）——排除点唱(唱/金曲)与音乐频道
    if _VOD_MS_OPERA.search(q) and not _VOD_MS_OPERA_BLOCK.search(q):
        return True
    # 7) 历史/上次看过 + 影视载体 —— 排除有声/广播剧/歌曲
    if _VOD_MS_HIST_CARRIER.search(q) and not _VOD_MS_HIST_BLOCK.search(q):
        return True
    # 8) 泛指纯播放视频/看电视（无内容载体）
    if _VOD_MS_PLAY_GENERIC.search(q):
        return True
    # 9) 观看历史(时间窗) —— 排除少儿/音乐/有声/播放记录
    if _VOD_MS_WATCH_HIST.search(q) and not _VOD_MS_WATCH_BLOCK.search(q):
        return True
    # 10) relate/推荐/追剧 + 影视载体 —— 排除有声/少儿/音乐
    if _VOD_MS_RELATE.search(q) and not _VOD_MS_RELATE_BLOCK.search(q):
        return True
    # 11) 裸个性化推荐（无内容载体）—— 排除任意内容域词
    if _VOD_MS_PERSONAL.search(q) and not _VOD_MS_PERSONAL_BLOCK.search(q):
        return True
    # 12) 演员/导演 + 影视行业动作 —— 排除歌曲/演唱会/MV
    if _VOD_MS_FILMPEOPLE.search(q) and not _VOD_MS_FILMPEOPLE_BLOCK.search(q):
        return True
    # 13) 春晚/晚会/奇妙游 + 节目/小视频/段子 —— 排除频道点歌
    if _VOD_MS_GALA.search(q) and not _VOD_MS_GALA_BLOCK.search(q):
        return True
    # 14) 综艺节目档期查询（更新时间/首播/下期直播/最新一期）—— 排除歌曲更新
    if _VOD_MS_SCHEDULE.search(q) and not _VOD_MS_SCHEDULE_BLOCK.search(q):
        return True
    # 15) 纯播放看片 / 裸推荐电视
    if _VOD_MS_WATCH_MOVIE.search(q):
        return True
    if _VOD_MS_REC_TV.search(q):
        return True
    return False


def _make_detect_rules():
    rules = []

    def _add(rid, prio, title, dec, group="detect.signal"):
        # group = 该规则所属的**层/域级开关组**（见 app/config.py::RULE_SWITCHES）。
        # 默认 detect.signal（高置信信号类，不属任何单一目标域）。
        rules.append(_DRule(id=rid, priority=prio, title=title, decide=dec,
                            scope="both", group=group))

    # 依序复刻原 if-栈（priority 越小越先评估，等价原先后顺序）
    _add("off_routing", 1, "息屏分诊链",
         lambda q, llm, tv: _off_routing(q, llm) if str(tv) == "6" else None, "detect.off")
    _add("badcase", 100, "精确句 badcase", lambda q, llm, tv: _badcases().get(q) or None,
         "detect.badcase")
    _add("vod_media_strong", 110, "亮屏影视体裁/综艺/片段/观看历史/短视频→vod",
         lambda q, llm, tv: "vod" if (str(tv) != "6" and _is_vod_media_strong(q)) else None,
         "detect.vod")
    _add("kids_channel", 115, "少儿/卡通/动画频道→children",
         lambda q, llm, tv: "children" if (str(tv) != "6" and _is_kids_channel(q)) else None,
         "detect.children")
    _add("music_override", 125, "高置信歌曲/音乐频道强信号→music",
         lambda q, llm, tv: ("music" if (str(tv) != "6" and _music_override(q)) else None),
         "detect.music")
    _add("media_knowledge_qa", 150, "亮屏媒体/明星信息咨询→qa",
         lambda q, llm, tv: "qa" if (str(tv) != "6" and llm in ("vod", "music", "qa") and _media_knowledge_qa(q)) else None,
         "detect.qa")
    _add("edu_bright_strong", 155, "亮屏教育强词→education",
         lambda q, llm, tv: "education" if (str(tv) == "0" and _edu_bright_strong(q)) else None,
         "detect.education")
    _add("vod_auteur_anim", 180, "署名动画作者影片→vod",
         lambda q, llm, tv: "vod" if _is_vod_auteur_anim(q) else None, "detect.vod")
    # 【已停用 2026-09-16】vod_donghua_region / vod_desc_search 两条规则：
    # 二者把「地区+动画」「找/搜+动画」一律抢判 vod，与 0901 sheet（当前基准）的标注冲突——
    # 0901_children 63 条动画用例 golden 全为 educ_*（educ_search 38 / fuzzy 15 / all 7 / relate 2 /
    # history 1），e80803_multiintent 的「俄罗斯拍的动画」content 子句 golden 亦为 educ_search_all，
    # 而 0821_serial（旧 sheet）同构句 golden 为 vod_*。删除后实测：
    #   children 94.2% → 99.7%（+18 条），audio 97.0% → 97.6%（+2），vod/其余 4 域 ±0；
    #   代价是 0821_serial 有 3 条「那个…的动画片」指代式描述回退（旧 sheet，另议）。
    # 定义（_VOD_DESC_SEARCH / _VOD_DONGHUA_REGION / _is_vod_desc_search / _is_vod_donghua_media）
    # 暂留供审计与回滚；确认 0821 serial 可废弃后再整体删除。
    _add("media_query", 200, "vod 媒资载体保护",
         lambda q, llm, tv: _media_query_decide(q, llm, tv), "detect.vod")
    _add("media_locator", 300, "台词/片段/集数定位",
         lambda q, llm, tv: "vod" if (llm in ("vod", "qa") and _is_media_locator(q)) else None,
         "detect.vod")
    _add("children_locator", 400, "children 内容定位",
         lambda q, llm, tv: "children" if (_match(q)[0] == "children" and re.search(r"绘本|动画|动漫|卡通|台词|哪部动画|哪个动画", q)) else None,
         "detect.children")
    _add("_audio_listen_carry", 480, "听/播放+有声载体→audio",
         lambda q, llm, tv: _audio_listen_carry_patch(q, llm, tv), "detect.audio")
    # plot_qa 提高优先级(395)到 children_locator(400) 之前，让「这部动画片讲的是…」剧情问答
    # (对某一部影视的结果提问剧情)归 qa，不被 children 抢走；有 block 不误伤真 children 播放/听。
    _add("plot_qa", 395, "具名影视剧情问答→qa",
         lambda q, llm, tv: "qa" if (str(tv) != "6" and _is_plot_qa(q)) else None, "detect.qa")
    _add("audio_discovery", 500, "audio 内容 Discovery",
         lambda q, llm, tv: "audio" if _audio_discovery(q) else None, "detect.audio")
    _add("music_discovery", 600, "music 内容 Discovery",
         lambda q, llm, tv: "music" if _music_discovery(q) else None, "detect.music")
    _add("sports_prediction", 700, "sports 赛事预测",
         lambda q, llm, tv: "sports" if _sports_prediction(q) else None, "detect.sports")
    # 运动项目名 + 强运动语境词 → sports。修正 LLM planner 把「山地自行车视频/速度轮滑视频/
    # 足球亚巡赛视频」判成 vod 的误判（这类查询无「比赛/球队」word，signal_match(1100) 抓不到，
    # 原先直落 llm_domain_keep 保 vod）。置于 signal_match 之前不改变后者已覆盖的 48 条；
    # 实测 7 域 testset：仅 sports 域新增命中 20 条（另 48 条 signal_match 已判 sports），
    # vod/children/music/education/audio/device 零误伤。
    _add("sports_project_media", 710, "运动项目+媒体语境→sports",
         lambda q, llm, tv: "sports" if _sports_project_media(q) else None, "detect.sports")
    _add("vod_recommend_genre", 790, "推荐类型片→vod",
         lambda q, llm, tv: "vod" if _vod_recommend_genre(q) else None, "detect.vod")
    _add("qa_open_knowledge", 800, "泛知识开放问答",
         lambda q, llm, tv: "qa" if _is_qa(q) else None, "detect.qa")
    _add("edu_no_anchor_qa", 900, "教育无锚问答→education",
         lambda q, llm, tv: ("education" if str(tv) == "0" else None)
         if _is_edu_no_anchor_qa(q, llm) else None, "detect.education")
    _add("children_ergou_bright", 1050, "亮屏具名/播放儿歌→children",
         lambda q, llm, tv: "children" if (str(tv) == "0" and _is_bright_children_erge(q)) else None,
         "detect.children")
    _add("signal_match", 1100, "高置信信号",
         lambda q, llm, tv: _signal_table_match(q), "detect.signal")
    _add("qa_greeting", 1150, "纯问候/闲聊→qa",
         lambda q, llm, tv: "qa" if _is_greeting(q) else None, "detect.qa")
    _add("llm_domain_keep", 1200, "保 LLM 兜底", lambda q, llm, tv: llm, "detect.keep")
    return _DetectRuleSet(rules)


_DETECT_RULES = _make_detect_rules()
