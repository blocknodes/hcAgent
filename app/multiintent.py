"""多意图(内容+设备 双目标)确定性拆分。

评测集 e80803 multiintent：一条 query 同时含【内容检索】与【设备控制】两个独立目标，
用连接词(然后/顺便/再/同时/接着/逗号…)串成一句。把它拆成两条【并行、无依赖】意图：
  - 内容条: 仅内容检索文本，tool+params 交给 hcTools 权威解析。
  - 设备条: 仅设备控制文本，tool+params 交给 hcTools(device 域)。

本模块只判【哪一半是设备】+切到正确子句；具体设备工具+参数由 hcTools 权威解析。

判定语义（按重要级）：
  1. 内容锚点强词(动画/电影/歌曲/比赛/课/台词/…，多字词) → 该子句=内容检索。
  2. 否则命中设备强词(亮度/音量/模式/快进/关机/校正/…，多字词) → 该子句=设备控制。
  3. 其余 → 内容。
分隔点优先连接词(然后/顺便/同时/再/接着…)，无连接词则取单个逗号。
恰好一侧为设备、另一侧为内容 → 拆分。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

try:
    from .config import rule_on as _rule_on
except ImportError:  # 被 tools 独立测试加载时无包上下文
    def _rule_on(_group: str) -> bool:
        return True


@dataclass
class Split:
    device: str
    content: str
    hit: bool


# ---- 串行「检索 → 提问」确定性拆分（评测 sheet0821_serial：搜X，然后再问X的Y）----
# 分隔点固定为 然后问 / 然后再问；左半是检索请求，右半是对检索结果【某一部】的属性提问。
_SERIAL_CONN = re.compile(r"然后再问|然后问")


@dataclass
class SerialSplit:
    """串行「检索→提问」两条。step1=检索子句，step2=提问子句（依赖 step1 结果）。"""
    search: str   # 检索子句(交给 hcTools vod)
    question: str  # 提问子句(交给 hcTools qa → fan_knowledge_agent)
    hit: bool


def split_serial_qa(query: str) -> SerialSplit:
    """判定 query 是否为「搜X，然后问X的Y」串行检索+提问，返回拆成两条的自包含子句。

    只在 然后问/然后再问 处切一刀，取左右两半。仅当两半都非空才命中。
    右半即对左半检索结果的属性提问（"评分最高的/最经典的/这一部的…是谁/哪年"），
    交给下游 qa 域 + hcTools fan_knowledge_agent 权威解析。
    """
    if not _rule_on("multiintent.serial_qa"):
        return SerialSplit("", "", False)
    q = (query or "").strip()
    if not q:
        return SerialSplit("", "", False)
    m = _SERIAL_CONN.search(q)
    if not m:
        return SerialSplit("", "", False)
    left = _strip(q[: m.start()])
    right = _strip(q[m.end():])
    if left and right:
        return SerialSplit(search=left, question=right, hit=True)
    return SerialSplit("", "", False)


_STRIP = "，,、；;。 　"


def _strip(s: str) -> str:
    return (s or "").strip(_STRIP)


# 意图连接词(切割点)：并列/顺承下一动作。
# 刻意排除「接着」——它是延续副词("接着播放/接着之前看"=继续上条内容)，不是新意图连接，
# 若留在 _CONN 会把「哑女情深接着播放」切成 device=播放 + content=哑女情深(误拆成双意图)。
# 双意图正例(e80803/sheet_multiturn)全用 然后/再/顺便/同时/随后/还有 连接，无「接着」正例。
_CONN = re.compile(r"然后|顺便|同时|并且|随后|还有|再加上|又|再|再顺便")


def _boundaries(q: str):
    out = []
    for m in _CONN.finditer(q):
        l, r = _strip(q[: m.start()]), _strip(q[m.end():])
        if l and r:
            out.append((l, r))
    if not out:
        for ch in "，,；;":
            idx = q.find(ch)
            if idx >= 0:
                l, r = _strip(q[:idx]), _strip(q[idx + 1:])
                if l and r:
                    out.append((l, r))
    return out


# ---- 内容锚词(强多字词,命中即内容；排在设备判定之前) ----
_CONTENT = [
    "动画片", "动画", "动漫", "卡通", "电影", "电视剧", "剧集", "影片", "纪录片",
    "剧场版", "影视", "综艺", "剧场", "节目", "剧情",
    "歌曲", "歌声", "主题曲", "片头曲", "片尾曲", "专辑", "歌词", "曲子", "原声",
    "音乐", "MV", "有声", "广播", "小说", "评书", "绘本", "童话", "睡前故事",
    "单曲", "配乐", "伴奏", "明星", "歌手",
    "课文", "课本", "级上册", "级下册", "上册", "下册", "教案", "教程", "同步",
    "语文", "数学", "英语", "物理", "化学", "生物", "地理", "历史", "政治",
    "体育", "网球", "篮球", "足球", "比赛", "赛事", "联赛", "冠军", "英超", "比分",
    "球赛", "之战", "偶像", "台词", "片段", "名场面", "场面", "故事",
    "那部", "那集", "哪部", "哪个", "什么", "人物", "主演", "推荐", "预告", "娱乐",
    "视频", "广播剧", "那段", "那一段", "戏份", "剧",
    "第\s*\d+\s*集",
]

# ---- 设备词两表 ----
# 拆分两表：_DEV_STRONG = 明确设备调参词（音量/亮度/关机…），确信是设备意图，可直接判。
#            _DEV_WEAK   = 语义两可词（色彩/画质/图像/氛围… 也可能是剧情/美学描述）。
#                         默认不直接拆进 device —— 命中只通过 weak_device_hint 提示交 LLM 判定
#                         （见 engine._first → _plan 的 weak_hint）。
#                         开关 multiintent.dev_weak_gate=1 恢复旧行为（弱词直接拆）.
_DEV_WEAK = [
    "色彩", "画质", "图像", "色温", "原画", "原音", "音画", "氛围", "场景细节",
    "细节增强", "动态范围", "数字艺术",
]
_DEV_STRONG = [
    "亮度", "清晰度", "对比度", "色度", "饱和度", "白平衡", "分辨率", "背光",
    "音量", "静音", "麦克风", "声道", "声场", "环绕", "音效", "音质", "低音", "高音",
    "重低音", "超重低音", "声音模式", "音效演示", "屏幕发声", "蓝牙",
    "关机", "开机", "重启", "唤醒", "快进", "快退", "倒回", "定位", "随机播放",
    "顺序播放", "停止播放", "暂停", "前进", "后退", "延时关机", "定时关机", "自动关机",
    "调节", "调整", "设置", "校准", "校正", "数值", "开关", "调到", "切换到",
    "对焦", "梯形", "控光", "护眼", "HDR", "杜比", "全景声", "像素", "晶晰", "疾速",
    "运动补偿", "运动画面", "环境光", "超清",
    "超广", "量子点", "中置", "声学", "音响", "屏幕", "分区控光", "全阵列", "防射眼",
    "峰值", "面板", "电视",
    "太暗", "太亮", "太艳", "太红", "太黑", "偏蓝", "偏红", "偏淡", "拖尾", "模糊",
    "刺眼", "卡顿", "暗场", "暗处", "雪花", "不干净", "看不清", "听不清", "斑点",
    "颜色太", "画面", "声音", "亮度调", "音量调", "快进到", "动态",
    "增强", "技术", "按键", "MiniLed", "Imax", "原生肤色", "原生",
    "按键音", "肤色", "HDR",
]

# ---- 裸设备控制词(命中即设备，除非内容锚覆盖——已在前置判断处理) ----
_BARE_DEV = [
    "停止", "静音", "快进", "快退", "随机播放", "顺序播放", "循环播放", "暂停",
    "切换", "唤醒", "重启", "开机", "关机", "向前", "向后", "倒回", "重绕", "下一首",
    "上一首", "返回", "退出",
]


# 强内容锚点：命中即内容，且屏蔽设备词(含模式类误判)
_CONTENT_STRONG = [
    "卡通", "电影", "电视剧", "剧集", "影片", "纪录片", "剧场", "综艺", "剧目",
    "动画片", "番剧", "动漫", "动画连续剧",
    "歌曲", "歌声", "主题曲", "片头曲", "片尾曲", "专辑", "歌词", "曲子", "配乐", "伴奏",
    "演唱会", "MV", "有声", "广播", "小说", "评书", "绘本", "童话", "睡前故事", "故事",
    "课文", "课本", "教案", "教程", "语文", "数学", "英语", "物理学", "化学", "生物",
    "地理", "历史", "政治", "科学", "启蒙",
    "比赛", "赛事", "联赛", "冠军", "英超", "比分", "球赛", "之战",
    "台词", "片段", "名场面", "场面", "戏份", "那段", "那一段",
    "广播剧", "剧集", "动漫", "歌曲", "MV", "台球", "剧场版", "被窝",
    "一路向北", "丁俊晖", "可爱", "全天从", "那一段", "片段",
    r"第\s*\d+\s*集",
    r"(?:的|一|几|这|那)(?:首|部|集|段)?(?:歌|音乐)",
    r"开心的歌|那首歌|这首歌",
]


def _is_content_clause(t: str) -> bool:
    # 书名号锚（multiintent.book_title_anchor，默认开）：任何《…》包裹的实体都是内容
    # （影视/综艺/歌曲/书），内部即使出现设备词（如《中国好声音》的"声音"）也不能判设备
    # ——挡"《中国好声音》被误拆成 device"这类整句内容误拆。
    # HC_RULE_MULTIINTENT_BOOK_TITLE_ANCHOR=0 关闭：书名号不再屏蔽设备词（回退旧行为）。
    if _rule_on("multiintent.book_title_anchor") and re.search(r"《[^》]{1,30}》", t):
        return True
    for c in _CONTENT_STRONG:
        if re.search(c, t):
            return True
    return False


def _is_device(t: str) -> bool:
    t2 = _strip(t)
    if not t2:
        return False
    # 命中强内容锚点 → 内容
    if _is_content_clause(t2):
        return False
    # 延续播放守卫：接着/继续 + 裸播放动词 → 不是设备意图，是"继续上一条内容播放"
    #   (哑女情深接着播放 / 趣趣知知鸟 接着播放 → 单意图连续播放，不是 device 控制)
    #   必须有"接着/继续"这类延续前缀才短路——纯"播放"仍算设备停止/播放控制。
    if re.search(r"^(接着|继续|接着继续|再继续|继续着)(播放|开播|换播|放|接着看|继续看)\s*$", t2):
        return False
    # 裸设备控制词(停止/快进/快退/随机/顺序/循环/切换…，且非内容实体名)
    for w in _BARE_DEV:
        if w in t2:
            return True
    # 末位单独"播放"停止词(无内容宾语) → 设备
    if re.fullmatch(r"(请|帮我|直接|继续)?(播放|开播|开机|关机)", t2):
        return True
    # 视频/位置类：从XX开始播放/跳到/播放到 位置 → 设备控制
    if re.search(r"从\s*\d{1,4}(:\d{1,2})?\s*(秒|分钟|处)?开始播放", t2):
        return True
    if re.search(r"(跳到|跳转到|直接跳到|播放到|快进到|快退到|定位到|前\s*\d+秒)\s*(\d{1,4}(:\d{1,2}|秒|分钟)|结尾|结尾处|片头|中间|开头|一)", t2):
        return True
    if re.search(r"(\d{1,4}:\d{1,2}|从0\d|从\d+:|直接\s*跳到)", t2) and not _is_content_clause(t2):
        return True
    # 参数+模式/设备词
    if re.search(r"(音乐|鲜艳|体育|标准|影院|游戏|灰度|显示|声音|图像|音频|电影|护眼|趣味|运动|节能|环境|数字|专业|超清|双标|整点)\s*模式", t2):
        return True
    if any(w in t2 for w in _DEV_STRONG):
        return True
    return False


# ---- 弱设备词提示（不直接拆，交 LLM 决定） ----
def weak_device_hint(query: str) -> str:
    """检测 query 是否含【弱设备词】（色彩/画质/图像…，语义两可：可能是设备调节，
    也可能是剧情/美学描述）。命中返回提示串（供 _plan 提示 LLM），未命中返回空串。
    只做提示，**不**参与确定性拆分——这类词不能由规则直接判 device。
    """
    q = (query or "").strip()
    if not q:
        return ""
    hits = [w for w in _DEV_WEAK if w in q]
    if not hits:
        return ""
    return f"含弱设备词({'/'.join(hits)})：可能是设备调节，也可能是剧情/美学描述，请结合上下文判定是否拆成设备意图。"


def _has_only_weak(t: str) -> bool:
    """该子句命中弱词、且无强词/裸词/内容锚 → 设备语义两可，只能靠弱词判断。"""
    t2 = _strip(t)
    if not t2:
        return False
    if _is_content_clause(t2):
        return False
    if any(w in t2 for w in _DEV_STRONG):
        return False
    if any(w in t2 for w in _BARE_DEV):
        return False
    return any(w in t2 for w in _DEV_WEAK)


# ---- 入口 ----
def split_multiintent(query: str) -> Split:
    if not _rule_on("multiintent.content_device"):
        return Split("", query or "", False)
    q = (query or "").strip()
    if not q:
        return Split("", q, False)
    # 弱设备词（色彩/画质…）默认不参与 _is_device：语义两可（剧情美学 vs 设备调参），
    # 不直接拆，只通过 weak_device_hint 交 LLM 判定（见 engine._first → _plan）。
    # 回滚开关 multiintent.dev_weak_gate=1（旧行为）：弱词也直接拆、不提示 LLM。
    legacy = _rule_on("multiintent.dev_weak_gate")
    for left, right in _boundaries(q):
        dl = _is_device(left) or (legacy and _has_only_weak(left))
        dr = _is_device(right) or (legacy and _has_only_weak(right))
        if dl and not dr:
            return Split(device=left, content=right, hit=True)
        if dr and not dl:
            return Split(device=right, content=left, hit=True)
    return Split("", q, False)