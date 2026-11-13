"""The fixed benchmark story: three days, four sessions a day, hand-written (no LLM, no randomness).

    python scripts/bench/story.py            # summary: lines per session / speaker / condition, term counts
    python scripts/bench/story.py --json     # the whole story as JSON (stdout)
    python scripts/bench/story.py --check    # exit 1 when validate() finds a problem

Everything is fictional. Five recurring characters (the owner "我", two colleagues, the partner and a friend who
is deliberately NOT enrolled) plus one narrator voice that is only ever heard through the system-audio channel
(a documentary at home, a news video at the office; two colleagues also join one meeting remotely).

Conventions
-----------
* Numerals are written the way mimo-v2.5-asr writes them (checked against real outputs): spoken numbers,
  amounts and dates in Chinese numerals ("两千六百九十九", "十一月九号", "百分之十"), Arabic digits only inside
  alphanumeric codes ("T30", "X7", "P0"). textnorm.normalize() makes both spellings equal anyway.
* A line is ``L(speaker, text, far=, sys=, overlap=, gap=)``:
    far      rendered as far-field speech (-20 dB, synthetic reverb) on the microphone channel
    sys      played through the system-audio channel (remote meeting participant / video narrator)
    overlap  seconds this line starts BEFORE the previous line ends (two people talking at once)
    gap      silence before the line in seconds (default: a fixed pattern, see build.default_gap)
* Ground truth for the memory layer sits next to each session (``memory``): the facts / todos / decisions that
  should be extracted, with the indexes of the supporting lines; ``done_in`` names the later session (and
  lines) that proves a todo done (None = still open at the end); ``superseded_in`` the later decision that
  replaces an earlier one. CONFLICTS lists the two pairs of statements that contradict each other without
  anybody noticing. MEDIA_DISTRACTORS are narrator statements that must NOT become memories.
* LEXICON is the gold lexicon: the hard proper nouns the lines repeat on purpose (nicknames, a project,
  English acronyms, product codes, near-homophones of common words). ``aliases`` is left empty: the
  mis-hearings are what the system under test has to learn. FIGURES are the amounts / dates scored the same way.
"""
from __future__ import annotations

import json
import sys
from typing import Any

STORY_ID = "bench-story-v1"
OWNER = "我"
NARRATOR = "旁白"

#: name -> voice + role. voice: ("preset", <MiMo preset voice>) or ("design", <voice description>): a designed
#: voice is synthesised ONCE as a reference clip and every line is then cloned from that reference.
CAST: dict[str, dict[str, Any]] = {
    "我": {"voice": ("preset", "苏打"), "enrolled": True, "self": True,
          "role": "录音设备的主人，产品经理，三十岁左右"},
    "周岚": {"voice": ("preset", "冰糖"), "enrolled": True, "self": False,
           "role": "同事，后端工程师，大家叫她岚姐"},
    "陈默": {"voice": ("preset", "白桦"), "enrolled": True, "self": False,
           "role": "团队负责人，语速快，爱下结论"},
    # designed, not a preset: the two female presets are near-twins for the speaker model (centroid cosine 0.79),
    # which would make every speaker number a statement about that pair
    "阿梨": {"voice": ("design", "二十八岁左右的年轻女性，声音偏低，温柔沉稳，略带一点鼻音，语速不快，像在家里和家人聊天"),
           "enrolled": True, "self": False,
           "role": "“我”的伴侣，晚上在家"},
    "老秦": {"voice": ("design", "三十五岁左右的男性，普通话标准，嗓音略微沙哑，语速偏快，说话爽朗随意，像朋友聊天"),
           "enrolled": False, "self": False,
           "role": "朋友，没有登记声纹的常客：应该被标成未知，并在第二次出现后被问“这是谁”"},
    NARRATOR: {"voice": ("design", "四十岁左右的男性纪录片旁白，声音低沉浑厚，语速偏慢，吐字清晰，标准播音腔"),
               "enrolled": False, "self": False, "system_only": True,
               "role": "视频旁白，只出现在系统声音里"},
}

#: read by every character once; they become the enrollment samples of the enrolled characters
ENROLL_TEXT = [
    "你好，这是一段用来登记声音的样本。我会用平时说话的语速和音量，读几句普通的话。",
    "今天天气不错，下午大概三点左右开会，记得把上周的数据整理一下带过来。",
]
#: text of the reference clip of a designed voice (never part of a session)
DESIGN_REFERENCE_TEXT = "你好，这是一段用来确定音色的参考录音。接下来我会用平稳的语速，读几句普通的话，方便后面保持同一个声音。"
#: style instruction for ordinary lines / for the narrator
STYLE = "平静自然的日常对话语气，正常语速"
STYLE_NARRATOR = "纪录片旁白的讲述语气，平稳清晰"


def L(speaker: str, text: str, *, far: bool = False, sys: bool = False, overlap: float = 0.0,
      gap: float | None = None) -> dict[str, Any]:
    return {"speaker": speaker, "text": text, "far": bool(far), "sys": bool(sys), "overlap": float(overlap), "gap": gap}


def M(mid: str, kind: str, title: str, lines: list[int], **extra: Any) -> dict[str, Any]:
    """One expected memory item. kind: fact | todo | decision. extra: subject, owner, due (YYYY-MM-DD),
    done_in=(session id, [line indexes]) or None, superseded_in=(session id, item id)."""
    d = {"id": mid, "kind": kind, "title": title, "lines": list(lines)}
    d.update(extra)
    return d


DAYS: list[dict[str, Any]] = [
    # ------------------------------------------------------------------------------------------ day 1
    {"date": "2026-11-02", "sessions": [
        {"id": "d1s1", "start": "09:30", "title": "晨会", "kind": "meeting", "lines": [
            L("我", "早，人齐了咱们就开始。今天主要过一下青禾App灰度发布的安排。"),
            L("周岚", "好。我先说后端，白鹭网关上周五合了主干，目前还剩二十三个bug。"),
            L("陈默", "二十三个有点多，P0的有几个？", far=True),
            L("周岚", "P0的两个，都跟支付回调有关，我今天能修完。"),
            L("我", "埋点SDK那边呢？上次说有数据丢失。"),
            L("周岚", "埋点SDK升到二点四就好了，不过要重新压测一遍。"),
            L("陈默", "那青禾的灰度先定在这周五，十一月六号，比例百分之五。", far=True),
            L("我", "行，周五灰度，百分之五。岚姐，压测你排在哪天？"),
            L("周岚", "周三下午压测，目标QPS三千五。"),
            L("陈默", "可以。周岚，压测报告出来发我一份。另外CDN的预算谁在跟？", far=True),
            L("我", "我在跟，下午评审会上一起说。"),
            L("陈默", "好，散会。", far=True),
            L("我", "嗯，辛苦。"),
        ], "memory": [
            M("d1s1-m1", "fact", "白鹭网关上周五合入主干，还剩二十三个bug", [1], subject="白鹭"),
            M("d1s1-m2", "fact", "P0的bug有两个，都和支付回调有关", [2, 3]),
            M("d1s1-m3", "todo", "周岚今天修完两个P0", [3], owner="周岚", due="2026-11-02", done_in=("d2s1", [1])),
            M("d1s1-m4", "fact", "埋点SDK升到二点四可以解决数据丢失，但要重新压测", [4, 5], subject="埋点SDK"),
            M("d1s1-m5", "decision", "青禾灰度定在周五十一月六号，比例百分之五", [6, 7], subject="青禾",
              superseded_in=("d2s1", "d2s1-m4")),
            M("d1s1-m6", "todo", "周岚周三下午压测，目标QPS三千五", [7, 8], owner="周岚", due="2026-11-04",
              done_in=("d3s3", [0])),
            M("d1s1-m7", "todo", "周岚把压测报告发给陈默", [9], owner="周岚", done_in=None),
            M("d1s1-m8", "todo", "我跟进CDN预算，下午评审会上说", [9, 10], owner="我", due="2026-11-02",
              done_in=("d1s3", [1])),
        ]},
        {"id": "d1s2", "start": "12:15", "title": "午饭", "kind": "conversation", "lines": [
            L("老秦", "这家鹿鸣咖啡的简餐还不错，你常来？"),
            L("我", "常来，就在公司楼下。最近陈默盯得紧，中午只能在这儿凑合。"),
            L("老秦", "我最近在看扫地机，朗月T30你听过没有？"),
            L("我", "听过，朗月T30，好像是两千六百九十九？"),
            L("老秦", "对对对，到手两千六百九十九，还送一套滤网。", overlap=0.6),
            L("我", "那还行。我家小满掉毛厉害，正想买一台。"),
            L("老秦", "小满就是你家那只橘猫吧？几岁了？"),
            L("我", "两岁半。对了，听说鹿鸣要开新店？"),
            L("老秦", "嗯，新店开在栖霞路，下个月十二号开业。"),
            L("我", "栖霞路啊，那离我家很近。", overlap=0.5),
            L("老秦", "开业那天第二杯半价，到时候叫上阿梨一起去。"),
            L("我", "好，我回去跟阿梨说。"),
            L("老秦", "行，那我先撤了，回头聊。"),
        ], "memory": [
            M("d1s2-m1", "fact", "老秦在考虑买扫地机朗月T30", [2], subject="老秦"),
            M("d1s2-m2", "fact", "朗月T30到手价两千六百九十九，送一套滤网", [3, 4], subject="朗月T30"),
            M("d1s2-m3", "fact", "我家的猫叫小满，是一只两岁半的橘猫", [5, 6, 7], subject="小满"),
            M("d1s2-m4", "fact", "鹿鸣咖啡的新店开在栖霞路，下个月十二号开业", [8], subject="鹿鸣"),
            M("d1s2-m5", "fact", "鹿鸣新店开业当天第二杯半价", [10], subject="鹿鸣"),
            M("d1s2-m6", "todo", "我回去跟阿梨说鹿鸣新店开业一起去", [10, 11], owner="我", done_in=("d1s4", [1, 2])),
        ]},
        {"id": "d1s3", "start": "15:00", "title": "线上评审", "kind": "meeting", "lines": [
            L("陈默", "都能听到吗？我们开始评审。", sys=True),
            L("我", "能听到。我先说CDN预算，供应商报价一年十二万五。"),
            L("陈默", "十二万五有点高，首期能不能少付一点？", sys=True),
            L("我", "谈过了，首期八万，剩下的四万五上线以后再付。"),
            L("周岚", "插一句，白鹭网关的压测环境要加两台机器。", sys=True),
            L("陈默", "两台够吗？目标QPS是三千五。", sys=True),
            L("周岚", "够了，上次单机压到一千八。", sys=True),
            L("我", "那预算里再加两台机器，一个月大概三千块。"),
            L("陈默", "行，CDN就按首期八万批。青禾的DAU目标呢？", sys=True),
            L("我", "青禾灰度期间DAU目标两万，全量以后到十五万。"),
            L("陈默", "好，写进OKR里。今天先到这儿。", sys=True),
            L("我", "好的，会议纪要我来发。"),
        ], "memory": [
            M("d1s3-m1", "fact", "CDN供应商报价一年十二万五", [1], subject="CDN"),
            M("d1s3-m2", "decision", "CDN首期八万，剩下的四万五上线以后再付", [3, 8], subject="CDN",
              superseded_in=("d2s3", "d2s3-m3")),
            M("d1s3-m3", "decision", "白鹭网关的压测环境加两台机器，一个月大概三千块", [4, 7], subject="白鹭"),
            M("d1s3-m4", "fact", "上次单机压测QPS到一千八", [6]),
            M("d1s3-m5", "fact", "青禾的DAU目标：灰度期间两万，全量以后十五万", [9], subject="青禾"),
            M("d1s3-m6", "todo", "我把DAU目标写进OKR", [10], owner="我", done_in=("d2s3", [3])),
            M("d1s3-m7", "todo", "我发会议纪要", [11], owner="我", done_in=("d2s1", [4])),
        ]},
        {"id": "d1s4", "start": "20:40", "title": "晚上在家", "kind": "conversation", "lines": [
            L("阿梨", "你回来啦？饭在锅里，自己热一下。", far=True),
            L("我", "好。今天老秦说鹿鸣咖啡要在栖霞路开新店，下个月十二号开业。"),
            L("阿梨", "栖霞路？那走路十分钟就到，开业我们去看看。"),
            L("我", "行。还有，他推荐了朗月T30扫地机，两千六百九十九。"),
            L("阿梨", "买吧，小满的毛我每天要扫两遍。"),
            L("我", "那我今晚就下单。青禾下周要灰度，这几天会比较忙。"),
            L("阿梨", "知道了。对了，小满该打疫苗了，周六上午你带它去？"),
            L("我", "周六上午可以，几点？"),
            L("阿梨", "十点，宠物医院我约好了。"),
            L("我", "好，周六十点。你妈生日是不是也快到了？"),
            L("阿梨", "十一月二十号，我想订个蛋糕。"),
            L("我", "蛋糕我来订，你把口味发给我。"),
            L("阿梨", "好，要栗子的。", far=True),
            L("我", "嗯，记下了。"),
        ], "memory": [
            M("d1s4-m1", "decision", "买朗月T30扫地机，我今晚下单", [3, 4, 5], subject="朗月T30"),
            M("d1s4-m2", "todo", "我今晚下单朗月T30", [5], owner="我", due="2026-11-02", done_in=("d2s4", [6, 7])),
            M("d1s4-m3", "todo", "我周六上午十点带小满去宠物医院打疫苗", [6, 7, 8, 9], owner="我", due="2026-11-07",
              subject="小满", done_in=None),
            M("d1s4-m4", "fact", "阿梨妈妈的生日是十一月二十号", [9, 10], subject="阿梨"),
            M("d1s4-m5", "todo", "我订栗子口味的生日蛋糕", [11, 12, 13], owner="我", done_in=("d3s4", [8])),
        ]},
    ]},
    # ------------------------------------------------------------------------------------------ day 2
    {"date": "2026-11-03", "sessions": [
        {"id": "d2s1", "start": "09:30", "title": "晨会", "kind": "meeting", "lines": [
            L("陈默", "早。先同步进度，周岚，bug还剩多少？"),
            L("周岚", "昨天修了十六个，还剩七个，两个P0都修完了。", far=True),
            L("陈默", "不错。埋点SDK升级了吗？"),
            L("周岚", "升到二点四了，数据丢失没再出现。", far=True),
            L("我", "会议纪要我昨天发了，CDN合同今天走流程。"),
            L("陈默", "好。有个变化，青禾周五灰度来不及，改到下周一。"),
            L("我", "下周一，十一月九号？比例还是百分之五吗？"),
            L("陈默", "十一月九号，比例提到百分之十。"),
            L("周岚", "那压测还是周三下午，时间够。", far=True),
            L("我", "好，我去通知运营，灰度改到十一月九号。"),
            L("陈默", "嗯，今天就通知。"),
            L("我", "收到。"),
        ], "memory": [
            M("d2s1-m1", "fact", "昨天修了十六个bug，还剩七个，两个P0都修完了", [1]),
            M("d2s1-m2", "fact", "埋点SDK已经升到二点四，数据丢失没再出现", [2, 3], subject="埋点SDK"),
            M("d2s1-m3", "todo", "我今天走CDN合同流程", [4], owner="我", due="2026-11-03", done_in=("d3s1", [0])),
            M("d2s1-m4", "decision", "青禾灰度从周五改到下周一十一月九号，比例提到百分之十", [5, 6, 7], subject="青禾"),
            M("d2s1-m5", "todo", "我今天通知运营灰度改到十一月九号", [9, 10], owner="我", due="2026-11-03",
              done_in=("d2s3", [1])),
        ]},
        {"id": "d2s2", "start": "12:10", "title": "午饭闲聊", "kind": "conversation", "lines": [
            L("我", "岚姐，中午吃什么？"),
            L("周岚", "去鹿鸣咖啡吧，我想喝他们的燕麦拿铁。"),
            L("我", "行。你那个显示器后来买了吗？"),
            L("周岚", "买了，极影X7，二十七寸，三千二百块。"),
            L("我", "极影X7？我也在看这个。", overlap=0.5),
            L("周岚", "挺好的，就是支架有点晃。"),
            L("我", "那我再想想。对了，OKR你写完了吗？"),
            L("周岚", "还没，陈默说周四之前交。"),
            L("我", "周四之前，那我今晚写。"),
            L("周岚", "你写完借我参考一下。", overlap=0.7),
            L("我", "没问题，岚姐。"),
            L("周岚", "走吧，再晚鹿鸣就没座了。"),
        ], "memory": [
            M("d2s2-m1", "fact", "周岚买了显示器极影X7，二十七寸，三千二百块", [2, 3], subject="周岚"),
            M("d2s2-m2", "fact", "极影X7的支架有点晃", [5], subject="极影X7"),
            M("d2s2-m3", "fact", "陈默要求OKR周四之前交", [6, 7], subject="OKR"),
            M("d2s2-m4", "todo", "我今晚写OKR", [8], owner="我", due="2026-11-03", done_in=("d3s1", [2])),
            M("d2s2-m5", "todo", "我写完OKR借周岚参考", [9, 10], owner="我", done_in=("d3s1", [2, 3])),
        ]},
        {"id": "d2s3", "start": "15:10", "title": "一对一", "kind": "meeting", "lines": [
            L("陈默", "坐。运营那边通知到了吗？"),
            L("我", "通知了，青禾灰度改到十一月九号，他们没意见。"),
            L("陈默", "好。你的OKR里DAU怎么定的？"),
            L("我", "灰度期间两万，全量以后十五万，已经写进去了。", far=True),
            L("陈默", "十五万可以。CDN那边，我想了一下，首期十万吧。"),
            L("我", "首期十万？昨天评审说的是首期八万。"),
            L("陈默", "供应商说首期十万可以再打九五折。"),
            L("我", "明白，那我按首期十万重新走合同。", far=True),
            L("陈默", "另外，压测要是出了P0，第一时间告诉我。"),
            L("我", "好，周三压测我全程盯着。"),
            L("陈默", "嗯。还有，白鹭网关的文档你催一下周岚。"),
            L("我", "行，我让岚姐周五前补完。", far=True),
            L("陈默", "好，就这些。"),
        ], "memory": [
            M("d2s3-m1", "fact", "运营对灰度改到十一月九号没有意见", [0, 1]),
            M("d2s3-m2", "fact", "供应商说首期十万可以再打九五折", [6], subject="CDN"),
            M("d2s3-m3", "decision", "CDN首期改成十万", [4, 5, 6, 7], subject="CDN"),
            M("d2s3-m4", "todo", "我按首期十万重新走CDN合同", [7], owner="我", done_in=("d3s1", [0])),
            M("d2s3-m5", "todo", "我周三全程盯压测，出了P0第一时间告诉陈默", [8, 9], owner="我", due="2026-11-04",
              done_in=("d3s3", [3, 4])),
            M("d2s3-m6", "todo", "我催周岚周五前补完白鹭网关的文档", [10, 11], owner="我", due="2026-11-06",
              subject="白鹭", done_in=None),
        ]},
        {"id": "d2s4", "start": "20:30", "title": "在家看纪录片", "kind": "media", "lines": [
            L(NARRATOR, "每年十一月，成群的白鹭从北方飞来，在这片湿地停留一百多天。", sys=True),
            L("阿梨", "这个纪录片拍得真好。"),
            L(NARRATOR, "白鹭喜欢浅水，它们常常单腿站立，一动不动地等待猎物。", sys=True),
            L("我", "我们青禾那个网关也叫白鹭，陈默起的名字。"),
            L("阿梨", "是吗？那还挺好听的。"),
            L(NARRATOR, "这片湿地面积约三千公顷，是候鸟迁徙路上重要的中转站。", sys=True),
            L("我", "对了，朗月T30今天到了，我放在门口了。"),
            L("阿梨", "这么快？昨晚才下的单，老秦推荐的还挺靠谱。"),
            L(NARRATOR, "到了傍晚，白鹭会成群飞回树林，那是一天中最壮观的时刻。", sys=True),
            L("阿梨", "明天我来拆，让小满先适应一下。"),
            L("我", "好。它估计会怕，第一次开低档。"),
            L(NARRATOR, "保护湿地，就是保护这些远道而来的客人。", sys=True),
            L("阿梨", "嗯，周末我们去湿地公园走走吧。"),
            L("我", "可以，周日去，周六要带小满打疫苗。"),
        ], "memory": [
            M("d2s4-m1", "fact", "白鹭网关的名字是陈默起的", [3], subject="白鹭"),
            M("d2s4-m2", "fact", "朗月T30今天到了", [6, 7], subject="朗月T30"),
            M("d2s4-m3", "todo", "阿梨明天拆朗月T30，让小满先适应", [9, 10], owner="阿梨", due="2026-11-04",
              done_in=("d3s4", [2, 4])),
            M("d2s4-m4", "decision", "周日去湿地公园", [12, 13]),
        ]},
    ]},
    # ------------------------------------------------------------------------------------------ day 3
    {"date": "2026-11-04", "sessions": [
        {"id": "d3s1", "start": "09:30", "title": "晨会", "kind": "meeting", "lines": [
            L("我", "早。先说我的，CDN合同昨天按首期十万重新提交了。"),
            L("陈默", "好。OKR都交了吗？"),
            L("我", "我昨晚写完交了，也发给岚姐参考了。"),
            L("周岚", "收到了，谢谢。我的今天下班前交。"),
            L("陈默", "行。周岚，bug呢？"),
            L("周岚", "七个全清了，现在是零bug。"),
            L("陈默", "零bug？不错。", overlap=0.4),
            L("周岚", "今天下午两点压测，目标QPS三千五，我和测试一起盯。"),
            L("我", "我也在，出了P0我马上同步。"),
            L("陈默", "好。青禾灰度还是十一月九号，百分之十，不变。"),
            L("我", "明白。"),
            L("周岚", "对了，埋点SDK我又发了个小版本，二点四点一。"),
            L("陈默", "可以，记得写进发布说明。"),
        ], "memory": [
            M("d3s1-m1", "fact", "CDN合同已经按首期十万重新提交", [0], subject="CDN"),
            M("d3s1-m2", "todo", "周岚今天下班前交OKR", [3], owner="周岚", due="2026-11-04", done_in=None),
            M("d3s1-m3", "fact", "七个bug全部清掉，现在零bug", [4, 5]),
            M("d3s1-m4", "fact", "今天下午两点压测，目标QPS三千五", [7]),
            M("d3s1-m5", "decision", "青禾灰度维持十一月九号、百分之十不变", [9], subject="青禾"),
            M("d3s1-m6", "fact", "埋点SDK发了小版本二点四点一", [11], subject="埋点SDK"),
            M("d3s1-m7", "todo", "周岚把埋点SDK二点四点一写进发布说明", [11, 12], owner="周岚", done_in=None),
        ]},
        {"id": "d3s2", "start": "12:20", "title": "午休看科技快讯", "kind": "media", "lines": [
            L(NARRATOR, "欢迎收看本周科技快讯。首先来看智能家居。", sys=True),
            L(NARRATOR, "朗月本周发布了新款扫地机T30 Pro，售价三千四百九十九。", sys=True),
            L("我", "T30 Pro？我刚买了T30。"),
            L(NARRATOR, "相比上一代的T30，新款的吸力提升了百分之四十。", sys=True),
            L("周岚", "你在看什么？"),
            L("我", "科技快讯，朗月出了T30 Pro，三千四百九十九。"),
            L("周岚", "比你那台贵八百块。"),
            L(NARRATOR, "另一条消息，极影下个月将推出X7的升级款X7S。", sys=True),
            L("周岚", "极影X7S？我的X7才买了一个星期。"),
            L("我", "哈哈，买早了。", overlap=0.5),
            L(NARRATOR, "以上就是本期的全部内容，我们下周再见。", sys=True),
            L("周岚", "走了，两点压测，别迟到。"),
            L("我", "好，马上来。"),
        ], "memory": [
            M("d3s2-m1", "fact", "朗月出了T30 Pro，售价三千四百九十九，比T30贵八百块", [5, 6], subject="朗月T30"),
            M("d3s2-m2", "fact", "周岚的极影X7才买了一个星期", [8], subject="周岚"),
        ]},
        {"id": "d3s3", "start": "15:30", "title": "压测复盘", "kind": "meeting", "lines": [
            L("周岚", "压测结果出来了，QPS最高到四千二，超过目标。"),
            L("陈默", "四千二？比三千五高不少。延迟呢？", far=True),
            L("周岚", "平均延迟九十毫秒，上周还是三百二十毫秒。"),
            L("我", "有没有出P0？"),
            L("周岚", "没有P0，只有一个埋点重复上报的小问题。"),
            L("陈默", "埋点的问题周五前修掉。", far=True),
            L("周岚", "好，周五前修掉。"),
            L("我", "白鹭网关的文档呢？"),
            L("周岚", "还差两章，周五前一定补完。"),
            L("陈默", "那就定了：青禾十一月九号灰度百分之十，十一月十八号全量。", far=True),
            L("我", "十一月十八号全量，我写进OKR。"),
            L("陈默", "好。这次压测辛苦了，周五我请大家喝鹿鸣咖啡。", far=True),
            L("周岚", "谢谢陈默，那我要大杯的。"),
            L("我", "谢谢。", overlap=0.4),
        ], "memory": [
            M("d3s3-m1", "fact", "压测QPS最高到四千二，超过目标三千五", [0, 1]),
            M("d3s3-m2", "fact", "平均延迟九十毫秒，上周是三百二十毫秒", [2]),
            M("d3s3-m3", "fact", "压测没有出P0，只有一个埋点重复上报的小问题", [3, 4]),
            M("d3s3-m4", "todo", "周岚周五前修掉埋点重复上报的问题", [5, 6], owner="周岚", due="2026-11-06", done_in=None),
            M("d3s3-m5", "fact", "白鹭网关的文档还差两章，周岚说周五前补完", [7, 8], subject="白鹭"),
            M("d3s3-m6", "decision", "青禾十一月九号灰度百分之十，十一月十八号全量", [9, 10], subject="青禾"),
            M("d3s3-m7", "todo", "我把十一月十八号全量写进OKR", [10], owner="我", done_in=None),
            M("d3s3-m8", "fact", "陈默周五请大家喝鹿鸣咖啡", [11], subject="陈默"),
        ]},
        {"id": "d3s4", "start": "20:30", "title": "晚上在家", "kind": "conversation", "lines": [
            L("老秦", "打扰了啊，路过你们家，上来坐坐。"),
            L("阿梨", "老秦来啦？快坐，我在切水果。", far=True),
            L("我", "正好，给你看看朗月T30，阿梨今天刚拆的。"),
            L("老秦", "真买了？用着怎么样？"),
            L("我", "吸力不错，就是小满一开始吓得躲到沙发底下。"),
            L("老秦", "哈哈哈，猫都这样。", overlap=0.6),
            L("阿梨", "对了，鹿鸣的新店我今天路过了，开在南湖路，下个月八号开业。", far=True),
            L("老秦", "那到时候一起去，我请客。"),
            L("我", "行。阿梨，你妈的生日蛋糕我订好了，栗子口味。"),
            L("阿梨", "这么快？我妈生日是二十二号，别送早了。"),
            L("我", "放心，订的是当天送。"),
            L("老秦", "你们家小满呢？让我看看。"),
            L("我", "在阳台。周六我还要带它去打疫苗。"),
            L("阿梨", "水果好了，过来吃吧。", far=True),
        ], "memory": [
            M("d3s4-m1", "fact", "鹿鸣咖啡的新店开在南湖路，下个月八号开业", [6], subject="鹿鸣"),
            M("d3s4-m2", "fact", "阿梨妈妈的生日是二十二号", [9], subject="阿梨"),
            M("d3s4-m3", "fact", "小满一开始被朗月T30吓得躲到沙发底下", [4], subject="小满"),
            M("d3s4-m4", "fact", "生日蛋糕已经订好，栗子口味，生日当天送", [8, 10]),
        ]},
    ]},
]

#: statements that contradict each other and that nobody in the story notices (conflict detection)
CONFLICTS: list[dict[str, Any]] = [
    {"id": "c1", "topic": "鹿鸣咖啡新店的地址和开业日期", "a": "d1s2-m4", "b": "d3s4-m1",
     "note": "老秦说栖霞路、下个月十二号；阿梨说南湖路、下个月八号"},
    {"id": "c2", "topic": "阿梨妈妈的生日", "a": "d1s4-m4", "b": "d3s4-m2",
     "note": "第一天说十一月二十号，第三天说二十二号"},
]

#: narrator statements (system audio, media sessions) that must not turn into memories of the owner
MEDIA_DISTRACTORS: list[dict[str, Any]] = [
    {"session": "d2s4", "lines": [0, 5], "text": "湿地面积约三千公顷，白鹭停留一百多天"},
    {"session": "d3s2", "lines": [3], "text": "新款的吸力提升了百分之四十"},
    {"session": "d3s2", "lines": [7], "text": "极影下个月将推出X7S"},
]

#: gold lexicon. kind follows the lexicon table (person | term | project | org | place | product | other).
#: aliases stay empty on purpose (the mis-hearings are to be learned); min = least number of occurrences.
LEXICON: list[dict[str, Any]] = [
    {"term": "陈默", "kind": "person", "note": "团队负责人；与“沉默”同音", "aliases": [], "min": 4},
    {"term": "周岚", "kind": "person", "note": "同事，后端工程师", "aliases": [], "min": 4},
    {"term": "岚姐", "kind": "person", "note": "周岚的昵称", "aliases": [], "min": 4},
    {"term": "阿梨", "kind": "person", "note": "“我”的伴侣", "aliases": [], "min": 4},
    {"term": "老秦", "kind": "person", "note": "朋友", "aliases": [], "min": 3},
    {"term": "小满", "kind": "other", "note": "家里的橘猫；也是节气名", "aliases": [], "min": 6},
    {"term": "青禾", "kind": "project", "note": "正在做的App，青禾App", "aliases": [], "min": 8},
    {"term": "白鹭", "kind": "project", "note": "支付网关的内部代号；与节气“白露”同音，纪录片里指鸟", "aliases": [], "min": 8},
    {"term": "鹿鸣", "kind": "org", "note": "公司楼下的咖啡店，鹿鸣咖啡", "aliases": [], "min": 7},
    {"term": "栖霞路", "kind": "place", "note": "鹿鸣新店的地址之一", "aliases": [], "min": 4},
    {"term": "南湖路", "kind": "place", "note": "与栖霞路矛盾的另一个地址", "aliases": [], "min": 1},
    {"term": "朗月", "kind": "product", "note": "扫地机品牌", "aliases": [], "min": 7},
    {"term": "T30", "kind": "product", "note": "朗月扫地机型号", "aliases": [], "min": 10},
    {"term": "极影", "kind": "product", "note": "显示器品牌", "aliases": [], "min": 4},
    {"term": "X7", "kind": "product", "note": "极影显示器型号", "aliases": [], "min": 5},
    {"term": "P0", "kind": "term", "note": "最高级别的故障 / bug", "aliases": [], "min": 7},
    {"term": "OKR", "kind": "term", "note": "目标与关键结果", "aliases": [], "min": 5},
    {"term": "SDK", "kind": "term", "note": "埋点SDK", "aliases": [], "min": 4},
    {"term": "QPS", "kind": "term", "note": "每秒请求数", "aliases": [], "min": 4},
    {"term": "CDN", "kind": "term", "note": "内容分发网络", "aliases": [], "min": 6},
    {"term": "DAU", "kind": "term", "note": "日活", "aliases": [], "min": 3},
    {"term": "灰度", "kind": "term", "note": "灰度发布", "aliases": [], "min": 11},
    {"term": "埋点", "kind": "term", "note": "数据上报；常被听成“买点 / 卖点”", "aliases": [], "min": 6},
    {"term": "压测", "kind": "term", "note": "压力测试", "aliases": [], "min": 12},
]

#: figures / amounts / dates that carry facts. Scored like lexicon terms after numeral normalisation.
FIGURES: list[dict[str, Any]] = [
    {"term": "二十三个", "note": "第一天的bug数", "min": 2},
    {"term": "两千六百九十九", "note": "朗月T30的到手价", "min": 3},
    {"term": "三千四百九十九", "note": "T30 Pro的售价", "min": 2},
    {"term": "十二万五", "note": "CDN一年的报价", "min": 2},
    {"term": "首期八万", "note": "CDN首期（后来被改掉）", "min": 3},
    {"term": "首期十万", "note": "CDN首期（最终）", "min": 5},
    {"term": "三千五", "note": "压测目标QPS", "min": 4},
    {"term": "四千二", "note": "压测实际QPS", "min": 2},
    {"term": "十五万", "note": "全量以后的DAU目标", "min": 3},
    {"term": "十一月九号", "note": "灰度日期（最终）", "min": 6},
    {"term": "十一月十八号", "note": "全量日期", "min": 2},
    {"term": "百分之五", "note": "最初的灰度比例", "min": 3},
    {"term": "百分之十", "note": "最终的灰度比例", "min": 3},
    {"term": "二点四", "note": "埋点SDK版本", "min": 2},
    {"term": "九十毫秒", "note": "压测后的平均延迟", "min": 1},
    {"term": "三百二十毫秒", "note": "压测前的平均延迟", "min": 1},
]


# ------------------------------------------------------------------------------------------------ accessors
def sessions() -> list[dict[str, Any]]:
    """Every session, in time order, with day / ISO start and indexed lines."""
    out: list[dict[str, Any]] = []
    for d in DAYS:
        for s in d["sessions"]:
            lines = [{"i": i, "session": s["id"], **ln} for i, ln in enumerate(s["lines"])]
            out.append({"id": s["id"], "day": d["date"], "start": f"{d['date']}T{s['start']}:00", "title": s["title"],
                        "kind": s["kind"], "lines": lines, "memory": [dict(m) for m in s.get("memory", [])]})
    return out


def session(session_id: str) -> dict[str, Any]:
    for s in sessions():
        if s["id"] == session_id:
            return s
    raise KeyError(session_id)


def all_lines() -> list[dict[str, Any]]:
    return [ln for s in sessions() for ln in s["lines"]]


def base_condition(line: dict[str, Any], next_line: dict[str, Any] | None = None) -> str:
    """near | far | overlap | system for a scripted line (the build may override it per variant)."""
    if line.get("sys"):
        return "system"
    if float(line.get("overlap") or 0) > 0 or (next_line is not None and float(next_line.get("overlap") or 0) > 0):
        return "overlap"
    if line.get("far"):
        return "far"
    return "near"


def enrolled_names() -> list[str]:
    return [n for n, c in CAST.items() if c.get("enrolled")]


def term_list(include_figures: bool = True) -> list[dict[str, Any]]:
    """[{term, group: name | term | number, kind}] for the proper-noun error rate."""
    out = []
    for t in LEXICON:
        group = "term" if t["kind"] == "term" else "name"
        out.append({"term": t["term"], "group": group, "kind": t["kind"]})
    if include_figures:
        out.extend({"term": f["term"], "group": "number", "kind": "number"} for f in FIGURES)
    return out


def memory_items() -> list[dict[str, Any]]:
    return [{**m, "session": s["id"], "day": s["day"]} for s in sessions() for m in s["memory"]]


def count_occurrences(term: str) -> int:
    return sum(ln["text"].count(term) for ln in all_lines())


def validate() -> list[str]:
    """Structural problems of the story (empty list = fine). Pure; the unit tests call it."""
    problems: list[str] = []
    ss = sessions()
    order = {s["id"]: k for k, s in enumerate(ss)}
    if len(DAYS) != 3 or any(len(d["sessions"]) != 4 for d in DAYS):
        problems.append("expected 3 days x 4 sessions")
    if len(order) != len(ss):
        problems.append("duplicate session id")
    items: dict[str, dict[str, Any]] = {}
    for s in ss:
        n = len(s["lines"])
        for k, ln in enumerate(s["lines"]):
            if ln["speaker"] not in CAST:
                problems.append(f"{s['id']}[{k}]: unknown speaker {ln['speaker']}")
            if not ln["text"].strip():
                problems.append(f"{s['id']}[{k}]: empty text")
            if CAST.get(ln["speaker"], {}).get("system_only") and not ln["sys"]:
                problems.append(f"{s['id']}[{k}]: the narrator must be system audio")
            if ln["sys"] and (ln["far"] or ln["overlap"]):
                problems.append(f"{s['id']}[{k}]: a system line cannot be far / overlapped")
            if ln["overlap"]:
                prev = s["lines"][k - 1] if k else None
                if prev is None or prev["speaker"] == ln["speaker"]:
                    problems.append(f"{s['id']}[{k}]: overlap needs a previous line of another speaker")
                elif prev["sys"] or prev["far"] or ln["far"]:
                    problems.append(f"{s['id']}[{k}]: overlap pairs are near-field microphone lines")
                elif prev["overlap"]:
                    problems.append(f"{s['id']}[{k}]: chained overlaps are not supported")
        for m in s["memory"]:
            if m["id"] in items:
                problems.append(f"duplicate memory id {m['id']}")
            items[m["id"]] = {**m, "session": s["id"]}
            if m["kind"] not in ("fact", "todo", "decision"):
                problems.append(f"{m['id']}: bad kind {m['kind']}")
            if not m["lines"] or any(not (0 <= i < n) for i in m["lines"]):
                problems.append(f"{m['id']}: line index out of range")
            if m["kind"] == "todo" and "done_in" not in m:
                problems.append(f"{m['id']}: todo without done_in (use None for still open)")
    for m in items.values():
        done = m.get("done_in")
        if done:
            sid, idx = done
            if sid not in order:
                problems.append(f"{m['id']}: done_in names unknown session {sid}")
                continue
            if order[sid] <= order[m["session"]]:
                problems.append(f"{m['id']}: done_in must be a later session")
            n = len(ss[order[sid]]["lines"])
            if any(not (0 <= i < n) for i in idx):
                problems.append(f"{m['id']}: done_in line index out of range")
        sup = m.get("superseded_in")
        if sup:
            sid, other = sup
            if other not in items or items[other]["session"] != sid or order.get(sid, -1) <= order[m["session"]]:
                problems.append(f"{m['id']}: bad superseded_in {sup}")
    for c in CONFLICTS:
        if c["a"] not in items or c["b"] not in items:
            problems.append(f"conflict {c['id']}: unknown memory id")
    for d in MEDIA_DISTRACTORS:
        if d["session"] not in order:
            problems.append(f"distractor names unknown session {d['session']}")
            continue
        lines = ss[order[d["session"]]]["lines"]
        if any(not (0 <= i < len(lines)) or lines[i]["speaker"] != NARRATOR for i in d["lines"]):
            problems.append(f"distractor in {d['session']} must point at narrator lines")
    for t in LEXICON + FIGURES:
        n = count_occurrences(t["term"])
        if n < int(t.get("min", 1)):
            problems.append(f"term {t['term']}: {n} occurrences < {t.get('min', 1)}")
    if any(t["aliases"] for t in LEXICON):
        problems.append("gold lexicon aliases must stay empty (they are to be learned)")
    return problems


def as_json() -> dict[str, Any]:
    cast = {n: {**{k: v for k, v in c.items() if k != "voice"}, "voice": list(c["voice"])} for n, c in CAST.items()}
    return {"story": STORY_ID, "owner": OWNER, "cast": cast, "sessions": sessions(), "conflicts": CONFLICTS,
            "media_distractors": MEDIA_DISTRACTORS, "lexicon": LEXICON, "figures": FIGURES}


def summary() -> dict[str, Any]:
    ss = sessions()
    by_speaker: dict[str, int] = {}
    by_cond: dict[str, int] = {}
    chars = 0
    for s in ss:
        for k, ln in enumerate(s["lines"]):
            nxt = s["lines"][k + 1] if k + 1 < len(s["lines"]) else None
            by_speaker[ln["speaker"]] = by_speaker.get(ln["speaker"], 0) + 1
            c = base_condition(ln, nxt)
            by_cond[c] = by_cond.get(c, 0) + 1
            chars += len(ln["text"])
    mem = memory_items()
    return {"sessions": len(ss), "lines": sum(len(s["lines"]) for s in ss), "chars": chars, "by_speaker": by_speaker,
            "by_condition": by_cond, "memory_items": len(mem),
            "todos": sum(1 for m in mem if m["kind"] == "todo"),
            "todos_done_later": sum(1 for m in mem if m["kind"] == "todo" and m.get("done_in")),
            "conflicts": len(CONFLICTS),
            "terms": {t["term"]: count_occurrences(t["term"]) for t in LEXICON + FIGURES}}


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="print the whole story as JSON")
    ap.add_argument("--check", action="store_true", help="validate and exit 1 on problems")
    args = ap.parse_args(argv)
    if args.json:
        print(json.dumps(as_json(), ensure_ascii=False, indent=1))
        return 0
    problems = validate()
    if args.check:
        for p in problems:
            print("!", p)
        return 1 if problems else 0
    print(json.dumps(summary(), ensure_ascii=False, indent=1))
    for p in problems:
        print("!", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
