"""Prompt templates for the daily analysis (map over session blocks -> reduce into one digest).

Both stages run in JSON mode. The schemas below are repeated verbatim inside the prompts so the
model sees exactly which keys to emit; `daily.py` normalises whatever comes back.

    map    : one block of transcript (<= ~6000 chars, with HH:MM times and speaker labels)
             -> {title, summary, topics[], decisions[], todos[], people[], mood, open_questions[]}
    reduce : all block results of a day
             -> {overview, topics[], decisions[], todos[{text, owner?, due?}],
                 people[{name, context}], mood, open_loops[], timeline[{time, event}]}

Both user prompts carry a 「人物」 line (people_line) naming the user themselves and how the other
enrolled speakers relate to them, so the model reads the user's own utterances in the first person.
"""
from __future__ import annotations

import json
from typing import Any

MAP_SCHEMA = (
    '{"title": "这段对话的小标题（<=20字）", '
    '"summary": "2-4 句话概括这段内容", '
    '"topics": ["讨论到的话题，每条<=30字"], '
    '"decisions": ["明确做出的决定或结论"], '
    '"todos": ["需要去做的事；尽量写清楚谁做、什么时候"], '
    '"people": ["这段里出现或被提到的人名（只写名字，不包含用户本人）"], '
    '"mood": "一句话描述说话人的整体情绪/氛围", '
    '"open_questions": ["提出了但没有解决的问题"]}'
)

REDUCE_SCHEMA = (
    '{"overview": "3-6 句话的全天概览，像给自己写的日记摘要", '
    '"topics": ["全天的主要话题，按重要性排序，每条<=40字"], '
    '"decisions": ["全天做出的决定/结论"], '
    '"todos": [{"text": "待办内容", "owner": "负责人（可省略）", "due": "截止时间（可省略）"}], '
    '"people": [{"name": "人名（不包含用户本人）", "context": "此人今天和什么事有关"}], '
    '"mood": "一句话描述全天的情绪走向", '
    '"open_loops": ["未闭环的事项/悬而未决的问题"], '
    '"timeline": [{"time": "HH:MM", "event": "发生了什么（<=30字）"}]}'
)

PEOPLE_NOTE = (
    "「人物」一行给出了记录里的说话人和他们与用户本人的关系：用户本人的话按第一人称理解（就是“我”说的），"
    "待办里负责人是本人的就是用户自己的待办；people 字段不要包含用户本人。\n"
)

MAP_SYSTEM = (
    "你是一个私人的语音记录整理助手。用户全天佩戴录音设备，下面是某一个时段的语音转写文本，"
    "每行格式为「HH:MM 说话人：内容」。转写可能有少量识别错误和口语化表达，请根据上下文理解。\n"
    + PEOPLE_NOTE +
    "请只根据文本内容作答，不要编造文本里没有的信息；没有的字段给空数组或空字符串。\n"
    "必须只输出一个 JSON 对象，不要输出其他内容。JSON 结构（键名固定，用中文填写内容）：\n"
    + MAP_SCHEMA
)

MAP_USER = (
    "日期：{day}\n"
    "时段：第 {index}/{total} 块，{time_from}–{time_to}\n"
    "{people}"
    "\n"
    "=== 转写文本开始 ===\n"
    "{text}\n"
    "=== 转写文本结束 ===\n"
    "\n"
    "请按约定的 JSON 结构总结这一块。"
)

REDUCE_SYSTEM = (
    "你是一个私人的语音记录整理助手。下面是用户某一天各个时段语音记录的分块总结（JSON 数组，按时间排序），"
    "请把它们合并成一份当天的日报。\n"
    + PEOPLE_NOTE +
    "要求：\n"
    "1. 合并重复的话题/待办/人物，去掉空泛的内容；待办尽量写清楚负责人与时间（没有就省略该键）。\n"
    "2. 人物只保留真实的人名，并说明今天和此人相关的事。\n"
    "3. 时间线按时间先后，用各块给出的 HH:MM 时间，挑选 5-15 个关键节点。\n"
    "4. 不要编造分块总结里没有的信息；没有的字段给空数组或空字符串。\n"
    "必须只输出一个 JSON 对象，不要输出其他内容。JSON 结构（键名固定，用中文填写内容）：\n"
    + REDUCE_SCHEMA
)

REDUCE_USER = (
    "日期：{day}\n"
    "统计：共 {utterance_count} 条语句，{session_count} 个时段，{block_count} 个分块；出现的说话人：{speakers}\n"
    "{people}"
    "\n"
    "=== 分块总结 JSON 开始 ===\n"
    "{blocks_json}\n"
    "=== 分块总结 JSON 结束 ===\n"
    "\n"
    "请按约定的 JSON 结构输出当天日报。"
)

SELF_MISSING = "用户本人：未登记（记录里没有标出哪位是本人）"


def people_line(people: list[dict[str, Any]], unknown_label: str = "未知") -> str:
    """One prompt line describing the enrolled speakers, e.g.
    人物：我=「张伟」（用户本人）；王敏（同事）；李强（朋友，自动推断）；小雨（关系未标）；未识别的说话人标为「未知」
    `people` is db.people_table() (self first). Without a self speaker the line starts with
    「用户本人：未登记（记录里没有标出哪位是本人）」."""
    parts: list[str] = []
    me = next((p for p in people if p.get("is_self")), None)
    if me is not None:
        parts.append(f"我=「{me.get('name')}」（用户本人）")
    else:
        parts.append(SELF_MISSING)
    for p in people:
        if p is me or p.get("is_self"):
            continue
        rel = str(p.get("relation") or "").strip()
        if not rel:
            tag = "关系未标"
        elif p.get("relation_source") == "auto":
            tag = f"{rel}，自动推断"
        else:
            tag = rel
        parts.append(f"{p.get('name')}（{tag}）")
    if len(parts) == 1:
        parts.append("没有登记其他说话人")
    parts.append(f"未识别的说话人标为「{unknown_label}」")
    return "人物：" + "；".join(parts)


def _people_block(people: str) -> str:
    people = (people or "").strip()
    return people + "\n" if people else ""


def map_prompt(day: str, index: int, total: int, time_from: str, time_to: str, text: str, people: str = "") -> tuple[str, str]:
    """(system, user) for one transcript block. `people` is a people_line() ('' = omit the line)."""
    return MAP_SYSTEM, MAP_USER.format(day=day, index=index, total=total, time_from=time_from, time_to=time_to, text=text,
                                       people=_people_block(people))


def reduce_prompt(day: str, blocks: list[dict[str, Any]], *, utterance_count: int, session_count: int,
                  speakers: list[str], people: str = "") -> tuple[str, str]:
    """(system, user) for merging the per-block results into the daily digest."""
    user = REDUCE_USER.format(
        day=day,
        utterance_count=utterance_count,
        session_count=session_count,
        block_count=len(blocks),
        speakers="、".join(speakers) if speakers else "（未识别）",
        people=_people_block(people),
        blocks_json=json.dumps(blocks, ensure_ascii=False, indent=1),
    )
    return REDUCE_SYSTEM, user
