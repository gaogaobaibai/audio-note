"""Relative-date parsing for Chinese questions ("昨天晚上", "上周三", "10月28号", "最近几天").

Used by the chat prefetch: a question that names a time range is retrieved with that day filter, so
"昨天晚上在家聊了什么" ranks yesterday's utterances instead of whatever is lexically closest. The
agentic tools still get the raw question; the model converts dates itself there.

parse_range(text, now) -> (day_from, day_to) as 'YYYY-MM-DD' strings, or None when the text
carries no date. Only the first expression found is used.

parse_hours(text) -> (hour_from, hour_to) for a part of the day ("上午", "中午", "晚上", "昨晚" ...), and
parse_when(text, now) -> When(day_from, day_to, hours) combines both: "昨天晚上" is yesterday, 18:00-24:00.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

_CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_WEEKDAY = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}

_RE_DATE = re.compile(r"(?:(\d{4})[年/\-.])?(\d{1,2})[月/\-.](\d{1,2})[日号]?")
_RE_WEEK = re.compile(r"(上上|上|这|本|下)?(?:周|星期|礼拜)([一二三四五六日天])")
_RE_WEEK_SPAN = re.compile(r"(上上|上|这|本)(?:周|星期|礼拜)(?![一二三四五六日天])")
_RE_LAST_N_DAYS = re.compile(r"(?:最近|过去|这)\s*(\d+|[一二两三四五六七八九十]+)\s*天")
_RE_MONTH = re.compile(r"(上个?|这个?|本)月")

_FIXED = {
    "大前天": (-3, -3), "前天": (-2, -2), "昨天": (-1, -1), "昨晚": (-1, -1), "今天": (0, 0), "今晚": (0, 0),
    "今早": (0, 0), "明天": (1, 1), "后天": (2, 2),
    "最近几天": (-6, 0), "这几天": (-6, 0), "最近": (-6, 0), "这两天": (-1, 0), "这三天": (-2, 0),
}


def _cn_int(s: str) -> int:
    if s.isdigit():
        return int(s)
    total = 0
    if "十" in s:
        a, _, b = s.partition("十")
        total = (_CN_NUM.get(a, 1) if a else 1) * 10 + (_CN_NUM.get(b, 0) if b else 0)
        return total
    return _CN_NUM.get(s, 0)


def _span(d0: date, d1: date) -> tuple[str, str]:
    return d0.isoformat(), d1.isoformat()


def parse_range(text: str, now: datetime | date | None = None) -> tuple[str, str] | None:
    """First date expression in `text` as an inclusive (day_from, day_to) pair, or None."""
    today = (now or datetime.now())
    today = today.date() if isinstance(today, datetime) else today
    t = (text or "").replace(" ", "")
    if not t:
        return None

    cands: list[tuple[int, tuple[str, str]]] = []

    # explicit dates: 10月28号 / 2026-10-28 / 10/28
    for m in _RE_DATE.finditer(t):
        y = int(m[1]) if m[1] else today.year
        try:
            d = date(y, int(m[2]), int(m[3]))
        except ValueError:
            continue
        if not m[1] and d > today + timedelta(days=1):   # "10月28号" said in January -> last year
            d = d.replace(year=y - 1)
        cands.append((m.start(), _span(d, d)))

    # weekdays: 周三 / 上周三 / 这周五 / 下周一
    for m in _RE_WEEK.finditer(t):
        wd = _WEEKDAY[m[2]]
        this_week_day = today - timedelta(days=today.weekday()) + timedelta(days=wd)
        prefix = m[1] or ""
        if prefix in ("这", "本"):
            d = this_week_day
        elif prefix == "上":
            d = this_week_day - timedelta(days=7)
        elif prefix == "上上":
            d = this_week_day - timedelta(days=14)
        elif prefix == "下":
            d = this_week_day + timedelta(days=7)
        else:   # bare 周三: the most recent one (today counts)
            d = this_week_day if this_week_day <= today else this_week_day - timedelta(days=7)
        cands.append((m.start(), _span(d, d)))

    # whole weeks: 上周 / 这周 / 本周
    for m in _RE_WEEK_SPAN.finditer(t):
        monday = today - timedelta(days=today.weekday())
        if m[1] == "上":
            monday -= timedelta(days=7)
        elif m[1] == "上上":
            monday -= timedelta(days=14)
        cands.append((m.start(), _span(monday, min(monday + timedelta(days=6), today))))

    # 最近 N 天 / 过去三天
    for m in _RE_LAST_N_DAYS.finditer(t):
        n = max(1, _cn_int(m[1]))
        cands.append((m.start(), _span(today - timedelta(days=n - 1), today)))

    # 上个月 / 这个月
    for m in _RE_MONTH.finditer(t):
        first = today.replace(day=1)
        if m[1] and m[1].startswith("上"):
            last_prev = first - timedelta(days=1)
            cands.append((m.start(), _span(last_prev.replace(day=1), last_prev)))
        else:
            cands.append((m.start(), _span(first, today)))

    # fixed words (longest first so 大前天 beats 前天, 最近几天 beats 最近)
    for word in sorted(_FIXED, key=len, reverse=True):
        i = t.find(word)
        if i >= 0 and not any(c[0] <= i < c[0] + len(word) for c in cands):
            a, b = _FIXED[word]
            cands.append((i, _span(today + timedelta(days=a), today + timedelta(days=b))))

    if not cands:
        return None
    cands.sort(key=lambda c: c[0])
    return cands[0][1]


# --- parts of the day --------------------------------------------------------------------------------
#: (words, (hour_from, hour_to)). The windows are generous on purpose: people say "晚上" for 18:30 and for
#: 23:00, and the filter only narrows a search that is ranked afterwards anyway.
_PARTS: tuple[tuple[tuple[str, ...], tuple[float, float]], ...] = (
    (("凌晨", "后半夜", "半夜"), (0.0, 6.0)),
    (("大清早", "清晨", "早晨", "早上", "一早", "今早", "明早", "早饭", "早餐"), (5.0, 11.0)),
    (("上午",), (6.0, 12.0)),
    (("中午", "正午", "午饭", "午餐", "午休"), (11.0, 14.0)),
    (("下午", "午后"), (12.0, 18.5)),
    (("傍晚", "黄昏", "晚饭", "晚餐"), (17.0, 20.5)),
    (("深夜",), (21.0, 24.0)),
    (("晚上", "夜里", "夜晚", "昨晚", "今晚", "明晚", "当晚", "昨夜", "今夜"), (18.0, 24.0)),
)


@dataclass(frozen=True)
class When:
    """The time a question is about: an inclusive day range and / or a part of the day (clock hours)."""
    day_from: str | None = None
    day_to: str | None = None
    hours: tuple[float, float] | None = None

    @property
    def label(self) -> str:
        """Short description for the prompt: '2026-10-29', '2026-10-27~2026-10-29 18:00-24:00', '11:00-14:00'."""
        parts: list[str] = []
        if self.day_from or self.day_to:
            a, b = self.day_from or "…", self.day_to or "…"
            parts.append(a if a == b else f"{a}~{b}")
        if self.hours:
            def hm(h: float) -> str:
                return f"{int(h):02d}:{int(round((h - int(h)) * 60)):02d}"
            parts.append(f"{hm(self.hours[0])}-{hm(self.hours[1])}")
        return " ".join(parts)


def parse_hours(text: str) -> tuple[float, float] | None:
    """Clock window of the first part-of-day word in `text` ("上午" -> (6, 12)), or None."""
    t = (text or "").replace(" ", "")
    best: tuple[int, int, tuple[float, float]] | None = None      # (position, -length, window)
    for words, window in _PARTS:
        for w in words:
            i = t.find(w)
            if i >= 0 and (best is None or (i, -len(w)) < best[:2]):
                best = (i, -len(w), window)
    return best[2] if best else None


def parse_when(text: str, now: datetime | date | None = None) -> When | None:
    """Day range (parse_range) and part of the day (parse_hours) of a question; None when it names neither."""
    span = parse_range(text, now)
    hours = parse_hours(text)
    if span is None and hours is None:
        return None
    return When(span[0] if span else None, span[1] if span else None, hours)
