"""Edit values in a hand-written config.yaml without losing its comments.

    text = set_values(text, {("mimo", "chat_model"): "mimo-v2.6-pro", ("push", "feishu", "webhook"): "https://…"})

Only what the settings page needs is supported: 2- or 3-level paths (section.key / section.sub.key)
whose values are scalars, flat lists or flat dicts. Block mappings, inline `{a: 1}` flow mappings and
trailing `# comments` are all handled; a missing key is appended to its section, a missing section
to the end of the file. The result is parsed back and every requested value is verified; if anything
does not match (exotic YAML), the file is re-dumped from data instead, which keeps the values but
drops comments — `set_values` reports that through `EditResult.comments_lost`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import yaml

Path_ = tuple[str, ...]


@dataclass
class EditResult:
    text: str
    comments_lost: bool = False
    problems: list[str] = field(default_factory=list)


def render(value: Any) -> str:
    """One-line YAML for a scalar / flat list / flat dict ('22:30' gets quoted, '' -> '')."""
    s = yaml.safe_dump(value, default_flow_style=True, allow_unicode=True, width=10 ** 6, sort_keys=False)
    s = s.strip()
    if s.endswith("..."):
        s = s[:-3].strip()
    return s


def split_value_comment(rest: str) -> tuple[str, str]:
    """'value   # comment' -> ('value', '   # comment'); quote- and brace-aware."""
    quote: str | None = None
    depth = 0
    for i, ch in enumerate(rest):
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth = max(0, depth - 1)
        elif ch == "#" and depth == 0 and (i == 0 or rest[i - 1] in " \t"):
            # i == 0: `key:   # comment` -- the regex already consumed the blanks before '#', so put one
            # back or the rebuilt line reads `key: value# comment` (the '#' would join a plain scalar)
            return rest[:i].rstrip(), rest[i - 1:] if i > 0 else " " + rest
    return rest.rstrip(), ""


_KEY_RE = re.compile(r"^(\s*)([A-Za-z0-9_.\-]+):(\s*)(.*)$")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _is_blank_or_comment(line: str) -> bool:
    s = line.strip()
    return not s or s.startswith("#")


def _find_key(lines: list[str], start: int, end: int, indent: int, key: str) -> int | None:
    """Index of the line `<indent spaces>key:` within lines[start:end]."""
    for i in range(start, end):
        m = _KEY_RE.match(lines[i])
        if m and len(m.group(1)) == indent and m.group(2) == key:
            return i
    return None


def _block_end(lines: list[str], start: int, indent: int) -> int:
    """First index after `start` whose content is at indent <= `indent` (ignoring blanks/comments)."""
    i = start + 1
    while i < len(lines):
        if not _is_blank_or_comment(lines[i]) and _indent(lines[i]) <= indent:
            return i
        i += 1
    return len(lines)


def _child_indent(lines: list[str], start: int, end: int, fallback: int) -> int:
    for i in range(start + 1, end):
        if not _is_blank_or_comment(lines[i]):
            return _indent(lines[i])
    return fallback


def _last_content(lines: list[str], start: int, end: int) -> int:
    """Index after the last non-blank line in [start, end) — where an appended key goes."""
    i = end
    while i > start + 1 and _is_blank_or_comment(lines[i - 1]) and not lines[i - 1].strip().startswith("#"):
        i -= 1
    return i


def _set_in_block(lines: list[str], start: int, end: int, indent: int, path: Path_, value: Any) -> tuple[list[str], int]:
    """Set `path` (relative to the mapping whose children sit at `indent` within lines[start:end]).
    Returns (lines, delta in line count)."""
    key, rest = path[0], path[1:]
    idx = _find_key(lines, start, end, indent, key)
    if idx is None:
        # append the key (and any nested keys) at the end of this block
        at = _last_content(lines, start - 1, end)
        new = _emit(indent, path, value)
        lines[at:at] = new
        return lines, len(new)
    m = _KEY_RE.match(lines[idx])
    assert m
    lead, _, _, after = m.groups()
    val, comment = split_value_comment(after)
    if not rest:
        if val == "" and _has_nested_block(lines, idx, indent):
            # a block (e.g. a list written as "- item" lines) is being replaced by an inline value
            nested_end = _block_end(lines, idx, indent)
            del lines[idx + 1:nested_end]
            removed = nested_end - idx - 1
        else:
            removed = 0
        lines[idx] = f"{lead}{key}: {render(value)}{comment}"
        return lines, -removed
    # nested path
    if val.startswith("{"):
        try:
            cur = yaml.safe_load(val) or {}
        except yaml.YAMLError:
            cur = {}
        if not isinstance(cur, dict):
            cur = {}
        _set_nested(cur, rest, value)
        lines[idx] = f"{lead}{key}: {render(cur)}{comment}"
        return lines, 0
    if val == "":
        nested_end = _block_end(lines, idx, indent)
        child = _child_indent(lines, idx, nested_end, indent + 2)
        return _set_in_block(lines, idx + 1, nested_end, child, rest, value)
    # scalar where a mapping is needed: replace it by a block mapping
    lines[idx] = f"{lead}{key}:{comment}"
    new = _emit(indent + 2, rest, value)
    lines[idx + 1:idx + 1] = new
    return lines, len(new)


def _has_nested_block(lines: list[str], idx: int, indent: int) -> bool:
    j = idx + 1
    while j < len(lines) and _is_blank_or_comment(lines[j]):
        j += 1
    return j < len(lines) and _indent(lines[j]) > indent


def _emit(indent: int, path: Path_, value: Any) -> list[str]:
    pad = " " * indent
    if len(path) == 1:
        return [f"{pad}{path[0]}: {render(value)}"]
    return [f"{pad}{path[0]}:"] + _emit(indent + 2, path[1:], value)


def _set_nested(d: dict, path: Path_, value: Any) -> None:
    cur = d
    for k in path[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[path[-1]] = value


def _get_nested(d: Any, path: Path_) -> Any:
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return _MISSING
        cur = cur[k]
    return cur


_MISSING = object()


def set_values(text: str, updates: dict[Path_, Any]) -> EditResult:
    """Apply every (path -> value) to the YAML text, preserving comments where possible."""
    if not updates:
        return EditResult(text)
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.split("\n")
    lines = [ln.rstrip("\r") for ln in lines]
    problems: list[str] = []
    for path, value in updates.items():
        path = tuple(str(p) for p in path)
        if len(path) < 2:
            problems.append(f"{'.'.join(path)}: 路径至少两级")
            continue
        try:
            section = path[0]
            idx = _find_key(lines, 0, len(lines), 0, section)
            if idx is None:
                if lines and lines[-1].strip():
                    lines.append("")
                lines.extend(_emit(0, path, value))
                continue
            m = _KEY_RE.match(lines[idx])
            assert m
            val, comment = split_value_comment(m.group(4))
            if val.startswith("{"):
                cur = yaml.safe_load(val) or {}
                if not isinstance(cur, dict):
                    cur = {}
                _set_nested(cur, path[1:], value)
                lines[idx] = f"{section}: {render(cur)}{comment}"
                continue
            if val != "":
                lines[idx] = f"{section}:{comment}"
            end = _block_end(lines, idx, 0)
            child = _child_indent(lines, idx, end, 2)
            lines, _ = _set_in_block(lines, idx + 1, end, child, path[1:], value)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{'.'.join(path)}: {e}")
    new_text = nl.join(lines)
    # verify
    ok = not problems
    if ok:
        try:
            data = yaml.safe_load(new_text) or {}
            for path, value in updates.items():
                got = _get_nested(data, tuple(str(p) for p in path))
                want = yaml.safe_load(render(value))
                if got is _MISSING or got != want:
                    ok = False
                    problems.append(f"{'.'.join(path)}: 写入后读回不一致 ({got!r} != {want!r})")
                    break
        except yaml.YAMLError as e:
            ok = False
            problems.append(f"结果不是合法 YAML: {e}")
    if ok:
        return EditResult(new_text, False, [])
    # fallback: structured rewrite (comments are lost)
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    for path, value in updates.items():
        _set_nested(data, tuple(str(p) for p in path), value)
    dumped = yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=10 ** 6)
    return EditResult(dumped.replace("\n", nl), True, problems)


def flatten(d: dict, prefix: Path_ = ()) -> dict[Path_, Any]:
    """{'a': {'b': 1, 'c': {'d': 2}}} -> {('a','b'): 1, ('a','c','d'): 2}; lists stay as leaves."""
    out: dict[Path_, Any] = {}
    for k, v in (d or {}).items():
        p = prefix + (str(k),)
        if isinstance(v, dict) and len(p) < 3:
            out.update(flatten(v, p))
        else:
            out[p] = v
    return out
