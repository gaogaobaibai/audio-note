"""The one parser for glossary.txt (correction.glossary_file).

Three modules used to read this file with three slightly different rules (pipeline.load_glossary,
correct.load_glossary, memory.index.load_glossary): a line such as `灰度发布 # 上线方式` was the term `灰度发布`
for one of them and the whole line for the two that were actually used, so the comment leaked into the correction
prompt and into jieba. They are meant to be thin wrappers around this module.

Format
------
- one term per line; surrounding whitespace is ignored; a UTF-8 BOM is tolerated;
- blank lines and lines starting with `#` are ignored;
- an inline comment starts at a `#` that follows whitespace (`术语 # 说明`); a `#` inside a term (`C#`) is kept;
- duplicates are dropped, the first occurrence keeps its place.

No project imports: any module may use it.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

_BOM = chr(0xFEFF)
_INLINE_COMMENT = re.compile(r"\s#.*$")


def parse_line(line: str) -> str | None:
    """The term on one line, or None for a blank / comment line."""
    s = line.strip().lstrip(_BOM).strip()
    if not s or s.startswith("#"):
        return None
    s = _INLINE_COMMENT.sub("", s).strip()
    return s or None


def parse_glossary(text: str) -> list[str]:
    """Terms of a glossary text, in file order, without duplicates."""
    out: list[str] = []
    seen: set[str] = set()
    for line in (text or "").splitlines():
        term = parse_line(line)
        if term is not None and term not in seen:
            seen.add(term)
            out.append(term)
    return out


def read_glossary(path: str | os.PathLike | None) -> list[str]:
    """Terms of the glossary file at `path`. No path / no such file -> []. Read and decoding errors propagate
    (OSError, UnicodeDecodeError): the caller decides whether a broken glossary is fatal."""
    if not path:
        return []
    p = Path(path)
    if not p.is_file():
        return []
    with open(p, "r", encoding="utf-8-sig") as f:
        return parse_glossary(f.read())


__all__ = ["parse_glossary", "parse_line", "read_glossary"]
