from __future__ import annotations

import re

_RE_WORD = re.compile(
    r"[A-Za-z0-9_]+|[\u4e00-\u9fff]",  # English/number tokens OR single CJK char
    flags=re.UNICODE,
)


def tokenize(text: str) -> list[str]:
    # Lowercase only for latin tokens.
    out: list[str] = []
    for t in _RE_WORD.findall(text):
        if len(t) == 1 and "\u4e00" <= t <= "\u9fff":
            out.append(t)
        else:
            out.append(t.lower())
    return out

