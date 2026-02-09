from __future__ import annotations


def _top_heading(heading_path: str) -> str:
    hp = (heading_path or "").strip()
    if not hp:
        return ""
    # Our heading_path uses " / " as a stack separator.
    return hp.split(" / ", 1)[0].strip()


def _format_context(hits: list[dict], max_chars: int = 12000) -> str:
    parts: list[str] = []
    used = 0
    for i, h in enumerate(hits, start=1):
        meta = h.get("meta", {}) or {}
        header = f"[{i}] source: {meta.get('source_path', '')}"
        top = _top_heading(meta.get("heading_path", ""))
        if top:
            header += f" | section: {top}"
        body = h.get("text", "")
        chunk = header + "\n" + body
        if used + len(chunk) > max_chars:
            break
        parts.append(chunk)
        used += len(chunk)
    return "\n\n---\n\n".join(parts)


def build_messages(
    user_query: str,
    history: list[dict],
    hits: list[dict],
) -> list[dict]:
    # Keep system prompt simple and strict: use context first, cite [n], don't hallucinate.
    system = (
        "你是我的个人知识库助手。你会先阅读我提供的“检索片段”（可能来自多篇论文/笔记），"
        "然后用中文回答问题。\n"
        "规则：\n"
        "1) 如果检索片段存在：优先基于检索片段回答；需要引用时，用 [1] [2] 这样的编号标注引用来源。\n"
        "2) 如果检索片段为空：也要给出可用的通用回答，但必须在回答开头明确标注“未命中知识库片段”。\n"
        "3) 不要编造不存在的论文、公式、数据或结论。\n"
        "4) 输出要求：\n"
        "   - 先给出“回答”。\n"
        "   - 最后给出“可参考定位：”并列出你实际引用到的编号，每条都写清楚对应的 source + section（大标题即可）。\n"
        "   - 若未命中片段：在“可参考定位”里写“（本次未命中）”，并建议去你认为最相关的论文的 REFERENCES/Related Work 追溯。\n"
    )

    ctx = _format_context(hits)
    user = (
        "问题：\n"
        f"{user_query}\n\n"
        "检索片段：\n"
        f"{ctx if ctx else '(无)'}\n"
    )

    # Only keep user/assistant roles in history
    trimmed = [m for m in history if m.get("role") in ("user", "assistant")]
    return [{"role": "system", "content": system}, *trimmed, {"role": "user", "content": user}]
