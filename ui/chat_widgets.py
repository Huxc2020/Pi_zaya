from __future__ import annotations

import base64
import html
import mimetypes
import os
import re
import time
from pathlib import Path

import streamlit as st

from ui.strings import S

def _resolve_sidebar_logo_path() -> Path | None:
    base = Path(__file__).resolve().parent
    env_logo = (os.environ.get("KB_SIDEBAR_LOGO") or "").strip().strip("'\"")
    candidates: list[Path] = []
    if env_logo:
        p = Path(env_logo).expanduser()
        if not p.is_absolute():
            candidates.append((base / p).resolve())
        candidates.append(p.resolve())
    candidates.extend(
        [
            base / "team_logo.png",
            base / "assets" / "team_logo.png",
            base / "assets" / "team_logo.jpg",
            base / "assets" / "team_logo.jpeg",
            base / "assets" / "team_logo.webp",
        ]
    )
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                return p
        except Exception:
            continue
    return None

def _sidebar_logo_data_uri(path: Path) -> str | None:
    try:
        raw = path.read_bytes()
        if not raw:
            return None
        mime, _ = mimetypes.guess_type(str(path))
        if not mime:
            mime = "image/png"
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None

def _resolve_ai_inline_logo_data_uri() -> str | None:
    cache_key = "_kb_ai_inline_logo_data_uri"
    miss_key = "__MISS__"
    cached = st.session_state.get(cache_key, None)
    if cached == miss_key:
        return None
    if isinstance(cached, str) and cached.startswith("data:"):
        return cached

    base = Path(__file__).resolve().parent
    env_logo = (os.environ.get("KB_INLINE_AI_LOGO") or "").strip().strip("'\"")
    candidates: list[Path] = []
    if env_logo:
        p = Path(env_logo).expanduser()
        if not p.is_absolute():
            candidates.append((base / p).resolve())
        candidates.append(p.resolve())
    candidates.extend(
        [
            base / "assets" / "pi_logo.png",
            base / "pi_logo.png",
            base / "assets" / "team_logo.png",
        ]
    )
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                uri = _sidebar_logo_data_uri(p)
                if uri:
                    st.session_state[cache_key] = uri
                    return uri
        except Exception:
            continue

    st.session_state[cache_key] = miss_key
    return None

def _render_app_title() -> None:
    title = str(S.get("title") or "").strip()
    if not title:
        return
    safe_title = html.escape(title)
    if bool(st.session_state.get("_hero_title_typed_once")):
        st.markdown(f"<h1 class='kb-hero-title'>{safe_title}</h1>", unsafe_allow_html=True)
        return

    holder = st.empty()
    acc: list[str] = []
    for ch in title:
        acc.append(ch)
        live = html.escape("".join(acc))
        holder.markdown(
            f"<h1 class='kb-hero-title'>{live}<span class='kb-title-caret'>▌</span></h1>",
            unsafe_allow_html=True,
        )
        time.sleep(0.020 if ord(ch) < 128 else 0.028)

    holder.markdown(f"<h1 class='kb-hero-title'>{safe_title}</h1>", unsafe_allow_html=True)
    st.session_state["_hero_title_typed_once"] = True

def _normalize_math_markdown(text: str) -> str:
    """
    Make math rendering more stable in Streamlit markdown.

    Goals:
    - Inline math: $...$
    - Display math: $$...$$
    - Avoid code spans wrapping math (backticks break KaTeX/MathJax).
    """
    if not text:
        return text

    import re

    s = text

    # Prefer $...$ and $$...$$ over \( \) and \[ \], but do it conservatively.
    # Avoid touching escaped citation brackets like \[24\].
    def _inline_math_repl(m: re.Match) -> str:
        inner = str(m.group(1) or "")
        return f"${inner}$"

    def _display_math_repl(m: re.Match) -> str:
        inner = str(m.group(1) or "")
        probe = inner.strip()
        # Keep citation-like escaped brackets untouched.
        if re.fullmatch(r"\d{1,4}(?:\s*[,;，、-]\s*\d{1,4})*", probe):
            return m.group(0)
        # Convert only when it reasonably looks like math/display content.
        looks_math = bool(
            ("\n" in inner)
            or re.search(
                r"[=^_{}]|\\(?:frac|sum|int|prod|sqrt|mathbf|mathbb|left|right|begin|end|alpha|beta|gamma|theta|lambda|cdot|times)",
                inner,
            )
        )
        if not looks_math:
            return m.group(0)
        return "$$" + inner + "$$"

    s = re.sub(r"\\\((.+?)\\\)", _inline_math_repl, s, flags=re.DOTALL)
    s = re.sub(r"\\\[(.+?)\\\]", _display_math_repl, s, flags=re.DOTALL)

    # Unwrap math that was mistakenly put in code spans.
    s = re.sub(r"`(\$[^`]+?\$)`", r"\1", s)
    s = re.sub(r"`(\$\$[\s\S]+?\$\$)`", r"\1", s)

    return s

def _md_to_plain_text(md: str) -> str:
    """
    Best-effort Markdown -> plain text for clipboard copy.
    Keeps formulas as their LaTeX source ($$...$$ / $...$).
    """
    if not md:
        return ""

    s = md
    # Remove code fences but keep their content.
    s = re.sub(r"```[^\n]*\n", "", s)
    s = s.replace("```", "")
    # Remove inline code backticks.
    s = re.sub(r"`([^`]+)`", r"\1", s)
    # Links: [text](url) -> text
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    # Images: ![alt](url) -> alt
    s = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", s)
    # Basic emphasis markers
    s = s.replace("**", "").replace("__", "").replace("*", "").replace("_", "")
    # Headings/list markers
    s = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", s)
    s = re.sub(r"(?m)^\s*[-*+]\s+", "", s)
    s = re.sub(r"(?m)^\s*\d+\.\s+", "", s)
    # Collapse extra blank lines
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s

def _render_answer_copy_bar(answer_md: str, *, key_ns: str) -> None:
    md = _normalize_math_markdown(answer_md or "")
    txt = _md_to_plain_text(md)
    md_id = f"{key_ns}_md"
    txt_id = f"{key_ns}_txt"
    ai_logo_uri = _resolve_ai_inline_logo_data_uri()
    logo_html = (
        f'<img class="kb-ai-inline-logo" src="{html.escape(ai_logo_uri)}" alt="AI"/>'
        if ai_logo_uri
        else ""
    )

    # Hidden payloads (large text lives here, buttons reference them by id).
    st.markdown(
        f"""
<textarea id="{html.escape(md_id)}" style="display:none">{html.escape(md)}</textarea>
<textarea id="{html.escape(txt_id)}" style="display:none">{html.escape(txt)}</textarea>
<div class="kb-copybar">
  {logo_html}
  <button class="kb-copybtn" type="button" data-target="{html.escape(txt_id)}">复制文本</button>
  <button class="kb-copybtn" type="button" data-target="{html.escape(md_id)}">复制 Markdown</button>
</div>
        """,
        unsafe_allow_html=True,
    )

def _render_ai_live_header(stage: str = "") -> None:
    logo_uri = _resolve_ai_inline_logo_data_uri()
    safe_stage = html.escape((stage or "").strip())
    logo_html = (
        f'<img class="kb-ai-live-logo" src="{html.escape(logo_uri)}" alt="AI"/>'
        if logo_uri
        else '<span class="msg-meta">AI</span>'
    )
    stage_html = f'<span class="kb-ai-live-stage">阶段：{safe_stage}</span>' if safe_stage else ""
    st.markdown(
        f"""
<div class="kb-ai-livebar">
  {logo_html}
  <span class="kb-ai-live-pill">生成中</span>
  {stage_html}
</div>
        """,
        unsafe_allow_html=True,
    )
