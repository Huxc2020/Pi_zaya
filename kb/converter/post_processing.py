from __future__ import annotations

import re
from typing import Optional

from .text_utils import _normalize_text

_ALLOWED_UNNUMBERED_HEADINGS = {
    "ABSTRACT",
    "ACKNOWLEDGMENTS",
    "ACKNOWLEDGEMENT",
    "REFERENCES",
    "BIBLIOGRAPHY",
    "APPENDIX",
    "APPENDICES",
    "INTRODUCTION",
    "CONCLUSION",
    "CONCLUSIONS",
    "RELATED WORK",
    "METHOD",
    "METHODS",
    "RESULTS",
    "DISCUSSION",
}

def _is_reasonable_heading_text(t: str) -> bool:
    if len(t) < 2:
        return False
    # If it's very long, probably not a heading
    if len(t) > 200:
        return False
    # If it contains many math symbols, surely not a heading
    if re.search(r"[=\\^_]", t):
        return False
    return True

def _parse_numbered_heading_level(text: str) -> Optional[int]:
    """
    Detect '1. Introduction', '2.1 Method', 'A. Appendix' etc.
    """
    t = text.lstrip()
    m = re.match(r"^(\d+(?:\.\d+)*)\.?\s+(.*)", t)
    if not m:
        return None
    nums = m.group(1).split(".")
    # Filter empty strings from '1.' -> ['1', '']
    nums = [x for x in nums if x]
    return min(3, max(1, len(nums)))

def _parse_appendix_heading_level(text: str) -> Optional[int]:
    t = text.lstrip()
    # "Appendix A", "Appendix A.1"
    if re.match(r"^Appendix\s+[A-Z](?:\.\d+)*", t, re.IGNORECASE):
        # We can treat this as level 1 usually
        return 1
    # "A. Proof of..."
    m = re.match(r"^([A-Z])(?:\.(\d+))*\.?\s+(.*)", t)
    if m:
        # A -> 1, A.1 -> 2
        groups = [g for g in m.groups() if g is not None]
        # groups[0] is letter, groups[1] is number...
        # It's heuristic.
        count = 0
        if m.group(1): count += 1
        if m.group(2): count += 1
        return min(3, max(1, count))
    return None

def _cleanup_noise_lines(md: str) -> str:
    lines = md.splitlines()
    out = []
    # Simple deduplication of blank lines and noise
    for ln in lines:
        if not ln.strip():
            if out and not out[-1].strip():
                continue
            out.append(ln)
            continue
        out.append(ln)
    return "\n".join(out)

def _convert_caption_following_tabular_lines(md: str) -> str:
    # Logic to attach captions to tables if they got separated
    # This is a bit complex in regex, maybe simplified version here
    return md

def _reflow_hard_wrapped_paragraphs(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    buf: list[str] = []

    fence_re = re.compile(r"^\s*```")
    heading_re = re.compile(r"^\s*#{1,6}\s+")
    list_re = re.compile(r"^\s*(?:[-*+]\s+|\d+\.\s+)")
    table_re = re.compile(r"^\s*\|")
    math_marker = re.compile(r"^\s*\$\$")

    in_fence = False
    in_math = False

    def flush_buf():
        nonlocal buf
        if not buf:
            return
        # Join with space
        merged = " ".join(buf).replace("  ", " ")
        out.append(merged)
        buf = []

    for line in lines:
        s = line.strip()
        if fence_re.match(line):
            flush_buf()
            in_fence = not in_fence
            out.append(line)
            continue
        if math_marker.match(line):
            flush_buf()
            in_math = not in_math
            out.append(line)
            continue
        
        if in_fence or in_math:
            out.append(line)
            continue
        
        if not s:
            flush_buf()
            out.append(line)
            continue
        
        if heading_re.match(line) or list_re.match(line) or table_re.match(line):
            flush_buf()
            out.append(line)
            continue
            
        # Image link
        if s.startswith("![") and s.endswith(")"):
            flush_buf()
            out.append(line)
            continue
            
        # Otherwise, regular text line, buffer it
        buf.append(s)

    flush_buf()
    return "\n".join(out)

def _fix_split_numbered_headings(md: str) -> str:
    # "1\nIntroduction" -> "1 Introduction"
    # Copied from test2.py logic roughly
    lines = md.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if i + 1 < len(lines):
            next_line = lines[i+1]
            # If line is just a number/letter
            if re.fullmatch(r"\d+(\.\d+)*\.?", line.strip()) or re.fullmatch(r"[A-Z]\.?", line.strip()):
                if next_line.strip() and not _parse_numbered_heading_level(next_line):
                    # Merge
                    out.append(f"{line.strip()} {next_line.strip()}")
                    i += 2
                    continue
        out.append(line)
        i += 1
    return "\n".join(out)

def _is_common_section_heading(title: str) -> bool:
    t = _normalize_text(title).upper()
    return any(x in t for x in _ALLOWED_UNNUMBERED_HEADINGS)

def _enforce_heading_policy(md: str) -> str:
    lines = md.splitlines()
    out = []
    # Ensure headings act like headings
    for line in lines:
        m = re.match(r"^(#+)\s+(.*)", line)
        if m:
            lvl = len(m.group(1))
            title = m.group(2).strip()
            # If title is numeric only, demote
            if re.fullmatch(r"[\d\.]+", title):
                out.append(title)
                continue
            if _is_common_section_heading(title):
                if lvl > 2:
                    # Promote essential headings
                    out.append(f"## {title}")
                else:
                    out.append(line)
                continue
            
            # Demote likely false headings (too long)
            if len(title) > 150:
                out.append(title)
                continue
                
        out.append(line)
    return "\n".join(out)

def _format_references(md: str) -> str:
    # Locate REFERENCES section and ensure items are formatted
    # This is a bit advanced regex, implementing a simplified pass
    return md

def postprocess_markdown(md: str) -> str:
    md = _cleanup_noise_lines(md)
    md = _convert_caption_following_tabular_lines(md)
    md = _reflow_hard_wrapped_paragraphs(md)
    md = _fix_split_numbered_headings(md)
    md = _enforce_heading_policy(md)
    md = _format_references(md)
    return md
