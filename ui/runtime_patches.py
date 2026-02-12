from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

_CHAT_DOCK_JS_PATH = Path(__file__).resolve().parent / "assets" / "chat_dock_runtime.js"
_CHAT_DOCK_JS_CACHE: str | None = None

def _init_theme_css(theme_mode: str = "dark") -> None:
    mode = "dark" if str(theme_mode or "").lower() == "dark" else "light"
    color_scheme = "dark" if mode == "dark" else "light"

    if mode == "dark":
        tokens = """
  --bg: #1f1f1f;
  --panel: #252526;
  --sidebar-bg: #181818;
  --line: rgba(168, 176, 189, 0.42);
  --muted: #d2d9e4;
  --text-main: #e7eaef;
  --text-soft: #e0e7f0;
  --sidebar-strong-text: #e9eff8;
  --sidebar-soft-text: #d5deea;
  --slider-tick-text: #d9e1ec;
  --accent: #4daafc;
  --blue-weak: rgba(77, 170, 252, 0.18);
  --blue-line: rgba(77, 170, 252, 0.58);
  --font-display: "LittleP", "Segoe UI", "Microsoft YaHei", "PingFang SC", system-ui, -apple-system, sans-serif;
  --font-body: "Segoe UI", "Microsoft YaHei", "PingFang SC", system-ui, -apple-system, sans-serif;
  --btn-bg: #2d2d30;
  --btn-border: #45494f;
  --btn-text: #e7eaef;
  --btn-hover: #37373d;
  --btn-active: #3f444c;
  --btn-shadow: 0 1px 0 rgba(0, 0, 0, 0.32), 0 12px 30px rgba(0, 0, 0, 0.42);
  --input-bg: #1f2632;
  --input-border: #505a6d;
  --msg-user-bg: rgba(77, 170, 252, 0.14);
  --msg-user-border: rgba(126, 179, 228, 0.40);
  --msg-user-text: #eaf3ff;
  --msg-ai-bg: #222934;
  --msg-ai-border: #3f4b5f;
  --snip-bg: rgba(148, 163, 184, 0.14);
  --snip-border: rgba(148, 163, 184, 0.34);
  --snip-text: #d7deea;
  --snip-quote-bg: rgba(77, 170, 252, 0.16);
  --snip-quote-border: rgba(77, 170, 252, 0.50);
  --snip-mark-bg: rgba(250, 204, 21, 0.28);
  --snip-mark-text: #f8fafc;
  --notice-text: #fde68a;
  --notice-bg: rgba(245, 158, 11, 0.20);
  --notice-border: rgba(245, 158, 11, 0.38);
  --ref-accent: rgba(77, 170, 252, 0.52);
  --dock-bg: linear-gradient(180deg, rgba(31, 31, 31, 0.72) 0%, rgba(31, 31, 31, 0.94) 20%, rgba(31, 31, 31, 0.98) 100%);
  --dock-border: rgba(148, 163, 184, 0.30);
  --dock-shadow: 0 -10px 28px rgba(0, 0, 0, 0.45);
  --copy-btn-bg: rgba(45, 45, 48, 0.94);
  --copy-btn-border: rgba(148, 163, 184, 0.34);
  --copy-btn-text: #dbe4f0;
  --toast-bg: rgba(36, 39, 45, 0.96);
  --toast-border: rgba(148, 163, 184, 0.30);
  --toast-text: #ebf1f8;
  --hint-text: #d2d9e4;
  --refs-title-text: #e7eaef;
  --refs-body-text: #dbe4f0;
  --code-bg: #171d28;
  --code-border: #3d4658;
  --code-text: #e6edf3;
  --code-inline-bg: rgba(77, 170, 252, 0.14);
  --code-syn-keyword: #c678dd;
  --code-syn-string: #98c379;
  --code-syn-comment: #7f848e;
  --code-syn-number: #d19a66;
  --code-syn-func: #61afef;
  --code-syn-type: #e5c07b;
  --code-syn-literal: #56b6c2;
  --code-syn-operator: #abb2bf;
"""
    else:
        tokens = """
  --bg: #fcfcfd;
  --panel: #ffffff;
  --sidebar-bg: #f7f8fa;
  --line: rgba(90, 98, 112, 0.24);
  --muted: rgba(55, 65, 81, 0.76);
  --text-main: #1f2329;
  --text-soft: #4b5563;
  --sidebar-strong-text: #1f2329;
  --sidebar-soft-text: #5a6472;
  --slider-tick-text: #5a6472;
  --accent: #0f6cbd;
  --blue-weak: rgba(15, 108, 189, 0.10);
  --blue-line: rgba(15, 108, 189, 0.40);
  --font-display: "LittleP", "Segoe UI", "Microsoft YaHei", "PingFang SC", system-ui, -apple-system, sans-serif;
  --font-body: "Segoe UI", "Microsoft YaHei", "PingFang SC", system-ui, -apple-system, sans-serif;
  --btn-bg: #ffffff;
  --btn-border: rgba(31, 35, 41, 0.16);
  --btn-text: #1f2329;
  --btn-hover: rgba(15, 108, 189, 0.08);
  --btn-active: rgba(15, 108, 189, 0.14);
  --btn-shadow: 0 1px 0 rgba(16, 24, 40, 0.04), 0 10px 24px rgba(16, 24, 40, 0.06);
  --input-bg: #ffffff;
  --input-border: rgba(31, 35, 41, 0.18);
  --msg-user-bg: #eef3f9;
  --msg-user-border: rgba(108, 134, 170, 0.34);
  --msg-user-text: #1f2a37;
  --msg-ai-bg: #ffffff;
  --msg-ai-border: rgba(49, 51, 63, 0.12);
  --snip-bg: rgba(49, 51, 63, 0.04);
  --snip-border: rgba(49, 51, 63, 0.12);
  --snip-text: rgba(15, 23, 42, 0.90);
  --snip-quote-bg: rgba(15, 108, 189, 0.08);
  --snip-quote-border: rgba(15, 108, 189, 0.30);
  --snip-mark-bg: rgba(251, 191, 36, 0.36);
  --snip-mark-text: #0f172a;
  --notice-text: rgba(120, 53, 15, 0.95);
  --notice-bg: rgba(245, 158, 11, 0.10);
  --notice-border: rgba(245, 158, 11, 0.20);
  --ref-accent: rgba(15, 108, 189, 0.24);
  --dock-bg: linear-gradient(180deg, rgba(252, 252, 253, 0.76) 0%, rgba(252, 252, 253, 0.96) 18%, rgba(252, 252, 253, 0.99) 100%);
  --dock-border: rgba(49, 51, 63, 0.12);
  --dock-shadow: 0 -8px 26px rgba(16, 24, 40, 0.08);
  --copy-btn-bg: rgba(255, 255, 255, 0.88);
  --copy-btn-border: rgba(49, 51, 63, 0.16);
  --copy-btn-text: rgba(31, 42, 55, 0.90);
  --toast-bg: rgba(255, 255, 255, 0.95);
  --toast-border: rgba(49, 51, 63, 0.16);
  --toast-text: rgba(31, 42, 55, 0.88);
  --hint-text: rgba(75, 85, 99, 0.62);
  --refs-title-text: #1f2329;
  --refs-body-text: #4b5563;
  --code-bg: #f5f7fb;
  --code-border: rgba(31, 35, 41, 0.16);
  --code-text: #1f2329;
  --code-inline-bg: rgba(15, 108, 189, 0.10);
  --code-syn-keyword: #a626a4;
  --code-syn-string: #50a14f;
  --code-syn-comment: #a0a1a7;
  --code-syn-number: #986801;
  --code-syn-func: #4078f2;
  --code-syn-type: #c18401;
  --code-syn-literal: #0184bc;
  --code-syn-operator: #383a42;
"""

    css = """
<style>
:root{
__TOKENS__
  --text-color: var(--text-main);
  --secondary-text-color: var(--text-soft);
  --body-text-color: var(--text-main);
  --content-max: 1220px;
}
html, body{
  background: var(--bg) !important;
  color: var(--text-main) !important;
  color-scheme: __SCHEME__;
  font-family: var(--font-body);
}
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"]{
  background: var(--bg) !important;
}
header[data-testid="stHeader"]{
  background: var(--bg) !important;
  border-bottom: 1px solid var(--line) !important;
}
div[data-testid="stToolbar"],
div[data-testid="stStatusWidget"]{
  background: transparent !important;
}
div[data-testid="stToolbar"] button,
div[data-testid="stStatusWidget"] *{
  color: var(--text-soft) !important;
}
[data-stale="true"]{
  opacity: 1 !important;
  visibility: visible !important;
  pointer-events: none !important;
  filter: none !important;
  transition: none !important;
}
body.kb-live-streaming [data-testid="stAppViewContainer"],
body.kb-live-streaming [data-testid="stAppViewContainer"]{
  opacity: 1 !important;
  filter: none !important;
}
body.kb-live-streaming [data-stale="true"]{
  opacity: 1 !important;
  visibility: visible !important;
  pointer-events: none !important;
  filter: none !important;
  transition: none !important;
}
body.kb-resizing [data-testid="stAppViewContainer"],
body.kb-resizing [data-testid="stAppViewContainer"]{
  opacity: 1 !important;
  filter: none !important;
}
body.kb-resizing [data-testid="stAppViewContainer"] *{
  filter: none !important;
}
body.kb-resizing [data-stale="true"]{
  opacity: 1 !important;
  visibility: visible !important;
  pointer-events: none !important;
  filter: none !important;
  transition: none !important;
}
body.kb-resizing section[data-testid="stSidebar"]{
  background: var(--sidebar-bg) !important;
}
body.kb-resizing section[data-testid="stSidebar"] > div,
body.kb-resizing section[data-testid="stSidebar"] > div > div{
  background: var(--sidebar-bg) !important;
}
body.kb-resizing section[data-testid="stSidebar"] div[style*="z-index"]{
  background: transparent !important;
  opacity: 0 !important;
}
.block-container{
  width: 100%;
  max-width: var(--content-max);
  margin-left: auto !important;
  margin-right: auto !important;
  padding-top: 1.6rem;
  padding-bottom: 12.2rem;
}
section[data-testid="stSidebar"] > div:first-child{
  background: var(--sidebar-bg) !important;
  border-right: 1px solid var(--line) !important;
}
section[data-testid="stSidebar"]{
  background: var(--sidebar-bg) !important;
  --text-color: var(--sidebar-strong-text) !important;
  --secondary-text-color: var(--sidebar-soft-text) !important;
  --body-text-color: var(--sidebar-strong-text) !important;
}
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] > div > div{
  background: var(--sidebar-bg) !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button,
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button[aria-label*="Close"],
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button[aria-label*="关闭"]{
  width: 34px !important;
  min-width: 34px !important;
  height: 34px !important;
  min-height: 34px !important;
  padding: 0 !important;
  border-radius: 10px !important;
  border: 1px solid var(--btn-border) !important;
  background: color-mix(in srgb, var(--sidebar-bg) 76%, var(--panel)) !important;
  box-shadow: none !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  position: relative !important;
  font-size: 0 !important;
  line-height: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button:hover,
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button[aria-label*="Close"]:hover,
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button[aria-label*="关闭"]:hover{
  background: var(--btn-hover) !important;
  border-color: var(--blue-line) !important;
  transform: none !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button:active,
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button[aria-label*="Close"]:active,
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button[aria-label*="关闭"]:active{
  background: var(--btn-active) !important;
  border-color: var(--blue-line) !important;
  transform: none !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button svg,
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button[aria-label*="Close"] svg,
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button[aria-label*="关闭"] svg,
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button [data-testid="stIcon"],
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button[aria-label*="Close"] [data-testid="stIcon"],
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] button[aria-label*="关闭"] [data-testid="stIcon"]{
  display: none !important;
}
section[data-testid="stSidebar"] .kb-close-glyph{
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  font-size: 24px !important;
  line-height: 1 !important;
  font-weight: 500 !important;
  color: var(--text-main) !important;
  transform: translateY(-1px);
  pointer-events: none !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] *{
  color: var(--sidebar-strong-text) !important;
  opacity: 1 !important;
}
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"] *{
  color: var(--sidebar-soft-text) !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label,
section[data-testid="stSidebar"] div[data-testid="stRadio"] label *,
section[data-testid="stSidebar"] div[data-testid="stRadio"] p,
section[data-testid="stSidebar"] div[data-testid="stRadio"] span,
section[data-testid="stSidebar"] [role="radiogroup"] label,
section[data-testid="stSidebar"] [role="radiogroup"] label *,
section[data-testid="stSidebar"] [role="radiogroup"] p,
section[data-testid="stSidebar"] [role="radiogroup"] span{
  color: var(--text-main) !important;
  fill: var(--text-main) !important;
  opacity: 1 !important;
}
section[data-testid="stSidebar"] div[data-testid="stCheckbox"] label,
section[data-testid="stSidebar"] div[data-testid="stCheckbox"] label *,
section[data-testid="stSidebar"] div[data-testid="stCheckbox"] p,
section[data-testid="stSidebar"] div[data-testid="stCheckbox"] span{
  color: var(--text-main) !important;
  opacity: 1 !important;
}
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stTickBarMin"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stTickBarMax"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stSliderTickBarMin"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stSliderTickBarMax"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stSliderTickBar"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stThumbValue"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stSliderValue"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-baseweb="slider"] *,
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid*="TickBar"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid*="tick"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] [class*="stSlider"] *,
section[data-testid="stSidebar"] div[data-testid="stSlider"] [style*="color"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] div[style*="color"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] small,
section[data-testid="stSidebar"] div[data-testid="stSlider"] p,
section[data-testid="stSidebar"] div[data-testid="stSlider"] span{
  color: var(--sidebar-soft-text) !important;
  -webkit-text-fill-color: var(--sidebar-soft-text) !important;
  fill: var(--sidebar-soft-text) !important;
  opacity: 1 !important;
}
section[data-testid="stSidebar"] div[data-testid="stSlider"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] > div,
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid*="TickBar"]{
  opacity: 1 !important;
}
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid*="TickBar"]::before,
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid*="TickBar"]::after{
  color: var(--sidebar-soft-text) !important;
  -webkit-text-fill-color: var(--sidebar-soft-text) !important;
  opacity: 1 !important;
}
div[data-testid="stSlider"] [data-testid="stTickBarMin"],
div[data-testid="stSlider"] [data-testid="stTickBarMax"],
div[data-testid="stSlider"] [data-testid="stSliderTickBarMin"],
div[data-testid="stSlider"] [data-testid="stSliderTickBarMax"],
div[data-testid="stSlider"] [data-testid="stSliderTickBar"],
div[data-testid="stSlider"] [data-testid*="TickBarMin"],
div[data-testid="stSlider"] [data-testid*="TickBarMax"],
div[data-testid="stSlider"] [class*="TickBarMin"],
div[data-testid="stSlider"] [class*="TickBarMax"],
div[data-testid="stSlider"] [class*="tickBarMin"],
div[data-testid="stSlider"] [class*="tickBarMax"],
div[data-testid="stSlider"] .stSliderTickBar,
div[data-testid="stSlider"] .stSliderTickBar *,
div[data-testid="stSlider"] [data-testid*="TickBar"]{
  color: var(--slider-tick-text) !important;
  -webkit-text-fill-color: var(--slider-tick-text) !important;
  fill: var(--slider-tick-text) !important;
  stroke: var(--slider-tick-text) !important;
  opacity: 1 !important;
  filter: brightness(1.12) contrast(1.08) !important;
}
div[data-testid="stSlider"] [data-testid*="ThumbValue"],
div[data-testid="stSlider"] [data-testid="stSliderValue"],
div[data-testid="stSlider"] [class*="ThumbValue"]{
  color: var(--accent) !important;
  -webkit-text-fill-color: var(--accent) !important;
  fill: var(--accent) !important;
  opacity: 1 !important;
}
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-testid="stThumbValue"]{
  color: var(--accent) !important;
  -webkit-text-fill-color: var(--accent) !important;
  fill: var(--accent) !important;
  opacity: 1 !important;
}
.stMarkdown .katex,
.stMarkdown .katex *,
.stMarkdown .katex-display,
.stMarkdown .katex-display *,
.msg-ai .katex,
.msg-ai .katex *,
.msg-ai .katex-display,
.msg-ai .katex-display *{
  color: var(--text-main) !important;
  fill: var(--text-main) !important;
  opacity: 1 !important;
}
.stMarkdown mjx-container,
.stMarkdown mjx-container *,
.msg-ai mjx-container,
.msg-ai mjx-container *{
  color: var(--text-main) !important;
  fill: currentColor !important;
  stroke: currentColor !important;
  opacity: 1 !important;
}
.kb-sidebar-logo-wrap{
  display: flex;
  justify-content: center;
  align-items: center;
  margin: -4.2rem 0 0.28rem 0;
}
.kb-sidebar-logo-img{
  width: 220px;
  max-width: 86%;
  height: auto;
  display: block;
  object-fit: contain;
  image-rendering: -webkit-optimize-contrast;
  image-rendering: crisp-edges;
  transform: translateZ(0);
}
h1, h2, h3, h4, h5{
  color: var(--text-main) !important;
  letter-spacing: -0.01em;
}
h1{
  font-family: var(--font-display);
  font-weight: 800;
}
p, li, td, th{ color: var(--text-main) !important; }
small, .stCaption, .msg-meta, .refbox, .genbox, .chat-empty-state{ color: var(--muted) !important; }
.kb-hero-title{
  margin: 0.18rem 0 0.66rem 0 !important;
  color: var(--text-main) !important;
  letter-spacing: -0.012em !important;
  line-height: 1.04 !important;
  font-size: clamp(2.12rem, 3.2vw, 3.05rem) !important;
  font-family: var(--font-display) !important;
  font-weight: 820 !important;
}
.kb-title-caret{
  display: inline-block;
  color: var(--blue-line);
  margin-left: 0.08rem;
  animation: kb-title-caret-blink 0.72s step-end infinite;
  font-weight: 700;
}
@keyframes kb-title-caret-blink{ 0%,100%{opacity:1;} 50%{opacity:0;} }
@media (prefers-reduced-motion: reduce){ .kb-title-caret{ animation: none !important; } }

div.stButton > button,
button[kind]{
  background: var(--btn-bg) !important;
  border: 1px solid var(--btn-border) !important;
  color: var(--btn-text) !important;
  border-radius: 12px !important;
  padding: 0.44rem 0.88rem !important;
  font-weight: 620 !important;
  box-shadow: 0 1px 0 rgba(16, 24, 40, 0.03);
}
section[data-testid="stSidebar"] div.stButton > button{ width: 100%; }
div.stButton > button:hover,
button[kind]:hover{
  background: var(--btn-hover) !important;
  border-color: var(--blue-line) !important;
  box-shadow: var(--btn-shadow);
}
div.stButton > button:active,
button[kind]:active{
  background: var(--btn-active) !important;
  border-color: var(--blue-line) !important;
}
div.stButton > button:focus,
div.stButton > button:focus-visible,
button[kind]:focus,
button[kind]:focus-visible{
  outline: none !important;
  box-shadow: 0 0 0 2px var(--blue-weak) !important;
}

textarea,
input,
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea,
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] [data-baseweb="select"] > div{
  background: var(--input-bg) !important;
  color: var(--text-main) !important;
  border: 1px solid var(--input-border) !important;
  border-radius: 12px !important;
  box-shadow: none !important;
}
ul[data-testid="stSelectboxVirtualDropdown"],
div[role="listbox"]{
  background: var(--panel) !important;
  border: 1px solid var(--line) !important;
}
li[data-testid="stSelectboxVirtualDropdownOption"],
div[role="option"]{ color: var(--text-main) !important; }

div[data-baseweb="tab-list"]{
  gap: 0.48rem !important;
  border-bottom: 1px solid var(--line) !important;
  padding: 0.04rem 0 0.16rem 0 !important;
}
div[data-baseweb="tab-highlight"]{
  display: none !important;
}
button[data-baseweb="tab"]{
  background: color-mix(in srgb, var(--btn-bg) 84%, transparent) !important;
  border: 1px solid var(--btn-border) !important;
  color: var(--text-soft) !important;
  border-radius: 12px !important;
  padding: 0.42rem 0.9rem !important;
  font-weight: 620 !important;
  transition: background 140ms ease, border-color 140ms ease, color 140ms ease !important;
}
button[data-baseweb="tab"]:hover{
  background: var(--btn-hover) !important;
  border-color: var(--blue-line) !important;
  color: var(--text-main) !important;
}
button[data-baseweb="tab"][aria-selected="true"]{
  background: var(--blue-weak) !important;
  border-color: var(--blue-line) !important;
  color: var(--text-main) !important;
}

details[data-testid="stExpander"]{
  background: var(--panel) !important;
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
}
details[data-testid="stExpander"] summary,
details[data-testid="stExpander"] summary *{ color: var(--text-main) !important; }

[data-testid="stFileUploaderDropzone"]{
  background: var(--panel) !important;
  border: 1px dashed var(--input-border) !important;
  border-radius: 14px !important;
}
[data-testid="stFileUploaderDropzone"] *{ color: var(--text-soft) !important; }

pre{
  position: relative;
  border-radius: 12px !important;
  background: var(--code-bg) !important;
  border: 1px solid var(--code-border) !important;
  color: var(--code-text) !important;
  overflow: auto !important;
  box-shadow: none !important;
}
pre code{
  background: transparent !important;
  color: var(--code-text) !important;
  text-decoration: none !important;
  border: 0 !important;
  box-shadow: none !important;
}
pre span{
  background: transparent !important;
  text-decoration: none !important;
  border: 0 !important;
  box-shadow: none !important;
}
pre, pre *{
  text-decoration: none !important;
  background-image: none !important;
}
.kb-plain-code{
  margin: 0 !important;
  white-space: pre !important;
  overflow-x: auto !important;
  border: 0 !important;
  border-radius: 10px !important;
  box-shadow: none !important;
}
.kb-plain-code code{
  display: block !important;
  white-space: pre !important;
  border: 0 !important;
  box-shadow: none !important;
  text-decoration: none !important;
  background-image: none !important;
}
.kb-plain-code,
.kb-plain-code *{
  border-bottom: 0 !important;
  text-decoration: none !important;
  text-decoration-line: none !important;
  box-shadow: none !important;
  background-image: none !important;
}
.kb-plain-code code,
.kb-plain-code code.hljs{
  font-family: "JetBrains Mono", "Fira Code", "Cascadia Code", "Source Code Pro", "Consolas", "SFMono-Regular", monospace !important;
  font-size: 0.93rem !important;
  line-height: 1.62 !important;
  letter-spacing: 0.002em;
  color: var(--code-text) !important;
  background: transparent !important;
}
.kb-plain-code .hljs-comment,
.kb-plain-code .hljs-quote{
  color: var(--code-syn-comment) !important;
  font-style: italic;
}
.kb-plain-code .hljs-keyword,
.kb-plain-code .hljs-selector-tag,
.kb-plain-code .hljs-doctag{
  color: var(--code-syn-keyword) !important;
}
.kb-plain-code .hljs-string,
.kb-plain-code .hljs-regexp,
.kb-plain-code .hljs-attr,
.kb-plain-code .hljs-template-tag{
  color: var(--code-syn-string) !important;
}
.kb-plain-code .hljs-number,
.kb-plain-code .hljs-literal{
  color: var(--code-syn-number) !important;
}
.kb-plain-code .hljs-title,
.kb-plain-code .hljs-title.function_,
.kb-plain-code .hljs-function .hljs-title{
  color: var(--code-syn-func) !important;
}
.kb-plain-code .hljs-type,
.kb-plain-code .hljs-class .hljs-title,
.kb-plain-code .hljs-built_in,
.kb-plain-code .hljs-params{
  color: var(--code-syn-type) !important;
}
.kb-plain-code .hljs-variable,
.kb-plain-code .hljs-symbol,
.kb-plain-code .hljs-bullet{
  color: var(--code-syn-literal) !important;
}
.kb-plain-code .hljs-operator,
.kb-plain-code .hljs-punctuation{
  color: var(--code-syn-operator) !important;
}
.kb-plain-code .kb-syn-comment{ color: var(--code-syn-comment) !important; font-style: italic; }
.kb-plain-code .kb-syn-keyword{ color: var(--code-syn-keyword) !important; }
.kb-plain-code .kb-syn-string{ color: var(--code-syn-string) !important; }
.kb-plain-code .kb-syn-number{ color: var(--code-syn-number) !important; }
.kb-plain-code .kb-syn-func{ color: var(--code-syn-func) !important; }
.kb-plain-code .kb-syn-type{ color: var(--code-syn-type) !important; }
.kb-plain-code .kb-syn-literal{ color: var(--code-syn-literal) !important; }
.kb-plain-code .kb-syn-operator{ color: var(--code-syn-operator) !important; }
div[data-testid="stCodeBlock"],
div[data-testid="stCode"],
.stCodeBlock{
  background: var(--code-bg) !important;
  border: 1px solid var(--code-border) !important;
  border-radius: 12px !important;
  overflow: hidden !important;
}
div[data-testid="stCodeBlock"] > div,
div[data-testid="stCodeBlock"] pre,
div[data-testid="stCode"] > div,
div[data-testid="stCode"] pre,
.stCodeBlock > div,
.stCodeBlock pre{
  background: transparent !important;
  border: 0 !important;
  border-radius: 12px !important;
  color: var(--code-text) !important;
  box-shadow: none !important;
}
div[data-testid="stCodeBlock"] code,
div[data-testid="stCodeBlock"] pre code,
div[data-testid="stCode"] code,
div[data-testid="stCode"] pre code,
.stCodeBlock code,
.stCodeBlock pre code,
.stMarkdown div[data-testid="stMarkdownContainer"] pre code,
.stMarkdown pre code,
.msg-ai pre code{
  background: transparent !important;
  color: var(--code-text) !important;
}
div[data-testid="stCodeBlock"] span,
div[data-testid="stCode"] span,
.stCodeBlock span,
.stMarkdown div[data-testid="stMarkdownContainer"] pre span,
.stMarkdown pre span,
.msg-ai pre span{
  background: transparent !important;
  border: 0 !important;
  border-bottom: 0 !important;
  box-shadow: none !important;
  text-decoration: none !important;
}
div[data-testid="stCodeBlock"] pre *,
div[data-testid="stCode"] pre *,
.stCodeBlock pre *{
  border: 0 !important;
  border-bottom: 0 !important;
  outline: 0 !important;
  box-shadow: none !important;
  text-decoration: none !important;
  background-image: none !important;
}
div[data-testid="stCodeBlock"] div,
div[data-testid="stCode"] div,
.stCodeBlock div,
div[data-testid="stCodeBlock"] [class*="line"],
div[data-testid="stCode"] [class*="line"],
.stCodeBlock [class*="line"],
div[data-testid="stCodeBlock"] [style*="border-bottom"],
div[data-testid="stCode"] [style*="border-bottom"],
.stCodeBlock [style*="border-bottom"]{
  border-bottom: 0 !important;
  box-shadow: none !important;
  text-decoration: none !important;
  background-image: none !important;
}
div[data-testid="stCodeBlock"] table,
div[data-testid="stCodeBlock"] tbody,
div[data-testid="stCodeBlock"] tr,
div[data-testid="stCodeBlock"] td,
div[data-testid="stCodeBlock"] th,
div[data-testid="stCode"] table,
div[data-testid="stCode"] tbody,
div[data-testid="stCode"] tr,
div[data-testid="stCode"] td,
div[data-testid="stCode"] th,
.stCodeBlock table,
.stCodeBlock tbody,
.stCodeBlock tr,
.stCodeBlock td,
.stCodeBlock th{
  border: 0 !important;
  border-bottom: 0 !important;
  box-shadow: none !important;
  background: transparent !important;
  background-image: none !important;
}
div[data-testid="stCodeBlock"] :where(div, span, td, th, p, code),
div[data-testid="stCode"] :where(div, span, td, th, p, code),
.stCodeBlock :where(div, span, td, th, p, code){
  text-decoration: none !important;
  text-decoration-line: none !important;
  text-decoration-thickness: 0 !important;
  text-underline-offset: 0 !important;
}
div[data-testid="stCodeBlock"] *::before,
div[data-testid="stCodeBlock"] *::after,
div[data-testid="stCode"] *::before,
div[data-testid="stCode"] *::after,
.stCodeBlock *::before,
.stCodeBlock *::after{
  border-bottom: 0 !important;
  box-shadow: none !important;
  background-image: none !important;
}
.stMarkdown :not(pre) > code,
.msg-ai :not(pre) > code{
  background: var(--code-inline-bg) !important;
  color: var(--code-text) !important;
  border: 1px solid var(--code-border) !important;
  border-radius: 6px !important;
  padding: 0.08em 0.32em !important;
}
.refbox code, .meta-kv{ color: var(--text-soft) !important; }
.ref-muted-note{
  font-size: 0.80rem;
  color: var(--muted) !important;
  margin: 0.08rem 0 0.40rem 0;
}
.ref-item{
  background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.00));
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 0.55rem 0.62rem 0.50rem 0.62rem;
  margin: 0.16rem 0 0.34rem 0;
}
.ref-item-top{
  display: flex;
  align-items: center;
  gap: 0.42rem;
  min-width: 0;
}
.ref-rank{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 2.25rem;
  height: 1.35rem;
  padding: 0 0.40rem;
  border-radius: 999px;
  font-size: 0.76rem;
  font-weight: 700;
  color: var(--accent) !important;
  border: 1px solid var(--blue-line);
  background: var(--blue-weak);
}
.ref-source{
  flex: 1 1 auto;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--text-main) !important;
  font-weight: 620;
  letter-spacing: 0.01em;
}
.ref-chip{
  display: inline-flex;
  align-items: center;
  height: 1.35rem;
  padding: 0 0.46rem;
  border-radius: 999px;
  font-size: 0.74rem;
  font-weight: 640;
  color: var(--text-soft) !important;
  border: 1px solid var(--line);
  background: rgba(148, 163, 184, 0.12);
}
.ref-score{
  display: inline-flex;
  align-items: center;
  height: 1.35rem;
  padding: 0 0.48rem;
  border-radius: 999px;
  font-size: 0.74rem;
  font-weight: 700;
  border: 1px solid transparent;
}
.ref-score-hi{
  color: #22c55e !important;
  border-color: rgba(34, 197, 94, 0.36);
  background: rgba(34, 197, 94, 0.14);
}
.ref-score-mid{
  color: #f59e0b !important;
  border-color: rgba(245, 158, 11, 0.36);
  background: rgba(245, 158, 11, 0.14);
}
.ref-score-low{
  color: var(--text-soft) !important;
  border-color: var(--line);
  background: rgba(148, 163, 184, 0.10);
}
.ref-item-sub{
  margin-top: 0.34rem;
  font-size: 0.82rem;
  color: var(--text-soft) !important;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ref-item-gap{ height: 0.26rem; }
.hr{ height: 1px; background: var(--line); margin: 1rem 0; }
.pill{
  display: inline-flex;
  align-items: center;
  padding: 0.12rem 0.52rem;
  border-radius: 999px;
  font-size: 0.76rem;
  font-weight: 650;
  border: 1px solid transparent;
}
.pill.ok{
  background: rgba(34, 197, 94, 0.14);
  color: #22c55e !important;
  border-color: rgba(34, 197, 94, 0.34);
}
.pill.warn{
  background: rgba(245, 158, 11, 0.16);
  color: #f59e0b !important;
  border-color: rgba(245, 158, 11, 0.36);
}
.pill.run{
  background: var(--blue-weak);
  color: var(--accent) !important;
  border-color: var(--blue-line);
}

.msg-user{
  background: var(--msg-user-bg);
  border: 1px solid var(--msg-user-border);
  color: var(--msg-user-text) !important;
  border-radius: 16px;
  padding: 10px 14px;
  width: fit-content;
  max-width: min(900px, 94%);
  white-space: pre-wrap;
  word-break: keep-all;
  overflow-wrap: anywhere;
  margin-left: 0;
}
.msg-user-wrap{
  display: flex;
  align-items: flex-start;
  justify-content: flex-end;
  gap: 0.42rem;
  width: fit-content;
  max-width: min(900px, 94%);
  margin-left: auto;
}
.msg-user-wrap .msg-user{
  margin-left: 0 !important;
}
.msg-user-wrap .msg-meta-user{
  margin: 0.14rem 0 0 0;
  text-align: right;
  white-space: nowrap;
  line-height: 1.2;
  font-weight: 560;
}
.msg-ai{ background: transparent; border: none; max-width: min(900px, 94%); }
.msg-ai-stream{
  background: var(--msg-ai-bg);
  border: 1px solid var(--msg-ai-border);
  border-radius: 14px;
  padding: 12px 14px;
}
.msg-refs{
  margin: 0.35rem 0 0.80rem 0;
  padding: 0.06rem 0 0.80rem 0;
  border-left: none !important;
  outline: none !important;
  box-shadow: none !important;
}
.msg-refs::before,
.msg-refs::after{
  display: none !important;
  content: none !important;
}
.msg-refs details[data-testid="stExpander"]{
  background: var(--panel) !important;
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
}
.msg-refs details[data-testid="stExpander"] summary,
.msg-refs details[data-testid="stExpander"] summary *,
.msg-refs details[data-testid="stExpander"] summary p,
.msg-refs details[data-testid="stExpander"] summary span{
  color: var(--refs-title-text) !important;
  opacity: 1 !important;
  -webkit-text-fill-color: var(--refs-title-text) !important;
}
.msg-refs [data-testid="stMarkdownContainer"] *,
.msg-refs .refbox,
.msg-refs .refbox *{
  color: var(--refs-body-text) !important;
  opacity: 1 !important;
  -webkit-text-fill-color: var(--refs-body-text) !important;
}
.msg-refs details[data-testid="stExpander"] summary [data-testid="stMarkdownContainer"] *,
.msg-refs details[data-testid="stExpander"] summary p,
.msg-refs details[data-testid="stExpander"] summary span,
.msg-refs details[data-testid="stExpander"] summary div{
  color: var(--refs-title-text) !important;
  -webkit-text-fill-color: var(--refs-title-text) !important;
  opacity: 1 !important;
}
.msg-refs details[data-testid="stExpander"] summary svg,
.msg-refs details[data-testid="stExpander"] summary path{
  fill: var(--refs-title-text) !important;
  stroke: var(--refs-title-text) !important;
}
.msg-refs .ref-rank{
  color: var(--accent) !important;
  -webkit-text-fill-color: var(--accent) !important;
}
.msg-refs .ref-source,
.msg-refs .ref-item-sub{
  color: var(--text-main) !important;
  -webkit-text-fill-color: var(--text-main) !important;
}
.msg-refs .ref-item-sub{
  color: var(--text-soft) !important;
  -webkit-text-fill-color: var(--text-soft) !important;
}
.msg-refs .ref-chip{
  color: var(--text-soft) !important;
  -webkit-text-fill-color: var(--text-soft) !important;
}
.msg-refs .ref-score-hi{
  color: #22c55e !important;
  -webkit-text-fill-color: #22c55e !important;
}
.msg-refs .ref-score-mid{
  color: #f59e0b !important;
  -webkit-text-fill-color: #f59e0b !important;
}
.msg-refs .ref-score-low{
  color: var(--text-soft) !important;
  -webkit-text-fill-color: var(--text-soft) !important;
}
.msg-refs div[data-testid="stButton"] > button{
  min-height: 2.06rem !important;
  height: 2.06rem !important;
  border-radius: 10px !important;
  border-color: var(--btn-border) !important;
  background: var(--btn-bg) !important;
  color: var(--btn-text) !important;
  font-weight: 650 !important;
  box-shadow: none !important;
  padding: 0 0.74rem !important;
}
.msg-refs div[data-testid="stButton"] > button:hover{
  background: var(--btn-hover) !important;
}
.msg-refs div[data-testid="stButton"] > button:disabled{
  opacity: 0.50 !important;
}

.snipbox{
  background: var(--snip-bg);
  border: 1px solid var(--snip-border);
  border-radius: 12px;
  padding: 10px 12px;
  margin: 0.35rem 0 0.55rem 0;
}
.snipbox pre{
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--snip-text) !important;
  background: transparent !important;
  border: none !important;
  border-radius: 0 !important;
}
.snipquote{ border-left: 3px solid var(--snip-quote-border); background: var(--snip-quote-bg); border-radius: 10px; padding: 9px 11px; margin: 0.15rem 0 0.45rem 0; }
.snipquote .snipquote-title{ font-size: 0.78rem; color: var(--muted) !important; margin: 0 0 0.25rem 0; }
.snipquote .snipquote-body{ font-size: 0.88rem; line-height: 1.42; color: var(--text-main) !important; }
.snipquote mark{ background: var(--snip-mark-bg); color: var(--snip-mark-text); }

.kb-notice{
  font-size: 0.84rem;
  color: var(--notice-text) !important;
  background: var(--notice-bg);
  border: 1px solid var(--notice-border);
  border-radius: 10px;
  padding: 0.35rem 0.55rem;
  margin: 0 0 0.55rem 0;
}

.stTextArea{ position: relative; }
.stTextArea::after,
div[data-testid="stTextArea"]::after{
  content: "Ctrl+Enter 发送";
  position: absolute;
  right: 14px;
  bottom: 10px;
  font-size: 12px;
  color: var(--hint-text);
  pointer-events: none;
}

.kb-input-dock{
  position: fixed !important;
  bottom: max(10px, env(safe-area-inset-bottom, 0px));
  z-index: 40;
  left: 50%;
  transform: translateX(-50%);
  width: min(var(--content-max), calc(100vw - 1.6rem));
  box-sizing: border-box;
  background: var(--dock-bg) !important;
  border: 1px solid var(--dock-border) !important;
  border-radius: 20px;
  padding: 0.48rem 0.56rem 0.18rem 0.56rem;
  box-shadow: var(--dock-shadow);
  backdrop-filter: blur(5px);
  margin: 0 !important;
}
.kb-input-dock,
.kb-input-dock > div,
.kb-input-dock div[data-testid="stForm"]{
  background: var(--dock-bg) !important;
  border-color: var(--dock-border) !important;
}
.kb-input-dock textarea,
.kb-input-dock div[data-testid="stTextArea"] textarea{
  border-color: var(--input-border) !important;
}
.kb-input-dock textarea:focus,
.kb-input-dock div[data-testid="stTextArea"] textarea:focus{
  border-color: var(--blue-line) !important;
  box-shadow: 0 0 0 1px var(--blue-weak) !important;
}
html[data-theme="dark"] .kb-input-dock,
body[data-theme="dark"] .kb-input-dock{
  background: var(--dock-bg) !important;
  border-color: var(--dock-border) !important;
}
body.kb-resizing .kb-input-dock,
body.kb-resizing .kb-input-dock *{
  border-color: var(--dock-border) !important;
}
.kb-input-dock.kb-dock-positioned{ max-width: none !important; }
.kb-input-dock div[data-testid="stForm"]{ margin-bottom: 0 !important; }
.kb-input-dock::before{
  content: "问点什么...（会先检索你的 Markdown 再回答）";
  display: block;
  font-size: 0.75rem;
  color: var(--muted);
  margin: 0 0 0.32rem 0.16rem;
}
.kb-input-dock div[data-testid="stTextArea"] label{ display: none !important; }
.kb-input-dock textarea{ min-height: 92px !important; border-radius: 14px !important; }
.kb-input-dock div[data-testid="stFormSubmitButton"]{ display: flex !important; justify-content: flex-end !important; margin-top: 0.38rem !important; }
.kb-input-dock div[data-testid="stFormSubmitButton"] > button{
  width: 40px !important;
  min-width: 40px !important;
  height: 40px !important;
  min-height: 40px !important;
  border-radius: 999px !important;
  padding: 0 !important;
}

.kb-copybar{
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: nowrap;
  margin: 6px 0 10px 0;
}
.kb-ai-inline-logo{
  height: 32px;
  width: auto;
  max-width: 88px;
  object-fit: contain;
  display: inline-block;
  flex: 0 0 auto;
}
.kb-ai-livebar{
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  margin: 0.12rem 0 0.42rem 0;
}
.kb-ai-live-logo{
  height: 22px;
  width: auto;
  max-width: 60px;
  object-fit: contain;
  display: inline-block;
}
.kb-ai-live-pill{
  display: inline-flex;
  align-items: center;
  padding: 0.14rem 0.52rem;
  border-radius: 999px;
  border: 1px solid var(--blue-line);
  background: var(--blue-weak);
  color: var(--text-main) !important;
  font-size: 0.76rem;
  font-weight: 650;
}
.kb-ai-live-stage{
  color: var(--text-soft) !important;
  font-size: 0.82rem;
}
.kb-ai-live-dots{
  margin: 0.02rem 0 0.46rem 0.08rem;
  color: var(--muted) !important;
  font-size: 0.88rem;
  letter-spacing: 0.28em;
  user-select: none;
}
.kb-copybtn{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 32px;
  min-height: 32px;
  padding: 0 10px;
  border-radius: 10px;
  border: 1px solid var(--copy-btn-border);
  background: var(--copy-btn-bg);
  color: var(--copy-btn-text);
  font-weight: 600;
  font-size: 12px;
  white-space: nowrap;
  cursor: pointer;
}
.kb-copybtn:hover{ background: var(--btn-hover); border-color: var(--blue-line); }
.kb-copybtn:active{ background: var(--btn-active); border-color: var(--blue-line); }
.kb-codecopy{
  position: absolute;
  top: 10px;
  right: 10px;
  padding: 4px 8px;
  border-radius: 10px;
  border: 1px solid var(--copy-btn-border);
  background: var(--copy-btn-bg);
  color: var(--copy-btn-text);
  font-weight: 600;
  font-size: 12px;
  cursor: pointer;
  z-index: 2;
}
.kb-codecopy:hover{ background: var(--btn-hover); border-color: var(--blue-line); }
div[data-testid="stCodeBlock"]:not([data-kb-normalized="1"]) .kb-codecopy,
div[data-testid="stCode"]:not([data-kb-normalized="1"]) .kb-codecopy,
.stCodeBlock:not([data-kb-normalized="1"]) .kb-codecopy{
  display: none !important;
}
.kb-toast{
  position: fixed;
  right: 18px;
  bottom: 18px;
  padding: 10px 12px;
  border-radius: 12px;
  border: 1px solid var(--toast-border);
  background: var(--toast-bg);
  color: var(--toast-text);
  font-weight: 600;
  font-size: 12px;
  opacity: 0;
  transform: translateY(6px);
  transition: opacity 120ms ease, transform 120ms ease;
  z-index: 999999;
  pointer-events: none;
}
.kb-toast.show{ opacity: 1; transform: translateY(0); }
html[data-theme="dark"] small,
html[data-theme="dark"] .stCaption,
html[data-theme="dark"] div[data-testid="stCaptionContainer"] *,
body[data-theme="dark"] small,
body[data-theme="dark"] .stCaption,
body[data-theme="dark"] div[data-testid="stCaptionContainer"] *{
  color: var(--text-soft) !important;
  opacity: 1 !important;
}
html[data-theme="dark"] div[data-testid="stWidgetLabel"],
html[data-theme="dark"] div[data-testid="stWidgetLabel"] *,
body[data-theme="dark"] div[data-testid="stWidgetLabel"],
body[data-theme="dark"] div[data-testid="stWidgetLabel"] *{
  color: var(--text-main) !important;
  opacity: 1 !important;
}
html[data-theme="dark"] section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] *,
html[data-theme="dark"] section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
html[data-theme="dark"] section[data-testid="stSidebar"] p,
html[data-theme="dark"] section[data-testid="stSidebar"] span,
body[data-theme="dark"] section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] *,
body[data-theme="dark"] section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
body[data-theme="dark"] section[data-testid="stSidebar"] p,
body[data-theme="dark"] section[data-testid="stSidebar"] span{
  color: var(--text-main) !important;
  opacity: 1 !important;
}
html[data-theme="dark"] div[data-testid="stRadio"] label,
html[data-theme="dark"] div[data-testid="stRadio"] label *,
html[data-theme="dark"] div[data-testid="stCheckbox"] label,
html[data-theme="dark"] div[data-testid="stCheckbox"] label *,
body[data-theme="dark"] div[data-testid="stRadio"] label,
body[data-theme="dark"] div[data-testid="stRadio"] label *,
body[data-theme="dark"] div[data-testid="stCheckbox"] label,
body[data-theme="dark"] div[data-testid="stCheckbox"] label *{
  color: var(--text-main) !important;
  opacity: 1 !important;
}
html[data-theme="dark"] div[data-testid="stSlider"] label,
html[data-theme="dark"] div[data-testid="stSlider"] label *,
html[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stTickBarMin"],
html[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stTickBarMax"],
html[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stSliderTickBarMin"],
html[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stSliderTickBarMax"],
html[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stSliderTickBar"],
html[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stThumbValue"],
html[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stSliderValue"],
html[data-theme="dark"] div[data-testid="stSlider"] [data-baseweb="slider"] *,
body[data-theme="dark"] div[data-testid="stSlider"] label,
body[data-theme="dark"] div[data-testid="stSlider"] label *,
body[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stTickBarMin"],
body[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stTickBarMax"],
body[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stSliderTickBarMin"],
body[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stSliderTickBarMax"],
body[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stSliderTickBar"],
body[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stThumbValue"],
body[data-theme="dark"] div[data-testid="stSlider"] [data-testid="stSliderValue"],
body[data-theme="dark"] div[data-testid="stSlider"] [data-baseweb="slider"] *{
  color: var(--text-soft) !important;
  opacity: 1 !important;
}
html[data-theme="dark"] details[data-testid="stExpander"] summary,
html[data-theme="dark"] details[data-testid="stExpander"] summary *,
body[data-theme="dark"] details[data-testid="stExpander"] summary,
body[data-theme="dark"] details[data-testid="stExpander"] summary *{
  color: var(--text-main) !important;
  opacity: 1 !important;
}
html[data-theme="dark"] [data-testid="stMarkdownContainer"] p,
html[data-theme="dark"] [data-testid="stMarkdownContainer"] li,
html[data-theme="dark"] [data-testid="stMarkdownContainer"] span,
body[data-theme="dark"] [data-testid="stMarkdownContainer"] p,
body[data-theme="dark"] [data-testid="stMarkdownContainer"] li,
body[data-theme="dark"] [data-testid="stMarkdownContainer"] span{
  color: var(--text-soft) !important;
  opacity: 1 !important;
}
html[data-theme="dark"] .msg-meta,
html[data-theme="dark"] .refbox,
html[data-theme="dark"] .genbox,
html[data-theme="dark"] .chat-empty-state,
body[data-theme="dark"] .msg-meta,
body[data-theme="dark"] .refbox,
body[data-theme="dark"] .genbox,
body[data-theme="dark"] .chat-empty-state{
  color: var(--text-soft) !important;
  opacity: 1 !important;
}
html[data-theme="dark"] section[data-testid="stSidebar"] div[data-testid="stSlider"] *,
body[data-theme="dark"] section[data-testid="stSidebar"] div[data-testid="stSlider"] *{
  color: var(--text-soft) !important;
  fill: var(--text-soft) !important;
  stroke: var(--text-soft) !important;
  opacity: 1 !important;
}
html[data-theme="dark"] section[data-testid="stSidebar"] div[data-testid="stSlider"] [style*="color"],
body[data-theme="dark"] section[data-testid="stSidebar"] div[data-testid="stSlider"] [style*="color"]{
  color: var(--text-soft) !important;
  opacity: 1 !important;
}
</style>
<script>
(function () {
  const host = window.parent || window;
  const doc = host.document || document;
  const mode = "__MODE__";
  try {
    doc.documentElement.setAttribute("data-theme", mode);
    if (doc.body) doc.body.setAttribute("data-theme", mode);
  } catch (e) {}
})();
</script>
"""
    st.markdown(
        css.replace("__TOKENS__", tokens).replace("__SCHEME__", color_scheme).replace("__MODE__", mode),
        unsafe_allow_html=True,
    )

def _inject_copy_js() -> None:
    """
    Attach clipboard behaviors to:
    - Answer-level copy buttons (text / markdown)
    - Per-code-block copy buttons
    - Click-to-copy for LaTeX formulas rendered by KaTeX/MathJax (best effort)
    """
    components.html(
        r"""
<script>
(function () {
  const host = window.parent || window;
  const root = host.document || document;
  const TOAST_ID = "kb_toast";
  const HLJS_KEY = "__kbHljsReady";
  const HLJS_LOADING_KEY = "__kbHljsLoading";

  function ensureToast() {
    let t = root.getElementById(TOAST_ID);
    if (!t) {
      t = root.createElement("div");
      t.id = TOAST_ID;
      t.className = "kb-toast";
      t.textContent = "已复制";
      root.body.appendChild(t);
    }
    return t;
  }

  function toast(msg) {
    const t = ensureToast();
    t.textContent = msg || "已复制";
    t.classList.add("show");
    clearTimeout(t._kbTimer);
    t._kbTimer = setTimeout(() => t.classList.remove("show"), 900);
  }

  function ensureHighlightJs() {
    try {
      if (host[HLJS_KEY] && host.hljs && typeof host.hljs.highlightElement === "function") {
        return Promise.resolve(host.hljs);
      }
      if (host[HLJS_LOADING_KEY]) {
        return host[HLJS_LOADING_KEY];
      }
      host[HLJS_LOADING_KEY] = new Promise((resolve, reject) => {
        const existing = root.querySelector('script[data-kb-hljs="1"]');
        if (existing && host.hljs && typeof host.hljs.highlightElement === "function") {
          host[HLJS_KEY] = true;
          resolve(host.hljs);
          return;
        }
        const script = existing || root.createElement("script");
        if (!existing) {
          script.src = "https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/lib/common.min.js";
          script.async = true;
          script.defer = true;
          script.dataset.kbHljs = "1";
          (root.head || root.body || root.documentElement).appendChild(script);
        }
        script.addEventListener("load", () => {
          if (host.hljs && typeof host.hljs.highlightElement === "function") {
            host[HLJS_KEY] = true;
            resolve(host.hljs);
          } else {
            reject(new Error("hljs unavailable"));
          }
        }, { once: true });
        script.addEventListener("error", () => reject(new Error("hljs load failed")), { once: true });
      }).catch(() => null);
      return host[HLJS_LOADING_KEY];
    } catch (e) {
      return Promise.resolve(null);
    }
  }

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      toast("已复制");
      return true;
    } catch (e) {
      // Fallback: execCommand
      try {
        const ta = root.createElement("textarea");
        ta.value = text;
        ta.setAttribute("readonly", "");
        ta.style.position = "fixed";
        ta.style.left = "-9999px";
        root.body.appendChild(ta);
        ta.select();
        root.execCommand("copy");
        root.body.removeChild(ta);
        toast("已复制");
        return true;
      } catch (e2) {
        toast("复制失败");
        return false;
      }
    }
  }

  function hookCopyButtons() {
    const btns = root.querySelectorAll("button.kb-copybtn");
    for (const b of btns) {
      if (b.dataset.kbHooked === "1") continue;
      b.dataset.kbHooked = "1";
      b.addEventListener("click", async (e) => {
        e.preventDefault();
        const targetId = b.getAttribute("data-target");
        if (!targetId) return;
        const ta = root.getElementById(targetId);
        if (!ta) return;
        await copyText(ta.value || "");
      });
    }
  }

  function hookCodeBlocks() {
    function normalizeNativeCodeBlocks() {
      const blocks = root.querySelectorAll('div[data-testid="stCodeBlock"], div[data-testid="stCode"], .stCodeBlock');
      for (const block of blocks) {
        if (!block || !block.dataset) continue;
        if (block.dataset.kbNormalized === "1") continue;
        let codeNode = null;
        try {
          codeNode = block.querySelector("pre code, code");
        } catch (e) {
          codeNode = null;
        }
        if (!codeNode) continue;
        const txt = String(codeNode.innerText || codeNode.textContent || "");
        const codeClass = String(codeNode.className || "");
        if (!txt.trim()) continue;
        try {
          block.innerHTML = "";
          const pre = root.createElement("pre");
          pre.className = "kb-plain-code";
          const code = root.createElement("code");
          if (codeClass) code.className = codeClass;
          code.textContent = txt.replace(/\r\n/g, "\n");
          pre.appendChild(code);
          block.appendChild(pre);
          block.dataset.kbNormalized = "1";
        } catch (e) {}
      }
    }

    function escapeHtml(s) {
      return String(s || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
    }

    function inferLang(raw, cls) {
      const c = String(cls || "").toLowerCase();
      const t = String(raw || "");
      if (c.includes("python") || c.includes("language-py") || c.includes("lang-py")) return "python";
      if (c.includes("javascript") || c.includes("language-js") || c.includes("lang-js") || c.includes("typescript")) return "javascript";
      if (/\b(def|import|from|return|lambda|None|True|False|async|await)\b/.test(t)) return "python";
      if (/\b(function|const|let|var|return|=>|async|await)\b/.test(t)) return "javascript";
      return "plain";
    }

    function simpleHighlight(raw, lang) {
      let s = escapeHtml(String(raw || "").replace(/\r\n/g, "\n"));
      const stash = [];
      function keep(regex, cls) {
        s = s.replace(regex, (m) => {
          const id = stash.length;
          stash.push(`<span class="${cls}">${m}</span>`);
          return `@@KBH${id}@@`;
        });
      }

      keep(/("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')/g, "kb-syn-string");
      keep(/(#[^\n]*|\/\/[^\n]*)/g, "kb-syn-comment");

      let kw = [];
      if (lang === "python") {
        kw = ["and","as","assert","async","await","break","class","continue","def","del","elif","else","except","False","finally","for","from","global","if","import","in","is","lambda","None","nonlocal","not","or","pass","raise","return","True","try","while","with","yield"];
      } else if (lang === "javascript") {
        kw = ["await","break","case","catch","class","const","continue","debugger","default","delete","do","else","export","extends","finally","for","function","if","import","in","instanceof","let","new","return","super","switch","this","throw","try","typeof","var","void","while","with","yield"];
      }
      if (kw.length) {
        const kwRe = new RegExp("\\\\b(" + kw.join("|") + ")\\\\b", "g");
        s = s.replace(kwRe, '<span class="kb-syn-keyword">$1</span>');
      }
      s = s.replace(/\b(\d+(?:\.\d+)?)\b/g, '<span class="kb-syn-number">$1</span>');
      s = s.replace(/\b([A-Za-z_][A-Za-z0-9_]*)\s*(?=\()/g, '<span class="kb-syn-func">$1</span>');

      s = s.replace(/@@KBH(\d+)@@/g, (_, idx) => stash[Number(idx)] || "");
      return s;
    }

    function applyTyporaHighlight() {
      const codes = root.querySelectorAll("pre.kb-plain-code > code, .msg-ai pre code, .stMarkdown pre code");
      ensureHighlightJs().then((hljs) => {
        for (const code of codes) {
          if (!code) continue;
          const raw = String(code.textContent || "");
          if (!raw.trim()) continue;
          const lang = inferLang(raw, code.className || "");
          const sig = String(raw.length) + ":" + raw.slice(0, 120) + ":" + lang;
          if (code.dataset.kbHlSig === sig) continue;
          code.dataset.kbHlSig = sig;
          try {
            code.removeAttribute("data-highlighted");
            code.classList.remove("hljs");
            code.textContent = raw;
            if (hljs && typeof hljs.highlightElement === "function") {
              hljs.highlightElement(code);
              code.classList.add("hljs");
            } else {
              code.innerHTML = simpleHighlight(raw, lang);
              code.classList.add("hljs");
            }
          } catch (e) {
            try {
              code.innerHTML = simpleHighlight(raw, lang);
              code.classList.add("hljs");
            } catch (e2) {}
          }
        }
      });
    }

    function hasNativeCopy(pre) {
      if (!pre) return false;
      try {
        const hostBlock = pre.closest('div[data-testid="stCodeBlock"], div[data-testid="stCode"], .stCodeBlock');
        if (hostBlock) {
          return String(hostBlock.dataset && hostBlock.dataset.kbNormalized || "") !== "1";
        }
        const host = pre.parentElement;
        if (!host) return false;
        const nativeBtn = host.querySelector(
          'button[aria-label*="copy" i], button[title*="copy" i], button[aria-label*="复制"], button[title*="复制"], [data-testid*="copy" i]'
        );
        return !!nativeBtn;
      } catch (e) {
        return false;
      }
    }

    normalizeNativeCodeBlocks();
    applyTyporaHighlight();

    const pres = root.querySelectorAll("pre");
    for (const pre of pres) {
      if (hasNativeCopy(pre)) {
        const oldBtn = pre.querySelector(".kb-codecopy");
        if (oldBtn) {
          try { oldBtn.remove(); } catch (e) {}
        }
        pre.dataset.kbCodeHooked = "1";
        continue;
      }
      if (pre.dataset.kbCodeHooked === "1") continue;
      const code = pre.querySelector("code");
      if (!code) continue;
      pre.dataset.kbCodeHooked = "1";
      const btn = root.createElement("button");
      btn.className = "kb-codecopy";
      btn.type = "button";
      btn.textContent = "复制代码";
      btn.addEventListener("click", async (e) => {
        e.preventDefault();
        e.stopPropagation();
        await copyText(code.innerText || "");
      });
      pre.appendChild(btn);
    }
  }

  function extractTexFromKaTeX(node) {
    try {
      const ann = node.querySelector('annotation[encoding="application/x-tex"]');
      if (ann && ann.textContent) return ann.textContent;
    } catch (e) {}
    return null;
  }

  function hookMathClickToCopy() {
    const mathNodes = root.querySelectorAll(".katex, .MathJax, mjx-container");
    for (const n of mathNodes) {
      if (n.dataset && n.dataset.kbMathHooked === "1") continue;
      if (n.dataset) n.dataset.kbMathHooked = "1";
      n.style.cursor = "copy";
      n.addEventListener("click", async (e) => {
        // Prefer KaTeX annotation if available.
        const tex = extractTexFromKaTeX(n) || (n.innerText || "").trim();
        if (!tex) return;
        await copyText(tex);
        toast("已复制 LaTeX");
      });
    }
  }

  function tick() {
    hookCopyButtons();
    hookCodeBlocks();
    hookMathClickToCopy();
  }

  tick();
  setInterval(tick, 900);
})();
</script>
        """,
        height=0,
    )

def _inject_runtime_ui_fixes(theme_mode: str) -> None:
    mode = "dark" if str(theme_mode or "").lower() == "dark" else "light"
    components.html(
        f"""
<script>
(function () {{
  const host = window.parent || window;
  const doc = host.document || document;
  const KEY = "__kbUiRuntimeFixV2";
  const mode = "{mode}";
  try {{
    if (host[KEY] && typeof host[KEY].destroy === "function") {{
      host[KEY].destroy();
    }}
  }} catch (e) {{}}

  function paint(el, color) {{
    if (!el || !el.style) return;
    try {{
      el.style.setProperty("color", color, "important");
      el.style.setProperty("-webkit-text-fill-color", color, "important");
      el.style.setProperty("fill", color, "important");
      el.style.setProperty("stroke", color, "important");
      el.style.setProperty("opacity", "1", "important");
      el.style.setProperty("filter", "none", "important");
    }} catch (e) {{}}
  }}

  function clearInlineThemeForRefs() {{
    try {{
      const nodes = doc.querySelectorAll(".msg-refs, .msg-refs *");
      for (const n of nodes) {{
        if (!n || !n.style) continue;
        n.style.removeProperty("color");
        n.style.removeProperty("-webkit-text-fill-color");
        n.style.removeProperty("fill");
        n.style.removeProperty("stroke");
        n.style.removeProperty("opacity");
        n.style.removeProperty("filter");
      }}
    }} catch (e) {{}}
  }}

  function normalizeSidebarCloseIcon() {{
    try {{
      const mainText = mode === "dark" ? "#e7eaef" : "#1f2329";
      const sidebars = doc.querySelectorAll('section[data-testid="stSidebar"]');
      for (const sidebar of sidebars) {{
        const sbRect = sidebar.getBoundingClientRect ? sidebar.getBoundingClientRect() : null;
        const btns = sidebar.querySelectorAll("button");
        for (const b of btns) {{
          const aria = String(b.getAttribute("aria-label") || "").toLowerCase();
          const rect = b.getBoundingClientRect ? b.getBoundingClientRect() : null;
          const nearTopRight = !!(sbRect && rect && rect.top <= (sbRect.top + 90) && rect.left >= (sbRect.right - 100));
          const maybeClose = aria.includes("close") || aria.includes("collapse") || aria.includes("关闭") || nearTopRight;
          if (!maybeClose) continue;
          try {{
            b.style.setProperty("width", "34px", "important");
            b.style.setProperty("height", "34px", "important");
            b.style.setProperty("min-width", "34px", "important");
            b.style.setProperty("min-height", "34px", "important");
            b.style.setProperty("padding", "0", "important");
            b.style.setProperty("font-size", "0", "important");
            b.style.setProperty("line-height", "0", "important");
            b.style.setProperty("display", "inline-flex", "important");
            b.style.setProperty("align-items", "center", "important");
            b.style.setProperty("justify-content", "center", "important");
          }} catch (e) {{}}
          const svgs = b.querySelectorAll("svg");
          for (const s of svgs) {{
            try {{ s.style.setProperty("display", "none", "important"); }} catch (e) {{}}
          }}
          const icons = b.querySelectorAll('[data-testid="stIcon"]');
          for (const ic of icons) {{
            try {{ ic.style.setProperty("display", "none", "important"); }} catch (e) {{}}
          }}
          let glyph = b.querySelector(".kb-close-glyph");
          if (!glyph) {{
            glyph = doc.createElement("span");
            glyph.className = "kb-close-glyph";
            glyph.textContent = "×";
            glyph.setAttribute("aria-hidden", "true");
            b.appendChild(glyph);
          }}
          paint(glyph, mainText);
        }}
      }}
    }} catch (e) {{}}
  }}

  function clearCodeLineArtifacts() {{
    try {{
      const blocks = doc.querySelectorAll('div[data-testid="stCodeBlock"], div[data-testid="stCode"], .stCodeBlock');
      for (const block of blocks) {{
        const hrs = block.querySelectorAll("hr");
        for (const h of hrs) {{
          if (!h || !h.style) continue;
          h.style.setProperty("display", "none", "important");
          h.style.setProperty("border", "0", "important");
          h.style.setProperty("border-bottom", "0", "important");
          h.style.setProperty("height", "0", "important");
          h.style.setProperty("margin", "0", "important");
          h.style.setProperty("padding", "0", "important");
        }}

        const nodes = block.querySelectorAll("*");
        for (const n of nodes) {{
          if (!n || !n.style) continue;
          const tag = String(n.tagName || "").toLowerCase();
          if (tag === "button" || n.closest("button")) continue;
          if (n.classList && n.classList.contains("kb-codecopy")) continue;
          n.style.setProperty("border-bottom", "0", "important");
          n.style.setProperty("box-shadow", "none", "important");
          n.style.setProperty("outline", "0", "important");
          n.style.setProperty("text-decoration", "none", "important");
          n.style.setProperty("text-decoration-line", "none", "important");
          n.style.setProperty("text-decoration-thickness", "0", "important");
          n.style.setProperty("text-underline-offset", "0", "important");
          n.style.setProperty("background-image", "none", "important");
        }}
      }}
    }} catch (e) {{}}
  }}

  function applyNow() {{
    clearInlineThemeForRefs();
    normalizeSidebarCloseIcon();
    clearCodeLineArtifacts();
  }}

  let raf = 0;
  function schedule() {{
    if (raf) return;
    raf = host.requestAnimationFrame(function () {{
      raf = 0;
      applyNow();
    }});
  }}

  let mo = null;
  function observe() {{
    if (typeof MutationObserver === "undefined") return;
    try {{
      mo = new MutationObserver(function () {{ schedule(); }});
      mo.observe(doc.body, {{ childList: true, subtree: true, attributes: true }});
    }} catch (e) {{}}
  }}

  function destroy() {{
    try {{ if (mo) mo.disconnect(); }} catch (e) {{}}
    try {{ if (raf) host.cancelAnimationFrame(raf); }} catch (e) {{}}
  }}

  host[KEY] = {{ destroy }};

  schedule();
  observe();
}})();
</script>
        """,
        height=0,
    )

def _teardown_chat_dock_runtime() -> None:
    components.html(
        """
<script>
(function () {
  try {
    const host = window.parent || window;
    const root = host.document;
    if (!root || !root.body) return;

    const NS = "__kbDockManagerStableV3";
    try {
      if (host[NS] && typeof host[NS].destroy === "function") host[NS].destroy();
      delete host[NS];
    } catch (e) {}

    try {
      root.body.classList.remove("kb-resizing");
      root.body.classList.remove("kb-live-streaming");
    } catch (e) {}

    const docks = root.querySelectorAll(".kb-input-dock, .kb-dock-positioned");
    docks.forEach(function (el) {
      try {
        el.classList.remove("kb-input-dock", "kb-dock-positioned");
        el.style.left = "";
        el.style.right = "";
        el.style.width = "";
        el.style.transform = "";
      } catch (e) {}
    });
  } catch (e) {}
})();
</script>
        """,
        height=0,
    )

def _set_live_streaming_mode(active: bool) -> None:
    on_flag = "true" if bool(active) else "false"
    components.html(
        f"""
<script>
(function () {{
  try {{
    const host = window.parent || window;
    const root = host.document;
    if (!root || !root.body) return;
    const on = {on_flag};
    if (on) root.body.classList.add("kb-live-streaming");
    else root.body.classList.remove("kb-live-streaming");
  }} catch (e) {{}}
}})();
</script>
        """,
        height=0,
    )


def _inject_chat_dock_runtime() -> None:
    global _CHAT_DOCK_JS_CACHE
    if _CHAT_DOCK_JS_CACHE is None:
        try:
            _CHAT_DOCK_JS_CACHE = _CHAT_DOCK_JS_PATH.read_text(encoding="utf-8")
        except Exception:
            _CHAT_DOCK_JS_CACHE = ""
    js = str(_CHAT_DOCK_JS_CACHE or "").strip()
    if not js:
        return
    components.html("<script>\n" + js + "\n</script>", height=0)


def _inject_auto_rerun_once(*, delay_ms: int = 3500) -> None:
    delay = max(300, int(delay_ms))
    components.html(
        f"""
<script>
(function () {{
  try {{
    const root = window.parent;
    if (!root) return;
    if (root._kbAutoRefreshTimer) return;
    root._kbAutoRefreshTimer = setTimeout(function () {{
      try {{
        root._kbAutoRefreshTimer = null;
        root.postMessage({{ isStreamlitMessage: true, type: "streamlit:rerunScript" }}, "*");
      }} catch (e) {{}}
    }}, {delay});
  }} catch (e) {{}}
}})();
</script>
        """,
        height=0,
    )
