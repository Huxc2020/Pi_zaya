from __future__ import annotations

import os
import json
import re
import time
import base64
import threading
from typing import Optional, List, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .config import ConvertConfig
from .models import TextBlock
from .text_utils import _normalize_text
from .tables import _is_markdown_table_sane

class LLMWorker:
    def __init__(self, cfg: ConvertConfig):
        self.cfg = cfg
        self._client = None
        # Global-ish concurrency limiter: one LLMWorker is shared across page threads in PDFConverter.
        # This prevents "N pages in parallel" from flooding the provider and stalling on throttling.
        self._llm_sem: threading.Semaphore | None = None
        try:
            raw = str(os.environ.get("KB_LLM_MAX_INFLIGHT", "") or "").strip()
            if raw:
                max_inflight = int(raw)
            else:
                # Default: 12 for better parallel processing in vision-direct mode
                # (was 2, but vision-direct benefits from higher concurrency)
                max_inflight = 12
            max_inflight = max(1, min(32, int(max_inflight)))
            self._llm_sem = threading.Semaphore(max_inflight)
        except Exception:
            self._llm_sem = threading.Semaphore(12)  # Default to 12 instead of 2
        # Small in-memory caches to avoid repeated calls for identical snippets.
        # These caches live per Streamlit process and reset on restart.
        self._cache_confirm_heading: dict[str, dict] = {}
        self._cache_repair_math: dict[str, str] = {}
        self._cache_max_items = 2048
        if self.cfg.llm:
            try:
                self._client = self._ensure_openai_class()(
                    api_key=self.cfg.llm.api_key,
                    base_url=self.cfg.llm.base_url,
                )
            except Exception as e:
                print(f"[WARN] Failed to init OpenAI client: {e}")
                self._client = None

    def _ensure_openai_class(self):
        if OpenAI is None:
            raise ImportError("openai module not installed.")
        return OpenAI

    def _llm_create(self, **kwargs):
        if not self._client:
            raise RuntimeError("LLM client not initialized")
        # Keep call-timeouts and retries configurable; defaults are conservative.
        timeout_s = 45.0
        max_retries = 0
        try:
            if self.cfg.llm:
                timeout_s = float(getattr(self.cfg.llm, "timeout_s", timeout_s) or timeout_s)
                max_retries = int(getattr(self.cfg.llm, "max_retries", max_retries) or max_retries)
        except Exception:
            timeout_s = 45.0
            max_retries = 0

        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                if self._llm_sem is None:
                    return self._client.chat.completions.create(
                        model=self.cfg.llm.model,
                        timeout=timeout_s,
                        **kwargs,
                    )
                # Increase semaphore acquire timeout for vision-direct mode (full-page screenshots take longer)
                # Default: wait up to 60 seconds for a slot (was 15s), or 2x the request timeout, whichever is larger
                sem_timeout = max(60.0, float(timeout_s) * 2.0)
                # But cap at 120s to avoid infinite waits
                sem_timeout = min(120.0, sem_timeout)
                acquired = self._llm_sem.acquire(timeout=sem_timeout)
                if not acquired:
                    raise TimeoutError(
                        f"LLM inflight slots saturated (KB_LLM_MAX_INFLIGHT). "
                        f"Waited {sem_timeout:.1f}s for a slot. Consider increasing KB_LLM_MAX_INFLIGHT."
                    )
                try:
                    return self._client.chat.completions.create(
                        model=self.cfg.llm.model,
                        timeout=timeout_s,
                        **kwargs,
                    )
                finally:
                    try:
                        self._llm_sem.release()
                    except Exception:
                        pass
            except TimeoutError as e:
                # For semaphore timeout, retry with backoff (up to max_retries)
                last_err = e
                if attempt < max_retries:
                    # Exponential backoff: 2s, 4s, 8s...
                    backoff = min(10.0, 2.0 * (2**attempt))
                    time.sleep(backoff)
                    continue
                # If all retries exhausted, raise the timeout
                raise
            except Exception as e:
                last_err = e
                if attempt >= max_retries:
                    break
                # Short exponential backoff; keeps UI responsive.
                time.sleep(min(1.2, 0.25 * (2**attempt)))
        assert last_err is not None
        raise last_err

    def _get_max_tokens_for_vision(self, speed_mode: str = 'normal') -> int:
        """Get max_tokens for vision calls, allowing override via environment variable."""
        try:
            raw = str(os.environ.get("KB_PDF_VISION_MAX_TOKENS", "") or "").strip()
            if raw:
                return max(1024, min(8192, int(raw)))  # Clamp between 1024-8192
        except Exception:
            pass
        # Default based on speed mode
        defaults = {
            'normal': 3072,
            'ultra_fast': 2048,
            'no_llm': 0,
        }
        default = defaults.get(speed_mode, 3072)
        config_val = int(getattr(self.cfg.llm, "max_tokens", 0) or 0)
        if config_val > 0:
            return min(config_val, 4096)  # Respect config but cap at 4096
        return default

    def _cache_set(self, cache: dict, key: str, val) -> None:
        cache[key] = val
        # Simple size bound (drop oldest insertion order in Py>=3.7 dict).
        if len(cache) > int(self._cache_max_items):
            try:
                for k in list(cache.keys())[: max(1, len(cache) // 3)]:
                    cache.pop(k, None)
            except Exception:
                cache.clear()

    def _extract_json_array(self, s: str) -> Optional[list]:
        if not s:
            return None
        start = s.find("[")
        end = s.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return None
        blob = s[start : end + 1]
        try:
            return json.loads(blob)
        except Exception:
            return None

    def call_llm_repair_table(self, raw_table: str, *, page_number: int, block_index: int) -> Optional[str]:
        if not self.cfg.llm:
            return None
        text = raw_table.strip()
        if not text:
            return None
        
        prompt = (
            f"Fix this broken text table from PDF page {page_number+1} into a clean Markdown table.\n"
            "Return ONLY the Markdown table. No other text.\n"
            "If it is definitely not a table, return 'NOT_A_TABLE'.\n\n"
            f"RAW TEXT:\n{text}\n"
        )
        if self.cfg.llm.request_sleep_s > 0:
            time.sleep(self.cfg.llm.request_sleep_s)
        try:
            resp = self._llm_create(
                messages=[
                    {"role": "system", "content": "You are a precise table fixer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=self.cfg.llm.max_tokens,
            )
        except Exception:
            return None
        
        out = (resp.choices[0].message.content or "").strip()
        if out == "NOT_A_TABLE":
            return None
        if "```" in out:
            # Extract content from code block
            m = re.search(r"```(?:\w+)?\n(.*?)```", out, re.DOTALL)
            if m:
                out = m.group(1).strip()
        return out

    def call_llm_repair_math(
        self,
        raw_math: str,
        *,
        page_number: int,
        block_index: int,
        context_before: str = "",
        context_after: str = "",
        eq_number: Optional[str] = None
    ) -> Optional[str]:
        if not self.cfg.llm:
            return None

        cache_key = None
        try:
            norm = _normalize_text(raw_math or "").strip()
            if norm:
                cache_key = f"math:{page_number}:{block_index}:{eq_number or ''}:{norm[:800]}"
                cached = self._cache_repair_math.get(cache_key)
                if isinstance(cached, str) and cached.strip():
                    return cached
        except Exception:
            cache_key = None
        
        ctx_prompt = ""
        if context_before:
            ctx_prompt += f"\nContext before:\n...{context_before[-300:]}\n"
        if context_after:
            ctx_prompt += f"\nContext after:\n{context_after[:300]}...\n"
            
        eq_hint = f"(Equation number: {eq_number})" if eq_number else ""
        prompt = (
            f"Recover this garbled math equation from PDF page {page_number+1} {eq_hint}.\n"
            "Return ONLY the LaTeX code (no $/$$ delimiters, no \\begin{equation}/align).\n"
            "CRITICAL fidelity rules:\n"
            "- Do NOT invent new variable names or symbols.\n"
            "- Preserve the original identifiers as much as possible (e.g., M vs A, C vs X).\n"
            "- Do NOT add explanatory prose.\n"
            "- If unsure about a piece, keep it minimal/faithful rather than guessing.\n"
            "If it's a display equation, return standard LaTeX for the equation body.\n"
            f"{ctx_prompt}\n"
            f"GARBLED BLOCK:\n{raw_math}\n"
        )
        if self.cfg.llm.request_sleep_s > 0:
            time.sleep(self.cfg.llm.request_sleep_s)
        try:
            resp = self._llm_create(
                messages=[
                    {"role": "system", "content": "You are a LaTeX math expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=self.cfg.llm.max_tokens,
            )
        except Exception:
            return None
            
        out = (resp.choices[0].message.content or "").strip()
        # Remove markdown fences if the model ignored instructions.
        if out.startswith("```"):
            try:
                m = re.search(r"```(?:\w+)?\n(.*?)```", out, re.DOTALL)
                if m:
                    out = (m.group(1) or "").strip()
            except Exception:
                pass
        # Strip $$...$$ or \[...\]
        if out.startswith("$$") and out.endswith("$$"):
            out = out[2:-2].strip()
        elif out.startswith("\\[") and out.endswith("\\]"):
            out = out[2:-2].strip()

        # Validate: if the model returned an explanation / prose, treat as failure.
        # (This prevents huge paragraphs being wrapped into $$ ... $$.)
        try:
            out_s = out.strip()
            # Disallow equation environments; we only want raw LaTeX math content.
            if re.search(r"\\begin\{equation\}|\\end\{equation\}|\\begin\{align\}|\\end\{align\}", out_s):
                return None
            word_n = len(re.findall(r"\b\w+\b", out_s))
            letters_n = len(re.findall(r"[A-Za-z]", out_s))
            has_sentence = (". " in out_s) or ("? " in out_s) or ("! " in out_s)
            if (len(out_s) >= 160 and word_n >= 22 and letters_n >= 80 and has_sentence):
                return None
            # Common refusal/explanation patterns
            bad_markers = [
                "the garbled block",
                "based on the notation",
                "it likely represents",
                "interpretation",
                "here is the latex",
                "here is the laTeX",
            ]
            low = out_s.lower()
            if any(x in low for x in bad_markers):
                return None
        except Exception:
            pass

        if cache_key and out:
            try:
                self._cache_set(self._cache_repair_math, cache_key, out)
            except Exception:
                pass
        return out

    def call_llm_repair_math_from_image(
        self,
        png_bytes: bytes,
        *,
        page_number: int,
        block_index: int,
        eq_number: Optional[str] = None,
    ) -> Optional[str]:
        """
        Vision-based math recovery: read the equation image and output LaTeX.
        This is optional and only works if the configured model/backend supports image inputs.
        """
        if not self.cfg.llm or not self._client:
            return None
        if not png_bytes:
            return None

        # Cache by content hash to avoid repeated vision calls.
        cache_key = None
        try:
            import hashlib
            h = hashlib.sha1(png_bytes).hexdigest()[:20]
            cache_key = f"math_vision:{self.cfg.llm.model}:{page_number}:{block_index}:{eq_number or ''}:{h}"
            cached = self._cache_repair_math.get(cache_key)
            if isinstance(cached, str) and cached.strip():
                return cached
        except Exception:
            cache_key = None

        b64 = base64.b64encode(png_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"
        eq_hint = f"(Equation number: {eq_number})" if eq_number else ""

        prompt = (
            f"Recover the LaTeX for this equation image from PDF page {page_number+1} {eq_hint}.\n"
            "Return ONLY the LaTeX for the equation body.\n"
            "- No $/$$ delimiters\n"
            "- No \\begin{equation}/align environments\n"
            "- No explanations\n"
            "Be exact and faithful to the image.\n"
        )
        if self.cfg.llm.request_sleep_s > 0:
            time.sleep(self.cfg.llm.request_sleep_s)
        debug_vision = False
        try:
            debug_vision = bool(int(os.environ.get("KB_PDF_DEBUG_VISION_MATH", "0") or "0")) or bool(
                getattr(self.cfg, "keep_debug", False)
            )
        except Exception:
            debug_vision = False
        try:
            resp = self._llm_create(
                messages=[
                    {"role": "system", "content": "You are a LaTeX math expert. Return only LaTeX."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
                temperature=0.0,
                max_tokens=min(900, int(getattr(self.cfg.llm, "max_tokens", 4096) or 4096)),
            )
        except Exception as e:
            if debug_vision:
                try:
                    m = str(getattr(self.cfg.llm, "model", "") or "")
                except Exception:
                    m = ""
                print(
                    f"[VISION_MATH] call failed: page={page_number+1} block={block_index+1} model={m!a} err={e!a}",
                    flush=True,
                )
            return None

        out = (resp.choices[0].message.content or "").strip()
        # Strip fences / delimiters if model disobeyed.
        if out.startswith("```"):
            try:
                m = re.search(r"```(?:\w+)?\n(.*?)```", out, re.DOTALL)
                if m:
                    out = (m.group(1) or "").strip()
            except Exception:
                pass
        if out.startswith("$$") and out.endswith("$$"):
            out = out[2:-2].strip()
        if re.search(r"\\begin\{equation\}|\\begin\{align\}", out):
            return None
        # Reject obvious non-LaTeX (tables / explanations) to avoid corrupting math blocks.
        try:
            out_s = out.strip()
            # Table-like output
            if re.search(r"(?m)^\s*\|.*\|\s*$", out_s) and out_s.count("|") >= 6:
                return None
            word_n = len(re.findall(r"\b\w+\b", out_s))
            letters_n = len(re.findall(r"[A-Za-z]", out_s))
            has_sentence = (". " in out_s) or ("? " in out_s) or ("! " in out_s)
            if (len(out_s) >= 180 and word_n >= 28 and letters_n >= 80 and has_sentence):
                return None
            bad_markers = ["where ", "denotes ", "this equation", "the equation", "we can see", "it represents"]
            low = out_s.lower()
            if any(x in low for x in bad_markers):
                return None
        except Exception:
            pass
        if cache_key and out:
            try:
                self._cache_set(self._cache_repair_math, cache_key, out)
            except Exception:
                pass
        return out or None

    def call_llm_page_to_markdown(
        self,
        png_bytes: bytes,
        *,
        page_number: int,
        total_pages: int = 0,
        hint: str = "",
        speed_mode: str = 'normal',
    ) -> Optional[str]:
        """
        Vision-based full-page conversion: send a page screenshot to the VL model
        and get back Markdown directly.  This bypasses all text-extraction / block-
        classification logic and relies entirely on the vision model's OCR + layout
        understanding.
        """
        if not self.cfg.llm or not self._client:
            return None
        if not png_bytes:
            return None

        b64 = base64.b64encode(png_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"

        page_hint = f" (page {page_number + 1}"
        if total_pages > 0:
            page_hint += f" of {total_pages}"
        page_hint += ")"
        extra = f"\nAdditional context: {hint}" if hint else ""

        prompt = (
            f"Convert this PDF page image{page_hint} into **Markdown**.{extra}\n\n"
            "**CRITICAL: Mathematical Formulas**\n"
            "Mathematical formulas are the MOST IMPORTANT part. You must convert them with 100% accuracy:\n"
            "- **Display equations** (centered, on their own line): Use $$...$$\n"
            "- **Inline equations** (within text): Use $...$\n"
            "- **Numbered equations**: Add \\tag{{N}} at the end, e.g. $$E=mc^2 \\tag{{1}}$$\n"
            "- **CRITICAL: Keep formulas COMPLETE and on ONE line** — do NOT split long formulas across multiple lines.\n"
            "  * If a formula is long, keep it in a single $$...$$ block, even if it wraps visually.\n"
            "  * Do NOT break formulas at operators like =, +, -, etc.\n"
            "- **Prime symbols and subscripts**: G'_i (NOT G' i), G'_{low} (NOT G' low)\n"
            "- **Every symbol must be exact**: Greek letters (α, β, γ, δ, ε, θ, λ, μ, π, σ, φ, ω, etc.) → \\alpha, \\beta, \\gamma, \\delta, \\epsilon, \\theta, \\lambda, \\mu, \\pi, \\sigma, \\phi, \\omega\n"
            "- **Subscripts are MANDATORY**: If you see a letter/number below the baseline, it MUST use _{...}\n"
            "  * Examples: α_j (NOT \\alphaj), c_i (NOT ci), x_{ij} (NOT xij), \\partial c_j (NOT \\partial c j)\n"
            "  * Single letter subscripts: use _{j}, _{i}, _{k}, etc.\n"
            "  * Multi-character subscripts: use {{ij}}, {{max}}, {{min}}, etc.\n"
            "- **Superscripts are MANDATORY**: If you see a letter/number above the baseline, it MUST use ^{...}\n"
            "  * Examples: x^2 (NOT x2), x^{n+1} (NOT xn+1), e^{-x} (NOT e-x)\n"
            "  * Single character superscripts: use ^2, ^n, ^T, etc.\n"
            "  * Multi-character superscripts: use {{n+1}}, {{-1}}, {{T}}, etc.\n"
            "- **Fractions**: Use \\frac{{numerator}}{{denominator}}\n"
            "- **Sums, integrals, products**: Use \\sum, \\int, \\prod with proper limits: \\sum_{{i=1}}^{{n}}, \\int_{{a}}^{{b}}\n"
            "- **Operators**: Use \\cdot, \\times, \\div, \\pm, \\mp, \\leq, \\geq, \\neq, \\approx, \\equiv\n"
            "- **Sets and vectors**: Use \\mathbb{{R}}, \\mathbb{{N}}, \\mathbf{{x}}, \\vec{{v}}\n"
            "- **Functions**: Use \\sin, \\cos, \\log, \\ln, \\exp, \\max, \\min\n"
            "- **Brackets**: Use \\left(, \\right), \\left[, \\right], \\left\\{{, \\right\\}}\n"
            "- **Special symbols**: \\infty, \\partial, \\nabla, \\forall, \\exists, \\in, \\subset, \\cup, \\cap\n"
            "- **DO NOT simplify or approximate formulas** — reproduce them exactly as shown.\n"
            "- **DO NOT replace symbols with words** — use LaTeX symbols only.\n"
            "- **DO NOT skip any part of a formula** — include every term, operator, and symbol.\n\n"
            "**CRITICAL: Tables**\n"
            "- **ALL tables MUST be converted to Markdown format** — do NOT skip tables or convert them to plain text.\n"
            "- Use proper Markdown table syntax:\n"
            "  | Header 1 | Header 2 | Header 3 |\n"
            "  | --- | --- | --- |\n"
            "  | Cell 1 | Cell 2 | Cell 3 |\n"
            "- **Every table row must be on a single line** — do NOT split cells across lines.\n"
            "- **Preserve all table content** — include every cell, even if it contains formulas or special symbols.\n"
            "- If a cell contains a formula, use $...$ or $$...$$ inside the cell.\n"
            "- **Align columns properly** — use consistent spacing.\n\n"
            "**CRITICAL: References Section**\n"
            "- **References section MUST be plain text** — NO formulas, NO code blocks, NO math notation.\n"
            "- **Each reference MUST be on a separate line** with a number label at the start.\n"
            "- **Format**: Use numbered list format: `[1] Author, Title, Journal, Year` or `1. Author, Title, Journal, Year`\n"
            "- **DO NOT use** `$$...$$`, `$...$`, or code blocks (```) in references.\n"
            "- **DO NOT split** a single reference across multiple lines — keep each reference on ONE line.\n"
            "- **If a reference spans multiple pages**, continue it on the same line (do NOT break it).\n"
            "- **Preserve reference numbering** — if the PDF shows [1], [2], [3], keep those numbers.\n"
            "- **Convert any formulas in references to plain text** — e.g., \"H_2O\" → \"H2O\" or \"H sub 2 O\".\n"
            "- **Example format**:\n"
            "  [1] Author A, Author B. Title of Paper. Journal Name, 2023.\n"
            "  [2] Author C et al. Another Title. Conference Name, 2024.\n"
            "  [3] Author D. Book Title. Publisher, 2022.\n\n"
            "**Other Content**\n"
            "1. Reproduce ALL text content faithfully — do NOT omit or summarise.\n"
            "2. Use proper heading levels (# / ## / ###) matching the document hierarchy.\n"
            "3. For figures / images: write ![Figure N](figure_N.png) with the caption below.\n"
            "4. Keep bullet / numbered lists as-is.\n"
            "5. Do NOT add any commentary, explanation, or notes of your own.\n"
            "6. Return ONLY the Markdown content.\n"
        )

        if self.cfg.llm.request_sleep_s > 0:
            time.sleep(self.cfg.llm.request_sleep_s)

        try:
            resp = self._llm_create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert document converter. "
                            "You convert PDF page images into clean, faithful Markdown with correct LaTeX math. "
                            "Return ONLY the Markdown. No explanations."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
                temperature=0.0,
                max_tokens=self._get_max_tokens_for_vision(),
            )
        except Exception as e:
            print(f"[VISION_PAGE] error page={page_number + 1} err={e!a}", flush=True)
            return None

        out = (resp.choices[0].message.content or "").strip()
        # Strip markdown fences if model wrapped the whole output
        if out.startswith("```markdown") or out.startswith("```md"):
            try:
                m = re.search(r"```(?:markdown|md)\n(.*?)```", out, re.DOTALL)
                if m:
                    out = (m.group(1) or "").strip()
            except Exception:
                pass
        elif out.startswith("```") and out.endswith("```"):
            out = out[3:]
            if out.endswith("```"):
                out = out[:-3]
            out = out.strip()
        return out or None

    def call_llm_confirm_and_level_heading(
        self,
        heading_text: str,
        *,
        page_number: int,
        suggested_level: Optional[int] = None
    ) -> Optional[dict]:
        """Use LLM to confirm if text is a heading and determine its level.
        
        Returns:
            dict with keys: 'is_heading' (bool), 'level' (int), 'text' (str)
            or None if LLM unavailable
        """
        if not self.cfg.llm or not self._client:
            return None

        cache_key = None
        try:
            norm = _normalize_text(heading_text or "").strip()
            if norm:
                cache_key = f"heading:{suggested_level or ''}:{norm[:400]}"
                cached = self._cache_confirm_heading.get(cache_key)
                if isinstance(cached, dict) and ("is_heading" in cached):
                    return dict(cached)
        except Exception:
            cache_key = None
        
        prompt = f"""You are an expert at identifying research paper section headings. Analyze this text carefully.

Text: "{heading_text}"

CRITICAL: If the text matches ANY of these patterns, it is DEFINITELY a heading:
- Starts with a number followed by a dot and space: "1. ", "2. ", "3. ", etc.
- Starts with number.number: "3.1. ", "4.2. ", etc.
- Starts with number.number.number: "3.1.1. ", etc.
- Starts with a capital letter followed by dot and space: "A. ", "B. ", etc.
- Common section names: "Introduction", "Related Work", "Method", "Experiments", "Conclusion", "Abstract", "References"

Rules for heading levels:
- "1. Introduction", "2. Related Work", "3. Method", "4. Experiments", "5. Conclusion" → Level 1 (#) - main sections
- "3.1. Background", "3.2. Method", "4.1. Experimental Setup" → Level 2 (##) - subsections  
- "3.1.1. Details", "3.1.2. Implementation" → Level 3 (###) - sub-subsections
- "A. Appendix", "B. Proof" → Level 2 (##) - appendix sections
- "A.1. Details" → Level 3 (###) - appendix subsections

REJECT as heading ONLY if:
- It's clearly an author name (e.g., "Yunhao Li", "John Smith")
- It's clearly a university/affiliation (e.g., "Zhejiang University")
- It's a table header with metrics (e.g., "PSNR ↑ SSIM ↑")
- It's a pure math expression with no words (e.g., "x^2 + y^2")
- It's very short (≤3 chars) with no context

Return JSON with:
- "is_heading": true/false
- "level": 1/2/3 (only if is_heading is true, null otherwise)
- "text": cleaned heading text

Examples:
- "1. Introduction" → {{"is_heading": true, "level": 1, "text": "1. Introduction"}}
- "2. Related Work" → {{"is_heading": true, "level": 1, "text": "2. Related Work"}}
- "3. Method" → {{"is_heading": true, "level": 1, "text": "3. Method"}}
- "3.1. Background" → {{"is_heading": true, "level": 2, "text": "3.1. Background"}}
- "4.1. Experimental Setup" → {{"is_heading": true, "level": 2, "text": "4.1. Experimental Setup"}}
- "x^2 + y^2" → {{"is_heading": false, "level": null, "text": "x^2 + y^2"}}
- "Yunhao Li" → {{"is_heading": false, "level": null, "text": "Yunhao Li"}}

Return ONLY valid JSON, no other text:"""
        
        try:
            resp = self._llm_create(
                messages=[
                    {"role": "system", "content": "You are an expert at identifying research paper section headings and determining their hierarchy. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=200,
            )
            result_text = (resp.choices[0].message.content or "").strip()
            # Remove markdown code fences if present
            if result_text.startswith("```"):
                result_text = re.sub(r'^```(?:\w+)?\n?', '', result_text)
                result_text = re.sub(r'\n?```$', '', result_text)
            
            import json
            result = json.loads(result_text)
            if cache_key and isinstance(result, dict):
                try:
                    self._cache_set(self._cache_confirm_heading, cache_key, dict(result))
                except Exception:
                    pass
            return result
        except Exception as e:
            return None

    def call_llm_polish_code(self, raw_code: str, *, page_number: int, block_index: int) -> Optional[str]:
        if not self.cfg.llm:
            return None
            
        prompt = (
            f"Fix this OCR-damaged pseudocode/code from PDF page {page_number+1}.\n"
            "Preserve indentation. Fix arrow symbols (<- ->), assignment (:=), and keywords.\n"
            "Return ONLY the fixed code text. No markdown fences.\n\n"
            f"RAW CODE:\n{raw_code}\n"
        )
        if self.cfg.llm.request_sleep_s > 0:
            time.sleep(self.cfg.llm.request_sleep_s)
        try:
            resp = self._llm_create(
                messages=[
                    {"role": "system", "content": "You are a code fixer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=self.cfg.llm.max_tokens,
            )
        except Exception:
            return None
        
        return (resp.choices[0].message.content or "").strip()

    def call_llm_repair_body_paragraph(self, text: str, *, page_number: int, block_index: int) -> Optional[str]:
        if not self.cfg.llm:
            return None
            
        prompt = (
            f"Fix this text paragraph from PDF page {page_number+1}.\n"
            "It may contain inline math that was OCR'd as garbage text.\n"
            "Convert inline math to LaTeX ($...$). Fix partial words.\n"
            "Return ONLY the fixed paragraph text.\n\n"
            f"RAW TEXT:\n{text}\n"
        )
        if self.cfg.llm.request_sleep_s > 0:
            time.sleep(self.cfg.llm.request_sleep_s)
        try:
            resp = self._llm_create(
                messages=[
                    {"role": "system", "content": "You are a text fixer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=self.cfg.llm.max_tokens,
            )
        except Exception:
            return None
            
        return (resp.choices[0].message.content or "").strip()

    def call_llm_convert_page(self, text_content: str, *, page_num: int) -> str:
        """
        Full page conversion when heuristics fail.
        """
        if not self.cfg.llm:
            return text_content
            
        prompt = (
            f"Convert this raw PDF text from page {page_num+1} into clean, structured Markdown.\n"
            "Fix headers, lists, tables, and math.\n"
            "Return ONLY the Markdown.\n\n"
            f"RAW TEXT:\n{text_content}\n"
        )
        if self.cfg.llm.request_sleep_s > 0:
            time.sleep(self.cfg.llm.request_sleep_s)
        try:
            resp = self._llm_create(
                messages=[
                    {"role": "system", "content": "You are a PDF to Markdown converter."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=self.cfg.llm_render_max_tokens or self.cfg.llm.max_tokens,
            )
        except Exception:
            return text_content
            
        return (resp.choices[0].message.content or text_content).strip()

    def call_llm_classify_blocks(self, blocks: list[TextBlock], page_number: int, page_wh: tuple[float, float]) -> Optional[list[dict]]:
        if not self.cfg.llm or not self._client or not self.cfg.llm_classify:
            return None

        W, H = page_wh
        llm_cfg = self.cfg.llm

        def pack(bs: list[TextBlock], offset: int) -> list[dict]:
            out = []
            for i, b in enumerate(bs):
                txt = b.text.strip()
                if len(txt) > 800:
                    txt = txt[:800] + "..."
                out.append(
                    {
                        "i": offset + i,
                        "text": txt,
                        "font": round(float(b.max_font_size), 2),
                        "bold": bool(b.is_bold),
                        "bbox": [round(float(x), 2) for x in b.bbox],
                        "page": page_number,
                        "page_wh": [round(W, 2), round(H, 2)],
                    }
                )
            return out

        system = "You are a strict PDF block classifier. Output JSON only."

        def make_prompt(items: list[dict]) -> str:
            return f"""
Classify each block from a research paper PDF page.

Return a JSON array with EXACTLY the same number of items as the input.
Each output item MUST be an object with keys:
- i: integer (copy input i)
- action: "keep" or "drop"
- kind: "heading" | "body" | "table" | "math" | "code" | "caption"
- heading_level: 1 | 2 | 3 | null  (only for kind=heading)
- text: string (cleaned text; keep meaning; fix mojibake/ligatures; keep spacing for tables)

STRICT RULES:
1) Headings: kind="heading" ONLY if text matches a real paper section heading:
   - Numbered: ^\\d+(\\.\\d+)*\\s+<LETTER>  (examples: "1 INTRODUCTION", "5.2 Adaptive Control").
   - Appendix: ^[A-Z](?:\\.\\d+)*\\s+<LETTER> (examples: "A DETAILS...", "B.1 ...").
   - The literal word "APPENDIX".
2) Drop boilerplate/noise: headers, footers, page numbers, copyright.
3) Table vs code vs math:
   - table: rows/columns of numbers.
   - math: equations/symbols.
   - code: pseudocode/algorithms (while/for/if, arrows etc).
4) Captions: kind="caption" if starts with "Fig." or "Table".
5) Never invent content.

INPUT JSON:
{json.dumps(items, ensure_ascii=False)}
""".strip()

        batch_size = max(10, int(self.cfg.classify_batch_size))
        all_results: list[dict] = []
        for start in range(0, len(blocks), batch_size):
            sub = blocks[start : start + batch_size]
            items = pack(sub, offset=start)
            if llm_cfg.request_sleep_s > 0:
                time.sleep(llm_cfg.request_sleep_s)
            try:
                resp = self._llm_create(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": make_prompt(items)},
                    ],
                    temperature=0.0,
                    max_tokens=llm_cfg.max_tokens,
                )
            except Exception:
                return None
            content = resp.choices[0].message.content or ""
            arr = self._extract_json_array(content)
            if not isinstance(arr, list) or len(arr) != len(items):
                return None
            all_results.extend(arr)
        return all_results

    def call_llm_translate_zh(self, md: str) -> str:
        if not self.cfg.llm or not self._client:
            return md
        llm_cfg = self.cfg.llm
        prompt = (
            "Translate to Chinese. Keep ALL Markdown structure (#, $$, images, code fences) exactly. "
            "Do not translate author names, venues, citations, or LaTeX.\n\n"
            + md
        )
        if llm_cfg.request_sleep_s > 0:
            time.sleep(llm_cfg.request_sleep_s)
        try:
            resp = self._llm_create(
                messages=[{"role": "system", "content": "Translator mode."}, {"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=llm_cfg.max_tokens,
            )
        except Exception:
            return md
        return (resp.choices[0].message.content or md).strip()

    def call_llm_split_references(self, ref_block: str, *, paper_name: str) -> Optional[str]:
        if not self.cfg.llm:
            return None
        prompt = (
            f"Split this aggregated references block from paper '{paper_name}' into individual reference items.\n"
            "Return them as a Markdown numbered list.\n"
            "Do not change the text content much, just separate them.\n\n"
            f"BLOCK:\n{ref_block}\n"
        )
        if self.cfg.llm.request_sleep_s > 0:
            time.sleep(self.cfg.llm.request_sleep_s)
        try:
            resp = self._llm_create(
                messages=[
                    {"role": "system", "content": "Reference splitter."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=self.cfg.llm.max_tokens,
            )
        except Exception:
            return None
        return (resp.choices[0].message.content or "").strip()

    def run_llm_jobs_parallel(self, jobs: list[tuple[str, int, Callable[[], Optional[str]]]]) -> list[Optional[str]]:
        if not jobs:
            return []
        worker_cap = max(1, int(self.cfg.llm_workers))
        if worker_cap <= 1 or len(jobs) <= 1:
            out_seq: list[Optional[str]] = []
            for _, _, fn in jobs:
                try:
                    out_seq.append(fn())
                except Exception:
                    out_seq.append(None)
            return out_seq

        max_workers = min(worker_cap, len(jobs))
        out: list[Optional[str]] = [None] * len(jobs)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            fut_to_idx = {executor.submit(fn): i for i, (_, _, fn) in enumerate(jobs)}
            for fut in as_completed(fut_to_idx):
                i = fut_to_idx[fut]
                try:
                    out[i] = fut.result()
                except Exception:
                    out[i] = None
        return out
