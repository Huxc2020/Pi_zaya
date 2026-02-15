from __future__ import annotations

import os
import shutil
import re
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import fitz
except ImportError:
    fitz = None

from .config import ConvertConfig
from .models import TextBlock
from .geometry_utils import _rect_area, _union_rect, _rect_intersection_area
from .text_utils import _normalize_text

# Greek letter to LaTeX mapping (expanded)
GREEK_TO_LATEX = {
    # Lowercase
    "\u03b1": r"\alpha",
    "\u03b2": r"\beta",
    "\u03b3": r"\gamma",
    "\u03b4": r"\delta",
    "\u03b5": r"\epsilon",
    "\u03b6": r"\zeta",
    "\u03b7": r"\eta",
    "\u03b8": r"\theta",
    "\u03b9": r"\iota",
    "\u03ba": r"\kappa",
    "\u03bb": r"\lambda",
    "\u03bc": r"\mu",
    "\u03bd": r"\nu",
    "\u03be": r"\xi",
    "\u03bf": r"o",  # omicron (rarely used)
    "\u03c0": r"\pi",
    "\u03c1": r"\rho",
    "\u03c3": r"\sigma",
    "\u03c4": r"\tau",
    "\u03c5": r"\upsilon",
    "\u03c6": r"\phi",
    "\u03c7": r"\chi",
    "\u03c8": r"\psi",
    "\u03c9": r"\omega",
    # Uppercase
    "\u0391": r"A",  # Alpha
    "\u0392": r"B",  # Beta
    "\u0393": r"\Gamma",
    "\u0394": r"\Delta",
    "\u0395": r"E",  # Epsilon
    "\u0396": r"Z",  # Zeta
    "\u0397": r"H",  # Eta
    "\u0398": r"\Theta",
    "\u0399": r"I",  # Iota
    "\u039a": r"K",  # Kappa
    "\u039b": r"\Lambda",
    "\u039c": r"M",  # Mu
    "\u039d": r"N",  # Nu
    "\u039e": r"\Xi",
    "\u039f": r"O",  # Omicron
    "\u03a0": r"\Pi",
    "\u03a1": r"P",  # Rho
    "\u03a3": r"\Sigma",
    "\u03a4": r"T",  # Tau
    "\u03a5": r"\Upsilon",
    "\u03a6": r"\Phi",
    "\u03a7": r"X",  # Chi
    "\u03a8": r"\Psi",
    "\u03a9": r"\Omega",
}

# Math symbol to LaTeX mapping
MATH_SYMBOL_TO_LATEX = {
    "\u2264": r"\leq",
    "\u2265": r"\geq",
    "\u2260": r"\neq",
    "\u2248": r"\approx",
    "\u2208": r"\in",
    "\u2209": r"\notin",
    "\u2211": r"\sum",
    "\u220f": r"\prod",
    "\u222b": r"\int",
    "\u221e": r"\infty",
    "\u2192": r"\to",
    "\u00d7": r"\times",
    "\u00b7": r"\cdot",
    "\u2212": r"-",  # minus sign
}
from .layout_analysis import (
    detect_body_font_size,
    build_repeated_noise_texts,
    _pick_render_scale,
    _collect_visual_rects,
    _is_frontmatter_noise_line,
    _pick_column_range,
    sort_blocks_reading_order,
    _detect_column_split_x,
)
from .geometry_utils import _bbox_width
from .heuristics import (
    _suggest_heading_level,
    _is_noise_line,
    _page_has_references_heading,
    _page_looks_like_references_content,
    detect_header_tag,
)
from .tables import _extract_tables_by_layout, _is_markdown_table_sane, _page_maybe_has_table_from_dict, table_text_to_markdown
from .block_classifier import _looks_like_math_block, _looks_like_code_block
from .llm_worker import LLMWorker
from .post_processing import postprocess_markdown
from .md_analyzer import MarkdownAnalyzer


class PDFConverter:
    def __init__(self, cfg: ConvertConfig):
        self.cfg = cfg
        self.llm_worker = LLMWorker(cfg)
        self.noise_texts: Set[str] = set()
        # Store optional config attributes
        self.dpi = getattr(cfg, 'dpi', 200)
        self.analyze_quality = getattr(cfg, 'analyze_quality', True)
        # Track seen headings to avoid duplicates
        self.seen_headings: Set[str] = set()
        # Track heading hierarchy across pages
        self.heading_stack: List[Tuple[int, str]] = []  # (level, text)

    def convert(self, pdf_path: str, save_dir: str) -> None:
        """Convert PDF to Markdown using the new converter."""
        print("=" * 60, flush=True)
        print("NEW PDFConverter starting...", flush=True)
        print("=" * 60, flush=True)
        
        if fitz is None:
            raise ImportError("PyMuPDF (fitz) not installed.")
            
        pdf_path = Path(pdf_path).resolve()
        save_dir = Path(save_dir).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        
        assets_dir = save_dir / "assets"
        assets_dir.mkdir(exist_ok=True)
        
        print(f"Opening PDF: {pdf_path}", flush=True)
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        print(f"PDF opened successfully, {total_pages} pages", flush=True)
        
        # Output total pages for progress tracking (must match expected format)
        print(f"Detected body font size: 12.0 | pages: {total_pages} | range: 1-{total_pages}", flush=True)
        print(f"Starting conversion of {total_pages} pages...", flush=True)
        
        # Pre-scan for noise
        self.noise_texts = build_repeated_noise_texts(doc)
        print(f"Detected {len(self.noise_texts)} repeated lines as noise.", flush=True)
        
        # Process pages
        # For simplicity, we'll use a sequential loop or simple batching here.
        # The original code had complex batching. We'll simplify to page-by-page for now
        # or use ThreadPool if configured.
        
        md_pages = [None] * total_pages
        speed_mode = getattr(self.cfg, 'speed_mode', 'normal')
        speed_config = self._get_speed_mode_config(speed_mode, total_pages)
        self._active_speed_config = speed_config
        
        # Use LLM config from cfg (already set by test2.py)
        use_llm = False
        llm_config = self.cfg.llm
        
        # If LLM config is provided, use it
        if llm_config:
            use_llm = True
            print(f"Using LLM ({llm_config.model}) for processing", flush=True)
        else:
            print("LLM not configured, using fast mode", flush=True)
        
        # Only use vision-direct mode: screenshot → VL → Markdown
        speed_mode = getattr(self.cfg, 'speed_mode', 'normal')
        
        if speed_mode == 'no_llm':
            # No LLM mode: use basic text extraction (fallback)
            print("[MODE] No LLM: basic text extraction only", flush=True)
            md_pages = self._process_batch_no_llm(doc, pdf_path, assets_dir)
        else:
            # Vision-direct mode with LLM (normal or ultra_fast)
            if not use_llm or not llm_config:
                print("[WARN] LLM not configured, falling back to no_llm mode", flush=True)
                md_pages = self._process_batch_no_llm(doc, pdf_path, assets_dir)
            else:
                mode_name = "ultra_fast" if speed_mode == "ultra_fast" else "normal"
                print(f"[MODE] Vision-direct ({mode_name}): each page screenshot → VL model → Markdown", flush=True)
                md_pages = self._process_batch_vision_direct(doc, pdf_path, assets_dir, speed_mode=speed_mode)
            
        final_md = "\n\n".join(filter(None, md_pages))

        # Post-process: minimal cleanup for vision-direct output
        final_md = postprocess_markdown(final_md)
        final_md = self._fix_heading_structure(final_md)
        final_md = self._fix_vision_formula_errors(final_md)
        final_md = self._fix_references_format(final_md)
        
        # Write output
        out_file = save_dir / "output.md"
        out_file.write_text(final_md, encoding="utf-8")
        print(f"Saved to {out_file}", flush=True)
        print(f"Conversion completed successfully!", flush=True)
        
        # Analyze quality and generate report
        if self.analyze_quality:
            analyzer = MarkdownAnalyzer()
            issues = analyzer.analyze(final_md, out_file)
            if issues:
                report = analyzer.generate_report()
                report_file = save_dir / "quality_report.md"
                report_file.write_text(report, encoding="utf-8")
                print(f"Quality report saved to {report_file}")
                print(f"Found {len(issues)} quality issues")
            else:
                print("[OK] No quality issues detected")

    # Removed: _process_batch_fast and _process_batch_llm (old text extraction methods)
    # Now only using vision-direct mode (_process_batch_vision_direct) and no-LLM mode (_process_batch_no_llm)

    # ------------------------------------------------------------------
    # Vision-direct mode: screenshot each page → VL model → Markdown
    # ------------------------------------------------------------------
    def _process_batch_vision_direct(self, doc, pdf_path: Path, assets_dir: Path, speed_mode: str = 'normal') -> List[Optional[str]]:
        """
        Bypass all text-extraction / block-classification logic.
        For every page: render a high-DPI screenshot, send it to the vision LLM,
        and collect the Markdown it returns.
        Supports parallel processing with ThreadPoolExecutor.
        """
        total_pages = len(doc)
        results: List[Optional[str]] = [None] * total_pages

        start = max(0, int(getattr(self.cfg, "start_page", 0) or 0))
        end = int(getattr(self.cfg, "end_page", -1) or -1)
        if end < 0:
            end = total_pages
        end = min(total_pages, end)
        if start >= end:
            return results

        # Get speed mode config
        import multiprocessing
        speed_config = self._get_speed_mode_config(speed_mode, total_pages)
        cpu_count = multiprocessing.cpu_count()
        
        # DPI from config or environment variable
        base_dpi = int(getattr(self, "dpi", 200) or 200)
        try:
            vision_dpi = int(os.environ.get("KB_PDF_VISION_DPI", "") or "")
            if vision_dpi > 0:
                dpi = max(150, min(600, vision_dpi))
            else:
                dpi = speed_config.get('dpi', 160)
        except Exception:
            dpi = speed_config.get('dpi', 160)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        # Determine number of workers for parallel processing
        raw_llm_pw = (os.environ.get("KB_PDF_LLM_PAGE_WORKERS") or "").strip()
        num_workers = int(raw_llm_pw) if raw_llm_pw else int(os.environ.get("KB_PDF_WORKERS", "0") or "0")
        if num_workers <= 0:
            max_parallel = speed_config.get('max_parallel_pages', min(32, cpu_count * 4))
            if total_pages <= 2:
                num_workers = 1
            else:
                num_workers = min(max_parallel, cpu_count, total_pages)

        # Get max_inflight from config
        try:
            raw_inflight = (os.environ.get("KB_LLM_MAX_INFLIGHT") or "").strip()
            if raw_inflight:
                max_inflight = int(raw_inflight)
            else:
                max_inflight = speed_config.get('max_inflight', 32)
            max_inflight = max(1, min(32, int(max_inflight)))
        except Exception:
            max_inflight = speed_config.get('max_inflight', 32)
        
        num_workers_before_cap = num_workers
        cap = None
        
        # If user explicitly set KB_PDF_LLM_PAGE_WORKERS, don't cap at all (trust the user)
        if raw_llm_pw:
            cap = None
            num_workers = min(int(num_workers), int(total_pages))
        else:
            # Only cap if user didn't explicitly set KB_PDF_LLM_PAGE_WORKERS
            try:
                raw_cap = (os.environ.get("KB_PDF_LLM_PAGE_WORKERS_CAP") or "").strip()
                if raw_cap:
                    cap = int(raw_cap)
                    cap = max(1, min(64, int(cap)))
                else:
                    # For normal mode, use aggressive parallelism (max_parallel_pages from config)
                    # For ultra_fast, use more conservative cap
                    if speed_mode == 'normal':
                        # Normal mode: use max_parallel_pages directly, don't cap too aggressively
                        cap = max_parallel  # Use the max_parallel from config (32 for normal mode)
                    else:
                        # Ultra-fast mode: use more conservative cap
                        cap_default = max(2, min(16, int(max_inflight) * 2))
                        cap = cap_default
                
                if cap is not None:
                    num_workers = min(int(num_workers), int(cap), int(total_pages))
                else:
                    num_workers = min(int(num_workers), int(total_pages))
            except Exception as e:
                # On error, just use num_workers as-is (don't cap)
                cap = None
                num_workers = min(int(num_workers), int(total_pages))

        # Debug: print worker count calculation
        try:
            print(f"[VISION_DIRECT] worker calculation: raw_llm_pw={raw_llm_pw!r}, num_workers_before={num_workers_before_cap}, final_num_workers={num_workers}, max_inflight={max_inflight}, cap={cap}, total_pages={total_pages}", flush=True)
            # Warn if max_inflight is less than num_workers (may cause timeouts)
            if max_inflight < num_workers:
                print(f"[VISION_DIRECT] WARNING: KB_LLM_MAX_INFLIGHT={max_inflight} < num_workers={num_workers}. "
                      f"This may cause timeout errors. Consider setting KB_LLM_MAX_INFLIGHT >= {num_workers}", flush=True)
        except Exception:
            pass

        if num_workers <= 1 or total_pages <= 1:
            # Sequential processing
            print(f"[VISION_DIRECT] Converting pages {start+1}–{end} via VL screenshots (dpi={dpi}, sequential)", flush=True)
            for i in range(start, end):
                t0 = time.time()
                print(f"Processing page {i+1}/{total_pages} (vision-direct) ...", flush=True)
                try:
                    page = doc.load_page(i)
                    
                    # Extract images BEFORE sending to LLM, so they're available when LLM references them
                    try:
                        visual_rects = _collect_visual_rects(page)
                        W = float(page.rect.width)
                        H = float(page.rect.height)
                        header_threshold = H * 0.12
                        footer_threshold = H * 0.88
                        side_margin = W * 0.05
                        spanning_threshold = W * 0.55
                        
                        img_count = 0
                        for rect_idx, rect in enumerate(visual_rects):
                            # Check if this is a full-width image
                            is_full_width = _rect_area(rect) >= (W * H * 0.40) or _bbox_width(tuple(rect)) >= spanning_threshold
                            
                            # Skip if in header/footer region (unless it's a large figure)
                            if rect.y1 < header_threshold or rect.y0 > footer_threshold:
                                if not is_full_width and _rect_area(rect) < (W * H * 0.15):
                                    continue
                            
                            # Skip small edge artifacts
                            if not is_full_width:
                                if rect.x0 < side_margin and rect.width < W * 0.1:
                                    continue
                                if rect.x1 > W - side_margin and rect.width < W * 0.1:
                                    continue
                            
                            # Crop rect slightly to avoid edge artifacts
                            crop_margin = 2.0 if not is_full_width else 1.0
                            cropped_rect = fitz.Rect(
                                max(0, rect.x0 + crop_margin),
                                max(0, rect.y0 + crop_margin),
                                min(W, rect.x1 - crop_margin),
                                min(H, rect.y1 - crop_margin)
                            )
                            
                            if cropped_rect.width <= 0 or cropped_rect.height <= 0:
                                continue
                            
                            # Save image
                            img_name = f"page_{i+1}_fig_{rect_idx+1}.png"
                            img_path = assets_dir / img_name
                            
                            try:
                                pix_img = page.get_pixmap(clip=cropped_rect, dpi=dpi)
                                pix_img.save(img_path)
                                # Verify file was saved
                                if img_path.exists() and img_path.stat().st_size >= 256:
                                    img_count += 1
                            except Exception:
                                # Fallback to original rect if crop fails
                                try:
                                    pix_img = page.get_pixmap(clip=rect, dpi=dpi)
                                    pix_img.save(img_path)
                                    if img_path.exists() and img_path.stat().st_size >= 256:
                                        img_count += 1
                                except Exception:
                                    continue
                        
                        if img_count > 0:
                            print(f"  [Page {i+1}] Extracted {img_count} images", flush=True)
                    except Exception as e:
                        print(f"  [Page {i+1}] Image extraction failed: {e}", flush=True)
                        # Continue even if image extraction fails
                    
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    png_bytes = pix.tobytes("png")
                    
                    # Compress PNG based on speed mode config
                    try:
                        compress_level_raw = os.environ.get("KB_PDF_VISION_COMPRESS", "").strip()
                        if compress_level_raw:
                            compress_level = int(compress_level_raw)
                        else:
                            compress_level = speed_config.get('compress', 3)
                        if compress_level > 0:
                            try:
                                from PIL import Image
                                import io
                                img = Image.open(io.BytesIO(png_bytes))
                                output = io.BytesIO()
                                img.save(output, format="PNG", optimize=True, compress_level=min(9, max(1, compress_level)))
                                png_bytes = output.getvalue()
                            except ImportError:
                                pass  # PIL not available, skip compression
                    except Exception:
                        pass

                    # Save screenshot for debugging if requested
                    try:
                        if bool(int(os.environ.get("KB_PDF_SAVE_PAGE_SCREENSHOTS", "0") or "0")):
                            dbg_dir = assets_dir / "page_screenshots"
                            dbg_dir.mkdir(exist_ok=True)
                            (dbg_dir / f"page_{i+1}.png").write_bytes(png_bytes)
                    except Exception:
                        pass

                    md = self.llm_worker.call_llm_page_to_markdown(
                        png_bytes,
                        page_number=i,
                        total_pages=total_pages,
                        speed_mode=speed_mode,
                    )
                    elapsed = time.time() - t0
                    if md:
                        results[i] = md
                        print(f"Finished page {i+1}/{total_pages} ({elapsed:.1f}s, {len(md)} chars)", flush=True)
                    else:
                        # Fallback: try old pipeline for this page
                        print(f"[VISION_DIRECT] VL returned empty for page {i+1}, falling back to extraction pipeline", flush=True)
                        results[i] = self._process_page(page, page_index=i, pdf_path=pdf_path, assets_dir=assets_dir)
                        print(f"Finished page {i+1}/{total_pages} (fallback, {time.time()-t0:.1f}s)", flush=True)
                except Exception as e:
                    print(f"[VISION_DIRECT] error page {i+1}: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    results[i] = None
            return results

        # Parallel processing
        print(f"[VISION_DIRECT] Converting pages {start+1}–{end} via VL screenshots (dpi={dpi}, {num_workers} workers)", flush=True)
        from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

        def process_single_page(i: int):
            try:
                print(f"Processing page {i+1}/{total_pages} (vision-direct) ...", flush=True)
                t0 = time.time()
                # Avoid sharing fitz.Document / Page across threads
                with fitz.open(str(pdf_path)) as local_doc:
                    page = local_doc.load_page(i)
                    
                    # Extract images BEFORE sending to LLM, so they're available when LLM references them
                    try:
                        visual_rects = _collect_visual_rects(page)
                        W = float(page.rect.width)
                        H = float(page.rect.height)
                        header_threshold = H * 0.12
                        footer_threshold = H * 0.88
                        side_margin = W * 0.05
                        spanning_threshold = W * 0.55
                        
                        img_count = 0
                        for rect_idx, rect in enumerate(visual_rects):
                            # Check if this is a full-width image
                            is_full_width = _rect_area(rect) >= (W * H * 0.40) or _bbox_width(tuple(rect)) >= spanning_threshold
                            
                            # Skip if in header/footer region (unless it's a large figure)
                            if rect.y1 < header_threshold or rect.y0 > footer_threshold:
                                if not is_full_width and _rect_area(rect) < (W * H * 0.15):
                                    continue
                            
                            # Skip small edge artifacts
                            if not is_full_width:
                                if rect.x0 < side_margin and rect.width < W * 0.1:
                                    continue
                                if rect.x1 > W - side_margin and rect.width < W * 0.1:
                                    continue
                            
                            # Crop rect slightly to avoid edge artifacts
                            crop_margin = 2.0 if not is_full_width else 1.0
                            cropped_rect = fitz.Rect(
                                max(0, rect.x0 + crop_margin),
                                max(0, rect.y0 + crop_margin),
                                min(W, rect.x1 - crop_margin),
                                min(H, rect.y1 - crop_margin)
                            )
                            
                            if cropped_rect.width <= 0 or cropped_rect.height <= 0:
                                continue
                            
                            # Save image
                            img_name = f"page_{i+1}_fig_{rect_idx+1}.png"
                            img_path = assets_dir / img_name
                            
                            try:
                                pix_img = page.get_pixmap(clip=cropped_rect, dpi=dpi)
                                pix_img.save(img_path)
                                # Verify file was saved
                                if img_path.exists() and img_path.stat().st_size >= 256:
                                    img_count += 1
                            except Exception:
                                # Fallback to original rect if crop fails
                                try:
                                    pix_img = page.get_pixmap(clip=rect, dpi=dpi)
                                    pix_img.save(img_path)
                                    if img_path.exists() and img_path.stat().st_size >= 256:
                                        img_count += 1
                                except Exception:
                                    continue
                        
                        if img_count > 0:
                            print(f"  [Page {i+1}] Extracted {img_count} images", flush=True)
                    except Exception as e:
                        print(f"  [Page {i+1}] Image extraction failed: {e}", flush=True)
                        # Continue even if image extraction fails
                    
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    png_bytes = pix.tobytes("png")
                    
                    # Compress PNG based on speed mode config
                    try:
                        compress_level_raw = os.environ.get("KB_PDF_VISION_COMPRESS", "").strip()
                        if compress_level_raw:
                            compress_level = int(compress_level_raw)
                        else:
                            compress_level = speed_config.get('compress', 3)
                        if compress_level > 0:
                            # Use PIL/Pillow to compress if available
                            try:
                                from PIL import Image
                                import io
                                img = Image.open(io.BytesIO(png_bytes))
                                output = io.BytesIO()
                                # compress_level: 1-9, higher = more compression (slower)
                                img.save(output, format="PNG", optimize=True, compress_level=min(9, max(1, compress_level)))
                                png_bytes = output.getvalue()
                            except ImportError:
                                pass  # PIL not available, skip compression
                    except Exception:
                        pass

                    # Save screenshot for debugging if requested
                    try:
                        if bool(int(os.environ.get("KB_PDF_SAVE_PAGE_SCREENSHOTS", "0") or "0")):
                            dbg_dir = assets_dir / "page_screenshots"
                            dbg_dir.mkdir(exist_ok=True)
                            (dbg_dir / f"page_{i+1}.png").write_bytes(png_bytes)
                    except Exception:
                        pass

                    md = self.llm_worker.call_llm_page_to_markdown(
                        png_bytes,
                        page_number=i,
                        total_pages=total_pages,
                        speed_mode=speed_mode,
                    )
                    elapsed = time.time() - t0
                    if md:
                        print(f"Finished page {i+1}/{total_pages} ({elapsed:.1f}s, {len(md)} chars)", flush=True)
                        return i, md
                    else:
                        # Fallback: try old pipeline for this page
                        print(f"[VISION_DIRECT] VL returned empty for page {i+1}, falling back to extraction pipeline", flush=True)
                        fallback_md = self._process_page(page, page_index=i, pdf_path=pdf_path, assets_dir=assets_dir)
                        print(f"Finished page {i+1}/{total_pages} (fallback, {time.time()-t0:.1f}s)", flush=True)
                        return i, fallback_md
            except Exception as e:
                print(f"[VISION_DIRECT] error page {i+1}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                return i, None

        executor = ThreadPoolExecutor(max_workers=num_workers)
        futures = {executor.submit(process_single_page, i): i for i in range(start, end)}
        pending = set(futures.keys())
        done_pages = set()

        # Heartbeat logging to avoid UI appearing frozen
        hb_every_s = 8.0
        try:
            hb_every_s = float(os.environ.get("KB_PDF_BATCH_HEARTBEAT_S", str(hb_every_s)) or hb_every_s)
            hb_every_s = max(2.0, min(60.0, hb_every_s))
        except Exception:
            hb_every_s = 8.0
        last_hb = time.time()

        try:
            while pending:
                now_ts = time.time()
                done, not_done = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                pending = set(not_done)
                try:
                    if (now_ts - last_hb) >= hb_every_s:
                        inflight_pages = sorted({int(futures[fut]) + 1 for fut in pending})
                        if inflight_pages:
                            head = inflight_pages[:12]
                            more = len(inflight_pages) - len(head)
                            extra = f" (+{more} more)" if more > 0 else ""
                            print(
                                f"[VISION_DIRECT] still running pages: {head}{extra} | workers={num_workers} llm_inflight={max_inflight}",
                                flush=True,
                            )
                        last_hb = now_ts
                except Exception:
                    pass
                for future in done:
                    i = futures[future]
                    try:
                        i2, result = future.result()
                        results[i2] = result
                        done_pages.add(i2 + 1)
                    except Exception as e:
                        print(f"[VISION_DIRECT] error processing page {i+1}: {e}", flush=True)
                        results[i] = None
        finally:
            if pending:
                for future in pending:
                    i = futures.get(future)
                    if i is not None and results[i] is None:
                        results[i] = f"<!-- kb_page: {i+1} -->\n\n[Page {i+1} conversion incomplete]"
                    try:
                        future.cancel()
                    except Exception:
                        pass
            executor.shutdown(wait=False, cancel_futures=True)

        return results

    def _process_page(self, page, page_index: int, pdf_path: Path, assets_dir: Path) -> str:
        import time
        page_start = time.time()
        
        # 1. Analyze Layout
        step_start = time.time()
        body_size = detect_body_font_size([page]) # Heuristic was on full doc, but per-page is okay fallback
        print(f"  [Page {page_index+1}] Step 1 (layout analysis): {time.time()-step_start:.2f}s", flush=True)
        
        # 2. Check if this is a references page (for special handling)
        step_start = time.time()
        is_references_page = _page_has_references_heading(page) or _page_looks_like_references_content(page)
        print(f"  [Page {page_index+1}] Step 2 (refs check): {time.time()-step_start:.2f}s", flush=True)
        
        # 3. Extract specific rects (excluding header/footer regions)
        step_start = time.time()
        W = float(page.rect.width)
        H = float(page.rect.height)
        header_threshold = H * 0.12
        footer_threshold = H * 0.88
        visual_rects = _collect_visual_rects(page)
        # Filter visual rects in header/footer (unless they're large figures)
        visual_rects = [
            r for r in visual_rects 
            if not (r.y1 < header_threshold or r.y0 > footer_threshold) or _rect_area(r) > (W * H * 0.15)
        ]
        print(f"  [Page {page_index+1}] Step 3 (visual rects): {time.time()-step_start:.2f}s, found {len(visual_rects)} rects", flush=True)
        
        # 4. Extract Tables
        step_start = time.time()
        # We need to map table rects to avoid processing them as text
        try:
            # Fast hint gate to enable more aggressive table strategies on table-heavy pages.
            # This significantly improves table extraction for dense CVPR/ICCV two-column PDFs.
            try:
                d0 = page.get_text("dict")
                has_table_hint = _page_maybe_has_table_from_dict(d0) if isinstance(d0, dict) else False
            except Exception:
                has_table_hint = False
            try:
                setattr(page, "has_table_hint", bool(has_table_hint))
            except Exception:
                pass

            tables_found = _extract_tables_by_layout(
                page, 
                pdf_path=pdf_path, 
                page_index=page_index,
                visual_rects=visual_rects,
                # Only enable pdfplumber fallback when page likely has a table.
                # If pdfplumber isn't installed, the fallback is a no-op.
                use_pdfplumber_fallback=bool(has_table_hint),
            )
            table_time = time.time() - step_start
            if table_time > 2.0:
                print(f"  [Page {page_index+1}] Step 4 (table extraction): {table_time:.2f}s (SLOW!), found {len(tables_found)} tables", flush=True)
            else:
                print(f"  [Page {page_index+1}] Step 4 (table extraction): {table_time:.2f}s, found {len(tables_found)} tables", flush=True)
        except Exception as e:
            print(f"  [Page {page_index+1}] Step 4 (table extraction) FAILED: {e}", flush=True)
            import traceback
            traceback.print_exc()
            tables_found = []
        
        # 5. Extract Text Blocks
        step_start = time.time()
        blocks = self._extract_text_blocks(
            page, 
            page_index=page_index, 
            body_size=body_size,
            tables=tables_found,
            visual_rects=visual_rects,
            assets_dir=assets_dir,
            is_references_page=is_references_page
        )
        print(f"  [Page {page_index+1}] Step 5 (text blocks): {time.time()-step_start:.2f}s, found {len(blocks)} blocks", flush=True)

        # 5.5 Merge split math fragments BEFORE any LLM work / rendering.
        # PDF extraction frequently splits a single equation into many tiny blocks ("N X", ", (5)", "r ∈ R").
        # If we try to repair each fragment in isolation, LaTeX quality collapses.
        step_start = time.time()
        try:
            blocks = self._merge_adjacent_math_fragments(blocks, page_wh=(page.rect.width, page.rect.height))
        except Exception:
            # Never fail conversion due to a heuristic merge.
            pass
        print(f"  [Page {page_index+1}] Step 5.5 (merge math frags): {time.time()-step_start:.2f}s, now {len(blocks)} blocks", flush=True)
        try:
            if bool(int(os.environ.get("KB_PDF_DEBUG_MATH", "0") or "0")):
                math_n0 = sum(1 for b in blocks if bool(getattr(b, "is_math", False)))
                code_n0 = sum(1 for b in blocks if bool(getattr(b, "is_code", False)))
                table_n0 = sum(1 for b in blocks if bool(getattr(b, "is_table", False)))
                print(
                    f"  [Page {page_index+1}] Debug: blocks math={math_n0} code={code_n0} table={table_n0}",
                    flush=True,
                )
        except Exception:
            pass
        
        # 6. LLM Classification / Repair
        step_start = time.time()
        speed_cfg = getattr(self, "_active_speed_config", None) or {}
        if self.cfg.llm and speed_cfg.get("use_llm_for_all", True):
            # Enhance blocks with LLM
            blocks = self._enhance_blocks_with_llm(blocks, page_index, page)
            print(f"  [Page {page_index+1}] Step 6 (LLM enhance): {time.time()-step_start:.2f}s", flush=True)
        else:
            print(f"  [Page {page_index+1}] Step 6 (LLM enhance): skipped (no LLM or disabled)", flush=True)
        try:
            if bool(int(os.environ.get("KB_PDF_DEBUG_MATH", "0") or "0")):
                math_n1 = sum(1 for b in blocks if bool(getattr(b, "is_math", False)))
                print(f"  [Page {page_index+1}] Debug: after enhance math={math_n1}", flush=True)
        except Exception:
            pass

        # Re-run math fragment merge AFTER LLM classify.
        # The classifier often flips is_math flags on blocks that heuristics missed; merging again
        # prevents one display equation from being rendered as many tiny $$ blocks (bad LaTeX).
        try:
            if self.cfg.llm and speed_cfg.get("use_llm_for_all", True):
                step_start2 = time.time()
                blocks = self._merge_adjacent_math_fragments(blocks, page_wh=(page.rect.width, page.rect.height))
                if bool(int(os.environ.get("KB_PDF_DEBUG_MATH", "0") or "0")):
                    math_n2 = sum(1 for b in blocks if bool(getattr(b, "is_math", False)))
                    print(
                        f"  [Page {page_index+1}] Debug: post-enhance merge math={math_n2} blocks={len(blocks)} ({time.time()-step_start2:.2f}s)",
                        flush=True,
                    )
        except Exception:
            pass
            
        # 7. Render
        step_start = time.time()
        result = self._render_blocks_to_markdown(blocks, page_index, page=page, assets_dir=assets_dir)
        print(f"  [Page {page_index+1}] Step 7 (render): {time.time()-step_start:.2f}s", flush=True)
        print(f"  [Page {page_index+1}] TOTAL: {time.time()-page_start:.2f}s", flush=True)
        return result

    def _merge_adjacent_math_fragments(self, blocks: List[TextBlock], *, page_wh: tuple[float, float]) -> List[TextBlock]:
        """
        Merge adjacent math fragments split by PDF extraction.

        Typical failure mode (your screenshot):
        - one display equation becomes multiple blocks: "N X", ", (5)", "r ∈ R", ...
        If we repair each in isolation, LaTeX becomes nonsense.
        """
        if not blocks:
            return blocks

        page_w = float(page_wh[0] or 1.0)

        def _t(b: TextBlock) -> str:
            return (b.text or "").strip()

        def _is_eq_no_text(s: str) -> bool:
            # "(5)" / "(EQNO 5)" / ", (5)"
            ss = s.strip()
            if re.fullmatch(r"[,;:]?\s*\(\s*(?:EQNO\s+)?\d{1,4}\s*\)\s*", ss, flags=re.IGNORECASE):
                return True
            if re.fullmatch(r"\(\s*\d{1,4}\s*\)", ss):
                return True
            return False

        def _is_tiny_connector(s: str) -> bool:
            ss = s.strip()
            return bool(re.fullmatch(r"[,.;:]", ss))

        def _looks_mathish_text(s: str) -> bool:
            ss = s.strip()
            if not ss:
                return False
            # very short math-y tokens
            if any(ch in ss for ch in ["=", "∈", "≤", "≥", "≈", "→", "←", "×", "·", "Σ", "∑", "∫"]):
                return True
            # LaTeX-ish / OCR math
            if "\\" in ss or "^" in ss or "_" in ss:
                return True
            # bracket-heavy + sparse words
            if len(ss) <= 28 and re.search(r"[(){}\[\]]", ss) and not re.search(r"[A-Za-z]{3,}", ss):
                return True
            # "N X" / "r ∈ R"
            if re.fullmatch(r"[A-Za-z]\s+[A-Za-z]", ss):
                return True
            if "∈" in ss and len(ss) <= 24:
                return True
            return False

        def _looks_proseish_text(s: str) -> bool:
            """
            Detect paragraph-like text that may contain a couple math symbols,
            but should NOT be merged into a formula block.
            """
            ss = s.strip()
            if len(ss) < 80:
                return False
            try:
                word_n = len(re.findall(r"\b[A-Za-z]{2,}\b", ss))
                math_sym_n = len(re.findall(r"[=+\-*/^_{}\\\\\\[\\]]|[∈≤≥≈×·Σ∑∫]", ss))
                has_sentence = (". " in ss) or ("? " in ss) or ("! " in ss)
                if word_n >= 14 and (math_sym_n <= 10 or has_sentence):
                    return True
            except Exception:
                return False
            return False

        def _y_overlap_ratio(r1: "fitz.Rect", r2: "fitz.Rect") -> float:
            y0 = max(float(r1.y0), float(r2.y0))
            y1 = min(float(r1.y1), float(r2.y1))
            ov = max(0.0, y1 - y0)
            denom = max(1e-6, min(float(r1.height), float(r2.height)))
            return ov / denom

        def _x_overlap_ratio(r1: "fitz.Rect", r2: "fitz.Rect") -> float:
            x0 = max(float(r1.x0), float(r2.x0))
            x1 = min(float(r1.x1), float(r2.x1))
            ov = max(0.0, x1 - x0)
            denom = max(1e-6, min(float(r1.width), float(r2.width)))
            return ov / denom

        merged: list[TextBlock] = []
        i = 0
        while i < len(blocks):
            b = blocks[i]
            bt = _t(b)
            if not bt:
                merged.append(b)
                i += 1
                continue

            # Start a merge group only when current block is math-ish (or already classified math).
            if not (b.is_math or _looks_mathish_text(bt)):
                merged.append(b)
                i += 1
                continue

            # Do not merge tables/code/captions/headings into math.
            if b.is_table or b.is_code or b.is_caption or b.heading_level:
                merged.append(b)
                i += 1
                continue

            group_texts = [bt]
            group_rect = fitz.Rect(b.bbox)
            last_rect = group_rect

            j = i + 1
            while j < len(blocks):
                nb = blocks[j]
                nt = _t(nb)
                if not nt:
                    j += 1
                    continue

                # Stop at structural blocks.
                if nb.is_table or nb.is_code or nb.is_caption or nb.heading_level:
                    break

                # Avoid swallowing normal paragraphs into an equation merge.
                # (The paragraph often contains symbols like "r∈R" which are math-ish, but it's prose.)
                if (not nb.is_math) and _looks_proseish_text(nt):
                    break

                nr = fitz.Rect(nb.bbox)

                # Same-line merge: strong y-overlap, small x-gap.
                y_ov = _y_overlap_ratio(last_rect, nr)
                x_gap = float(nr.x0) - float(last_rect.x1)
                same_line = (y_ov >= 0.65) and (x_gap <= max(10.0, page_w * 0.03))

                # Stacked-line merge (multi-line equation): x overlap, small y-gap.
                y_gap = float(nr.y0) - float(last_rect.y1)
                x_ov = _x_overlap_ratio(group_rect, nr)
                stacked = (y_gap >= -2.0) and (y_gap <= max(14.0, (float(last_rect.height) + float(nr.height)) * 0.35)) and (x_ov >= 0.25)

                # Allow merging tiny connector / equation-number fragments near math.
                is_ok_token = nb.is_math or _looks_mathish_text(nt) or _is_eq_no_text(nt) or _is_tiny_connector(nt)

                if is_ok_token and (same_line or stacked):
                    # Preserve line breaks for stacked parts; spaces for same-line.
                    if stacked and not same_line:
                        group_texts.append("\n" + nt)
                    else:
                        group_texts.append(" " + nt)
                    group_rect = _union_rect(group_rect, nr)
                    last_rect = nr
                    j += 1
                    continue

                break

            if j == i + 1:
                merged.append(b)
                i += 1
                continue

            merged_text = "".join(group_texts).strip()
            bd = b.model_dump()
            bd["bbox"] = tuple(group_rect)
            bd["text"] = merged_text
            bd["is_math"] = True
            bd["is_code"] = False
            bd["is_caption"] = False
            bd["heading_level"] = None
            merged.append(TextBlock(**bd))
            i = j

        return merged

    def _extract_text_blocks(
        self, 
        page, 
        page_index: int, 
        body_size: float, 
        tables: List[Tuple[fitz.Rect, str]], 
        visual_rects: List[fitz.Rect],
        assets_dir: Path,
        is_references_page: bool = False
    ) -> List[TextBlock]:
        import time
        extract_start = time.time()
        
        text_blocks = []
        # Get raw blocks
        step_start = time.time()
        page_dict = page.get_text("dict")
        raw_blocks = page_dict.get("blocks", [])
        print(f"    [Page {page_index+1}] get_text('dict'): {time.time()-step_start:.2f}s, {len(raw_blocks)} raw blocks", flush=True)
        
        W = float(page.rect.width)
        H = float(page.rect.height)
        
        # Mask out tables and figures
        ignore_rects = [r for r, _ in tables] + visual_rects
        
        for b in raw_blocks:
            bbox = fitz.Rect(b["bbox"])
            
            # Check overlap with tables/figures
            is_masked = False
            for ig in ignore_rects:
                if _rect_intersection_area(bbox, ig) > _rect_area(bbox) * 0.5:
                    is_masked = True
                    break
            if is_masked:
                continue
                
            # Process lines: keep MuPDF line structure. A single raw block often contains
            # both display equations and a following prose explanation ("where ...").
            # If we flatten everything with spaces, the equation and prose become inseparable.
            max_size = 0.0
            is_bold = False

            if "lines" not in b:
                continue

            line_items: list[tuple[fitz.Rect, str]] = []
            for l in b["lines"]:
                spans = l.get("spans") or []
                parts: list[str] = []
                for s in spans:
                    t = (s.get("text") or "")
                    if not t.strip():
                        continue
                    # Track font size/bold
                    try:
                        size = float(s.get("size") or 0.0)
                    except Exception:
                        size = 0.0
                    try:
                        font = str(s.get("font") or "").lower()
                    except Exception:
                        font = ""
                    if size > max_size:
                        max_size = size
                    if ("bold" in font) or ("dubai-bold" in font):
                        is_bold = True
                    parts.append(t)
                line_text = " ".join(parts).strip()
                if not line_text:
                    continue
                try:
                    lb = fitz.Rect(l.get("bbox"))
                except Exception:
                    lb = bbox
                line_items.append((lb, line_text))

            if not line_items:
                continue

            # Split this raw block into sub-blocks by line-type (math-like vs prose-like).
            # This prevents "equation + where paragraph" from becoming one giant math block.
            from .heuristics import _looks_like_equation_text, _is_caption_like_text

            def _is_math_line(txt: str) -> bool:
                tt = (txt or "").strip()
                if not tt:
                    return False
                low = tt.lower()
                # Captions should never be merged into equations.
                try:
                    if _is_caption_like_text(tt):
                        return False
                except Exception:
                    pass
                # Prose explanation lines commonly start with "where/with/and" and include many words.
                try:
                    word_n0 = len(re.findall(r"\b[A-Za-z]{2,}\b", tt))
                    if (low.startswith("where ") or low.startswith("with ") or low.startswith("and ")) and word_n0 >= 6:
                        return False
                    # Long sentence-like lines with few hard equation anchors are usually prose.
                    if word_n0 >= 14 and ("=" not in tt) and ("\\sum" not in tt) and ("\\int" not in tt):
                        return False
                except Exception:
                    pass
                if _looks_like_math_block([tt]):
                    return True
                if _looks_like_equation_text(tt):
                    return True
                # Extra: short symbol-heavy lines are usually equation lines
                sym_n = len(re.findall(r"[=+\-*/^_{}\\\\\\[\\]]|[∈≤≥≈×·Σ∑∫]", tt))
                word_n = len(re.findall(r"\b[A-Za-z]{2,}\b", tt))
                if len(tt) <= 80 and sym_n >= 3 and word_n <= 10:
                    return True
                return False

            groups: list[tuple[bool, list[tuple[fitz.Rect, str]]]] = []
            cur_is_math = _is_math_line(line_items[0][1])
            cur: list[tuple[fitz.Rect, str]] = []
            for lr, lt in line_items:
                m = _is_math_line(lt)
                if cur and (m != cur_is_math):
                    groups.append((cur_is_math, cur))
                    cur = []
                    cur_is_math = m
                cur.append((lr, lt))
            if cur:
                groups.append((cur_is_math, cur))

            for group_is_math_hint, group_lines in groups:
                try:
                    rects = [rr for rr, _ in group_lines]
                    group_bbox = _union_rect(rects) or rects[0]
                except Exception:
                    group_bbox = group_lines[0][0]

                # For heuristics like heading/caption/noise detection we want a flat string.
                full_text_space = " ".join(x for _, x in group_lines).strip()
                if not full_text_space:
                    continue
            
                # Check if this looks like author information (should not be heading)
                # Pattern: "Name 1 , Name 2 , Name 3 , and Name 4 ,"
                if re.search(r'^\s*[A-Z][a-z]+\s+\d+\s*,.*\d+\s*,.*and.*\d+\s*,?\s*$', full_text_space):
                    tb = TextBlock(
                        bbox=tuple(group_bbox),
                        text=full_text_space,
                        max_font_size=max_size,
                        is_bold=is_bold,
                        heading_level=None,
                        is_math=False,
                        is_code=False,
                        is_caption=False
                    )
                    text_blocks.append(tb)
                    continue
            
                # Detect math formulas, code blocks, and captions early
                is_math = False
                is_code = False
                is_caption = False

                # Check if this is a caption (Figure/Table caption)
                from .heuristics import _is_caption_like_text
                is_caption = _is_caption_like_text(full_text_space)

                # Determine math/codelike. Prefer line-level hint to avoid misclassifying prose lines.
                if not is_caption:
                    group_line_texts = [x for _, x in group_lines]
                    is_math = bool(group_is_math_hint) or _looks_like_math_block(group_line_texts)
                    if not is_math:
                        is_code = _looks_like_code_block(group_line_texts)
                # Preserve line breaks for math blocks for better repair/rendering.
                full_text = ("\n".join(x for _, x in group_lines) if is_math else full_text_space).strip()
                if not full_text:
                    continue
            
                # Filter noise - check if in header/footer region
                is_header_footer = False
                header_threshold = H * 0.12  # Top 12% of page
                footer_threshold = H * 0.88  # Bottom 12% of page
                if group_bbox.y1 < header_threshold or group_bbox.y0 > footer_threshold:
                    is_header_footer = True
                    # Still allow if it's a major heading (e.g., section title at top of page)
                    if not (max_size > body_size + 1.5 and is_bold and len(full_text_space) < 100):
                        if full_text_space in self.noise_texts or _is_noise_line(full_text_space):
                            continue
            
                if full_text_space in self.noise_texts:
                    continue
                if _is_frontmatter_noise_line(full_text_space):
                    continue
                if _is_noise_line(full_text_space) and not is_header_footer:
                    continue
                
                # Detect heading (heuristic-based, LLM will refine later if available)
                # Skip heading detection if this is already classified as math, code, or caption
                heading_level = None
                if not is_math and not is_code and not is_caption:
                    # Additional check: very short text with numbers is likely formula, not heading
                    text_stripped = full_text_space.strip()
                    if len(text_stripped) <= 25:
                        # Check for formula-like patterns more strictly
                        if re.search(r'^\s*[A-Z]?\s*\d+\s*[a-z]', text_stripped) or \
                           re.search(r'^\s*\d+\s*[a-z]+\s*[+\-]', text_stripped) or \
                           re.search(r'^\s*[a-z]\s*\d+', text_stripped) or \
                           (re.search(r'\d+.*[a-z]|[a-z].*\d+', text_stripped) and not re.search(r'[A-Z]{2,}', text_stripped) and '=' not in text_stripped):
                            is_math = True  # Reclassify as math
                            is_caption = False
                            full_text = ("\n".join(x for _, x in group_lines)).strip()

                    if not is_math:
                        heading_tag = detect_header_tag(
                            page_index=page_index,
                            text=full_text_space,
                            max_size=max_size,
                            is_bold=is_bold,
                            body_size=body_size,
                            page_width=W,
                            bbox=tuple(group_bbox),
                        )
                        if heading_tag:
                            heading_level = heading_tag
                
                # Create Block with detected types
                tb = TextBlock(
                    bbox=tuple(group_bbox),
                    text=full_text,
                    max_font_size=max_size,
                    is_bold=is_bold,
                    heading_level=heading_level,
                    is_math=is_math,
                    is_code=is_code,
                    is_caption=is_caption
                )
                text_blocks.append(tb)

        # Insert Tables as Blocks
        for rect, md in tables:
            tb = TextBlock(
                bbox=tuple(rect),
                text="[TABLE]",
                max_font_size=body_size,
                is_table=True,
                table_markdown=md
            )
            text_blocks.append(tb)
            
        # Insert Images (Visual Rects) as Blocks
        step_start = time.time()
        # Filter out header/footer regions and crop properly
        header_threshold = H * 0.12
        footer_threshold = H * 0.88
        side_margin = W * 0.05  # 5% margin on sides
        
        # Detect column layout for proper image handling
        col_split = _detect_column_split_x(text_blocks, page_width=W) if text_blocks else None
        spanning_threshold = W * 0.55  # Full-width images span both columns
        
        img_count = 0
        for rect_idx, rect in enumerate(visual_rects):
            img_step_start = time.time()
            # Check if this is a full-width image (spans both columns or most of page)
            is_full_width = _rect_area(rect) >= (W * H * 0.40) or _bbox_width(tuple(rect)) >= spanning_threshold
            
            # Skip if in header/footer region (likely page numbers, headers)
            if rect.y1 < header_threshold or rect.y0 > footer_threshold:
                # Allow if it's a large figure (likely a real figure, not header/footer)
                if not is_full_width and _rect_area(rect) < (W * H * 0.15):  # Less than 15% of page area
                    continue
            
            # Skip small edge artifacts (unless it's a full-width image)
            if not is_full_width:
                if rect.x0 < side_margin and rect.width < W * 0.1:
                    continue
                if rect.x1 > W - side_margin and rect.width < W * 0.1:
                    continue
            
            # For full-width images in double-column layout, ensure we capture the full width
            if is_full_width and col_split:
                # Expand rect to full width if it's close to spanning
                if _bbox_width(tuple(rect)) < W * 0.85:
                    # It might be a full-width image that was detected as two separate rects
                    # Keep original rect but ensure proper cropping
                    pass
            
            # Crop rect slightly to avoid edge artifacts (but preserve full-width images)
            crop_margin = 2.0 if not is_full_width else 1.0  # Less cropping for full-width
            cropped_rect = fitz.Rect(
                max(0, rect.x0 + crop_margin),
                max(0, rect.y0 + crop_margin),
                min(W, rect.x1 - crop_margin),
                min(H, rect.y1 - crop_margin)
            )
            
            if cropped_rect.width <= 0 or cropped_rect.height <= 0:
                continue
                
            # Save image with proper DPI from config
            # Use a stable per-page index to avoid filename collisions/overwrites on Windows.
            img_name = f"page_{page_index+1}_fig_{rect_idx+1}.png"
            img_path = assets_dir / img_name
            
            # Use configured DPI
            dpi = self.dpi
            try:
                pixmap_start = time.time()
                pix = page.get_pixmap(clip=cropped_rect, dpi=dpi)
                pixmap_time = time.time() - pixmap_start
                save_start = time.time()
                pix.save(img_path)
                save_time = time.time() - save_start
                if pixmap_time > 1.0 or save_time > 1.0:
                    print(f"      [Page {page_index+1}] Image {rect_idx+1}/{len(visual_rects)}: get_pixmap={pixmap_time:.2f}s, save={save_time:.2f}s, size={cropped_rect.width:.0f}x{cropped_rect.height:.0f}", flush=True)
            except Exception as e:
                print(f"      [Page {page_index+1}] Image {rect_idx+1} get_pixmap failed: {e}", flush=True)
                # Fallback to original rect if crop fails
                try:
                    pix = page.get_pixmap(clip=rect, dpi=dpi)
                    pix.save(img_path)
                except Exception:
                    continue

            # Guard: don't emit broken markdown links.
            try:
                if (not img_path.exists()) or (img_path.stat().st_size < 256):
                    continue
            except Exception:
                continue
            
            tb = TextBlock(
                bbox=tuple(rect),
                text=f"![Figure](./assets/{img_name})",
                max_font_size=body_size
            )
            text_blocks.append(tb)
            img_count += 1
            if time.time() - img_step_start > 0.5:
                print(f"      [Page {page_index+1}] Image {rect_idx+1} processing took {time.time()-img_step_start:.2f}s", flush=True)
        
        print(f"    [Page {page_index+1}] Image processing: {time.time()-step_start:.2f}s, processed {img_count}/{len(visual_rects)} images", flush=True)

        # Sort reading order
        sort_start = time.time()
        sorted_blocks = sort_blocks_reading_order(text_blocks, page_width=W)
        print(f"    [Page {page_index+1}] Sort: {time.time()-sort_start:.2f}s", flush=True)
        print(f"    [Page {page_index+1}] _extract_text_blocks TOTAL: {time.time()-extract_start:.2f}s", flush=True)
        return sorted_blocks

    def _enhance_blocks_with_llm(self, blocks: List[TextBlock], page_index: int, page) -> List[TextBlock]:
        # Call classifier
        classified = self.llm_worker.call_llm_classify_blocks(
            blocks, 
            page_number=page_index, 
            page_wh=(page.rect.width, page.rect.height)
        )
        if not classified:
            return blocks
        
        # Index classifications by block index to avoid O(n^2) scans.
        cls_by_i: dict[int, dict] = {}
        try:
            for it in classified:
                ii = it.get("i")
                if isinstance(ii, int):
                    cls_by_i[ii] = it
        except Exception:
            cls_by_i = {}

        # Apply classifications - create new blocks since TextBlock is immutable
        enhanced_blocks = []
        for i, b in enumerate(blocks):
            item = cls_by_i.get(i)
            
            if item:
                kind = item.get("kind")
                # Create new block with updated properties
                block_dict = b.model_dump()
                if kind == "heading":
                    lvl = item.get("heading_level")
                    if lvl:
                        block_dict["heading_level"] = f"[H{lvl}]"
                    # Clear other flags when classified as heading
                    block_dict["is_math"] = False
                    block_dict["is_code"] = False
                    block_dict["is_caption"] = False
                elif kind == "table":
                    block_dict["is_table"] = True
                    block_dict["is_math"] = False
                    block_dict["is_code"] = False
                elif kind == "code":
                    block_dict["is_code"] = True
                    block_dict["is_math"] = False
                    block_dict["is_caption"] = False
                elif kind == "math":
                    block_dict["is_math"] = True
                    block_dict["is_code"] = False
                    block_dict["heading_level"] = None  # Clear heading if it's math
                elif kind == "caption":
                    block_dict["is_caption"] = True
                    block_dict["is_math"] = False
                    block_dict["is_code"] = False
                    block_dict["heading_level"] = None  # Clear heading if it's caption
                enhanced_blocks.append(TextBlock(**block_dict))
            else:
                enhanced_blocks.append(b)
                    
        return enhanced_blocks

    def _convert_formula_to_latex(self, text: str) -> str:
        """Convert formula text to LaTeX format."""
        if not text:
            return ""
        
        # Fast path: if text is very long, skip expensive processing
        if len(text) > 1000:
            # For very long formulas, just do basic cleanup and return
            t = text.strip()
            t = re.sub(r'\(\s*(?:EQNO\s+)?\d+\s*\)\s*$', '', t).strip()
            return t
        
        # Normalize text but preserve line structure for multi-line formulas
        # Skip expensive _normalize_text for math formulas - just basic cleanup
        lines = text.splitlines()
        normalized_lines = []
        for line in lines:
            t = line.strip()
            if t:
                normalized_lines.append(t)
        
        if not normalized_lines:
            return ""
        
        t = " ".join(normalized_lines)
        
        # Remove equation numbers at the end like "(8)" or "(EQNO 8)"
        t = re.sub(r'\(\s*(?:EQNO\s+)?\d+\s*\)\s*$', '', t).strip()
        
        # Replace Greek letters first (before other processing)
        # Use simple replace (fast enough for small dicts)
        for greek, latex in GREEK_TO_LATEX.items():
            if greek in t:  # Only check if present (faster)
                t = t.replace(greek, latex)
        
        # Replace math symbols
        for symbol, latex in MATH_SYMBOL_TO_LATEX.items():
            if symbol in t:  # Only check if present (faster)
                t = t.replace(symbol, latex)

        # Fix common OCR spacing around parentheses: "C ( r )" -> "C(r)"
        # Do this early so subsequent sub/superscript rules see a cleaner token stream.
        try:
            t = re.sub(r"\s*\(\s*", "(", t)
            t = re.sub(r"\s*\)\s*", ")", t)
        except Exception:
            pass

        # Fix superscripts in a safe, targeted way:
        # - "C(r) 2" -> "C(r)^2"
        # Only trigger when a digit follows a closing bracket (very likely exponent, not index).
        try:
            t = re.sub(r"([\)\]\}])\s+(\d{1,2})\b", r"\1^{\2}", t)
        except Exception:
            pass

        # Fix "hat" when OCR emits a caret with whitespace: "^ C" or "ˆ C" (normalized to "^ C")
        # This avoids confusing "^" exponent usage because we REQUIRE whitespace after caret.
        try:
            t = re.sub(r"\^\s+([A-Za-z])\b", r"\\hat{\1}", t)
        except Exception:
            pass
        
        # Normalize whitespace but be careful with subscripts
        # First, protect potential subscripts/superscripts
        # Pattern: letter followed by space and then digit or lowercase letter
        # But only if not already in LaTeX format
        
        # Fix common math functions - compile regex once for speed
        if not hasattr(self, '_math_func_regexes'):
            self._math_func_regexes = {
                'log': re.compile(r'\blog\b'),
                'exp': re.compile(r'\bexp\b'),
                'sin': re.compile(r'\bsin\b'),
                'cos': re.compile(r'\bcos\b'),
                'tan': re.compile(r'\btan\b'),
                'max': re.compile(r'\bmax\b'),
                'min': re.compile(r'\bmin\b'),
                'ln': re.compile(r'\bln\b'),
                'sqrt': re.compile(r'\bsqrt\b'),
            }
            self._math_func_replacements = {
                'log': r'\\log',
                'exp': r'\\exp',
                'sin': r'\\sin',
                'cos': r'\\cos',
                'tan': r'\\tan',
                'max': r'\\max',
                'min': r'\\min',
                'ln': r'\\ln',
                'sqrt': r'\\sqrt',
            }
        
        for func, regex in self._math_func_regexes.items():
            t = regex.sub(self._math_func_replacements[func], t)
        
        # Fix subscripts more carefully
        # Pattern: variable name (single letter or word) followed by space and digit/single letter
        # Only if it's not already in LaTeX format
        if '_{' not in t and '_' not in t:
            # Simple case: "x 1" -> "x_1", but be careful
            # Only do this for single letters followed by single digits/letters
            # Use compiled regex for speed
            if not hasattr(self, '_subscript_regex1'):
                self._subscript_regex1 = re.compile(r'\b([a-zA-Z])\s+(\d+)\b')
                self._subscript_regex2 = re.compile(r'\b([a-zA-Z])\s+([a-z])\b(?!\w)')
            t = self._subscript_regex1.sub(r'\1_{\2}', t)
            # For single lowercase letters as subscripts: "x i" -> "x_i" (but not "x in")
            t = self._subscript_regex2.sub(r'\1_{\2}', t)
        
        # Normalize remaining whitespace - use compiled regex
        if not hasattr(self, '_whitespace_regex'):
            self._whitespace_regex = re.compile(r'\s+')
            self._operator_regexes = {
                '=': re.compile(r'\s*=\s*'),
                '+': re.compile(r'\s*\+\s*'),
                '-': re.compile(r'\s*-\s*'),
                '*': re.compile(r'\s*\*\s*'),
                '/': re.compile(r'\s*/\s*'),
            }
        t = self._whitespace_regex.sub(' ', t).strip()
        
        # Fix common operators that might have been split
        t = self._operator_regexes['='].sub(' = ', t)
        t = self._operator_regexes['+'].sub(' + ', t)
        t = self._operator_regexes['-'].sub(' - ', t)
        # Replace * with \cdot, but escape the backslash properly
        t = self._operator_regexes['*'].sub(r' \\cdot ', t)
        t = self._operator_regexes['/'].sub(' / ', t)
        
        # Clean up extra spaces around operators
        t = self._whitespace_regex.sub(' ', t)
        
        return t

    def _render_blocks_to_markdown(self, blocks: List[TextBlock], page_index: int, *, page=None, assets_dir: Path | None = None) -> str:
        import time
        render_start = time.time()
        llm_call_count = 0
        llm_total_time = 0.0

        eq_img_idx = 0

        def _ctx_from_neighbor_blocks(idx: int, *, direction: int) -> str:
            """
            Build a small, useful context snippet for LLM math repair from nearby raw blocks.
            direction: -1 for before, +1 for after
            """
            try:
                acc: list[str] = []
                j = idx + direction
                # Take up to 2 blocks of context
                while 0 <= j < len(blocks) and len(acc) < 2:
                    nb = blocks[j]
                    t = (nb.text or "").strip()
                    if not t:
                        j += direction
                        continue
                    # Skip structural noise
                    if nb.is_table or nb.is_code:
                        j += direction
                        continue
                    # Avoid dumping huge paragraphs
                    if len(t) > 240:
                        t = t[:240] + "..."
                    acc.append(t)
                    j += direction
                return "\n".join(acc) if direction > 0 else "\n".join(reversed(acc))
            except Exception:
                return ""

        def _extract_math_raw_from_page(page, bbox: tuple[float, float, float, float]) -> str:
            """
            Extract math-like text from `page` inside `bbox` using span-level geometry (rawdict),
            attempting to preserve superscript/subscript structure.

            This is critical for full_llm quality: `get_text('dict')` often loses layout cues,
            producing garbled math that even an LLM can't reliably reconstruct.
            """
            try:
                clip = fitz.Rect(bbox)
                if clip.width <= 2 or clip.height <= 2:
                    return ""
            except Exception:
                return ""

            try:
                d = page.get_text("rawdict")
            except Exception:
                try:
                    d = page.get_text("dict")
                except Exception:
                    return ""

            spans: list[tuple[float, float, float, float, float, str]] = []
            try:
                for b0 in (d.get("blocks") or []):
                    for ln in (b0.get("lines") or []):
                        for sp in (ln.get("spans") or []):
                            txt = str(sp.get("text") or "")
                            if not txt.strip():
                                # Some PDFs store glyphs only in `chars` with empty `text`.
                                chars = sp.get("chars") or []
                                if chars:
                                    try:
                                        txt = "".join(str(ch.get("c") or "") for ch in chars)
                                    except Exception:
                                        txt = ""
                            if not txt.strip():
                                continue
                            sb = sp.get("bbox")
                            if not sb:
                                continue
                            try:
                                r = fitz.Rect(tuple(float(x) for x in sb))
                            except Exception:
                                continue
                            if not r.intersects(clip):
                                continue
                            size = float(sp.get("size") or 0.0)
                            spans.append((float(r.x0), float(r.y0), float(r.x1), float(r.y1), size, txt))
            except Exception:
                spans = []

            if not spans:
                return ""

            # Sort top-to-bottom, left-to-right
            spans.sort(key=lambda x: (x[1], x[0]))

            # Group spans into lines by y proximity
            lines: list[list[tuple[float, float, float, float, float, str]]] = []
            cur: list[tuple[float, float, float, float, float, str]] = []
            cur_y = None
            for sp in spans:
                y0, y1, size = sp[1], sp[3], sp[4]
                cy = (y0 + y1) / 2.0
                if cur_y is None:
                    cur_y = cy
                    cur = [sp]
                    continue
                tol = max(2.0, (size or 10.0) * 0.65)
                if abs(cy - cur_y) <= tol:
                    cur.append(sp)
                    # update running center y
                    cur_y = (cur_y * 0.7) + (cy * 0.3)
                else:
                    lines.append(cur)
                    cur = [sp]
                    cur_y = cy
            if cur:
                lines.append(cur)

            def _median(xs: list[float]) -> float:
                if not xs:
                    return 0.0
                xs2 = sorted(xs)
                mid = len(xs2) // 2
                return xs2[mid] if (len(xs2) % 2 == 1) else (xs2[mid - 1] + xs2[mid]) / 2.0

            out_lines: list[str] = []
            for ln in lines:
                ln.sort(key=lambda x: x[0])
                centers = [((x[1] + x[3]) / 2.0) for x in ln]
                sizes = [float(x[4] or 0.0) for x in ln]
                base_c = _median(centers)
                base_s = _median([s for s in sizes if s > 0.0]) or 0.0
                # Estimate line height
                heights = [max(1.0, float(x[3] - x[1])) for x in ln]
                lh = _median(heights) or 10.0

                parts: list[str] = []
                prev_x1 = None
                for x0, y0, x1, y1, size, txt in ln:
                    t = str(txt)
                    # Normalize some common math glyphs early
                    t = t.replace("⊙", r"\odot").replace("∈", r"\in").replace("×", r"\times").replace("·", r"\cdot")
                    t = re.sub(r"\s+", " ", t).strip()
                    if not t:
                        continue
                    c = (y0 + y1) / 2.0
                    is_small = (base_s > 0.0) and (size > 0.0) and (size <= base_s * 0.88)
                    # Sup/sub classification by relative center y
                    super_th = base_c - (lh * 0.22)
                    sub_th = base_c + (lh * 0.22)
                    if is_small and c < super_th:
                        t = "^{" + t + "}"
                    elif is_small and c > sub_th:
                        t = "_{" + t + "}"

                    # Insert a space if there is a large horizontal gap between spans
                    if prev_x1 is not None:
                        gap = float(x0) - float(prev_x1)
                        if gap > max(2.0, (base_s or 10.0) * 0.4):
                            parts.append(" ")
                    parts.append(t)
                    prev_x1 = x1

                line_s = "".join(parts).strip()
                if line_s:
                    out_lines.append(line_s)

            return "\n".join(out_lines).strip()

        def _save_eq_image(bbox: tuple[float, float, float, float]) -> str | None:
            nonlocal eq_img_idx
            if (not self.cfg.eq_image_fallback) or (page is None) or (assets_dir is None):
                return None
            try:
                assets_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                return None
            try:
                r = fitz.Rect(bbox)
            except Exception:
                return None
            # Pad a bit to include equation number / surrounding symbols
            try:
                pad_x = max(2.0, float(r.width) * 0.04)
                pad_y = max(2.0, float(r.height) * 0.10)
                clip = fitz.Rect(
                    max(0.0, float(r.x0) - pad_x),
                    max(0.0, float(r.y0) - pad_y),
                    min(float(page.rect.width), float(r.x1) + pad_x),
                    min(float(page.rect.height), float(r.y1) + pad_y),
                )
                if clip.width <= 2 or clip.height <= 2:
                    return None
            except Exception:
                clip = r
            eq_img_idx += 1
            img_name = f"page_{page_index+1}_eq_{eq_img_idx}.png"
            img_path = assets_dir / img_name
            try:
                pix = page.get_pixmap(clip=clip, dpi=int(getattr(self, "dpi", 200) or 200))
                pix.save(img_path)
                if (not img_path.exists()) or (img_path.stat().st_size < 256):
                    return None
            except Exception:
                return None
            return f"![Equation](./assets/{img_name})"

        def _vision_math_enabled() -> bool:
            """
            Whether to use vision-capable math recovery from equation screenshots.
            Priority:
            - explicit env KB_PDF_LLM_VISION_MATH
            - otherwise auto-enable for VL/vision models (e.g. qwen3-vl-plus)
            """
            try:
                if (not self.cfg.llm) or (not self.llm_worker) or (not getattr(self.llm_worker, "_client", None)):
                    return False
                if page is None:
                    return False
                raw = str(os.environ.get("KB_PDF_LLM_VISION_MATH", "") or "").strip().lower()
                if raw:
                    return raw in {"1", "true", "yes", "y", "on"}
                m = str(getattr(self.cfg.llm, "model", "") or "").strip().lower()
                return ("vl" in m) or ("vision" in m)
            except Exception:
                return False

        def _looks_like_broken_display_math(math_src: str, latex_text: str) -> bool:
            """
            Decide if a non-empty latex_text is still likely wrong enough to justify a vision retry.
            Keep conservative to control cost; only for display-ish math.
            """
            ms = (math_src or "").strip()
            lt = (latex_text or "").strip()
            if not ms or not lt:
                return False
            if ("=" in ms) and ("=" not in lt):
                return True
            if len(ms) >= 55 and len(lt) <= max(24, int(len(ms) * 0.45)):
                return True
            if any(x in ms for x in ["∑", "Σ", "\\sum", "∫", "\\int", "||", "‖"]) and not any(
                y in lt for y in ["\\sum", "\\int", "\\left\\|", "\\|", "\\lVert", "\\rVert"]
            ):
                return True
            if len(lt) >= 120 and len(re.findall(r"\b[A-Za-z]{3,}\b", lt)) >= 10:
                return True
            return False

        def _debug_vision_math() -> bool:
            try:
                return bool(int(os.environ.get("KB_PDF_DEBUG_VISION_MATH", "0") or "0")) or bool(
                    getattr(self.cfg, "keep_debug", False)
                )
            except Exception:
                return False

        def _vision_math_policy() -> str:
            """
            Control when to use vision math recovery.
            - env KB_PDF_VISION_MATH_POLICY: off|fallback|prefer|force
            - additionally, KB_PDF_LLM_VISION_MATH can be set to "prefer"/"force" (backward compatible with boolean).
            """
            try:
                raw2 = str(os.environ.get("KB_PDF_VISION_MATH_POLICY", "") or "").strip().lower()
                if raw2 in {"off", "0", "false", "none"}:
                    return "off"
                if raw2 in {"fallback", "prefer", "force"}:
                    return raw2
                raw = str(os.environ.get("KB_PDF_LLM_VISION_MATH", "") or "").strip().lower()
                if raw in {"prefer", "force"}:
                    return raw
                return "fallback"
            except Exception:
                return "fallback"

        def _expand_math_group_bbox(idx: int, bb: tuple[float, float, float, float]) -> "fitz.Rect":
            """
            Expand a bbox to include neighboring math-ish fragments that belong to the same display equation.
            This dramatically improves VL quality on PDFs where a single equation is split into many tiny blocks.
            """
            try:
                if fitz is None:
                    return fitz.Rect(bb)  # type: ignore[union-attr]
            except Exception:
                return fitz.Rect(bb)  # type: ignore[union-attr]
            try:
                base = fitz.Rect(bb)
            except Exception:
                base = fitz.Rect(0, 0, 0, 0)
            if base.width <= 1 or base.height <= 1:
                return base

            def _is_mathish_block(b2: TextBlock) -> bool:
                try:
                    if bool(getattr(b2, "is_math", False)):
                        return True
                except Exception:
                    pass
                t2 = (getattr(b2, "text", "") or "").strip()
                if not t2:
                    return False
                if len(t2) <= 10 and re.fullmatch(r"[A-Za-z0-9\s\(\)\[\]\{\}=+\-*/^_.,\\]+", t2):
                    return True
                if re.fullmatch(r"[A-Za-z]\s+[A-Za-z]", t2):
                    return True
                if any(ch in t2 for ch in ["∈", "≤", "≥", "≈", "×", "·", "Σ", "∑", "∫", "∞", "→", "←", "⇔", "⇒"]):
                    return True
                if ("\\" in t2) or ("^" in t2) or ("_" in t2):
                    return True
                return False

            def _can_merge(r0: "fitz.Rect", r1: "fitz.Rect") -> bool:
                try:
                    y_ol = min(float(r0.y1), float(r1.y1)) - max(float(r0.y0), float(r1.y0))
                    y_gap = max(0.0, max(float(r1.y0) - float(r0.y1), float(r0.y0) - float(r1.y1)))
                    x_ol = min(float(r0.x1), float(r1.x1)) - max(float(r0.x0), float(r1.x0))
                    x_ol = max(0.0, float(x_ol))
                    min_w = max(1.0, min(float(r0.width), float(r1.width)))
                    x_ratio = x_ol / min_w
                    tiny = (float(r0.width) < 60.0) or (float(r1.width) < 60.0)
                    return (y_ol > 0.0) or (y_gap <= max(3.0, min(float(r0.height), float(r1.height)) * 0.95) and (x_ratio >= 0.12 or tiny))
                except Exception:
                    return False

            r = base
            # Look forward/backward a few blocks.
            max_hops = 6
            for j in range(idx + 1, min(len(blocks), idx + 1 + max_hops)):
                nb = blocks[j]
                if not _is_mathish_block(nb):
                    break
                try:
                    r2 = fitz.Rect(getattr(nb, "bbox", None) or bb)
                except Exception:
                    break
                if _can_merge(r, r2):
                    r |= r2
                    continue
                # Stop if it jumps too far downward.
                try:
                    if float(r2.y0) - float(r.y1) > max(18.0, float(r.height) * 1.8):
                        break
                except Exception:
                    break
            for j in range(idx - 1, max(-1, idx - 1 - max_hops), -1):
                pb = blocks[j]
                if not _is_mathish_block(pb):
                    break
                try:
                    r2 = fitz.Rect(getattr(pb, "bbox", None) or bb)
                except Exception:
                    break
                if _can_merge(r, r2):
                    r |= r2
                    continue
                try:
                    if float(r.y0) - float(r2.y1) > max(18.0, float(r.height) * 1.8):
                        break
                except Exception:
                    break
            return r
        
        out = []
        try:
            if bool(int(os.environ.get("KB_PDF_DEBUG_MATH_BLOCKS", "0") or "0")):
                print(f"[DEBUG] Page {page_index+1} blocks dump (n={len(blocks)}):", flush=True)
                for i0, b0 in enumerate(blocks[:180]):
                    try:
                        t0 = (b0.text or "").strip().replace("\n", "\\n")
                    except Exception:
                        t0 = ""
                    try:
                        bb0 = getattr(b0, "bbox", None)
                    except Exception:
                        bb0 = None
                    try:
                        is_m = bool(getattr(b0, "is_math", False))
                    except Exception:
                        is_m = False
                    if (not is_m) and (not re.search(r"[=^_\\]|[∈≤≥≈×·Σ∑∫∞→←]", t0)):
                        continue
                    print(f"[DEBUG]  idx={i0:03d} is_math={int(is_m)} bbox={bb0} text={ascii(t0[:140])}", flush=True)
        except Exception:
            pass
        block_times = []
        for block_idx, b in enumerate(blocks):
            block_start = time.time()
            # Check if this is an image block (images are stored as text blocks with markdown image syntax)
            if b.text and (b.text.startswith("![") or re.match(r'^!\[.*?\]\(.*?\)', b.text)):
                # This is an image block - output it directly
                out.append(b.text)
                out.append("")  # Add blank line after image
                block_time = time.time() - block_start
                if block_time > 0.1:
                    block_times.append((block_idx, "image", block_time))
                continue
            
            if b.heading_level:
                # [H1] -> #
                lvl = int(b.heading_level.replace("[H", "").replace("]", ""))
                heading_text = b.text.strip()
                # Render-stage LLM calls are expensive and often redundant with Step 6 (LLM classify).
                # Keep render deterministic: apply strict heuristics only.
                if '@' in heading_text or re.search(r'\b(?:university|dept|department|institute|email|zhejiang|westlake)\b', heading_text, re.IGNORECASE):
                    out.append(heading_text)
                    continue

                if re.search(r'[↑↓]', heading_text) or re.search(r'\b(?:PSNR|SSIM|LPIPS)\b', heading_text, re.IGNORECASE):
                    out.append(heading_text)
                    continue

                # Quick heuristic: very short or math-like -> not heading
                if len(heading_text) <= 5 or not re.search(r'[A-Za-z]{3,}', heading_text):
                    out.append(heading_text)
                    continue

                math_symbol_count = len(re.findall(r'[+\-*/=^_{}\[\]()]', heading_text))
                if math_symbol_count > 2:
                    out.append(heading_text)
                    continue

                # Normalize for duplicate check
                normalized_heading = re.sub(r'^[IVX]+\.\s*', '', heading_text, flags=re.IGNORECASE).strip()
                normalized_heading = re.sub(r'^[A-Z]\.\s*', '', normalized_heading).strip()
                normalized_heading = re.sub(r'^\d+\.\s*', '', normalized_heading).strip()

                if normalized_heading in self.seen_headings:
                    out.append(heading_text)
                    continue

                # Heuristic level determination based on numbering pattern
                numbered_match = re.match(r'^(\d+(?:\.\d+)*)\.?\s+', heading_text)
                if numbered_match:
                    num_parts = numbered_match.group(1).split('.')
                    if len(num_parts) == 1:
                        lvl = 1
                    elif len(num_parts) == 2:
                        lvl = 2
                    else:
                        lvl = 3
                else:
                    letter_match = re.match(r'^[A-Z]\.\s+', heading_text)
                    if letter_match:
                        lvl = 2
                    else:
                        lvl = min(3, lvl)

                while self.heading_stack and self.heading_stack[-1][0] >= lvl:
                    self.heading_stack.pop()
                self.heading_stack.append((lvl, heading_text))
                self.seen_headings.add(normalized_heading)
                out.append("#" * lvl + " " + heading_text)
            elif b.is_table:
                if b.table_markdown:
                    out.append(b.table_markdown)
                else:
                    out.append(b.text) # Fallback
            elif b.is_code:
                out.append("```\n" + b.text + "\n```")
            elif b.is_math:
                math_block_start = time.time()
                math_text_len = len(b.text)
                text_stripped = b.text.strip()

                # Some PDFs pack "equation + where explanation + figure caption" into one block/line.
                # If we treat the whole thing as math, it becomes a giant broken $$...$$ block.
                # Split out obvious prose/caption tails and only repair/render the math prefix.
                prose_tail = ""
                math_src = text_stripped
                # For display equations, prefer a rawdict span-based extraction inside the bbox.
                # This preserves superscript/subscript cues and improves LLM repair quality.
                try:
                    if page is not None:
                        bb = getattr(b, "bbox", None)
                        if bb and isinstance(bb, tuple) and len(bb) == 4:
                            raw2 = _extract_math_raw_from_page(page, bb)
                            # Only replace if it looks non-trivial.
                            if raw2 and len(raw2) >= max(12, int(len(math_src) * 0.85)):
                                math_src = raw2
                                text_stripped = math_src.strip()
                except Exception:
                    pass
                try:
                    cap_m = re.search(r"(?i)(\*?Figure\s+\d+\b|\bFig\.?\s*\d+\b|\bTable\s+\d+\b)", math_src)
                    if cap_m:
                        prose_tail = math_src[cap_m.start():].strip()
                        math_src = math_src[:cap_m.start()].strip()
                except Exception:
                    pass
                try:
                    low0 = math_src.lower()
                    wpos = -1
                    for tok in (" where ", "\nwhere ", "\r\nwhere "):
                        p = low0.find(tok)
                        if p >= 0:
                            wstart = p + tok.find("where")
                            if wpos < 0 or wstart < wpos:
                                wpos = wstart
                    if wpos < 0 and low0.startswith("where "):
                        wpos = 0
                    if wpos >= 0:
                        tail = math_src[wpos:].strip()
                        tail_words = len(re.findall(r"\b[A-Za-z]{2,}\b", tail))
                        if tail_words >= 8:
                            prose_tail = (tail + ("\n" + prose_tail if prose_tail else "")).strip()
                            math_src = math_src[:wpos].strip()
                except Exception:
                    pass

                if not math_src and prose_tail:
                    out.append(prose_tail)
                    out.append("")
                    continue
                if math_src:
                    text_stripped = math_src

                def _strip_prose_tail_from_math(s: str) -> str:
                    ss = (s or "").strip()
                    if not ss:
                        return ss
                    try:
                        cap_m2 = re.search(r"(?i)(\*?Figure\s+\d+\b|\bFig\.?\s*\d+\b|\bTable\s+\d+\b)", ss)
                        if cap_m2 and cap_m2.start() > 0:
                            ss = ss[:cap_m2.start()].strip()
                    except Exception:
                        pass
                    try:
                        low1 = ss.lower()
                        wpos2 = -1
                        for tok in (" where ", "\nwhere ", "\r\nwhere "):
                            p = low1.find(tok)
                            if p >= 0:
                                wstart = p + tok.find("where")
                                if wpos2 < 0 or wstart < wpos2:
                                    wpos2 = wstart
                        if wpos2 < 0 and low1.startswith("where "):
                            wpos2 = 0
                        if wpos2 > 0:
                            tail2 = ss[wpos2:].strip()
                            tail_words2 = len(re.findall(r"\b[A-Za-z]{2,}\b", tail2))
                            if tail_words2 >= 8:
                                ss = ss[:wpos2].strip()
                    except Exception:
                        pass
                    return ss

                # Fast heuristic: some headings can be misclassified as math. Handle without LLM.
                looks_like_heading = bool(
                    re.match(r'^(?:\d+(?:\.\d+)*\.?|[A-Z]|[IVX]+)\.?\s+\S+', text_stripped)
                    or re.match(r'^(?:abstract|introduction|related work|method|methods|experiments?|results?|discussion|conclusion|references|appendix)\b', text_stripped, re.IGNORECASE)
                )
                if looks_like_heading and (len(text_stripped) >= 6) and re.search(r'[A-Za-z]{3,}', text_stripped) and (len(re.findall(r'[+\-*/=^_{}\[\]()]', text_stripped)) <= 1):
                    heading_text = text_stripped
                    # Determine level by numbering pattern (keep it simple here)
                    lvl2 = 2
                    numbered_match = re.match(r'^(\d+(?:\.\d+)*)\.?\s+', heading_text)
                    if numbered_match:
                        n_parts = numbered_match.group(1).split('.')
                        lvl2 = 1 if len(n_parts) == 1 else (2 if len(n_parts) == 2 else 3)
                    elif re.match(r'^[A-Z]\.\s+', heading_text):
                        lvl2 = 2
                    normalized_heading = re.sub(r'^[IVX]+\.\s*', '', heading_text, flags=re.IGNORECASE).strip()
                    normalized_heading = re.sub(r'^[A-Z]\.\s*', '', normalized_heading).strip()
                    normalized_heading = re.sub(r'^\d+(?:\.\d+)*\.?\s*', '', normalized_heading).strip()
                    if normalized_heading not in self.seen_headings:
                        while self.heading_stack and self.heading_stack[-1][0] >= lvl2:
                            self.heading_stack.pop()
                        self.heading_stack.append((lvl2, heading_text))
                        self.seen_headings.add(normalized_heading)
                        out.append("#" * lvl2 + " " + heading_text)
                        continue
                
                # Guard: if a "math" block is actually prose (common misclassification),
                # render it as plain text and skip all math repair.
                try:
                    word_n = len(re.findall(r"\b\w+\b", text_stripped))
                    letters_n = len(re.findall(r"[A-Za-z]", text_stripped))
                    math_sym_n = len(re.findall(r"[=+\-*/^_{}\\\[\]]", text_stripped))
                    has_sentence = (". " in text_stripped) or ("? " in text_stripped) or ("! " in text_stripped)
                    # Long, wordy, low-math-symbol content is almost surely not an equation.
                    if (
                        len(text_stripped) >= 120
                        and word_n >= 18
                        and letters_n >= 60
                        and math_sym_n <= 6
                        and has_sentence
                    ):
                        out.append(text_stripped)
                        continue
                except Exception:
                    pass

                # LLM said it's not a heading, or LLM unavailable - proceed with math rendering
                # Quick heuristic: very short or no letters -> definitely math
                if len(text_stripped) <= 5 or not re.search(r'[A-Za-z]', text_stripped):
                    # Too short or no letters - definitely math
                    pass

                # If the extracted "math" looks extremely fragmentary (common in PDF text extraction),
                # avoid LLM guessing; rely on lightweight rule fixes instead.
                try:
                    frag = False
                    if len(text_stripped) <= 14 and re.search(r"[A-Za-z]", text_stripped):
                        frag = True
                    # e.g., "N X", "r ∈ R", "C(r) 2"
                    if re.match(r"^[A-Za-z]\s+[A-Za-z]$", text_stripped):
                        frag = True
                    if "∈" in text_stripped and len(text_stripped) <= 20:
                        frag = True
                except Exception:
                    pass
                
                # It's actually math, proceed with math rendering
                # ALWAYS try LLM repair first for better quality (user said speed is OK)
                latex_text = None
                speed_cfg = getattr(self, "_active_speed_config", None) or {}
                # Check both use_llm_for_all and use_llm_in_render (balanced mode disables render LLM)
                use_llm_in_render = speed_cfg.get("use_llm_in_render", speed_cfg.get("use_llm_for_all", True)) if self.cfg.llm else False
                # Don't ask the LLM to "repair" tiny fragments; it tends to hallucinate.
                prefer_llm_repair = (
                    (len(text_stripped) >= 18)
                    or ("\n" in b.text)
                    or ("=" in text_stripped)
                    or any(x in text_stripped for x in ["\\sum", "\\int", "\\frac", "\\sqrt"])
                )
                
                if prefer_llm_repair and use_llm_in_render and self.cfg.llm and self.llm_worker._client:
                    try:
                        # First attempt: standard repair
                        ctx_before = _ctx_from_neighbor_blocks(block_idx, direction=-1)
                        ctx_after = _ctx_from_neighbor_blocks(block_idx, direction=+1)
                        t0 = time.time()
                        repaired = self.llm_worker.call_llm_repair_math(
                            math_src,
                            page_number=page_index,
                            block_index=block_idx,
                            context_before=ctx_before,
                            context_after=ctx_after,
                        )
                        llm_call_count += 1
                        llm_total_time += (time.time() - t0)
                        if repaired:
                            latex_text = repaired
                        else:
                            # Second attempt: more aggressive repair for inline math
                            # Check if it looks like inline math (short, no line breaks)
                            if len(b.text.strip()) <= 50 and "\n" not in b.text:
                                # Try with a more specific prompt for inline math
                                prompt = f"""Convert this garbled inline math expression to proper LaTeX.
The expression is: {math_src}

Requirements:
- Use proper LaTeX syntax (e.g., \\hat{{C}} not ˆ C, C(r)^2 not C ( r ) 2)
- Remove extra spaces
- Fix subscripts and superscripts properly
- Return ONLY the LaTeX code without $ delimiters

LaTeX:"""
                                try:
                                    t1 = time.time()
                                    resp = self.llm_worker._llm_create(
                                        messages=[
                                            {"role": "system", "content": "You are a LaTeX math expert specializing in inline math expressions."},
                                            {"role": "user", "content": prompt}
                                        ],
                                        temperature=0.0,
                                        max_tokens=200,
                                    )
                                    llm_call_count += 1
                                    llm_total_time += (time.time() - t1)
                                    repaired2 = (resp.choices[0].message.content or "").strip()
                                    # Remove $ if present
                                    if repaired2.startswith("$") and repaired2.endswith("$"):
                                        repaired2 = repaired2[1:-1].strip()
                                    if repaired2:
                                        latex_text = repaired2
                                except Exception:
                                    pass
                    except Exception:
                        pass

                # Vision math recovery (if enabled): screenshot the equation and ask a VL model for exact LaTeX.
                # Important: do NOT require latex_text to be empty; we also retry when it looks suspiciously broken.
                if _vision_math_enabled() and self.cfg.llm and self.llm_worker._client and page is not None:
                    try:
                        bb = getattr(b, "bbox", None)
                        if bb and isinstance(bb, tuple) and len(bb) == 4:
                            # "display-ish" heuristic (controls cost): include common non-'=' display equations too.
                            # Many papers have short display equations without '=' (e.g., sums / norms / constraints).
                            ms = (math_src or "").strip()
                            sym_n = len(re.findall(r"[=+\-*/^_{}\\\[\]]|[∈≤≥≈×·Σ∑∫∞→←⇔⇒]", ms))
                            complex_tok = bool(
                                re.search(
                                    r"[∑Σ∫∞≤≥≈≠→←⇔⇒√]|\\(?:frac|sqrt|sum|int|prod|log|exp|left|right|begin)\b",
                                    ms,
                                )
                            )
                            r = _expand_math_group_bbox(block_idx, bb)
                            is_wide = False
                            try:
                                is_wide = float(r.width) >= float(page.rect.width) * 0.55
                            except Exception:
                                is_wide = False
                            displayish = (
                                ("\n" in ms)
                                or ("=" in ms)
                                or (len(ms) >= 60)
                                or complex_tok
                                or (sym_n >= 10 and len(ms) >= 25)
                                or (is_wide and len(ms) >= 18)
                            )
                            policy = _vision_math_policy()
                            should_try = False
                            if policy == "off":
                                should_try = False
                            elif policy == "force":
                                # Force: try vision whenever we have a bbox and the snippet looks math-ish enough.
                                should_try = bool(displayish or complex_tok or sym_n >= 6 or len(ms) >= 18)
                            elif policy == "prefer":
                                # Prefer: for display-ish math, try vision first even if text repair produced something.
                                should_try = bool(displayish)
                            else:
                                # Fallback: only when text repair failed or looks broken.
                                should_try = bool(displayish and (latex_text is None or _looks_like_broken_display_math(ms, latex_text)))
                            if _debug_vision_math() and not should_try:
                                try:
                                    mname = str(getattr(self.cfg.llm, "model", "") or "")
                                except Exception:
                                    mname = ""
                                why = []
                                if not displayish:
                                    why.append(f"not_displayish(sym_n={sym_n},wide={int(is_wide)},len={len(ms)})")
                                if (latex_text is not None) and (not _looks_like_broken_display_math(ms, latex_text)):
                                    why.append("latex_not_suspicious")
                                why.append(f"policy={policy}")
                                why_s = ",".join(why) or "unknown"
                                print(
                                    f"[VISION_MATH] skip page={page_index+1} block={block_idx+1} model={mname!a} reason={why_s} src_snip={ms[:80]!a}",
                                    flush=True,
                                )

                            if should_try:
                                # pad to capture delimiters / equation number area
                                pad_x = max(2.0, float(r.width) * 0.06)
                                pad_y = max(2.0, float(r.height) * 0.20)
                                # If policy is force and the extracted math looks fragmentary, expand the crop more.
                                # This helps when PDF text extraction misses most of the equation but the glyphs
                                # are still visible on the page.
                                try:
                                    fraggy = (policy == "force") and (
                                        (len(ms) <= 40)
                                        or (float(r.width) < float(page.rect.width) * 0.38)
                                        or (float(r.height) < 18.0)
                                    )
                                except Exception:
                                    fraggy = False
                                if fraggy:
                                    try:
                                        pad_x = max(pad_x, float(page.rect.width) * 0.10)
                                        pad_y = max(pad_y, 28.0)
                                    except Exception:
                                        pad_x = max(pad_x, 24.0)
                                        pad_y = max(pad_y, 28.0)
                                clip = fitz.Rect(
                                    max(0.0, float(r.x0) - pad_x),
                                    max(0.0, float(r.y0) - pad_y),
                                    min(float(page.rect.width), float(r.x1) + pad_x),
                                    min(float(page.rect.height), float(r.y1) + pad_y),
                                )
                                if clip.width > 4 and clip.height > 4:
                                    # Qwen VL models may enforce minimum image dimensions (e.g. >10px).
                                    # Expand clip to satisfy a conservative minimum in pixels at the chosen DPI.
                                    try:
                                        dpi0 = int(getattr(self, "dpi", 200) or 200)
                                    except Exception:
                                        dpi0 = 200
                                    try:
                                        min_px = int(os.environ.get("KB_PDF_VISION_MIN_PX", "12") or "12")
                                    except Exception:
                                        min_px = 12
                                    try:
                                        min_pt = (72.0 * float(min_px)) / max(50.0, float(dpi0))
                                        if clip.width < min_pt or clip.height < min_pt:
                                            ex = max(0.0, (min_pt - float(clip.width)) / 2.0)
                                            ey = max(0.0, (min_pt - float(clip.height)) / 2.0)
                                            clip = fitz.Rect(
                                                max(0.0, float(clip.x0) - ex),
                                                max(0.0, float(clip.y0) - ey),
                                                min(float(page.rect.width), float(clip.x1) + ex),
                                                min(float(page.rect.height), float(clip.y1) + ey),
                                            )
                                    except Exception:
                                        pass
                                    if _debug_vision_math():
                                        try:
                                            mname = str(getattr(self.cfg.llm, "model", "") or "")
                                        except Exception:
                                            mname = ""
                                        print(
                                            f"[VISION_MATH] call page={page_index+1} block={block_idx+1} model={mname!a} policy={policy} clip=({clip.x0:.1f},{clip.y0:.1f},{clip.x1:.1f},{clip.y1:.1f}) src_len={len(ms)} dpi={dpi0}",
                                            flush=True,
                                        )
                                    pix = page.get_pixmap(clip=clip, dpi=dpi0)
                                    try:
                                        if (int(getattr(pix, "width", 0) or 0) < int(min_px)) or (
                                            int(getattr(pix, "height", 0) or 0) < int(min_px)
                                        ):
                                            if _debug_vision_math():
                                                print(
                                                    f"[VISION_MATH] skip_small_image page={page_index+1} block={block_idx+1} pix=({pix.width}x{pix.height}) min_px={min_px}",
                                                    flush=True,
                                                )
                                            pix = None
                                    except Exception:
                                        pass
                                    if pix is not None:
                                        v_start = time.time()
                                        png = pix.tobytes("png")
                                        repaired_v = self.llm_worker.call_llm_repair_math_from_image(
                                            png,
                                            page_number=page_index,
                                            block_index=block_idx,
                                        )
                                        llm_call_count += 1
                                        llm_total_time += (time.time() - v_start)
                                        if repaired_v:
                                            if _debug_vision_math():
                                                print(
                                                    f"[VISION_MATH] ok page={page_index+1} block={block_idx+1} out_len={len(repaired_v)}",
                                                    flush=True,
                                                )
                                            latex_text = repaired_v
                    except Exception as e:
                        if _debug_vision_math():
                            try:
                                mname = str(getattr(self.cfg.llm, "model", "") or "")
                            except Exception:
                                mname = ""
                            print(
                                f"[VISION_MATH] error page={page_index+1} block={block_idx+1} model={mname!a} err={e!a}",
                                flush=True,
                            )
                
                # If LLM didn't help, use heuristic conversion
                if not latex_text:
                    convert_start = time.time()
                    latex_text = self._convert_formula_to_latex(math_src)
                    convert_time = time.time() - convert_start
                    if convert_time > 0.5:
                        print(f"      [Page {page_index+1}] Block {block_idx+1} _convert_formula_to_latex: {convert_time:.2f}s (SLOW!), text_len={len(b.text)}", flush=True)

                # If still looks bad, fall back to equation image to preserve correctness.
                try:
                    if self.cfg.eq_image_fallback:
                        bad = False
                        s = (latex_text or "").strip()
                        # too long text-like output or contains many normal words
                        if len(s) >= 140 and len(re.findall(r"\b[A-Za-z]{3,}\b", s)) >= 10:
                            bad = True
                        # contains obvious prose markers
                        if any(x in s.lower() for x in ["the ", "appears", "likely", "interpretation", "here is"]):
                            bad = True
                        if bad:
                            img_md = _save_eq_image(getattr(b, "bbox", None) or getattr(b, "bbox", (0, 0, 0, 0)))
                            if img_md:
                                out.append(img_md)
                                out.append("")
                                continue
                except Exception:
                    pass
                
                if latex_text:
                    # Clean up latex_text: remove trailing commas, fix nested $, etc.
                    latex_text = latex_text.strip()
                    latex_text = _strip_prose_tail_from_math(latex_text)
                    # Remove trailing commas and spaces
                    latex_text = re.sub(r',\s*$', '', latex_text)
                    # Fix nested $ symbols (should not have $ inside $...$)
                    latex_text = latex_text.replace('$', '')
                    
                    # Check if it's inline or display math
                    # Display math if:
                    # - Contains line breaks
                    # - Is long (> 60 chars)
                    # - Contains = (equations)
                    # - Contains complex structures (sum, integral, etc.)
                    has_break = "\n" in math_src
                    has_equals = "=" in latex_text
                    has_complex = any(op in latex_text for op in ['\\sum', '\\int', '\\prod', '\\frac', '\\sqrt', '\\exp', '\\log'])
                    is_long = len(latex_text) > 60
                    
                    is_inline = not (has_break or has_equals or has_complex or is_long)
                    
                    if is_inline:
                        # Inline math: use LLM to polish for better quality (only if enabled in speed mode)
                        speed_cfg = getattr(self, "_active_speed_config", None) or {}
                        use_llm_in_render = speed_cfg.get("use_llm_in_render", speed_cfg.get("use_llm_for_all", True)) if self.cfg.llm else False
                        # Avoid polishing very short expressions; LLM often over-edits/hallucinates.
                        if use_llm_in_render and self.cfg.llm and self.llm_worker._client and len(latex_text.strip()) >= 12:
                            try:
                                # More aggressive LLM polish for inline math
                                polish_prompt = f"""Convert this inline math expression to proper LaTeX. The expression may contain garbled characters or incorrect formatting.

Input: {latex_text}

Requirements:
1. Fix garbled characters (e.g., "ˆ C" -> "\\hat{{C}}", "C ( r ) 2" -> "C(r)^2")
2. Use proper LaTeX syntax:
   - Subscripts: x_i not x i
   - Superscripts: x^2 not x 2
   - Functions: \\hat{{C}} not ˆ C
   - Remove extra spaces
3. Ensure proper grouping with braces when needed
4. Return ONLY the cleaned LaTeX code without $ delimiters

LaTeX:"""
                                resp = self.llm_worker._llm_create(
                                    messages=[
                                        {"role": "system", "content": "You are a LaTeX math expert specializing in inline math expressions. Always return clean, correct LaTeX."},
                                        {"role": "user", "content": polish_prompt}
                                    ],
                                    temperature=0.0,
                                    max_tokens=300,
                                )
                                polished = (resp.choices[0].message.content or "").strip()
                                # Remove $ if present
                                if polished.startswith("$") and polished.endswith("$"):
                                    polished = polished[1:-1].strip()
                                elif polished.startswith("$$") and polished.endswith("$$"):
                                    polished = polished[2:-2].strip()
                                # Remove any markdown code fences
                                if polished.startswith("```"):
                                    polished = re.sub(r'^```(?:\w+)?\n?', '', polished)
                                    polished = re.sub(r'\n?```$', '', polished)
                                if polished and len(polished) > 0:
                                    latex_text = polished
                            except Exception as e:
                                # If LLM fails, use the original
                                pass
                        
                        # Inline math: ensure no nested $ and proper formatting
                        out.append(f"${latex_text}$")
                    else:
                        # Display math
                        out.append(f"$$\n{latex_text}\n$$")
                else:
                    # Fallback to original text if conversion fails
                    # Clean up the text first
                    fallback_text = math_src.strip()
                    fallback_text = _strip_prose_tail_from_math(fallback_text)
                    fallback_text = re.sub(r',\s*$', '', fallback_text)
                    fallback_text = fallback_text.replace('$', '')
                    out.append(f"$$\n{fallback_text}\n$$")
                # Emit any prose/caption tail AFTER the math block.
                if prose_tail:
                    out.append(prose_tail)
                math_block_time = time.time() - math_block_start
                if math_block_time > 0.1:
                    block_times.append((block_idx, f"math({math_text_len}chars)", math_block_time))
            elif b.is_caption:
                # Captions: italicize and add proper spacing
                caption_text = b.text.strip()
                # Remove leading "Fig." or "Figure" if already in markdown format
                if not caption_text.startswith("*"):
                    out.append(f"*{caption_text}*")
                else:
                    out.append(caption_text)
            else:
                # Regular text - use LLM to fix mojibake if available
                text = b.text
                if self.cfg.llm and hasattr(self.llm_worker, '_client') and self.llm_worker._client:
                    # Check if text has mojibake
                    if any(pattern in text for pattern in ['ďŹ', 'Ď', 'Î´', 'Îą', 'â']):
                        try:
                            t2 = time.time()
                            repaired = self.llm_worker.call_llm_repair_body_paragraph(
                                text,
                                page_number=page_index,
                                block_index=len(out)
                            )
                            llm_call_count += 1
                            llm_total_time += (time.time() - t2)
                            if repaired:
                                text = repaired
                        except Exception:
                            pass
                out.append(text)
            out.append("")
        
        render_time = time.time() - render_start
        if llm_call_count > 0:
            avg_llm_time = llm_total_time / llm_call_count
            print(f"    [Page {page_index+1}] Render: {render_time:.2f}s total, {llm_call_count} LLM calls, {llm_total_time:.2f}s LLM time (avg {avg_llm_time:.2f}s/call)", flush=True)
        else:
            print(f"    [Page {page_index+1}] Render: {render_time:.2f}s (no LLM calls)", flush=True)
        
        # Report slow blocks
        if block_times:
            slow_blocks = sorted(block_times, key=lambda x: x[2], reverse=True)[:5]
            for block_idx, block_type, block_time in slow_blocks:
                print(f"      [Page {page_index+1}] Slow block {block_idx+1} ({block_type}): {block_time:.2f}s", flush=True)
        
        return "\n".join(out)

    def _merge_split_formulas(self, md: str) -> str:
        """Merge consecutive formula blocks that were split across lines."""
        lines = md.splitlines()
        result = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Check for display math block start
            if stripped == "$$":
                # Collect formula content
                formula_parts = []
                i += 1
                
                # Collect until we find closing $$
                while i < len(lines):
                    next_line = lines[i]
                    next_stripped = next_line.strip()
                    
                    if next_stripped == "$$":
                        # Found closing $$
                        i += 1
                        break
                    elif next_stripped:
                        formula_parts.append(next_stripped)
                    i += 1
                
                # Merge formula parts
                if formula_parts:
                    merged = " ".join(formula_parts)
                    # Clean up: remove duplicate spaces, fix common issues
                    merged = re.sub(r'\s+', ' ', merged)
                    result.append("$$")
                    result.append(merged)
                    result.append("$$")
                    result.append("")
            else:
                # Regular line - check if it's inline math
                if re.search(r'\$[^$]+\$', line):
                    result.append(line)
                else:
                    result.append(line)
                i += 1
        
        # Second pass: merge consecutive $$ blocks and inline math that should be together
        final_result = []
        i = 0
        while i < len(result):
            line = result[i]
            stripped = line.strip()
            
            if stripped == "$$":
                # Start collecting consecutive formula blocks
                formula_blocks = []
                i += 1
                
                # Collect this formula
                current_formula = []
                while i < len(result) and result[i].strip() != "$$":
                    if result[i].strip():
                        current_formula.append(result[i].strip())
                    i += 1
                
                if i < len(result) and result[i].strip() == "$$":
                    i += 1
                    if current_formula:
                        formula_blocks.append(" ".join(current_formula))
                
                # Check if next blocks are also formulas (within 3 lines, including inline math)
                blank_count = 0
                inline_math_blocks = []
                while i < len(result) and blank_count < 3:
                    next_line = result[i]
                    next_stripped = next_line.strip()
                    
                    if next_stripped == "":
                        blank_count += 1
                        i += 1
                    elif next_stripped == "$$":
                        # Another display math block - merge it
                        i += 1
                        next_formula = []
                        while i < len(result) and result[i].strip() != "$$":
                            if result[i].strip():
                                next_formula.append(result[i].strip())
                            i += 1
                        if i < len(result) and result[i].strip() == "$$":
                            i += 1
                            if next_formula:
                                formula_blocks.append(" ".join(next_formula))
                        blank_count = 0
                    elif re.match(r'^\$[^$]+\$$', next_stripped):
                        # Inline math - collect it if it looks like part of a larger formula
                        inline_math_blocks.append(next_stripped)
                        i += 1
                        blank_count = 0
                    else:
                        # Check if this line looks like a formula fragment (short, contains math symbols)
                        if len(next_stripped) < 50 and re.search(r'[+\-*/=<>≤≥∈∑∫αβγδεθλμπστφω]', next_stripped):
                            # Might be a formula fragment
                            inline_math_blocks.append(next_stripped)
                            i += 1
                            blank_count = 0
                        else:
                            break
                
                # Merge all formula blocks
                all_parts = formula_blocks + inline_math_blocks
                if all_parts:
                    merged = " ".join(all_parts)
                    merged = re.sub(r'\s+', ' ', merged)
                    final_result.append("$$")
                    final_result.append(merged)
                    final_result.append("$$")
                    final_result.append("")
            else:
                final_result.append(line)
                i += 1
        
        return "\n".join(final_result)

    def _fix_heading_structure(self, md: str) -> str:
        """Fix heading hierarchy to ensure proper structure."""
        lines = md.splitlines()
        out = []
        heading_stack = [0]  # Track heading levels
        
        for line in lines:
            stripped = line.strip()
            
            # Check if it's a heading
            match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            if match:
                level = len(match.group(1))
                text = match.group(2).strip()
                
                # Skip if heading looks like a formula (more patterns)
                is_formula = False
                if re.search(r'^\s*[A-Z]?\s*\d+\s*[a-z]', text) or \
                   re.search(r'^\s*\d+\s*[a-z]+\s*[+\-]', text) or \
                   (len(text) <= 15 and re.search(r'\d+.*[a-z]|[a-z].*\d+', text) and '=' in text) or \
                   re.search(r'[αβγδεζηθικλμνξοπρστυφχψω]', text, re.IGNORECASE) or \
                   (len(text) <= 20 and re.search(r'[+\-*/^_{}\[\]()]', text) and not re.search(r'[A-Z]{3,}', text)):
                    # This is likely a formula, not a heading - convert to math
                    out.append(f"$$\n{text}\n$$")
                    continue
                
                # Fix skipped levels (e.g., H1 -> H3 should become H1 -> H2)
                if level > heading_stack[-1] + 1:
                    # Skip was detected, fix it
                    level = heading_stack[-1] + 1
                
                # Update stack
                while len(heading_stack) > 0 and heading_stack[-1] >= level:
                    heading_stack.pop()
                heading_stack.append(level)
                
                # Ensure heading text is reasonable
                if len(text) > 200:
                    # Very long heading - might be body text
                    out.append(text)
                else:
                    out.append("#" * level + " " + text)
            else:
                out.append(line)
        
        return "\n".join(out)

    def _fix_vision_formula_errors(self, md: str) -> str:
        """
        Fix common formula errors from vision model output:
        - Missing subscripts: \alphaj -> \alpha_j
        - Missing superscripts: x2 -> x^2
        - Split formulas (merged back together)
        - Prime symbols with subscripts: G' i -> G'_i
        - Unrendered symbols (□, etc.)
        - Formatting issues
        - Chinese text mixed in formulas
        - Table formatting
        """
        lines = md.splitlines()
        result = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Merge split display formulas ($$...$$ that were broken across lines)
            if line.strip().startswith('$$') and not line.strip().endswith('$$'):
                # Start of a split formula - collect until we find the closing $$
                formula_parts = [line]
                i += 1
                while i < len(lines) and not lines[i].strip().endswith('$$'):
                    formula_parts.append(lines[i])
                    i += 1
                if i < len(lines):
                    formula_parts.append(lines[i])
                # Merge into single line
                merged = ' '.join(p.strip() for p in formula_parts)
                result.append(merged)
                i += 1
                continue
            
            # Only process lines that contain formulas
            if '$' in line:
                # Fix prime symbols with subscripts: G' i -> G'_i, G'_{low} -> G'_{low} (already correct)
                line = re.sub(r"([A-Za-z])'\s+([a-z])(?![_^{])", r"\1'_{\2}", line)
                line = re.sub(r"([A-Za-z])'\s+_\{([^}]+)\}", r"\1'_{\2}", line)
                
                # Fix missing subscripts in Greek letters: \alphaj -> \alpha_j
                line = re.sub(r'\\(alpha|beta|gamma|delta|epsilon|theta|lambda|mu|nu|pi|rho|sigma|tau|phi|chi|psi|omega)([a-z])(?![_^{])', r'\\\1_{\2}', line)
                
                # Fix missing subscripts: \partial c j -> \partial c_j
                line = re.sub(r'\\(partial)\s+([a-z])\s+([a-z])(?![_^{])', r'\\\1 \2_{\3}', line)
                
                # Fix: \alphaj (no backslash before j) -> \alpha_j
                line = re.sub(r'\\(alpha|beta|gamma|delta|epsilon|theta|lambda|mu|nu|pi|rho|sigma|tau|phi|chi|psi|omega)([a-z])(?![_^{\\])', r'\\\1_{\2}', line)
                
                # Remove Chinese text from inside formulas
                line = re.sub(r'(\$[^$]*)[公式]+([^$]*\$)', r'\1\2', line)
                line = re.sub(r'(\$\$[^$]*)[公式]+([^$]*\$\$)', r'\1\2', line)
                
                # Remove unrendered box symbols (□) from formulas
                line = re.sub(r'(\$[^$]*)[□]+([^$]*\$)', r'\1\2', line)
                line = re.sub(r'(\$\$[^$]*)[□]+([^$]*\$\$)', r'\1\2', line)
                
                # Fix spaces before subscripts/superscripts: x _i -> x_i, x ^2 -> x^2
                line = re.sub(r'([a-zA-Z])\s+_(\w)', r'\1_\2', line)
                line = re.sub(r'([a-zA-Z])\s+\^(\w)', r'\1^\2', line)
                
                # Fix inline formulas that were split: $... $ ...$ -> $... ...$
                if line.count('$') >= 2 and line.count('$') % 2 == 0:
                    # Try to merge split inline formulas on the same line
                    line = re.sub(r'\$\s+([^$]+)\s+\$', r'$ \1 $', line)
            
            # Fix table formatting: ensure proper separator and alignment
            if '|' in line and line.strip().startswith('|'):
                # Check if this looks like a table row
                cells = [c.strip() for c in line.split('|') if c.strip()]
                if len(cells) >= 2:
                    # Ensure proper table format
                    if '---' not in line and i + 1 < len(lines):
                        # Check if next line is separator
                        next_line = lines[i + 1].strip()
                        if '|' in next_line and '---' not in next_line:
                            # Missing separator - insert one
                            sep = '| ' + ' | '.join(['---'] * len(cells)) + ' |'
                            result.append(line)
                            result.append(sep)
                            i += 1
                            continue
            
            result.append(line)
            i += 1
        
        return "\n".join(result)

    def _fix_references_format(self, md: str) -> str:
        """
        Fix references section formatting:
        - Remove formula blocks ($$...$$) and code blocks (```...```) from references
        - Ensure each reference is on a separate line
        - Ensure references are numbered (add numbers if missing)
        - Convert formulas in references to plain text
        """
        lines = md.splitlines()
        result = []
        in_references = False
        ref_start_idx = None
        ref_lines = []
        
        # Find References section
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Check if this is a References heading
            if re.match(r'^#+\s+References?\s*$', stripped, re.IGNORECASE) or \
               re.match(r'^References?\s*$', stripped, re.IGNORECASE):
                in_references = True
                ref_start_idx = i
                result.append(line)  # Keep the heading
                continue
            
            if in_references:
                # Check if we've reached the end of references (new major heading)
                if stripped.startswith('#') and not re.match(r'^#+\s+References?\s*$', stripped, re.IGNORECASE):
                    # Check if it's a major section (not just a subheading in references)
                    heading_level = len(stripped) - len(stripped.lstrip('#'))
                    if heading_level <= 2:  # H1 or H2 - likely end of references
                        # Process collected references
                        result.extend(self._format_references_block(ref_lines))
                        ref_lines = []
                        in_references = False
                        result.append(line)  # Add the new heading
                        continue
                
                # Collect reference lines
                ref_lines.append((i, line))
            else:
                result.append(line)
        
        # Process any remaining references
        if ref_lines:
            result.extend(self._format_references_block(ref_lines))
        
        return "\n".join(result)
    
    def _format_references_block(self, ref_lines: list[tuple[int, str]]) -> list[str]:
        """Format a block of reference lines."""
        formatted = []
        current_ref = []
        ref_num = 1
        
        for i, line in ref_lines:
            stripped = line.strip()
            
            # Skip empty lines (will add proper spacing later)
            if not stripped:
                if current_ref:
                    # End of current reference
                    ref_text = ' '.join(current_ref)
                    formatted.append(self._format_single_reference(ref_text, ref_num))
                    current_ref = []
                    ref_num += 1
                continue
            
            # Remove code blocks
            if stripped.startswith('```'):
                continue
            
            # Remove display math blocks ($$...$$)
            if stripped.startswith('$$'):
                if stripped.endswith('$$') and len(stripped) > 2:
                    # Single-line formula - convert to text
                    formula_text = stripped[2:-2].strip()
                    plain_text = self._formula_to_plain_text(formula_text)
                    if plain_text:
                        current_ref.append(plain_text)
                # Multi-line formulas are handled by collecting until closing $$
                continue
            
            # Remove inline math ($...$) but keep the content as text
            if '$' in stripped:
                # Replace $...$ with plain text
                stripped = re.sub(r'\$([^$]+)\$', lambda m: self._formula_to_plain_text(m.group(1)), stripped)
                stripped = re.sub(r'\$\$([^$]+)\$\$', lambda m: self._formula_to_plain_text(m.group(1)), stripped)
            
            # Check if this line starts a new reference (has a number at the start)
            ref_match = re.match(r'^(\[?\d+\]?)[\.\s]+(.+)$', stripped)
            if ref_match and current_ref:
                # This starts a new reference - finish the previous one
                ref_text = ' '.join(current_ref)
                formatted.append(self._format_single_reference(ref_text, ref_num))
                current_ref = []
                ref_num += 1
                # Add the new reference content
                current_ref.append(ref_match.group(2))
            elif ref_match:
                # First reference or continuation
                current_ref.append(ref_match.group(2))
            else:
                # Continuation of current reference
                current_ref.append(stripped)
        
        # Add the last reference
        if current_ref:
            ref_text = ' '.join(current_ref)
            formatted.append(self._format_single_reference(ref_text, ref_num))
        
        return formatted
    
    def _format_single_reference(self, text: str, num: int) -> str:
        """Format a single reference with proper numbering."""
        # Clean up the text
        text = text.strip()
        
        # Remove any remaining math notation
        text = re.sub(r'\$([^$]+)\$', lambda m: self._formula_to_plain_text(m.group(1)), text)
        text = re.sub(r'\$\$([^$]+)\$\$', lambda m: self._formula_to_plain_text(m.group(1)), text)
        
        # Check if it already has a number
        if re.match(r'^\[?\d+\]?\s+', text):
            # Already numbered, just ensure proper format
            text = re.sub(r'^\[?(\d+)\]?\s+', r'[\1] ', text)
            return text
        
        # Add number if missing
        return f"[{num}] {text}"
    
    def _formula_to_plain_text(self, formula: str) -> str:
        """Convert LaTeX formula to plain text for references."""
        if not formula:
            return ""
        
        # Remove LaTeX commands but keep the content
        text = formula
        
        # Convert subscripts: x_i -> x i or xi
        text = re.sub(r'_\{([^}]+)\}', r' \1', text)
        text = re.sub(r'_([a-z0-9])', r' \1', text)
        
        # Convert superscripts: x^2 -> x2
        text = re.sub(r'\^\{([^}]+)\}', r'\1', text)
        text = re.sub(r'\^([a-z0-9])', r'\1', text)
        
        # Remove LaTeX commands but keep Greek letter names
        text = re.sub(r'\\alpha', 'alpha', text)
        text = re.sub(r'\\beta', 'beta', text)
        text = re.sub(r'\\gamma', 'gamma', text)
        text = re.sub(r'\\delta', 'delta', text)
        text = re.sub(r'\\[a-z]+\{([^}]+)\}', r'\1', text)  # \command{content} -> content
        text = re.sub(r'\\[a-zA-Z]+', '', text)  # Remove remaining commands
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _llm_fix_misclassified_headings(self, md: str) -> str:
        """Fix headings that were misclassified as math formulas, and remove non-headings."""
        lines = md.splitlines()
        fixed_lines = []
        seen_headings = set()
        heading_stack = []
        
        print("Checking for misclassified headings...")
        fixed_count = 0
        removed_count = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Check if this line looks like a misclassified heading (starts and ends with $, single line)
            if stripped.startswith('$') and stripped.endswith('$'):
                # Count $ symbols - should be exactly 2 (one at start, one at end)
                dollar_count = stripped.count('$')
                if dollar_count == 2:
                    text_content = stripped[1:-1].strip()
                    
                    # First, use aggressive heuristic check for obvious headings
                    is_heading = False
                    level = 1
                    
                    # Pattern: "1. Introduction", "2. Related Work", etc.
                    match1 = re.match(r'^(\d+)\.\s+(.+)$', text_content)
                    if match1:
                        is_heading = True
                        level = 1
                        heading_text = text_content
                    
                    # Pattern: "3.1. Background", "4.1. Experimental Setup", etc.
                    match2 = re.match(r'^(\d+)\.(\d+)\.\s+(.+)$', text_content)
                    if match2:
                        is_heading = True
                        level = 2
                        heading_text = text_content
                    
                    # Pattern: "3.1.1. Details", etc.
                    match3 = re.match(r'^(\d+)\.(\d+)\.(\d+)\.\s+(.+)$', text_content)
                    if match3:
                        is_heading = True
                        level = 3
                        heading_text = text_content
                    
                    # Pattern: "A. Appendix", etc.
                    match4 = re.match(r'^([A-Z])\.\s+(.+)$', text_content)
                    if match4:
                        is_heading = True
                        level = 2
                        heading_text = text_content
                    
                    # If heuristic says it's a heading, use LLM to confirm and refine
                    if is_heading:
                        if self.cfg.llm and hasattr(self.llm_worker, '_client') and self.llm_worker._client:
                            try:
                                llm_result = self.llm_worker.call_llm_confirm_and_level_heading(
                                    heading_text,
                                    page_number=0
                                )
                                if llm_result and llm_result.get('is_heading'):
                                    heading_text = llm_result.get('text', heading_text)
                                    level = llm_result.get('level', level)
                            except Exception:
                                pass  # Use heuristic level if LLM fails
                        
                        # Normalize for duplicate check
                        normalized = re.sub(r'^[IVX]+\.\s*', '', heading_text, flags=re.IGNORECASE).strip()
                        normalized = re.sub(r'^[A-Z]\.\s*', '', normalized).strip()
                        normalized = re.sub(r'^\d+(?:\.\d+)*\.?\s*', '', normalized).strip()
                        
                        if normalized not in seen_headings:
                            # Update heading stack
                            while heading_stack and heading_stack[-1] >= level:
                                heading_stack.pop()
                            heading_stack.append(level)
                            seen_headings.add(normalized)
                            fixed_lines.append("#" * level + " " + heading_text)
                            fixed_count += 1
                            continue
            
            # Check if this is a heading that should be removed (author/affiliation lines)
            if stripped.startswith('#'):
                heading_match = re.match(r'^#+\s+(.+)$', stripped)
                if heading_match:
                    heading_text = heading_match.group(1).strip()
                    # Check if it's an author/affiliation line
                    is_author_line = (
                        '@' in heading_text or
                        re.search(r'\b(?:university|dept|department|institute|email|zhejiang|westlake)\b', heading_text, re.IGNORECASE) or
                        re.search(r'^\w+\s+\w+.*\d+.*\d+', heading_text)  # "Name 1, 2 Name 2" pattern
                    )
                    
                    if is_author_line:
                        # Remove heading markdown, keep as regular text
                        fixed_lines.append(heading_text)
                        removed_count += 1
                        continue
            
            # Not a misclassified heading, keep original line
            fixed_lines.append(line)
        
        if fixed_count > 0:
            print(f"Fixed {fixed_count} misclassified headings")
        if removed_count > 0:
            print(f"Removed {removed_count} non-heading titles")
        
        return "\n".join(fixed_lines)

    def _llm_fix_inline_formulas(self, md: str) -> str:
        """Fix inline formulas with batch LLM processing for speed."""
        if not self.cfg.llm:
            return md
        
        # Ensure LLM client is initialized
        if not hasattr(self.llm_worker, '_client') or not self.llm_worker._client:
            try:
                self.llm_worker._client = self.llm_worker._ensure_openai_class()(
                    api_key=self.cfg.llm.api_key,
                    base_url=self.cfg.llm.base_url,
                )
            except Exception as e:
                print(f"Failed to initialize LLM client for inline formulas: {e}")
                return md
        
        # Collect all inline formulas first
        import re
        pattern = r'\$([^$]+)\$'
        all_formulas = []
        formula_positions = []  # List of (line_idx, match_start, match_end, formula_text)
        
        lines = md.splitlines()
        for line_idx, line in enumerate(lines):
            if '$' in line and not line.strip().startswith('$$'):
                matches = list(re.finditer(pattern, line))  # Use original line, not stripped
                for match in matches:
                    formula_text = match.group(1)
                    # Skip if it's a display formula marker
                    if formula_text.startswith('$') or formula_text.endswith('$'):
                        continue
                    # Skip if it looks like already correct LaTeX (has backslashes and no garbled chars)
                    # But include if it has garbled chars like ˆ
                    has_garbled = any(c in formula_text for c in ['ˆ', ' ']) and '\\' not in formula_text
                    if '\\' in formula_text and '\\hat' in formula_text and not has_garbled:
                        continue
                    all_formulas.append(formula_text)
                    formula_positions.append((line_idx, match.start(), match.end(), formula_text))
        
        if not all_formulas:
            return md
        
        print(f"Fixing {len(all_formulas)} inline formulas in batch...")
        
        # Batch process ALL formulas at once for speed
        fixed_formulas = {}
        try:
            prompt = f"""Fix these {len(all_formulas)} inline math expressions to proper LaTeX. Return a JSON array with the fixed formulas.

CRITICAL FIXES for each formula:
1. Fix garbled Unicode: "ˆ" -> "\\hat", "C ( r )" -> "C(r)", "C ( r ) 2" -> "C(r)^2"
2. Use proper LaTeX: subscripts x_i, superscripts x^2, functions \\hat{{C}}
3. Remove ALL extra spaces between symbols
4. Group properly with braces

Input formulas:
{json.dumps(all_formulas, ensure_ascii=False)}

Return JSON array with fixed formulas in the same order, e.g.:
["\\hat{{C}}(r) - C(r)^2", "x_1 + x_2", ...]

Return ONLY the JSON array, no other text:"""
            
            resp = self.llm_worker._llm_create(
                messages=[
                    {"role": "system", "content": "You are a LaTeX math expert. Return only valid JSON array with fixed formulas."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=3000,
            )
            result_text = (resp.choices[0].message.content or "").strip()
            # Remove markdown code fences
            if result_text.startswith("```"):
                result_text = re.sub(r'^```(?:\w+)?\n?', '', result_text)
                result_text = re.sub(r'\n?```$', '', result_text)
            
            fixed_batch = json.loads(result_text)
            
            # Map back to original formulas
            for i, fixed_formula in enumerate(fixed_batch):
                if i < len(all_formulas):
                    original = all_formulas[i]
                    # Remove $ if present
                    fixed = str(fixed_formula).strip()
                    if fixed.startswith("$") and fixed.endswith("$"):
                        fixed = fixed[1:-1].strip()
                    elif fixed.startswith("$$") and fixed.endswith("$$"):
                        fixed = fixed[2:-2].strip()
                    # Additional cleanup for common issues
                    fixed = fixed.replace('\\hat C', '\\hat{C}').replace('\\hat{ C', '\\hat{C')
                    fixed = re.sub(r'C\s*\(\s*r\s*\)\s*2', r'C(r)^2', fixed)
                    fixed = re.sub(r'C\s*\(\s*r\s*\)', r'C(r)', fixed)
                    fixed = re.sub(r'\s+', ' ', fixed)  # Remove extra spaces
                    fixed_formulas[original] = fixed
        except json.JSONDecodeError as e:
            # If JSON parsing fails, try to extract formulas manually
            print(f"JSON parse failed, trying manual extraction: {e}")
            # Try to extract from text response
            if '[' in result_text and ']' in result_text:
                try:
                    # Try to find JSON array in the response
                    start = result_text.find('[')
                    end = result_text.rfind(']') + 1
                    if start >= 0 and end > start:
                        json_text = result_text[start:end]
                        fixed_batch = json.loads(json_text)
                        for i, fixed_formula in enumerate(fixed_batch):
                            if i < len(all_formulas):
                                original = all_formulas[i]
                                fixed = str(fixed_formula).strip()
                                if fixed.startswith("$"):
                                    fixed = fixed.strip('$').strip()
                                fixed = fixed.replace('\\hat C', '\\hat{C}')
                                fixed = re.sub(r'C\s*\(\s*r\s*\)\s*2', r'C(r)^2', fixed)
                                fixed = re.sub(r'C\s*\(\s*r\s*\)', r'C(r)', fixed)
                                fixed_formulas[original] = fixed
                except Exception:
                    pass
        except Exception as e:
            # If batch fails, keep originals
            print(f"Failed to fix inline formulas: {e}")
            pass
        
        # Apply fixes to lines - use positions for efficient replacement
        fixed_lines = lines.copy()
        fixed_count = 0
        # Process from end to start to preserve indices
        for line_idx, match_start, match_end, formula_text in reversed(formula_positions):
            if formula_text in fixed_formulas:
                fixed = fixed_formulas[formula_text]
                if fixed and fixed != formula_text:
                    # Replace in the line
                    line = fixed_lines[line_idx]
                    fixed_lines[line_idx] = line[:match_start] + f"${fixed}$" + line[match_end:]
                    fixed_count += 1
        
        if fixed_count > 0:
            print(f"Fixed {fixed_count} inline formulas")
        
        return "\n".join(fixed_lines)

    def _llm_fix_references(self, md: str) -> str:
        """Fix references section formatting with LLM."""
        if not self.cfg.llm:
            return md
        
        # Ensure LLM client is initialized
        if not hasattr(self.llm_worker, '_client') or not self.llm_worker._client:
            try:
                self.llm_worker._client = self.llm_worker._ensure_openai_class()(
                    api_key=self.cfg.llm.api_key,
                    base_url=self.cfg.llm.base_url,
                )
            except Exception as e:
                print(f"Failed to initialize LLM client for references: {e}")
                return md
        
        # Find References section
        lines = md.splitlines()
        ref_start = None
        for i, line in enumerate(lines):
            if re.match(r'^#+\s+References', line, re.IGNORECASE):
                ref_start = i
                break
        
        if ref_start is None:
            return md
        
        # Quick check: if references already formatted (has [1] pattern at start), skip
        # Check the FIRST non-empty line after References
        first_ref_line = None
        for idx in range(ref_start+1, min(ref_start+10, len(lines))):
            line = lines[idx].strip()
            if line:
                first_ref_line = line
                break
        if first_ref_line and re.match(r'^\[\d+\]', first_ref_line):
            # Check if all or most references are formatted (sample check)
            formatted_count = 0
            total_count = 0
            for idx in range(ref_start+1, min(ref_start+30, len(lines))):
                line = lines[idx].strip()
                if line:
                    total_count += 1
                    if re.match(r'^\[\d+\]', line):
                        formatted_count += 1
            # If 80%+ are formatted, skip
            if total_count > 0 and formatted_count / total_count >= 0.8:
                return md  # Already formatted
        
        print("Fixing references formatting...")
        # Extract references section
        ref_lines = lines[ref_start+1:]
        ref_text = "\n".join(ref_lines)
        
        # Process references in larger chunks for speed (reduce LLM calls)
        max_ref_chunk = 20000  # Larger chunks = fewer LLM calls
        if len(ref_text) <= max_ref_chunk:
            # Single chunk - fastest
            formatted_refs = self._llm_format_references_chunk(ref_text)
            if formatted_refs:
                return "\n".join(lines[:ref_start+1]) + "\n\n" + formatted_refs
        else:
            # Multiple chunks - but limit to max 2 chunks for speed
            ref_chunks = []
            # Split into 2 chunks max
            mid_point = len(ref_lines) // 2
            ref_chunks.append("\n".join(ref_lines[:mid_point]))
            ref_chunks.append("\n".join(ref_lines[mid_point:]))
            
            # Process chunks in parallel to reduce tail latency when page conversion is already at 100%.
            formatted_chunks = [None] * len(ref_chunks)
            with ThreadPoolExecutor(max_workers=min(2, len(ref_chunks))) as ex:
                fut_to_idx = {
                    ex.submit(self._llm_format_references_chunk, chunk): i
                    for i, chunk in enumerate(ref_chunks)
                }
                for fut in as_completed(fut_to_idx):
                    i = fut_to_idx[fut]
                    print(f"Processing references chunk {i+1}/{len(ref_chunks)}...", end='\r')
                    try:
                        formatted = fut.result()
                    except Exception:
                        formatted = None
                    formatted_chunks[i] = formatted if formatted else ref_chunks[i]
            print()  # New line
            
            if formatted_chunks:
                return "\n".join(lines[:ref_start+1]) + "\n\n" + "\n\n".join(formatted_chunks)
        
        return md
    
    def _llm_format_references_chunk(self, ref_text: str) -> Optional[str]:
        """Format a chunk of references using LLM."""
        try:
            prompt = f"""Format this references section properly. Each reference should be on its own line, properly formatted.

Requirements:
1. Each reference should start with [number] followed by a space
2. Format should be: [number] Author names. Title. Conference/Journal, pages, year.
3. Fix any garbled text and mojibake (e.g., "Miloš Hašan" not garbled)
4. Ensure proper spacing and punctuation
5. Keep all citation numbers exactly as they appear
6. Each reference on a separate line
7. Remove any duplicate references
8. Fix special characters properly (e.g., "®" not garbled)

References section:
{ref_text}

Return ONLY the formatted references, one per line, no explanations:"""
            
            resp = self.llm_worker._llm_create(
                messages=[
                    {"role": "system", "content": "You are an expert at formatting academic references. Return properly formatted references with correct Unicode characters."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=6000,
            )
            formatted_refs = (resp.choices[0].message.content or "").strip()
            # Remove markdown code fences if present
            if formatted_refs.startswith("```"):
                formatted_refs = re.sub(r'^```(?:\w+)?\n?', '', formatted_refs)
                formatted_refs = re.sub(r'\n?```$', '', formatted_refs)
            
            return formatted_refs if formatted_refs else None
        except Exception as e:
            print(f"Failed to format references chunk: {e}")
            return None

    def _llm_fix_tables(self, md: str) -> str:
        """Fix tables formatting with LLM."""
        if not self.cfg.llm:
            return md
        
        # Ensure LLM client is initialized
        if not hasattr(self.llm_worker, '_client') or not self.llm_worker._client:
            try:
                self.llm_worker._client = self.llm_worker._ensure_openai_class()(
                    api_key=self.cfg.llm.api_key,
                    base_url=self.cfg.llm.base_url,
                )
            except Exception as e:
                print(f"Failed to initialize LLM client for tables: {e}")
                return md
        
        lines = md.splitlines()
        fixed_lines = []
        in_table = False
        table_lines = []
        table_start = None
        fixed_table_count = 0
        
        print("Checking for tables to fix...")
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Check if this is a table line (starts with |)
            if stripped.startswith('|') and '|' in stripped[1:]:
                if not in_table:
                    in_table = True
                    table_start = i
                    table_lines = [line]
                else:
                    table_lines.append(line)
            else:
                # End of table
                if in_table:
                    # Process the table
                    table_text = "\n".join(table_lines)
                    # Quick check: if table already looks good (has --- separator), skip LLM
                    if '---' in table_text and len(table_lines) >= 3:
                        fixed_lines.extend(table_lines)
                    elif len(table_lines) >= 3:  # At least header, separator, one row
                        try:
                            prompt = f"""Fix this Markdown table. Ensure proper formatting, alignment, and fix any garbled text.

Requirements:
1. Ensure all rows have the same number of columns
2. Fix any garbled text or mojibake
3. Ensure proper alignment with | separators
4. Keep header row and separator row (---)
5. Fix any spacing issues

Table:
{table_text}

Return ONLY the fixed table in Markdown format, no explanations:"""
                            
                            resp = self.llm_worker._llm_create(
                                messages=[
                                    {"role": "system", "content": "You are an expert at formatting Markdown tables. Return properly formatted tables."},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.0,
                                max_tokens=2000,
                            )
                            fixed_table = (resp.choices[0].message.content or "").strip()
                            # Remove markdown code fences if present
                            if fixed_table.startswith("```"):
                                fixed_table = re.sub(r'^```(?:\w+)?\n?', '', fixed_table)
                                fixed_table = re.sub(r'\n?```$', '', fixed_table)
                            
                            if fixed_table and fixed_table != table_text:
                                fixed_lines.extend(fixed_table.splitlines())
                                fixed_table_count += 1
                            else:
                                fixed_lines.extend(table_lines)
                        except Exception as e:
                            # Keep original if LLM fails
                            fixed_lines.extend(table_lines)
                    else:
                        fixed_lines.extend(table_lines)
                    
                    in_table = False
                    table_lines = []
                
                fixed_lines.append(line)
        
        # Handle table at end of document
        if in_table and table_lines:
            fixed_lines.extend(table_lines)
        
        if fixed_table_count > 0:
            print(f"Fixed {fixed_table_count} tables")
        
        return "\n".join(fixed_lines)
    
    def _llm_fix_tables_with_screenshot(self, md: str, pdf_path: Path, save_dir: Path) -> str:
        """Fix tables formatting with LLM, or screenshot if too difficult."""
        if not self.cfg.llm:
            return self._llm_fix_tables(md)
        
        # Ensure LLM client is initialized
        if not hasattr(self.llm_worker, '_client') or not self.llm_worker._client:
            try:
                self.llm_worker._client = self.llm_worker._ensure_openai_class()(
                    api_key=self.cfg.llm.api_key,
                    base_url=self.cfg.llm.base_url,
                )
            except Exception as e:
                print(f"Failed to initialize LLM client for tables: {e}")
                return md
        
        lines = md.splitlines()
        fixed_lines = []
        in_table = False
        table_lines = []
        table_start = None
        table_caption = None
        fixed_table_count = 0
        screenshot_count = 0
        table_num = 1
        
        print("Checking for tables to fix (with screenshot fallback)...")
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Check for table caption (before table)
            if i > 0 and (stripped.lower().startswith('table ') or '*Table' in stripped or 'Table' in stripped):
                table_caption = stripped
            # Check if this is a table line (starts with |)
            if stripped.startswith('|') and '|' in stripped[1:]:
                if not in_table:
                    in_table = True
                    table_start = i
                    table_lines = [line]
                else:
                    table_lines.append(line)
            else:
                # End of table
                if in_table:
                    table_text = "\n".join(table_lines)
                    # Quick check: if table already looks good (has --- separator), skip
                    has_separator = any('---' in tl for tl in table_lines)
                    has_multiple_rows = len([l for l in table_lines if l.strip().startswith('|')]) >= 3
                    if has_separator and has_multiple_rows:
                        fixed_lines.extend(table_lines)
                    elif len(table_lines) >= 3:  # At least header, separator, one row
                        try:
                            # Try LLM fix first
                            prompt = f"""Fix this Markdown table. Ensure proper formatting, alignment, and fix any garbled text.

Requirements:
1. Ensure all rows have the same number of columns
2. Fix any garbled text or mojibake
3. Ensure proper alignment with | separators
4. Keep header row and separator row (---)
5. Fix any spacing issues

Table:
{table_text}

Return ONLY the fixed table in Markdown format, no explanations. If the table is too complex or corrupted, return "SCREENSHOT" as the only word:"""
                            
                            resp = self.llm_worker._llm_create(
                                messages=[
                                    {"role": "system", "content": "You are an expert at formatting Markdown tables. Return properly formatted tables or 'SCREENSHOT' if too difficult."},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.0,
                                max_tokens=2000,
                            )
                            fixed_table = (resp.choices[0].message.content or "").strip()
                            # Remove markdown code fences if present
                            if fixed_table.startswith("```"):
                                fixed_table = re.sub(r'^```(?:\w+)?\n?', '', fixed_table)
                                fixed_table = re.sub(r'\n?```$', '', fixed_table)
                            
                            # Check if LLM says to screenshot
                            if fixed_table.upper() == "SCREENSHOT" or "SCREENSHOT" in fixed_table.upper():
                                # Screenshot the table from PDF
                                screenshot_path = self._screenshot_table_from_pdf(pdf_path, table_num, save_dir, table_caption)
                                if screenshot_path:
                                    fixed_lines.append(f"![Table {table_num}](./assets/{screenshot_path.name})")
                                    if table_caption:
                                        fixed_lines.append(f"*{table_caption}*")
                                    screenshot_count += 1
                                    table_num += 1
                                else:
                                    # Fallback: keep original
                                    fixed_lines.extend(table_lines)
                            elif fixed_table and fixed_table != table_text:
                                fixed_lines.extend(fixed_table.splitlines())
                                if table_caption:
                                    fixed_lines.append(f"*{table_caption}*")
                                fixed_table_count += 1
                            else:
                                # LLM couldn't fix, try screenshot
                                screenshot_path = self._screenshot_table_from_pdf(pdf_path, table_num, save_dir, table_caption)
                                if screenshot_path:
                                    fixed_lines.append(f"![Table {table_num}](./assets/{screenshot_path.name})")
                                    if table_caption:
                                        fixed_lines.append(f"*{table_caption}*")
                                    screenshot_count += 1
                                    table_num += 1
                                else:
                                    fixed_lines.extend(table_lines)
                        except Exception as e:
                            # LLM failed, try screenshot
                            print(f"LLM table fix failed, trying screenshot: {e}")
                            screenshot_path = self._screenshot_table_from_pdf(pdf_path, table_num, save_dir, table_caption)
                            if screenshot_path:
                                fixed_lines.append(f"![Table {table_num}](./assets/{screenshot_path.name})")
                                if table_caption:
                                    fixed_lines.append(f"*{table_caption}*")
                                screenshot_count += 1
                                table_num += 1
                            else:
                                fixed_lines.extend(table_lines)
                    else:
                        fixed_lines.extend(table_lines)
                    
                    in_table = False
                    table_lines = []
                    table_caption = None
                
                fixed_lines.append(line)
        
        # Add any remaining table lines
        if in_table and table_lines:
            fixed_lines.extend(table_lines)
        
        if fixed_table_count > 0:
            print(f"Fixed {fixed_table_count} tables")
        if screenshot_count > 0:
            print(f"Screenshot {screenshot_count} difficult tables")
        
        return "\n".join(fixed_lines)
    
    def _screenshot_table_from_pdf(self, pdf_path: Path, table_num: int, save_dir: Path, caption: Optional[str] = None) -> Optional[Path]:
        """Screenshot a table from PDF (simplified - would need page detection in real implementation)."""
        if fitz is None:
            return None
        try:
            doc = fitz.open(pdf_path)
            assets_dir = save_dir / "assets"
            assets_dir.mkdir(exist_ok=True)
            
            # For now, just take a screenshot of the first page (in real implementation, would detect table location)
            # This is a placeholder - real implementation would need to detect which page has the table
            if len(doc) > 0:
                page = doc[0]  # Placeholder: use first page
                img_name = f"table_{table_num}.png"
                img_path = assets_dir / img_name
                pix = page.get_pixmap(dpi=self.dpi)
                pix.save(img_path)
                doc.close()
                return img_path
        except Exception as e:
            print(f"Failed to screenshot table: {e}")
        return None
    
    def _llm_fix_references_with_crossref(self, md: str, save_dir: Path) -> str:
        """Fix references formatting with LLM and enrich with Crossref metadata."""
        if not self.cfg.llm:
            return self._llm_fix_references(md)
        
        # First, format references normally
        formatted_md = self._llm_fix_references(md)
        
        # Then, enrich with Crossref metadata
        try:
            from kb.citation_meta import fetch_best_crossref_meta
            
            # Find References section
            lines = formatted_md.splitlines()
            ref_start = None
            for i, line in enumerate(lines):
                if re.match(r'^#+\s+References', line, re.IGNORECASE):
                    ref_start = i
                    break
            
            if ref_start is None:
                return formatted_md
            
            # Extract references and enrich with Crossref
            ref_lines = lines[ref_start+1:]
            enriched_refs = []
            crossref_metadata = {}
            
            print("Enriching references with Crossref metadata...")
            for ref_line in ref_lines[:50]:  # Limit to first 50 for speed
                ref_line = ref_line.strip()
                if not ref_line or not re.match(r'^\[\d+\]', ref_line):
                    enriched_refs.append(ref_line)
                    continue
                
                # Extract title from reference (simplified)
                # Try to extract title between first period and next period or comma
                title_match = re.search(r'\]\s*[^.]*\.\s*([^.,]+(?:\.|,))', ref_line)
                if title_match:
                    title = title_match.group(1).strip().rstrip('.,')
                    if len(title) > 10:  # Reasonable title length
                        try:
                            meta = fetch_best_crossref_meta(query_title=title, min_score=0.85)
                            if meta:
                                ref_num = re.match(r'^\[(\d+)\]', ref_line).group(1)
                                crossref_metadata[ref_num] = meta
                                # Add DOI if available
                                if meta.get('doi'):
                                    ref_line = f"{ref_line} DOI: {meta['doi']}"
                        except Exception:
                            pass
                
                enriched_refs.append(ref_line)
            
            # Save Crossref metadata to JSON file
            if crossref_metadata:
                metadata_file = save_dir / "crossref_metadata.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(crossref_metadata, f, indent=2, ensure_ascii=False)
                print(f"Saved Crossref metadata for {len(crossref_metadata)} references to {metadata_file}")
            
            # Reconstruct markdown
            return "\n".join(lines[:ref_start+1]) + "\n\n" + "\n".join(enriched_refs) + "\n\n" + "\n".join(lines[ref_start+1+len(ref_lines):])
        except Exception as e:
            print(f"Crossref enrichment failed: {e}, using formatted references")
            return formatted_md

    def _llm_fix_display_math(self, md: str) -> str:
        """Fix display math blocks - remove nested $ symbols, fix formatting, and add equation numbers."""
        if not self.cfg.llm:
            return md

        # Safety: by default, do NOT ask the LLM to rewrite display math blocks.
        # Some models may "explain" formulas or convert norms (|...|) into Markdown tables.
        # Enable explicitly if you really want it:
        #   KB_PDF_ENABLE_LLM_DISPLAY_MATH_FIX=1
        def _env_bool(name: str, default: bool = False) -> bool:
            try:
                raw = str(os.environ.get(name, "") or "").strip().lower()
                if not raw:
                    return bool(default)
                return raw in {"1", "true", "yes", "y", "on"}
            except Exception:
                return bool(default)

        enable_llm_fix = _env_bool("KB_PDF_ENABLE_LLM_DISPLAY_MATH_FIX", False)
        
        # Ensure LLM client is initialized
        if not hasattr(self.llm_worker, '_client') or not self.llm_worker._client:
            try:
                self.llm_worker._client = self.llm_worker._ensure_openai_class()(
                    api_key=self.cfg.llm.api_key,
                    base_url=self.cfg.llm.base_url,
                )
            except Exception as e:
                print(f"Failed to initialize LLM client for display math: {e}")
                return md
        
        lines = md.splitlines()
        fixed_lines = []
        in_display_math = False
        math_lines = []
        fixed_count = 0
        # NOTE: We do NOT auto-number equations here. PDFs already have numbering and
        # hallucinated renumbering is worse than leaving it as-is.
        
        print("Fixing display math blocks...")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "$$":
                if in_display_math:
                    # End of display math block
                    math_text = "\n".join(math_lines)
                    
                    # Check if there are nested $ symbols or \[ \]
                    if enable_llm_fix and ('$' in math_text or '\\[' in math_text or '\\]' in math_text):
                        try:
                            prompt = f"""Fix this display math block. Remove any nested $ symbols and fix formatting.

Input:
{math_text}

Requirements:
1. Remove ALL $ symbols inside the math block (display math uses $$...$$, no $ inside)
2. Remove any \\[ or \\] symbols (use $$ only for display math)
3. Fix garbled characters (e.g., "ˆ" -> "\\hat", "C ( r )" -> "C(r)")
4. Use proper LaTeX syntax (e.g., "Z_{{t}} f" -> "\\int_{{t_0}}^{{t_1}}" where t_0 and t_1 are time bounds)
5. Return ONLY the cleaned math content without $$ or \\[ \\] delimiters

LaTeX:"""
                            
                            resp = self.llm_worker._llm_create(
                                messages=[
                                    {"role": "system", "content": "You are a LaTeX math expert. Fix display math blocks by removing nested $ symbols and fixing formatting."},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.0,
                                max_tokens=1000,
                            )
                            fixed_math = (resp.choices[0].message.content or "").strip()
                            # Remove $ if present
                            if fixed_math.startswith("$$") and fixed_math.endswith("$$"):
                                fixed_math = fixed_math[2:-2].strip()
                            elif fixed_math.startswith("$") and fixed_math.endswith("$"):
                                fixed_math = fixed_math[1:-1].strip()
                            # Remove markdown code fences
                            if fixed_math.startswith("```"):
                                fixed_math = re.sub(r'^```(?:\w+)?\n?', '', fixed_math)
                                fixed_math = re.sub(r'\n?```$', '', fixed_math)
                            
                            if fixed_math and fixed_math != math_text:
                                # Remove any \[ or \] if present (should use $$ only)
                                fixed_math = fixed_math.replace('\\[', '').replace('\\]', '')
                                fixed_lines.append("$$")
                                fixed_lines.append(fixed_math)
                                fixed_lines.append("$$")
                                fixed_count += 1
                            else:
                                # Just remove nested $ symbols and \[ \] manually
                                cleaned_math = math_text.replace('$', '').replace('\\[', '').replace('\\]', '')
                                fixed_lines.append("$$")
                                fixed_lines.append(cleaned_math)
                                fixed_lines.append("$$")
                        except Exception:
                            # Fallback: just remove nested $ symbols and \[ \]
                            cleaned_math = math_text.replace('$', '').replace('\\[', '').replace('\\]', '')
                            fixed_lines.append("$$")
                            fixed_lines.append(cleaned_math)
                            fixed_lines.append("$$")
                    else:
                        # Clean up: remove empty lines and fix formatting
                        cleaned_lines = [l for l in math_lines if l.strip()]
                        if cleaned_lines:
                            # Basic cleanup only: remove nested delimiters if they leaked into the block.
                            cleaned_lines = [ln.replace('$', '').replace('\\[', '').replace('\\]', '') for ln in cleaned_lines]
                            # Only add $$ if we actually have content
                            fixed_lines.append("$$")
                            fixed_lines.extend(cleaned_lines)
                            fixed_lines.append("$$")
                        else:
                            # Empty math block, skip it - don't add anything
                            pass
                    
                    in_display_math = False
                    math_lines = []
                else:
                    # Start of display math block
                    # Check if next line is also $$ (duplicate)
                    if i + 1 < len(lines) and lines[i + 1].strip() == "$$":
                        # Skip this $$, it's a duplicate - don't add it
                        continue
                    in_display_math = True
                    # Don't add $$ yet, wait for content
            elif in_display_math:
                math_lines.append(line)
            else:
                fixed_lines.append(line)
        
        if fixed_count > 0:
            print(f"Fixed {fixed_count} display math blocks")
        
        return "\n".join(fixed_lines)

    def _llm_light_cleanup(self, md: str) -> str:
        """Light LLM cleanup - only fix remaining mojibake."""
        if not self.cfg.llm:
            return md
        
        # Ensure LLM client is initialized
        if not hasattr(self.llm_worker, '_client') or not self.llm_worker._client:
            try:
                self.llm_worker._client = self.llm_worker._ensure_openai_class()(
                    api_key=self.cfg.llm.api_key,
                    base_url=self.cfg.llm.base_url,
                )
            except Exception as e:
                print(f"Failed to initialize LLM client: {e}")
                return md
        
        # Only process if there are obvious mojibake issues
        mojibake_patterns = ['ďŹ', 'Ď', 'Î´', 'Îą', 'âĽ', 'âĺ¤', 'â', 'ˆ', 'âĺ']
        has_mojibake = any(pattern in md for pattern in mojibake_patterns)
        
        if not has_mojibake:
            return md
        
        # Process in smaller chunks for speed
        max_chunk_size = 12000
        if len(md) <= max_chunk_size:
            return self._llm_cleanup_chunk(md)
        
        lines = md.splitlines()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1
            if current_size + line_size > max_chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size
        
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        
        repaired_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"LLM cleanup chunk {i+1}/{len(chunks)}...", end='\r')
            repaired = self._llm_cleanup_chunk(chunk)
            repaired_chunks.append(repaired)
        print()  # New line
        
        return "\n\n".join(repaired_chunks)
    
    def _llm_cleanup_chunk(self, md_chunk: str) -> str:
        """Light cleanup of a chunk - fix mojibake and improve inline formulas."""
        prompt = f"""Fix issues in this Markdown chunk:

1. FIX MOJIBAKE (garbled Unicode):
   - "ďŹ" -> "fi"
   - "Ď" -> "σ" or "τ"
   - "Î´" -> "δ"
   - "Îą" -> "α"
   - "â" -> correct symbol (—, ∈, ∥, ≤, ≥, etc.)
   - "ˆ" -> "\\hat" (in math context)

2. FIX INLINE FORMULAS (single $...$):
   - Fix garbled math: "ˆ C ( r ) - C ( r ) 2" -> "\\hat{{C}}(r) - C(r)^2"
   - Remove extra spaces: "C ( r )" -> "C(r)"
   - Fix subscripts/superscripts: "x 2" -> "x^2" or "x_2"
   - Use proper LaTeX: "ˆ" -> "\\hat", "α" -> "\\alpha"

3. PRESERVE:
   - All headings (do NOT change)
   - All display math blocks ($$...$$)
   - All tables, images, code blocks
   - Document structure

Return ONLY the fixed Markdown, no explanations.

INPUT:
{md_chunk}
"""
        
        try:
            resp = self.llm_worker._llm_create(
                messages=[
                    {"role": "system", "content": "You are a mojibake fixer. Only fix garbled characters."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=min(16384, self.cfg.llm.max_tokens),
            )
            result = resp.choices[0].message.content or md_chunk
            if result.startswith("```"):
                result = re.sub(r'^```(?:\w+)?\n', '', result)
                result = re.sub(r'\n```$', '', result)
            return result.strip()
        except Exception as e:
            print(f"LLM cleanup failed: {e}")
            return md_chunk

    def _llm_postprocess_markdown(self, md: str) -> str:
        """Use LLM to fix mojibake, formulas, and structure in the final markdown."""
        if not self.cfg.llm:
            return md
        
        # Ensure LLM client is initialized
        if not hasattr(self.llm_worker, '_client') or not self.llm_worker._client:
            try:
                self.llm_worker._client = self.llm_worker._ensure_openai_class()(
                    api_key=self.cfg.llm.api_key,
                    base_url=self.cfg.llm.base_url,
                )
            except Exception as e:
                print(f"Failed to initialize LLM client for post-processing: {e}")
                return md
        
        # Split into chunks if too long (LLM has token limits)
        max_chunk_size = 8000  # Leave room for prompt
        if len(md) <= max_chunk_size:
            return self._llm_repair_markdown_chunk(md)
        
        # Process in chunks
        lines = md.splitlines()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            if current_size + line_size > max_chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size
        
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        
        # Repair each chunk
        repaired_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"LLM post-processing chunk {i+1}/{len(chunks)}...")
            repaired = self._llm_repair_markdown_chunk(chunk)
            repaired_chunks.append(repaired)
        
        return "\n\n".join(repaired_chunks)
    
    def _llm_repair_markdown_chunk(self, md_chunk: str) -> str:
        """Repair a chunk of markdown using LLM."""
        prompt = f"""You are fixing a Markdown document converted from a PDF. The conversion has many errors that need to be fixed.

CRITICAL TASKS:

1. FIX ALL MOJIBAKE/GARBLED TEXT (This is the MOST IMPORTANT):
   - "ďŹ" -> "fi" (e.g., "ďŹexible" -> "flexible", "ďŹrst" -> "first", "ďŹxed" -> "fixed")
   - "Ď" -> "σ" (sigma) when in math context, or "τ" (tau) when appropriate
   - "Î´" -> "δ" (delta)
   - "Îą" -> "α" (alpha)
   - "â" -> various symbols: "âĽ" -> "∥" (norm), "âĺ¤" -> "≤" (leq), "âĺ¥" -> "≥" (geq), "â" -> "—" (em dash), "â" -> "∈" (element of)
   - "Ă" -> "×" (times)
   - "Ě" -> "≠" (not equal) or other symbols
   - Fix ALL garbled characters systematically

2. FIX ALL FORMULAS (CRITICAL):
   - Convert ALL Unicode math symbols to LaTeX:
     * α, β, γ, δ, ε, θ, λ, μ, π, σ, τ, φ, ω -> \\alpha, \\beta, \\gamma, \\delta, \\epsilon, \\theta, \\lambda, \\mu, \\pi, \\sigma, \\tau, \\phi, \\omega
     * ≤ -> \\leq, ≥ -> \\geq, ≠ -> \\neq, ∈ -> \\in, ∉ -> \\notin
     * ∑ -> \\sum, ∏ -> \\prod, ∫ -> \\int, ∞ -> \\infty
   - Fix subscripts: "\\tau 1" -> "\\tau_1", "x i" -> "x_i", "I j" -> "I_j"
   - Fix superscripts: "x 2" -> "x^2" when appropriate
   - Display math (block): use $$...$$ for equations that should be centered
   - Inline math: use $...$ for formulas within text
   - Merge split formulas that belong together
   - Fix spacing: "log k" -> "\\log k", "exp" -> "\\exp", etc.
   - Remove equation numbers from inside formulas

3. FIX HEADING STRUCTURE (CRITICAL):
   - "I. INTRODUCTION" -> "## I. INTRODUCTION" (H2, not H1)
   - "II. MAIN RESULT" -> "## II. MAIN RESULT"
   - "III. SEQUENTIAL..." -> "## III. SEQUENTIAL..."
   - "A. Structured..." -> "### A. Structured..."
   - "B. Sensing..." -> "### B. Sensing..."
   - "IV. PROOF..." -> "## IV. PROOF..."
   - "V. ACKNOWLEDGEMENTS" -> "## V. ACKNOWLEDGEMENTS"
   - "APPENDIX" -> "## APPENDIX" (H2)
   - "A. Proof of..." -> "### A. Proof of..." (H3)
   - "REFERENCES" -> "## REFERENCES" (H2)
   - Remove author names from headings (they should be plain text)
   - Remove duplicate headings completely (if you see the same heading twice, keep only the first occurrence)
   - Remove any heading that appears in the middle of a paragraph (headings should be on their own line)
   - Ensure proper hierarchy: H1 for title only, H2 for main sections (I, II, III, IV, V, APPENDIX, REFERENCES), H3 for subsections (A, B, C, etc.)

4. FIX TEXT CONTENT:
   - "â" -> "—" (em dash) in text
   - Fix all ligatures and special characters
   - Preserve mathematical notation in text

5. PRESERVE:
   - All tables (keep markdown table format)
   - All images (keep ![alt](path) format)
   - All code blocks (keep ``` format)
   - All citations [1], [2], etc.
   - All references

6. OUTPUT REQUIREMENTS:
   - Return ONLY the fixed Markdown
   - NO explanations, NO comments, NO code fences
   - Maintain original paragraph structure
   - Keep all content, just fix the errors
   - Remove duplicate headings (if the same heading appears twice, keep only the first occurrence)
   - Ensure headings appear in logical order (I, II, III, IV, V, then APPENDIX, then REFERENCES)

INPUT MARKDOWN (fix all errors):
{md_chunk}
"""
        
        try:
            resp = self.llm_worker._llm_create(
                messages=[
                    {"role": "system", "content": "You are an expert at fixing PDF-to-Markdown conversion errors, especially mojibake, formula formatting, and document structure."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=self.cfg.llm.max_tokens,
            )
            result = resp.choices[0].message.content or md_chunk
            # Remove any markdown code fences if LLM added them
            if result.startswith("```"):
                result = re.sub(r'^```(?:\w+)?\n', '', result)
                result = re.sub(r'\n```$', '', result)
            return result.strip()
        except Exception as e:
            print(f"LLM post-processing failed: {e}, using original")
            return md_chunk
    
    def _llm_final_quality_check(self, md: str) -> str:
        """Final comprehensive quality check for full_llm mode - ensure everything is perfect."""
        if not self.cfg.llm:
            return md
        
        # Ensure LLM client is initialized
        if not hasattr(self.llm_worker, '_client') or not self.llm_worker._client:
            try:
                self.llm_worker._client = self.llm_worker._ensure_openai_class()(
                    api_key=self.cfg.llm.api_key,
                    base_url=self.cfg.llm.base_url,
                )
            except Exception as e:
                print(f"Failed to initialize LLM client for final quality check: {e}")
                return md
        
        # Process in chunks if too long
        max_chunk_size = 12000
        if len(md) <= max_chunk_size:
            return self._llm_final_quality_check_chunk(md)
        
        # Split into chunks at logical boundaries (sections)
        lines = md.splitlines()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1
            # Start new chunk at major section boundaries (H2 headings)
            if line.strip().startswith('## ') and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            elif current_size + line_size > max_chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size
        
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        
        # Process each chunk
        checked_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"Final quality check chunk {i+1}/{len(chunks)}...", end='\r')
            checked = self._llm_final_quality_check_chunk(chunk)
            checked_chunks.append(checked)
        print()  # New line
        
        return "\n\n".join(checked_chunks)
    
    def _llm_final_quality_check_chunk(self, md_chunk: str) -> str:
        """Final comprehensive quality check for a chunk - ensure everything is perfect."""
        prompt = f"""You are performing a FINAL COMPREHENSIVE QUALITY CHECK on a Markdown document converted from PDF. 
This is the LAST pass to ensure EVERYTHING is PERFECT. Fix ALL remaining issues.

CRITICAL QUALITY REQUIREMENTS (must be PERFECT):

1. TITLE/HEADING STRUCTURE (MUST BE PERFECT):
   - Ensure proper hierarchy: H1 for title only, H2 for main sections (I, II, III, IV, V, APPENDIX, REFERENCES), H3 for subsections (A, B, C, etc.)
   - Remove any duplicate headings (keep only first occurrence)
   - Remove author names from headings (they should be plain text)
   - Ensure headings are on their own lines (not in middle of paragraphs)
   - Fix any misclassified headings (e.g., headings wrapped in $...$ should be proper markdown headings)
   - Ensure logical order: I, II, III, IV, V, then APPENDIX, then REFERENCES

2. FORMULAS - BOTH INLINE AND DISPLAY (MUST BE PERFECT):
   - Inline formulas ($...$): Must be proper LaTeX, no garbled Unicode, correct subscripts/superscripts
   - Display formulas ($$...$$): Must be proper LaTeX, no nested $ symbols, correct formatting
   - Fix ALL Unicode math symbols to LaTeX (α -> \\alpha, ≤ -> \\leq, etc.)
   - Remove equation numbers from inside formulas (they should be outside or use \\tag)
   - Merge split formulas that belong together
   - Fix spacing: "log k" -> "\\log k", "exp" -> "\\exp", etc.
   - Ensure proper grouping with braces

3. IMAGES (MUST BE PERFECT):
   - All images must have proper markdown syntax: ![Figure](./assets/filename.png)
   - Ensure figure captions are properly formatted (bold "Fig. X." prefix)
   - Remove any duplicate image references
   - Ensure images are on their own lines with blank lines around them

4. TABLES (MUST BE PERFECT):
   - All tables must have proper Markdown format with | separators
   - All rows must have the same number of columns
   - Must have proper header row and separator row (---)
   - Fix any garbled text in tables
   - Ensure proper alignment

5. REFERENCES (MUST BE PERFECT):
   - Each reference must start with [number] followed by a space
   - Format: [number] Author names. Title. Conference/Journal, pages, year.
   - Fix all garbled text and mojibake
   - Ensure proper spacing and punctuation
   - Each reference on a separate line
   - Remove duplicate references

6. BODY TEXT (MUST BE PERFECT):
   - Fix ALL mojibake/garbled text (ďŹ -> fi, Ď -> σ, Î´ -> δ, etc.)
   - Fix all ligatures and special characters
   - Ensure proper paragraph structure
   - Preserve mathematical notation in text
   - Fix em dashes: "â" -> "—"

7. OVERALL STRUCTURE:
   - Ensure proper spacing between sections
   - Remove any empty or duplicate content
   - Ensure logical flow

OUTPUT REQUIREMENTS:
- Return ONLY the perfected Markdown
- NO explanations, NO comments, NO code fences
- Maintain all content, just fix errors and improve quality
- Ensure EVERYTHING is perfect - this is the final pass

INPUT MARKDOWN (make it PERFECT):
{md_chunk}
"""
        
        try:
            resp = self.llm_worker._llm_create(
                messages=[
                    {"role": "system", "content": "You are an expert at perfecting PDF-to-Markdown conversions. This is the final quality check - ensure EVERYTHING is perfect."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=self.cfg.llm.max_tokens,
            )
            result = resp.choices[0].message.content or md_chunk
            # Remove any markdown code fences if LLM added them
            if result.startswith("```"):
                result = re.sub(r'^```(?:\w+)?\n', '', result)
                result = re.sub(r'\n```$', '', result)
            return result.strip()
        except Exception as e:
            print(f"Final quality check failed: {e}, using original")
            return md_chunk
    
    def _get_speed_mode_config(self, speed_mode: str, total_pages: int) -> dict:
        """Get configuration for speed mode."""
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        
        configs = {
            'normal': {
                # 普通模式：截图识别，最大并行度
                'max_parallel_pages': min(64, max(32, cpu_count * 4), total_pages),  # 最大页面并行数，至少32
                'max_inflight': 64,  # 最大并发请求数，提高以支持更多并行
                'dpi': 160,  # DPI设置
                'compress': 3,  # 图片压缩级别
                'max_tokens': 3072,  # Max tokens
            },
            'ultra_fast': {
                # 超快模式：截图识别，降低质量换取速度
                'max_parallel_pages': min(32, max(16, cpu_count * 2), total_pages),
                'max_inflight': 32,  # 提高以支持更多并行
                'dpi': 150,  # 更低DPI
                'compress': 5,  # 更高压缩
                'max_tokens': 2048,  # 更少tokens
            },
            'no_llm': {
                # 无LLM模式：不使用（此配置不会被使用，因为no_llm走不同路径）
                'max_parallel_pages': min(8, cpu_count),
                'max_inflight': 1,
                'dpi': 200,
                'compress': 0,
                'max_tokens': 0,
            }
        }
        
        # 默认使用 normal 模式
        return configs.get(speed_mode, configs['normal'])
