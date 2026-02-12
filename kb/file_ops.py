from __future__ import annotations

import hashlib
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import Optional

def _resolve_md_output_paths(out_root: Path, pdf_path: Path) -> tuple[Path, Path, bool]:
    pdf = Path(pdf_path)
    md_folder = Path(out_root) / pdf.stem
    md_main = md_folder / f"{pdf.stem}.en.md"
    md_exists = md_main.exists()
    if (not md_exists) and md_folder.exists():
        try:
            any_md = next(iter(sorted(md_folder.glob("*.md"))), None)
            if any_md:
                md_main = any_md
                md_exists = True
        except Exception:
            pass
    return md_folder, md_main, md_exists

def _next_pdf_dest_path(pdf_dir: Path, base_name: str, *, max_suffix: int = 100) -> Path:
    dest_pdf = Path(pdf_dir) / f"{base_name}.pdf"
    if not dest_pdf.exists():
        return dest_pdf
    k = 2
    while (Path(pdf_dir) / f"{base_name}-{k}.pdf").exists() and k < int(max_suffix):
        k += 1
    return Path(pdf_dir) / f"{base_name}-{k}.pdf"

def _persist_upload_pdf(tmp_path: Path, dest_pdf: Path, data: bytes) -> None:
    try:
        if tmp_path.exists() and tmp_path.resolve() != dest_pdf.resolve():
            tmp_path.replace(dest_pdf)
            return
    except Exception:
        pass
    dest_pdf.write_bytes(data)

def _write_tmp_upload(pdf_dir: Path, filename: str, data: bytes) -> Path:
    stem = (Path(filename).stem or "upload").strip() or "upload"
    tmp = pdf_dir / f"__upload__{stem}.pdf"
    tmp.write_bytes(data)
    return tmp

def _sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()

def _pick_directory_dialog(initial_dir: str) -> Optional[str]:
    """
    Open a native folder picker on the local machine.
    """
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        sel = filedialog.askdirectory(initialdir=initial_dir or None, title="选择目录")
        try:
            root.destroy()
        except Exception:
            pass
        sel = (sel or "").strip()
        return sel or None
    except Exception:
        return None

def _cleanup_tmp_uploads(pdf_dir: Path) -> int:
    n = 0
    try:
        for p in Path(pdf_dir).glob("__upload__*.pdf"):
            try:
                p.unlink()
                n += 1
            except Exception:
                pass
    except Exception:
        pass
    return n

def _list_pdf_paths_fast(pdf_dir: Path) -> list[Path]:
    """
    Fast non-recursive PDF listing.
    Avoids per-file Path.stat() calls (important on large folders / slow disks).
    """
    pdf_dir = Path(pdf_dir)
    out: list[Path] = []
    try:
        with os.scandir(pdf_dir) as it:
            for e in it:
                try:
                    if not e.is_file():
                        continue
                    if not e.name.lower().endswith(".pdf"):
                        continue
                    out.append(Path(e.path))
                except Exception:
                    continue
    except Exception:
        # Fallback for unusual FS errors.
        try:
            out = [x for x in pdf_dir.glob("*.pdf") if x.is_file()]
        except Exception:
            out = []
    return out
