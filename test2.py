# -*- coding: utf-8 -*-
"""Compatibility shim.

Deprecated entrypoint kept for older scripts. Use `pdf_to_md.py` directly.
"""

from pdf_to_md import main


if __name__ == "__main__":
    print("[DEPRECATED] test2.py has been renamed to pdf_to_md.py. Please update your scripts.", flush=True)
    main()
