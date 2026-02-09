from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_prefs(path: Path) -> dict[str, Any]:
    """Tiny local persistence for UI settings (paths, etc)."""
    try:
        p = Path(path)
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_prefs(path: Path, prefs: dict[str, Any]) -> None:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(prefs, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return
