from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from kb.reference_index import build_reference_index

_LOCK = threading.Lock()
_STATE: dict[str, Any] = {}


def snapshot() -> dict[str, Any]:
    with _LOCK:
        return dict(_STATE)


def is_running_snapshot(snap: dict[str, Any] | None = None) -> bool:
    s = snap if snap is not None else snapshot()
    return str(s.get("status") or "").strip().lower() == "running"


def start_reference_sync(
    *,
    src_root: str | Path,
    db_dir: str | Path,
    incremental: bool = True,
    enable_title_lookup: bool = True,
    crossref_time_budget_s: float = 45.0,
    pdf_root: str | Path | None = None,
    library_db_path: str | Path | None = None,
) -> dict[str, object]:
    with _LOCK:
        if str(_STATE.get("status") or "").lower() == "running":
            return {"already_running": True}
        run_id = int(_STATE.get("run_id", 0) or 0) + 1
        _STATE.update(
            run_id=run_id, status="running", message="",
            error="", docs_done=0, docs_total=0,
            current="", stage="init",
            started_at=time.time(),
        )

    def _progress(info: dict[str, Any]) -> None:
        with _LOCK:
            _STATE.update({k: v for k, v in info.items() if k != "run_id"})

    def _worker() -> None:
        try:
            build_reference_index(
                src_root=Path(src_root),
                db_dir=Path(db_dir),
                incremental=incremental,
                enable_title_lookup=enable_title_lookup,
                crossref_time_budget_s=crossref_time_budget_s,
                pdf_root=Path(pdf_root) if pdf_root else None,
                library_db_path=Path(library_db_path) if library_db_path else None,
                progress_cb=_progress,
            )
            with _LOCK:
                _STATE["status"] = "done"
                _STATE["message"] = "参考文献索引同步完成。"
        except Exception as exc:
            with _LOCK:
                _STATE["status"] = "error"
                _STATE["error"] = str(exc)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return {"run_id": _STATE.get("run_id", 0)}
