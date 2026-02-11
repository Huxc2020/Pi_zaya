from __future__ import annotations

from pathlib import Path
from threading import Lock
import time
from typing import Any


def enqueue(state: dict[str, Any], lock: Lock, task: dict[str, Any]) -> None:
    with lock:
        if (not bool(state.get("running"))) and (not state.get("queue")):
            state["done"] = 0
            state["total"] = 0
            state["last"] = ""
        state.setdefault("queue", []).append(task)
        state["total"] = int(state.get("total", 0)) + 1


def remove_queued_tasks_for_pdf(state: dict[str, Any], lock: Lock, pdf_path: Path) -> int:
    target = str(Path(pdf_path))
    removed = 0
    with lock:
        queue = list(state.get("queue") or [])
        kept: list[dict[str, Any]] = []
        for task in queue:
            try:
                if str(task.get("pdf") or "") == target:
                    removed += 1
                else:
                    kept.append(task)
            except Exception:
                kept.append(task)
        state["queue"] = kept
        done = int(state.get("done", 0) or 0)
        total = int(state.get("total", 0) or 0) - int(removed)
        state["total"] = max(done, total)
    return removed


def cancel_all(state: dict[str, Any], lock: Lock, message: str) -> None:
    with lock:
        state["cancel"] = True
        state["cur_page_msg"] = message


def snapshot(state: dict[str, Any], lock: Lock) -> dict[str, Any]:
    with lock:
        snap = dict(state)
        try:
            snap["queue"] = list(state.get("queue") or [])
        except Exception:
            snap["queue"] = []
        return snap


def begin_next_task_or_idle(state: dict[str, Any], lock: Lock) -> dict[str, Any] | None:
    with lock:
        if state.get("cancel"):
            state.setdefault("queue", []).clear()
            state["running"] = False
            state["current"] = ""
            state["cancel"] = False
            state["total"] = int(state.get("done", 0))

        queue = state.get("queue") or []
        if queue:
            task = queue.pop(0)
            state["running"] = True
            state["current"] = str(task.get("name") or "")
            state["cur_task_id"] = str(task.get("_tid") or "")
            state["cur_page_done"] = 0
            state["cur_page_total"] = 0
            state["cur_page_msg"] = ""
            state["cur_profile"] = ""
            state["cur_llm_profile"] = ""
            state["cur_log_tail"] = []
            return task

        state["running"] = False
        state["current"] = ""
        state["cur_task_id"] = ""
        state["cur_page_done"] = 0
        state["cur_page_total"] = 0
        state["cur_page_msg"] = ""
        state["cur_profile"] = ""
        state["cur_llm_profile"] = ""
        state["cur_log_tail"] = []
        return None


def update_page_progress(
    state: dict[str, Any],
    lock: Lock,
    page_done: int,
    page_total: int,
    msg: str = "",
    *,
    task_id: str = "",
) -> None:
    with lock:
        cur_tid = str(state.get("cur_task_id") or "")
        tid = str(task_id or "")
        if tid:
            if (not cur_tid) or (tid != cur_tid):
                # Ignore stale updates from older/foreign worker threads.
                return

        old_done = int(state.get("cur_page_done", 0) or 0)
        old_total = int(state.get("cur_page_total", 0) or 0)
        new_done = max(0, int(page_done or 0))
        new_total = max(0, int(page_total or 0))

        # Keep UI progress monotonic within one task.
        total = max(old_total, new_total)
        done = max(old_done, new_done)
        if total > 0:
            done = min(done, total)
        state["cur_page_done"] = int(done)
        state["cur_page_total"] = int(total)
        line = str(msg or "")[:220]
        is_profile = line.startswith("converter profile:") or line.startswith("LLM concurrency:")
        regressed = (new_done < old_done) and (new_total <= old_total) and (not is_profile)
        if regressed:
            # Keep a stable message when stale lines arrive out of order.
            line = str(state.get("cur_page_msg") or "")
        state["cur_page_msg"] = line

        if line.startswith("converter profile:"):
            state["cur_profile"] = line
            state["cur_profile_ts"] = float(time.time())
        elif line.startswith("LLM concurrency:"):
            state["cur_llm_profile"] = line
            state["cur_llm_profile_ts"] = float(time.time())

        tail = list(state.get("cur_log_tail") or [])
        if line and (not regressed):
            tail.append(line)
            if len(tail) > 24:
                tail = tail[-24:]
        state["cur_log_tail"] = tail


def should_cancel(state: dict[str, Any], lock: Lock) -> bool:
    with lock:
        return bool(state.get("cancel"))


def finish_task(state: dict[str, Any], lock: Lock, message: str, *, task_id: str = "") -> None:
    with lock:
        cur_tid = str(state.get("cur_task_id") or "")
        tid = str(task_id or "")
        if tid:
            if (not cur_tid) or (tid != cur_tid):
                return
        state["done"] = int(state.get("done", 0)) + 1
        done = int(state.get("done", 0) or 0)
        total = int(state.get("total", 0) or 0)
        if done > total:
            state["total"] = done
        state["last"] = message
        state["cur_task_id"] = ""


def is_running_snapshot(snap: dict[str, Any]) -> bool:
    if bool(snap.get("running")):
        return True
    if str(snap.get("current") or "").strip():
        return True
    if list(snap.get("queue") or []):
        return True
    return False
