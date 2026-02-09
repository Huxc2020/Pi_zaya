from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path


class ChatStore:
    """
    A tiny local chat persistence layer.
    - One sqlite file
    - Multiple conversations
    - Append-only messages
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        # WAL helps concurrent reads while Streamlit reruns.
        conn = sqlite3.connect(str(self._db_path), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                  id TEXT PRIMARY KEY,
                  title TEXT NOT NULL,
                  created_at REAL NOT NULL,
                  updated_at REAL NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  conv_id TEXT NOT NULL,
                  role TEXT NOT NULL,
                  content TEXT NOT NULL,
                  created_at REAL NOT NULL,
                  FOREIGN KEY(conv_id) REFERENCES conversations(id)
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv_id ON messages(conv_id);")

    def create_conversation(self, title: str = "新对话") -> str:
        conv_id = uuid.uuid4().hex
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conv_id, title.strip() or "新对话", now, now),
            )
        return conv_id

    def list_conversations(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_conversation(self, conv_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE conv_id = ?", (conv_id,))
            conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))

    def get_messages(self, conv_id: str, limit: int | None = None) -> list[dict]:
        sql = "SELECT role, content, created_at FROM messages WHERE conv_id = ? ORDER BY id ASC"
        params: tuple = (conv_id,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (conv_id, int(limit))

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def append_message(self, conv_id: str, role: str, content: str) -> None:
        role = (role or "").strip()
        if role not in ("user", "assistant", "system"):
            role = "user"
        content = (content or "").strip()
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (conv_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (conv_id, role, content, now),
            )
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id))

    def set_title_if_default(self, conv_id: str, new_title: str) -> None:
        new_title = (new_title or "").strip()
        if not new_title:
            return
        new_title = new_title.replace("\n", " ").strip()
        new_title = new_title[:80]

        with self._connect() as conn:
            row = conn.execute("SELECT title FROM conversations WHERE id = ?", (conv_id,)).fetchone()
            if not row:
                return
            if (row["title"] or "").strip() not in ("新对话", ""):
                return
            now = time.time()
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (new_title, now, conv_id),
            )

