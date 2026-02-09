# -*- coding: utf-8 -*-
from __future__ import annotations

import sqlite3
import time
from pathlib import Path


class LibraryStore:
    """
    Minimal PDF library index:
    - keyed by sha1 to detect duplicates quickly
    - stores final pdf path and created_at
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pdf_files (
                  sha1 TEXT PRIMARY KEY,
                  path TEXT NOT NULL,
                  created_at REAL NOT NULL
                );
                """
            )

    def get_by_sha1(self, sha1: str) -> dict | None:
        sha1 = (sha1 or "").strip().lower()
        if not sha1:
            return None
        with self._connect() as conn:
            row = conn.execute("SELECT sha1, path, created_at FROM pdf_files WHERE sha1 = ?", (sha1,)).fetchone()
        return dict(row) if row else None

    def upsert(self, sha1: str, path: Path) -> None:
        sha1 = (sha1 or "").strip().lower()
        path_s = str(Path(path))
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO pdf_files (sha1, path, created_at) VALUES (?, ?, ?) "
                "ON CONFLICT(sha1) DO UPDATE SET path=excluded.path",
                (sha1, path_s, now),
            )

