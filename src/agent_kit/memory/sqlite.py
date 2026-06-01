from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .keyword import _metadata_matches, _tokenize
from .types import MemoryItem, MemorySearchResult


class SqliteMemoryStore:
    """Persistent memory store backed by SQLite.

    All database I/O runs on a dedicated single-thread executor so async
    callers are never blocked on the event loop thread.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agentkit_sqlite")
        # Create the connection on the executor thread; all subsequent calls
        # also run there, so check_same_thread is satisfied.
        self._conn: sqlite3.Connection = self._executor.submit(self._create_db).result()

    def _create_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                namespace TEXT NOT NULL,
                id TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                created_at REAL,
                updated_at REAL,
                PRIMARY KEY (namespace, id)
            )
            """
        )
        conn.commit()
        return conn

    async def upsert(self, items: list[MemoryItem], **kwargs: Any) -> None:
        now = time.time()
        rows = []
        for item in items:
            created_at = item.created_at if item.created_at is not None else now
            updated_at = now
            item.created_at = created_at
            item.updated_at = updated_at
            rows.append(
                (
                    item.namespace or "",
                    item.id,
                    item.content,
                    json.dumps(item.metadata, sort_keys=True),
                    created_at,
                    updated_at,
                )
            )
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._sync_upsert, rows)

    def _sync_upsert(self, rows: list[tuple[Any, ...]]) -> None:
        self._conn.executemany(
            """
            INSERT INTO memories(namespace, id, content, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(namespace, id) DO UPDATE SET
                content=excluded.content,
                metadata=excluded.metadata,
                updated_at=excluded.updated_at
            """,
            rows,
        )
        self._conn.commit()

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        namespace: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[MemorySearchResult]:
        query_terms = _tokenize(query)
        if not query_terms or limit <= 0:
            return []

        loop = asyncio.get_running_loop()
        raw_rows: list[dict[str, Any]] = await loop.run_in_executor(
            self._executor, self._sync_fetch, namespace
        )

        results: list[MemorySearchResult] = []
        for row in raw_rows:
            metadata = json.loads(row["metadata"] or "{}")
            if metadata_filter and not _metadata_matches(metadata, metadata_filter):
                continue
            item = MemoryItem(
                id=row["id"],
                content=row["content"],
                metadata=metadata,
                namespace=row["namespace"] or None,
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            item_terms = _tokenize(item.content)
            overlap = query_terms & item_terms
            if not overlap:
                continue
            score = len(overlap) / len(query_terms)
            results.append(
                MemorySearchResult(
                    item=item,
                    score=score,
                    metadata={"matched_terms": sorted(overlap)},
                )
            )

        results.sort(key=lambda result: (result.score or 0.0, result.item.id), reverse=True)
        return results[:limit]

    def _sync_fetch(self, namespace: str | None) -> list[dict[str, Any]]:
        if namespace is None:
            cursor = self._conn.execute("SELECT * FROM memories")
        else:
            cursor = self._conn.execute(
                "SELECT * FROM memories WHERE namespace = ?",
                (namespace or "",),
            )
        return [dict(row) for row in cursor.fetchall()]

    def close(self) -> None:
        self._executor.submit(self._conn.close).result()
        self._executor.shutdown(wait=False)

    def __enter__(self) -> SqliteMemoryStore:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
