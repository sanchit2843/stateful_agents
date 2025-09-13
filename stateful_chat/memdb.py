"""
SQLite metadata + FAISS vector index for memories.
Creates/loads DB and index; adds/list/deletes memories; searches top-k.
Run: python stateful_chat/mem_cli.py list u1
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _norm_key(s: str) -> str:
    return "_".join(" ".join(s.lower().strip().split()).split(" "))


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


@dataclass
class Memory:
    id: int
    created_at: str
    memory_type: str
    key: str
    value: str
    expires_at: Optional[str]


class MemDB:
    def __init__(
        self,
        db_path: str,
        index_path: str,
        duplicate_threshold: float = 0.75,
        conflict_threshold: float = 0.75,
    ) -> None:
        self.db_path = Path(db_path)
        self.index_path = Path(index_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        self.dup_thr = float(duplicate_threshold)
        self.conf_thr = float(conflict_threshold)
        self.conn = sqlite3.connect(str(self.db_path))
        self._create_schema()

        # Lazy-build FAISS index; load if exists
        self.index = None
        self.embed_dim = self._get_meta("embed_dim")
        if self.index_path.exists():
            import faiss  # type: ignore

            self.index = faiss.read_index(str(self.index_path))
            self.embed_dim = str(self.index.d)
        self._faiss = None

    # ------------ schema/meta ------------
    def _create_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              created_at TEXT NOT NULL,
              memory_type TEXT NOT NULL,
              key TEXT NOT NULL,
              value TEXT NOT NULL,
              sha1 TEXT NOT NULL UNIQUE,
              expires_at TEXT,
              seen_count INTEGER DEFAULT 1,
              last_seen_at TEXT,
              active INTEGER DEFAULT 1
            );
            """
        )
        # Upgrade path: add columns if missing
        cur.execute("PRAGMA table_info('memories')")
        cols = {r[1] for r in cur.fetchall()}
        if "expires_at" not in cols:
            cur.execute("ALTER TABLE memories ADD COLUMN expires_at TEXT")
        if "seen_count" not in cols:
            cur.execute("ALTER TABLE memories ADD COLUMN seen_count INTEGER DEFAULT 1")
        if "last_seen_at" not in cols:
            cur.execute("ALTER TABLE memories ADD COLUMN last_seen_at TEXT")
        if "active" not in cols:
            cur.execute("ALTER TABLE memories ADD COLUMN active INTEGER DEFAULT 1")
        # Backfill NULLs for newly added columns
        cur.execute("UPDATE memories SET active=1 WHERE active IS NULL")
        cur.execute("UPDATE memories SET seen_count=1 WHERE seen_count IS NULL")
        cur.execute("UPDATE memories SET last_seen_at=COALESCE(last_seen_at, created_at) WHERE last_seen_at IS NULL")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def _get_meta(self, key: str) -> Optional[str]:
        cur = self.conn.cursor()
        cur.execute("SELECT value FROM meta WHERE key=?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self.conn.commit()

    def _ensure_faiss(self):
        if self._faiss is None:
            import faiss  # type: ignore

            self._faiss = faiss
        return self._faiss

    def _ensure_index(self, dim: int) -> None:
        f = self._ensure_faiss()
        if self.index is None:
            self.index = f.IndexIDMap(f.IndexFlatIP(dim))
            self._set_meta("embed_dim", str(dim))

    # ------------ operations ------------
    def list_memories(self, limit: int = 100) -> List[Memory]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id, created_at, memory_type, key, value, expires_at FROM memories ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [Memory(*row) for row in cur.fetchall()]

    def delete_memory(self, mid: int) -> bool:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM memories WHERE id=?", (mid,))
        removed = cur.rowcount > 0
        self.conn.commit()
        if removed and self.index is not None:
            f = self._ensure_faiss()
            ids = np.array([mid], dtype=np.int64)
            self.index.remove_ids(ids)
            f.write_index(self.index, str(self.index_path))
        return removed

    def clear(self) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM memories")
        self.conn.commit()
        if self.index is not None:
            dim = self.index.d
            f = self._ensure_faiss()
            self.index = f.IndexIDMap(f.IndexFlatIP(dim))
            f.write_index(self.index, str(self.index_path))

    def _insert_row(self, memory_type: str, key: str, value: str, sha1: str, expires_at: Optional[str]) -> int:
        cur = self.conn.cursor()
        now = datetime.utcnow().isoformat()
        cur.execute(
            "INSERT INTO memories(created_at, memory_type, key, value, sha1, expires_at, seen_count, last_seen_at, active) VALUES(?,?,?,?,?,?,?,?,?)",
            (now, memory_type, key, value, sha1, expires_at, 1, now, 1),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def add_memory(
        self, memory_type: str, key: str, value: str, embedding: np.ndarray, expires_at: Optional[str] = None
    ) -> Tuple[int, bool, str]:
        mt = memory_type.lower().strip()
        k = _norm_key(key)
        v = " ".join(value.strip().split())
        sig = _sha1(f"{mt}|{k}|{v}")
        # Exact duplicate
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM memories WHERE sha1=?", (sig,))
        row = cur.fetchone()
        if row:
            mid = int(row[0])
            cur.execute(
                "UPDATE memories SET seen_count = COALESCE(seen_count,1)+1, last_seen_at=? WHERE id=?",
                (datetime.utcnow().isoformat(), mid),
            )
            self.conn.commit()
            return mid, False, "exact-duplicate"
        # Near-duplicate (same type+key; cosine >= thr) — check top-K neighbors
        if self.index is not None and self.index.ntotal > 0:
            q = embedding.astype(np.float32).reshape(1, -1)
            k_neighbors = int(min(10, max(1, self.index.ntotal)))
            D, I = self.index.search(q, k_neighbors)
            best_sim = -1.0
            best_id = -1
            for idx, sim in zip(I[0], D[0]):
                hit_id = int(idx)
                if hit_id == -1:
                    continue
                cur.execute(
                    "SELECT memory_type, key FROM memories WHERE id=?",
                    (hit_id,),
                )
                r2 = cur.fetchone()
                if r2 and r2[0].lower() == mt and _norm_key(r2[1]) == k:
                    if float(sim) > best_sim:
                        best_sim = float(sim)
                        best_id = hit_id
            if best_id != -1 and best_sim >= self.dup_thr:
                cur.execute(
                    "UPDATE memories SET seen_count = COALESCE(seen_count,1)+1, last_seen_at=?, active=1 WHERE id=?",
                    (datetime.utcnow().isoformat(), best_id),
                )
                self.conn.commit()
                return best_id, False, "near-duplicate"
            # If there are existing active rows for same type+key but similarity is low, deactivate them
            cur.execute("SELECT id FROM memories WHERE memory_type=? AND key=? AND active=1", (mt, k))
            stale_ids = [int(r[0]) for r in cur.fetchall()]
            if stale_ids and (best_sim == -1.0 or best_sim <= self.conf_thr):
                cur.execute("UPDATE memories SET active=0 WHERE memory_type=? AND key=?", (mt, k))
                self.conn.commit()
                if self.index is not None:
                    f = self._ensure_faiss()
                    self.index.remove_ids(np.array(stale_ids, dtype=np.int64))
                    f.write_index(self.index, str(self.index_path))
        # Insert and add to FAISS
        d = int(embedding.shape[0])
        if self.index is None:
            self._ensure_index(d)
        assert d == self.index.d, "Embedding dim mismatch"
        new_id = self._insert_row(mt, k, v, sig, expires_at)
        ids = np.array([new_id], dtype=np.int64)
        self.index.add_with_ids(embedding.reshape(1, -1).astype(np.float32), ids)
        f = self._ensure_faiss()
        f.write_index(self.index, str(self.index_path))
        return new_id, True, "inserted"

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[Memory, float]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        q = query_embedding.astype(np.float32).reshape(1, -1)
        assert q.shape[1] == self.index.d, "Embedding dim mismatch"
        D, I = self.index.search(q, top_k)
        ids = [int(i) for i in I[0] if i != -1]
        if not ids:
            return []
        cur = self.conn.cursor()
        qmarks = ",".join(["?"] * len(ids))
        cur.execute(
            f"SELECT id, created_at, memory_type, key, value, expires_at FROM memories WHERE id IN ({qmarks}) AND active=1",
            ids,
        )
        rows = {int(r[0]): Memory(*r) for r in cur.fetchall()}
        out: List[Tuple[Memory, float]] = []
        today = datetime.utcnow().date().isoformat()
        for i, d in zip(I[0], D[0]):
            iid = int(i)
            if iid in rows:
                m = rows[iid]
                if m.expires_at is not None and m.expires_at < today:
                    continue
                out.append((m, float(d)))
        return out
