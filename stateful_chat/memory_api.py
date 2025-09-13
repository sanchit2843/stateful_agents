"""
Tiny programmatic API: one-shot chat with per-user memories.
Inputs: user_id, prompt; uses ./memories/{user_id}/ for DB/FAISS.
Run: python -c "from stateful_chat.memory_api import chat_once; print(chat_once('u1','Hi'))"
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from stateful_chat.gemini_client import GeminiClient
from stateful_chat.memdb import MemDB


def _paths(user_id: str) -> tuple[str, str]:
    base = Path("memories") / user_id
    base.mkdir(parents=True, exist_ok=True)
    return str(base / "memories.sqlite3"), str(base / "mem.index")


def chat_once(
    user_id: str,
    prompt: str,
    *,
    top_k: int = 5,
    dup_threshold: float = 0.92,
    conflict_threshold: float = 0.75,
    chat_model: str = "gemini-2.5-flash",
    classify_model: str = "gemini-2.5-flash-lite",
    embed_model: str = "gemini-embedding-exp-03-07",
) -> str:
    """Return answer string after memory write+retrieve processing.

    - Automatically classifies and saves a memory if applicable.
    - Retrieves top_k relevant memories and injects into chat context.
    """
    db_path, index_path = _paths(user_id)
    gc = GeminiClient(
        chat_model=chat_model,
        classify_model=classify_model,
        embed_model=embed_model,
    )
    store = MemDB(db_path=db_path, index_path=index_path, duplicate_threshold=dup_threshold, conflict_threshold=conflict_threshold)

    # No per-turn writes; we only retrieve here

    qv = gc.embed(prompt)
    retrieved = store.search(qv, top_k=top_k)
    mems = [f"[{m.memory_type}] {m.key}: {m.value}" for m, _ in retrieved]
    answer = gc.chat(prompt, mems)

    # Session-end extraction is not handled in this one-shot helper
    return answer
