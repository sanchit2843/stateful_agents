"""
Simple interaction CLI: per-user RAG chat with two memory modes.
Inputs: user_id, prompt, optional --mem-mode {session|user}.
Run: python chat_cli.py u1 "I love matcha latte" --mem-mode=user
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stateful_chat.gemini_client import GeminiClient
from stateful_chat.memdb import MemDB


def _paths(user_id: str) -> tuple[str, str]:
    base = Path("memories") / user_id
    base.mkdir(parents=True, exist_ok=True)
    return str(base / "memories.sqlite3"), str(base / "mem.index")


def _mem_lines(items):
    return [f"[{m.memory_type}] {m.key}: {m.value}" for m, _ in items]


def main() -> None:
    p = argparse.ArgumentParser(description="Chat with per-user memories")
    p.add_argument("user_id")
    p.add_argument("prompt")
    p.add_argument("--mem-mode", choices=["session", "user"], default="session")
    args = p.parse_args()
    user_id, prompt, mem_mode = args.user_id, args.prompt, args.mem_mode

    db_path, index_path = _paths(user_id)
    gc = GeminiClient()
    store = MemDB(db_path=db_path, index_path=index_path, duplicate_threshold=0.92, conflict_threshold=0.75)
    history: list[dict] = []

    def handle(msg: str) -> str:
        qv = gc.embed(msg)
        retrieved = store.search(qv, top_k=5)
        answer = gc.chat_with_history(history, msg, _mem_lines(retrieved))
        # append this turn to history
        user_payload = ("User: " + msg)
        history.append({"role": "user", "parts": [user_payload]})
        history.append({"role": "model", "parts": [answer]})
        # Per-turn user-only memory extraction (after reply)
        if mem_mode == "user":
            cls = gc.classify_memory(msg)
            saved = merged = 0
            for m in cls:
                if not bool(m.get("should_write")):
                    continue
                et = " ".join(str(m["value"]).split())
                triple = f"{m['memory_type']} | {m['key']} | {et}"
                ev = gc.embed(triple)
                expires_at = m.get("expires_at")
                _id, inserted, _reason = store.add_memory(
                    str(m["memory_type"]), str(m["key"]), et, ev, expires_at=expires_at if expires_at else None
                )
                if inserted:
                    saved += 1
                else:
                    merged += 1
            if saved or merged:
                print(f"[memories] Saved {saved} (merged/updated {merged})")
        return answer

    print(handle(prompt))
    try:
        while True:
            nxt = input("You: ").strip()
            if not nxt or nxt.lower() in {"exit", "quit"}:
                break
            print(handle(nxt))
    except KeyboardInterrupt:
        pass
    # Session-end extraction from full transcript (both roles)
    if history and mem_mode == "session":
        transcript_lines: list[str] = []
        for item in history:
            role = item.get("role", "user")
            text = "\n".join([str(p) for p in item.get("parts", [])])
            transcript_lines.append(("User: " if role == "user" else "Assistant: ") + text)
        transcript = "\n".join(transcript_lines)
        extracted = gc.extract_memories(transcript, max_items=10)
        saved = merged = 0
        for m in extracted:
            if not bool(m.get("should_write")):
                continue
            et = " ".join(str(m["value"]).split())
            triple = f"{m['memory_type']} | {m['key']} | {et}"
            ev = gc.embed(triple)
            expires_at = m.get("expires_at")
            _id, inserted, reason = store.add_memory(
                str(m["memory_type"]), str(m["key"]), et, ev, expires_at=expires_at if expires_at else None
            )
            if inserted:
                saved += 1
            else:
                merged += 1
        print(f"[memories] Saved {saved} (merged/updated {merged})")


if __name__ == "__main__":
    main()
