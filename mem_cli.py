"""
Memory management CLI: list/delete/clear per-user memories.
Inputs: user_id (+ subcommand flags). Prints plain text.
Run: python stateful_chat/mem_cli.py list u1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stateful_chat.memdb import MemDB


def _paths(user_id: str) -> tuple[str, str]:
    base = Path("memories") / user_id
    base.mkdir(parents=True, exist_ok=True)
    return str(base / "memories.sqlite3"), str(base / "mem.index")


def cmd_list(args) -> None:
    db, idx = _paths(args.user_id)
    store = MemDB(db_path=db, index_path=idx)
    for m in store.list_memories(limit=args.limit):
        print(f"{m.id}\t{m.created_at}\t[{m.memory_type}] {m.key}: {m.value}, {m.expires_at}")


def cmd_delete(args) -> None:
    db, idx = _paths(args.user_id)
    store = MemDB(db_path=db, index_path=idx)
    ok = store.delete_memory(args.id)
    print("deleted" if ok else "not found")


def cmd_clear(args) -> None:
    db, idx = _paths(args.user_id)
    store = MemDB(db_path=db, index_path=idx)
    assert args.yes, "Pass --yes to confirm clearing all memories"
    store.clear()
    print("cleared")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mem", description="Per-user memory management")
    sub = p.add_subparsers(dest="cmd", required=True)

    l = sub.add_parser("list", help="List memories")
    l.add_argument("user_id")
    l.add_argument("--limit", type=int, default=100)
    l.set_defaults(func=cmd_list)

    d = sub.add_parser("delete", help="Delete a memory by id")
    d.add_argument("user_id")
    d.add_argument("--id", type=int, required=True)
    d.set_defaults(func=cmd_delete)

    c = sub.add_parser("clear", help="Clear all memories for user")
    c.add_argument("user_id")
    c.add_argument("--yes", action="store_true")
    c.set_defaults(func=cmd_clear)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
