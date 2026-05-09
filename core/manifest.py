"""File-level manifest of everything we've ever scanned in CHATS_DIR.

Stores per-file: is_chat, mtime, size, hash, checked_at, chat_id, topics, tags.
Shared across embedding models — `is_chat` does not depend on the embedder.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

MANIFEST_VERSION = 1

_UNSET = object()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Manifest:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.files: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if data.get("version") != MANIFEST_VERSION:
            return
        files = data.get("files")
        if isinstance(files, dict):
            self.files = files

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": MANIFEST_VERSION, "files": self.files}
        fd, tmp = tempfile.mkstemp(prefix=".manifest.", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def get(self, file_path: str) -> Optional[dict]:
        return self.files.get(file_path)

    def stat_unchanged(self, file_path: str, mtime: float, size: int) -> bool:
        entry = self.files.get(file_path)
        if not entry:
            return False
        return entry.get("mtime") == mtime and entry.get("size") == size

    def update(
        self,
        file_path: str,
        *,
        is_chat: bool,
        mtime: float,
        size: int,
        hash=_UNSET,
        chat_id=_UNSET,
        system_instruction=_UNSET,
    ) -> dict:
        existing = self.files.get(file_path, {})
        entry = {
            "is_chat": is_chat,
            "mtime": mtime,
            "size": size,
            "hash": existing.get("hash") if hash is _UNSET else hash,
            "checked_at": _now_iso(),
            "chat_id": existing.get("chat_id") if chat_id is _UNSET else chat_id,
            "system_instruction": (
                existing.get("system_instruction")
                if system_instruction is _UNSET
                else system_instruction
            ),
            "topics": existing.get("topics", []),
            "tags": existing.get("tags", []),
        }
        self.files[file_path] = entry
        return entry

    def set_tags(self, file_path: str, tags: list[str]) -> None:
        if file_path in self.files:
            self.files[file_path]["tags"] = list(tags)

    def set_topics(self, file_path: str, topics: list[str]) -> None:
        if file_path in self.files:
            self.files[file_path]["topics"] = list(topics)

    def chat_paths(self) -> list[str]:
        return [p for p, e in self.files.items() if e.get("is_chat")]
