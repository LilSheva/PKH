from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator

_SNIFF_BYTES = 4096
_BINARY_SIGS = (
    b"\x89PNG", b"\xff\xd8\xff", b"GIF8",
    b"PK\x03\x04", b"\x1f\x8b", b"%PDF",
    b"\x7fELF", b"BM",
)


def _read_head(path: Path) -> bytes:
    try:
        with path.open("rb") as f:
            return f.read(_SNIFF_BYTES)
    except (OSError, PermissionError):
        return b""


def is_chat_file(path: Path) -> bool:
    """JSON dump of an AI Studio chat — detected by content, not extension.

    AI Studio dumps come without an extension. We sniff the first few KB
    looking for `{` and the `chunkedPrompt` marker.
    """
    head = _read_head(path)
    if not head:
        return False
    if any(head.startswith(sig) for sig in _BINARY_SIGS):
        return False
    if b"{" not in head[:512]:
        return False
    return b'"chunkedPrompt"' in head


def file_hash(path: Path, chunk_kb: int = 64) -> str:
    h = hashlib.sha1()
    try:
        with path.open("rb") as f:
            h.update(f.read(chunk_kb * 1024))
    except OSError:
        return ""
    return h.hexdigest()


def iter_files(root: Path) -> Iterator[Path]:
    """Yield every regular file under root. No filtering — that's the manifest's job."""
    for p in root.rglob("*"):
        if p.is_file():
            yield p
