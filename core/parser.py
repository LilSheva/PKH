from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from pydantic import BaseModel

from . import embeddings as emb
from .cleaner import clean_text


class DialogBlock(BaseModel):
    block_id: str
    chat_id: str
    chunk_index: int
    user_text: str
    model_text: str
    cleaned_content: str
    content_hash: str
    file_path: str
    file_hash: str


@dataclass
class ChatPayload:
    chat_id: str
    file_path: str
    file_hash: str
    raw: dict
    system_instruction: Optional[str]


def _block_id(chat_id: str, chunk_index: int, user_text: str) -> str:
    raw = f"{chat_id}::{chunk_index}::{user_text[:200]}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _content_hash(text: str) -> str:
    normalized = " ".join(text.split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _strip_thought_signature(obj):
    if isinstance(obj, dict):
        return {
            k: _strip_thought_signature(v)
            for k, v in obj.items()
            if k != "thoughtSignature"
        }
    if isinstance(obj, list):
        return [_strip_thought_signature(x) for x in obj]
    return obj


def _extract_system_instruction(raw: dict) -> Optional[str]:
    """AI Studio packs systemInstruction as {"parts": [{"text": ...}]} or {"text": ...}.
    Empty dict is the AI Studio default for "not set".
    """
    si = raw.get("systemInstruction")
    if not isinstance(si, dict) or not si:
        return None
    parts = si.get("parts")
    if isinstance(parts, list):
        joined = "\n".join(
            p.get("text", "") for p in parts if isinstance(p, dict)
        ).strip()
        return joined or None
    text = si.get("text")
    if isinstance(text, str):
        return text.strip() or None
    return None


def load_chat(path: Path, file_hash_value: str) -> Optional[ChatPayload]:
    """Load + sanitize a chat file once. Returns None on broken JSON."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None
    raw = _strip_thought_signature(raw)
    return ChatPayload(
        chat_id=path.stem,
        file_path=str(path),
        file_hash=file_hash_value,
        raw=raw,
        system_instruction=_extract_system_instruction(raw),
    )


def _chunk_text(chunk: dict) -> str:
    val = chunk.get("text")
    if isinstance(val, str):
        return val
    parts = chunk.get("parts")
    if isinstance(parts, list):
        return "\n".join(p.get("text", "") for p in parts if isinstance(p, dict))
    return ""


def _filter_thought_smart(text: str, user_query: str, sim_threshold: float,
                          embedding_key: str) -> str:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs or not user_query:
        return ""
    import numpy as np
    qv = np.array(emb.embed_query(embedding_key, user_query))
    pvs = emb.embed_passages(embedding_key, paragraphs)
    kept = [
        paragraphs[i]
        for i, pv in enumerate(pvs)
        if float(np.dot(qv, np.array(pv))) >= sim_threshold
    ]
    return "\n\n".join(kept)


def _process_model_chunks(model_chunks: list[dict], user_query: str,
                          mode: str, long_threshold: int, sim_threshold: float,
                          embedding_key: str) -> str:
    out: list[str] = []
    for ch in model_chunks:
        text = _chunk_text(ch)
        if not text:
            continue
        is_thought = bool(ch.get("isThought"))
        if not is_thought:
            out.append(text)
            continue
        if mode == "OFF":
            continue
        if mode == "ON":
            out.append(text)
            continue
        if mode == "SMART":
            if len(text) <= long_threshold:
                out.append(text)
            else:
                kept = _filter_thought_smart(text, user_query, sim_threshold, embedding_key)
                if kept:
                    out.append(kept)
    return "\n\n".join(out)


def parse_json_chat(
    path: Path,
    file_hash_value: str,
    *,
    thought_mode: str,
    long_threshold: int,
    sim_threshold: float,
    embedding_key: str,
    max_token_len: int,
) -> Iterator[DialogBlock]:
    payload = load_chat(path, file_hash_value)
    if payload is None:
        return
    yield from iter_blocks(
        payload,
        thought_mode=thought_mode,
        long_threshold=long_threshold,
        sim_threshold=sim_threshold,
        embedding_key=embedding_key,
        max_token_len=max_token_len,
    )


def iter_blocks(
    payload: ChatPayload,
    *,
    thought_mode: str,
    long_threshold: int,
    sim_threshold: float,
    embedding_key: str,
    max_token_len: int,
) -> Iterator[DialogBlock]:
    chat_id = payload.chat_id
    chunks = payload.raw.get("chunkedPrompt", {}).get("chunks", [])
    if not isinstance(chunks, list):
        return

    pair_index = 0
    i = 0
    n = len(chunks)
    while i < n:
        ch = chunks[i]
        if not isinstance(ch, dict) or ch.get("role") != "user":
            i += 1
            continue
        user_text = _chunk_text(ch).strip()
        i += 1
        model_chunks: list[dict] = []
        while i < n and isinstance(chunks[i], dict) and chunks[i].get("role") == "model":
            model_chunks.append(chunks[i])
            i += 1
        model_text = _process_model_chunks(
            model_chunks, user_text, thought_mode,
            long_threshold, sim_threshold, embedding_key,
        ).strip()
        if not user_text and not model_text:
            continue
        merged = f"USER: {user_text}\n\nMODEL: {model_text}"
        cleaned = clean_text(merged, max_token_len=max_token_len)
        yield DialogBlock(
            block_id=_block_id(chat_id, pair_index, user_text),
            chat_id=chat_id,
            chunk_index=pair_index,
            user_text=user_text,
            model_text=model_text,
            cleaned_content=cleaned,
            content_hash=_content_hash(cleaned),
            file_path=payload.file_path,
            file_hash=payload.file_hash,
        )
        pair_index += 1
