"""LLM-driven chat tagging — fills manifest entries with `topics` and `tags`.

Provider-agnostic: caller passes an `llm_call(prompt: str) -> str` callable.
Wire it to Gemini / Claude / OpenAI / a local model — tagger doesn't care.

Usage:
    from sniper import ContextSniper
    sniper = ContextSniper()
    def call(prompt: str) -> str:
        # your LLM client here
        ...
    sniper.tag_chats(call, only_untagged=True)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from . import parser as chat_parser
from .manifest import Manifest


LLMCall = Callable[[str], str]


# Keep the prompt simple and language-neutral. The model returns JSON; we parse defensively.
PROMPT_TEMPLATE = """Ты получаешь диалог пользователя с ИИ-ассистентом и (опционально) системную инструкцию чата.
Темы и подтемы могут меняться по ходу диалога, и в середину могут быть вставлены важные побочные обсуждения — учитывай ВЕСЬ фрагмент целиком, а не только начало.
Если в диалог встроен маркер `[... пропущено ...]`, это значит, что середина усечена и тебе показаны репрезентативные семплы — реальные темы могут быть шире, чем кажется по обрывкам.

Задача: выдать JSON c двумя полями:
  - "topics": 1-5 крупных тем (короткие существительные/именные группы на русском, без решёток), напр. ["MaxPatrol SIEM", "обнаружение IoC", "интеграция с Kaspersky"]
  - "tags": 3-10 тегов в формате "#область_подобласть" на русском snake_case, напр. ["#siem_maxpatrol", "#обнаружение_ioc", "#kaspersky", "#инциденты"]

Отвечай СТРОГО валидным JSON без преамбулы и комментариев.

{si_block}### Фрагмент диалога:
{snippet}

JSON-ответ:"""


SAMPLE_MARKER = "[... пропущено ...]"


@dataclass
class TagResult:
    file_path: str
    chat_id: Optional[str]
    topics: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    error: Optional[str] = None


def _dialog_pieces(payload: chat_parser.ChatPayload) -> list[str]:
    """Linear list of [ROLE] text entries, skipping thoughts and empty chunks."""
    chunks = payload.raw.get("chunkedPrompt", {}).get("chunks", [])
    if not isinstance(chunks, list):
        return []
    out: list[str] = []
    for c in chunks:
        if not isinstance(c, dict) or c.get("isThought"):
            continue
        text = c.get("text")
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        role = (c.get("role") or "?").upper()
        out.append(f"[{role}] {text}")
    return out


def _sample_strategic(pieces: list[str], max_chars: int) -> str:
    """Pick a subset of pieces that fits in `max_chars`, anchored to first/last,
    with evenly-spaced samples from the middle. Inserts SAMPLE_MARKER on gaps."""
    if not pieces:
        return ""
    sep = "\n\n"
    sep_len = len(sep)
    marker_overhead = sep_len + len(SAMPLE_MARKER) + sep_len

    avg_len = sum(len(p) for p in pieces) / len(pieces)
    # Conservative: assume a marker between every pair (worst case)
    target_count = max(2, int((max_chars + marker_overhead) / (avg_len + marker_overhead)))
    target_count = min(target_count, len(pieces))

    if target_count >= len(pieces):
        picks = list(range(len(pieces)))
    else:
        step = (len(pieces) - 1) / (target_count - 1)
        picks = sorted({round(i * step) for i in range(target_count)})

    out: list[str] = [pieces[picks[0]]]
    for j in range(1, len(picks)):
        if picks[j] != picks[j - 1] + 1:
            out.append(SAMPLE_MARKER)
        out.append(pieces[picks[j]])

    # Safety: drop from the middle until under budget — preserves first/last anchors.
    while len(sep.join(out)) > max_chars and len(out) > 2:
        out.pop(len(out) // 2)

    # Collapse any consecutive / trailing markers left by middle drops.
    cleaned: list[str] = []
    for item in out:
        if item == SAMPLE_MARKER and cleaned and cleaned[-1] == SAMPLE_MARKER:
            continue
        cleaned.append(item)
    while cleaned and cleaned[-1] == SAMPLE_MARKER:
        cleaned.pop()
    while cleaned and cleaned[0] == SAMPLE_MARKER:
        cleaned.pop(0)

    return sep.join(cleaned)


def _build_snippet(payload: chat_parser.ChatPayload, max_chars: int) -> str:
    pieces = _dialog_pieces(payload)
    if not pieces:
        return ""
    full = "\n\n".join(pieces)
    if len(full) <= max_chars:
        return full
    return _sample_strategic(pieces, max_chars)


def _build_prompt(payload: chat_parser.ChatPayload, snippet_chars: int) -> str:
    si = (payload.system_instruction or "").strip()
    si_block = f"### Системная инструкция чата:\n{si}\n\n" if si else ""
    snippet = _build_snippet(payload, max_chars=snippet_chars)
    return PROMPT_TEMPLATE.format(si_block=si_block, snippet=snippet)


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_llm_response(text: str) -> tuple[list[str], list[str]]:
    """Extract topics/tags from a JSON LLM response, tolerating markdown fences."""
    if not text:
        return [], []
    candidate = text.strip()
    m = _JSON_FENCE_RE.search(candidate)
    if m:
        candidate = m.group(1)
    else:
        # Fallback: take the substring between first { and last }
        first = candidate.find("{")
        last = candidate.rfind("}")
        if first != -1 and last > first:
            candidate = candidate[first : last + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return [], []
    topics = [str(t).strip() for t in data.get("topics", []) if str(t).strip()]
    raw_tags = [str(t).strip() for t in data.get("tags", []) if str(t).strip()]
    tags = [t if t.startswith("#") else f"#{t}" for t in raw_tags]
    return topics, tags


def tag_chats(
    manifest: Manifest,
    llm_call: LLMCall,
    *,
    only_untagged: bool = True,
    snippet_chars: int = 200_000,
    on_progress: Optional[Callable[[TagResult], None]] = None,
    paths: Optional[Iterable[str]] = None,
) -> list[TagResult]:
    """Walk chat entries in the manifest, ask LLM for topics+tags, write back.

    `snippet_chars` is the LLM input budget (default 200K chars ≈ 50K tokens).
    Most chats fit whole; longer ones get head + evenly-spaced middle + tail
    sampling so topic drifts and mid-chat pivots are still visible to the LLM.
    """
    targets = list(paths) if paths is not None else manifest.chat_paths()
    results: list[TagResult] = []

    for spath in targets:
        entry = manifest.get(spath)
        if not entry or not entry.get("is_chat"):
            continue
        if only_untagged and (entry.get("tags") or entry.get("topics")):
            continue

        path = Path(spath)
        if not path.exists():
            res = TagResult(file_path=spath, chat_id=entry.get("chat_id"),
                            error="file missing")
            results.append(res)
            if on_progress:
                on_progress(res)
            continue

        payload = chat_parser.load_chat(path, entry.get("hash") or "")
        if payload is None:
            res = TagResult(file_path=spath, chat_id=entry.get("chat_id"),
                            error="failed to load chat JSON")
            results.append(res)
            if on_progress:
                on_progress(res)
            continue

        prompt = _build_prompt(payload, snippet_chars=snippet_chars)
        try:
            response = llm_call(prompt)
        except Exception as e:  # noqa: BLE001 — surface any provider error per-file
            res = TagResult(file_path=spath, chat_id=payload.chat_id,
                            error=f"llm_call raised: {type(e).__name__}: {e}")
            results.append(res)
            if on_progress:
                on_progress(res)
            continue

        topics, tags = _parse_llm_response(response)
        if not topics and not tags:
            res = TagResult(file_path=spath, chat_id=payload.chat_id,
                            error="empty/invalid LLM response")
            results.append(res)
            if on_progress:
                on_progress(res)
            continue

        manifest.set_topics(spath, topics)
        manifest.set_tags(spath, tags)
        res = TagResult(file_path=spath, chat_id=payload.chat_id,
                        topics=topics, tags=tags)
        results.append(res)
        if on_progress:
            on_progress(res)

    return results
