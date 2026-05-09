"""Synthetic smoke test for core/parser.py — runs without GPU/network/sentence-transformers.

Covers:
- chunkedPrompt -> DialogBlock pair extraction
- thoughtSignature is stripped recursively
- THOUGHT_MODE = OFF skips isThought blocks
- THOUGHT_MODE = ON includes them
- block_id is deterministic across runs
- chunk_index increments per (User, Model) pair
- cleaner removes long alphanumeric blobs
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import parser
from core.cleaner import clean_text


SAMPLE = {
    "chunkedPrompt": {
        "chunks": [
            {"role": "user", "text": "Как настроить ChromaDB на Colab?"},
            {
                "role": "model",
                "text": "Сначала разберём, что нужно сделать пошагово...",
                "isThought": True,
                "thoughtSignature": "AAAA_BASE64_GARBAGE_AAAA",
            },
            {"role": "model", "text": "Установи chromadb через pip и используй PersistentClient."},
            {"role": "user", "text": "А база переживёт рестарт рантайма?"},
            {
                "role": "model",
                "text": "Да, если path указывает на /content/drive/.",
                "thoughtSignature": "ZZZZ",
            },
        ]
    }
}


def _write_sample(tmpdir: Path, name: str = "chat_alpha") -> Path:
    p = tmpdir / name
    p.write_text(json.dumps(SAMPLE), encoding="utf-8")
    return p


def _parse(path: Path, mode: str) -> list:
    return list(parser.parse_json_chat(
        path,
        file_hash_value="deadbeef",
        thought_mode=mode,
        long_threshold=10_000,
        sim_threshold=0.4,
        embedding_key="bge-m3",
        max_token_len=100,
    ))


def test_pair_extraction_off():
    with tempfile.TemporaryDirectory() as td:
        path = _write_sample(Path(td))
        blocks = _parse(path, "OFF")
    assert len(blocks) == 2, f"expected 2 pairs, got {len(blocks)}"
    assert blocks[0].chunk_index == 0
    assert blocks[1].chunk_index == 1
    assert blocks[0].chat_id == "chat_alpha"
    assert "ChromaDB" in blocks[0].user_text
    assert "Установи chromadb" in blocks[0].model_text
    # OFF: thinking text must NOT be in model_text
    assert "разберём" not in blocks[0].model_text


def test_thought_mode_on_includes_thinking():
    with tempfile.TemporaryDirectory() as td:
        path = _write_sample(Path(td))
        blocks = _parse(path, "ON")
    assert "разберём" in blocks[0].model_text
    assert "Установи chromadb" in blocks[0].model_text


def test_thought_signature_is_stripped():
    with tempfile.TemporaryDirectory() as td:
        path = _write_sample(Path(td))
        blocks_off = _parse(path, "OFF")
        blocks_on = _parse(path, "ON")
    for b in (*blocks_off, *blocks_on):
        assert "AAAA_BASE64_GARBAGE" not in b.cleaned_content
        assert "ZZZZ" not in b.cleaned_content


def test_block_id_deterministic():
    with tempfile.TemporaryDirectory() as td:
        path = _write_sample(Path(td))
        b1 = _parse(path, "OFF")
        b2 = _parse(path, "OFF")
    assert [b.block_id for b in b1] == [b.block_id for b in b2]
    # different chunks -> different ids
    assert b1[0].block_id != b1[1].block_id


def test_content_hash_stable_across_whitespace():
    h1 = parser._content_hash("hello   world\n\n")
    h2 = parser._content_hash("hello world")
    assert h1 == h2


def test_cleaner_strips_long_blob():
    junk = "A" * 200
    out = clean_text(f"normal text {junk} more text", max_token_len=100)
    assert "A" * 200 not in out
    assert "[STRIPPED]" in out
    assert "normal text" in out and "more text" in out


def test_skips_when_no_chunked_prompt():
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "not_a_chat"
        bad.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
        blocks = _parse(bad, "OFF")
    assert blocks == []


def test_handles_orphan_user_without_model_reply():
    sample = {"chunkedPrompt": {"chunks": [{"role": "user", "text": "lonely question"}]}}
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "lonely"
        p.write_text(json.dumps(sample), encoding="utf-8")
        blocks = _parse(p, "OFF")
    assert len(blocks) == 1
    assert blocks[0].user_text == "lonely question"
    assert blocks[0].model_text == ""


def test_system_instruction_extraction():
    from core.parser import _extract_system_instruction
    assert _extract_system_instruction({}) is None
    assert _extract_system_instruction({"systemInstruction": {}}) is None
    assert _extract_system_instruction({"systemInstruction": None}) is None
    assert _extract_system_instruction(
        {"systemInstruction": {"text": "  be terse  "}}
    ) == "be terse"
    assert _extract_system_instruction(
        {"systemInstruction": {"parts": [{"text": "role: SOC analyst"}, {"text": "tone: dry"}]}}
    ) == "role: SOC analyst\ntone: dry"
    assert _extract_system_instruction(
        {"systemInstruction": {"parts": [{"text": ""}]}}
    ) is None


def test_load_chat_returns_payload_with_system_instruction():
    from core.parser import load_chat
    sample = {
        "systemInstruction": {"parts": [{"text": "You are a SIEM expert."}]},
        "chunkedPrompt": {"chunks": [
            {"role": "user", "text": "hi"},
            {"role": "model", "text": "hello"},
        ]},
    }
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "with_si"
        p.write_text(json.dumps(sample), encoding="utf-8")
        payload = load_chat(p, "fhash")
    assert payload is not None
    assert payload.system_instruction == "You are a SIEM expert."
    assert payload.chat_id == "with_si"


def test_load_chat_handles_missing_si_in_real_templates():
    from core.parser import load_chat
    from core import ingestion
    tdir = Path(__file__).resolve().parents[1] / "chats teamplates"
    if not tdir.is_dir():
        return  # skip if templates aren't checked out
    seen_any = False
    for p in tdir.iterdir():
        if not p.is_file() or not ingestion.is_chat_file(p):
            continue
        seen_any = True
        payload = load_chat(p, "x")
        assert payload is not None, f"failed to load {p.name}"
        # Every template in the snapshot has empty systemInstruction -> None
        assert payload.system_instruction is None, f"{p.name}: {payload.system_instruction!r}"
    assert seen_any, "no chat templates found to validate against"


TESTS = [
    test_pair_extraction_off,
    test_thought_mode_on_includes_thinking,
    test_thought_signature_is_stripped,
    test_block_id_deterministic,
    test_content_hash_stable_across_whitespace,
    test_cleaner_strips_long_blob,
    test_skips_when_no_chunked_prompt,
    test_handles_orphan_user_without_model_reply,
    test_system_instruction_extraction,
    test_load_chat_returns_payload_with_system_instruction,
    test_load_chat_handles_missing_si_in_real_templates,
]


def main() -> int:
    failed = 0
    for t in TESTS:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
        else:
            print(f"OK    {t.__name__}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
