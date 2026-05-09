"""Smoke test for core/tagger.py — uses a fake llm_call, no network."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import ingestion
from core.manifest import Manifest
from core.parser import ChatPayload
from core.tagger import (
    SAMPLE_MARKER,
    _build_snippet,
    _parse_llm_response,
    _sample_strategic,
    tag_chats,
)


def test_parse_llm_response_clean_json():
    topics, tags = _parse_llm_response('{"topics": ["SIEM"], "tags": ["#siem", "kaspersky"]}')
    assert topics == ["SIEM"]
    assert tags == ["#siem", "#kaspersky"]


def test_parse_llm_response_markdown_fenced():
    raw = "```json\n{\"topics\": [\"X\"], \"tags\": [\"#a\"]}\n```"
    topics, tags = _parse_llm_response(raw)
    assert topics == ["X"]
    assert tags == ["#a"]


def test_parse_llm_response_with_preamble():
    raw = 'Here you go: {"topics": ["A"], "tags": ["#b"]} hope it helps.'
    topics, tags = _parse_llm_response(raw)
    assert topics == ["A"]
    assert tags == ["#b"]


def test_parse_llm_response_garbage_returns_empty():
    assert _parse_llm_response("not json at all") == ([], [])
    assert _parse_llm_response("") == ([], [])


def test_tag_chats_writes_to_manifest():
    sample = {
        "systemInstruction": {"parts": [{"text": "Ты SOC-аналитик."}]},
        "chunkedPrompt": {"chunks": [
            {"role": "user", "text": "Что такое event_src.ip?"},
            {"role": "model", "text": "Поле сетевого источника в SIEM."},
        ]},
    }
    with tempfile.TemporaryDirectory() as td:
        chat_path = Path(td) / "chat_alpha"
        chat_path.write_text(json.dumps(sample), encoding="utf-8")

        manifest = Manifest(Path(td) / "m.json")
        st = chat_path.stat()
        manifest.update(
            str(chat_path), is_chat=True, mtime=st.st_mtime, size=st.st_size,
            hash="h", chat_id=chat_path.stem, system_instruction="Ты SOC-аналитик.",
        )

        captured_prompt = {}
        def fake_llm(prompt: str) -> str:
            captured_prompt["p"] = prompt
            return '{"topics": ["SIEM", "MaxPatrol"], "tags": ["#siem", "#siem_field"]}'

        results = tag_chats(manifest, fake_llm)
        assert len(results) == 1
        r = results[0]
        assert r.error is None
        assert r.topics == ["SIEM", "MaxPatrol"]
        assert r.tags == ["#siem", "#siem_field"]

        entry = manifest.get(str(chat_path))
        assert entry["topics"] == ["SIEM", "MaxPatrol"]
        assert entry["tags"] == ["#siem", "#siem_field"]

        # Prompt must include the system instruction and the dialog snippet
        prompt = captured_prompt["p"]
        assert "Ты SOC-аналитик" in prompt
        assert "event_src.ip" in prompt


def test_tag_chats_skips_already_tagged_when_only_untagged():
    with tempfile.TemporaryDirectory() as td:
        chat_path = Path(td) / "c"
        chat_path.write_text(json.dumps({"chunkedPrompt": {"chunks": []}}), encoding="utf-8")
        manifest = Manifest(Path(td) / "m.json")
        st = chat_path.stat()
        manifest.update(str(chat_path), is_chat=True, mtime=st.st_mtime, size=st.st_size)
        manifest.set_tags(str(chat_path), ["#existing"])

        called = []
        def fake_llm(p: str) -> str:
            called.append(1)
            return '{"topics":[],"tags":[]}'

        results = tag_chats(manifest, fake_llm, only_untagged=True)
        assert results == []
        assert called == []


def test_tag_chats_records_llm_error_per_file():
    with tempfile.TemporaryDirectory() as td:
        chat_path = Path(td) / "c"
        chat_path.write_text(json.dumps({"chunkedPrompt": {"chunks": []}}), encoding="utf-8")
        manifest = Manifest(Path(td) / "m.json")
        st = chat_path.stat()
        manifest.update(str(chat_path), is_chat=True, mtime=st.st_mtime, size=st.st_size)

        def boom(p: str) -> str:
            raise RuntimeError("provider down")

        results = tag_chats(manifest, boom)
        assert len(results) == 1
        assert "provider down" in results[0].error


def _payload_with_chunks(role_text_pairs):
    raw = {"chunkedPrompt": {"chunks": [
        {"role": r, "text": t} for r, t in role_text_pairs
    ]}}
    return ChatPayload(chat_id="t", file_path="/x", file_hash="h",
                       raw=raw, system_instruction=None)


def test_build_snippet_returns_full_when_fits():
    pairs = [("user", "hello"), ("model", "hi")]
    p = _payload_with_chunks(pairs)
    out = _build_snippet(p, max_chars=10_000)
    assert "[USER] hello" in out and "[MODEL] hi" in out
    assert SAMPLE_MARKER not in out


def test_build_snippet_samples_when_too_long():
    import re
    # 200 pieces of ~110 chars each -> ~22K chars. Budget 4K -> must sample.
    pairs = [("user" if i % 2 == 0 else "model",
              f"piece {i:03d} " + "x" * 90) for i in range(200)]
    p = _payload_with_chunks(pairs)
    out = _build_snippet(p, max_chars=4_000)
    assert len(out) <= 4_000
    assert SAMPLE_MARKER in out
    # First and last pieces must always be present (anchors)
    assert "piece 000" in out
    assert "piece 199" in out
    # Several pieces from the middle of the chat must show up (drift detection)
    seen = sorted({int(m) for m in re.findall(r"piece (\d{3})", out)})
    middle_seen = [i for i in seen if 30 <= i <= 170]
    assert len(middle_seen) >= 5, f"too few middle samples: {middle_seen}"


def test_build_snippet_skips_thoughts():
    raw = {"chunkedPrompt": {"chunks": [
        {"role": "user", "text": "real question"},
        {"role": "model", "text": "noisy thinking", "isThought": True},
        {"role": "model", "text": "real answer"},
    ]}}
    p = ChatPayload(chat_id="t", file_path="/x", file_hash="h",
                    raw=raw, system_instruction=None)
    out = _build_snippet(p, max_chars=10_000)
    assert "real question" in out and "real answer" in out
    assert "noisy thinking" not in out


def test_sample_strategic_keeps_anchors_and_under_budget():
    pieces = [f"p{i}_" + "y" * 50 for i in range(100)]
    out = _sample_strategic(pieces, max_chars=2_000)
    assert len(out) <= 2_000
    assert "p0_" in out
    assert "p99_" in out
    assert SAMPLE_MARKER in out


TESTS = [
    test_parse_llm_response_clean_json,
    test_parse_llm_response_markdown_fenced,
    test_parse_llm_response_with_preamble,
    test_parse_llm_response_garbage_returns_empty,
    test_tag_chats_writes_to_manifest,
    test_tag_chats_skips_already_tagged_when_only_untagged,
    test_tag_chats_records_llm_error_per_file,
    test_build_snippet_returns_full_when_fits,
    test_build_snippet_samples_when_too_long,
    test_build_snippet_skips_thoughts,
    test_sample_strategic_keeps_anchors_and_under_budget,
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
