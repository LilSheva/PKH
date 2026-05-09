"""Smoke test for core/manifest.py and the new ingestion contract.

Runs without GPU/sentence-transformers/chromadb.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import ingestion
from core.manifest import Manifest


def test_manifest_persistence_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        mpath = Path(td) / "manifest.json"
        m = Manifest(mpath)
        assert m.files == {}
        m.update("/x/a", is_chat=True, mtime=1.0, size=10, hash="h1", chat_id="a")
        m.update("/x/b", is_chat=False, mtime=2.0, size=20)
        m.save()
        assert mpath.exists()

        m2 = Manifest(mpath)
        assert "/x/a" in m2.files and "/x/b" in m2.files
        assert m2.files["/x/a"]["chat_id"] == "a"
        assert m2.files["/x/a"]["topics"] == []
        assert m2.files["/x/a"]["tags"] == []
        assert m2.files["/x/b"]["is_chat"] is False


def test_stat_unchanged_detects_modifications():
    with tempfile.TemporaryDirectory() as td:
        m = Manifest(Path(td) / "m.json")
        m.update("/x/a", is_chat=True, mtime=10.0, size=100, hash="h", chat_id="a")
        assert m.stat_unchanged("/x/a", 10.0, 100)
        assert not m.stat_unchanged("/x/a", 10.0, 101)
        assert not m.stat_unchanged("/x/a", 11.0, 100)
        assert not m.stat_unchanged("/x/missing", 10.0, 100)


def test_topics_tags_preserved_across_update():
    with tempfile.TemporaryDirectory() as td:
        m = Manifest(Path(td) / "m.json")
        m.update("/x/a", is_chat=True, mtime=1.0, size=10, hash="h", chat_id="a")
        m.set_tags("/x/a", ["#python", "#siem"])
        m.set_topics("/x/a", ["MaxPatrol"])
        m.update("/x/a", is_chat=True, mtime=2.0, size=20, hash="h2", chat_id="a")
        assert m.files["/x/a"]["tags"] == ["#python", "#siem"]
        assert m.files["/x/a"]["topics"] == ["MaxPatrol"]


def test_system_instruction_explicit_set_and_clear():
    with tempfile.TemporaryDirectory() as td:
        m = Manifest(Path(td) / "m.json")
        m.update("/x/a", is_chat=True, mtime=1.0, size=10, hash="h", chat_id="a",
                 system_instruction="be terse")
        assert m.files["/x/a"]["system_instruction"] == "be terse"
        # explicit None overwrites (user removed the SI in the chat)
        m.update("/x/a", is_chat=True, mtime=2.0, size=11, hash="h2", chat_id="a",
                 system_instruction=None)
        assert m.files["/x/a"]["system_instruction"] is None


def test_system_instruction_unset_keeps_existing():
    with tempfile.TemporaryDirectory() as td:
        m = Manifest(Path(td) / "m.json")
        m.update("/x/a", is_chat=True, mtime=1.0, size=10, hash="h", chat_id="a",
                 system_instruction="role: SOC")
        # not passing system_instruction -> should preserve
        m.update("/x/a", is_chat=True, mtime=2.0, size=11)
        assert m.files["/x/a"]["system_instruction"] == "role: SOC"


def test_corrupt_manifest_is_ignored():
    with tempfile.TemporaryDirectory() as td:
        mpath = Path(td) / "m.json"
        mpath.write_text("{ not valid json", encoding="utf-8")
        m = Manifest(mpath)
        assert m.files == {}


def test_is_chat_file_detects_real_format():
    sample = {"chunkedPrompt": {"chunks": [{"role": "user", "text": "hi"}]}}
    with tempfile.TemporaryDirectory() as td:
        good = Path(td) / "no_extension_chat"
        good.write_text(json.dumps(sample), encoding="utf-8")
        assert ingestion.is_chat_file(good)

        bad = Path(td) / "random.txt"
        bad.write_text("just some prose, no json here", encoding="utf-8")
        assert not ingestion.is_chat_file(bad)

        binary = Path(td) / "binary"
        binary.write_bytes(b"\x89PNG\x0d\x0a\x1a\x0a" + b"\x00" * 100)
        assert not ingestion.is_chat_file(binary)


def test_iter_files_walks_recursively():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "sub").mkdir()
        (root / "a").write_text("x")
        (root / "sub" / "b").write_text("y")
        found = sorted(p.name for p in ingestion.iter_files(root))
        assert found == ["a", "b"]


TESTS = [
    test_manifest_persistence_roundtrip,
    test_stat_unchanged_detects_modifications,
    test_topics_tags_preserved_across_update,
    test_system_instruction_explicit_set_and_clear,
    test_system_instruction_unset_keeps_existing,
    test_corrupt_manifest_is_ignored,
    test_is_chat_file_detects_real_format,
    test_iter_files_walks_recursively,
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
