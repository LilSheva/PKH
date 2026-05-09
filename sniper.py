from __future__ import annotations

from pathlib import Path
from typing import Optional

import config
from core import embeddings as emb, ingestion, parser
from core.manifest import Manifest
from core.tagger import LLMCall, TagResult, tag_chats
from core.vector_db import VectorStore


class ContextSniper:
    """Search-and-extract over a personal archive of AI Studio chats."""

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        db_dir: Optional[Path] = None,
        manifest_path: Optional[Path] = None,
    ):
        self.embedding_key = embedding_model or config.EMBEDDING_MODEL
        emb.get_spec(self.embedding_key)
        self.db_dir = Path(db_dir) if db_dir else config.DB_DIR
        self.store = VectorStore(self.db_dir, self.embedding_key)
        self.manifest = Manifest(manifest_path or config.MANIFEST_PATH)

    def ingest(self, root: str | Path) -> dict:
        root = Path(root)
        stats = {
            "files_seen": 0,
            "files_skipped_non_chat": 0,     # known non-chat, skipped via manifest
            "files_skipped_unchanged": 0,    # chat, file unchanged since last run
            "files_sniffed": 0,              # had to read+sniff (new or modified)
            "files_ingested": 0,             # actually parsed and upserted
            "blocks_added": 0,
            "blocks_skipped_dupe_content": 0,
        }

        for path in ingestion.iter_files(root):
            stats["files_seen"] += 1
            spath = str(path)
            try:
                st = path.stat()
            except OSError:
                continue
            mtime, size = st.st_mtime, st.st_size

            # Fast path: manifest says nothing changed -> use cached verdict
            if self.manifest.stat_unchanged(spath, mtime, size):
                entry = self.manifest.get(spath)
                if not entry["is_chat"]:
                    stats["files_skipped_non_chat"] += 1
                    continue
                # is_chat=True, file unchanged -> skip ingest entirely
                stats["files_skipped_unchanged"] += 1
                continue

            # Slow path: new or modified -> sniff + maybe ingest
            stats["files_sniffed"] += 1
            is_chat = ingestion.is_chat_file(path)
            if not is_chat:
                self.manifest.update(spath, is_chat=False, mtime=mtime, size=size)
                continue

            fhash = ingestion.file_hash(path)
            payload = parser.load_chat(path, fhash)
            if payload is None:
                # Looked like a chat but JSON is malformed — record the sniff verdict,
                # don't try to parse blocks.
                self.manifest.update(
                    spath, is_chat=True, mtime=mtime, size=size,
                    hash=fhash, chat_id=path.stem, system_instruction=None,
                )
                continue

            self.manifest.update(
                spath, is_chat=True, mtime=mtime, size=size,
                hash=fhash, chat_id=payload.chat_id,
                system_instruction=payload.system_instruction,
            )

            blocks = list(parser.iter_blocks(
                payload,
                thought_mode=config.THOUGHT_MODE,
                long_threshold=config.THOUGHT_LONG_THRESHOLD,
                sim_threshold=config.THOUGHT_SIM_THRESHOLD,
                embedding_key=self.embedding_key,
                max_token_len=config.MAX_TOKEN_LEN,
            ))

            # Layer B: exact content-hash dedup
            fresh = []
            for b in blocks:
                if self.store.content_exists(b.content_hash):
                    stats["blocks_skipped_dupe_content"] += 1
                    continue
                # Layer C: semantic dedup (stub — enable via config when needed)
                if config.ENABLE_SEMANTIC_DEDUP:
                    pass  # TODO: top-1 query, skip if cosine > SEMANTIC_DEDUP_THRESHOLD
                fresh.append(b)

            self.store.delete_stale_for_file(spath, fhash)
            stats["blocks_added"] += self.store.upsert_blocks(fresh)
            stats["files_ingested"] += 1

        self.manifest.save()
        return stats

    def search_context(self, meta_query: str, top_k: Optional[int] = None) -> dict:
        return self.store.query(meta_query, top_k or config.DEFAULT_TOP_K)

    def generate_super_prompt(
        self,
        main_prompt: str,
        meta_query: str,
        top_k: Optional[int] = None,
    ) -> str:
        res = self.search_context(meta_query, top_k=top_k)
        docs = res.get("documents", [[]])[0] if res.get("documents") else []
        ctx = "\n\n---\n\n".join(docs)
        return f"## CONTEXT\n{ctx}\n\n## TASK\n{main_prompt}"

    def stats(self) -> dict:
        return {
            "embedding_model": self.embedding_key,
            "collection": emb.collection_name(self.embedding_key),
            "blocks": self.store.count(),
            "manifest_files": len(self.manifest.files),
            "manifest_chats": len(self.manifest.chat_paths()),
        }

    def tag_chats(
        self,
        llm_call: LLMCall,
        *,
        only_untagged: bool = True,
        snippet_chars: int = 200_000,
        on_progress=None,
        paths=None,
    ) -> list[TagResult]:
        """LLM-classify chats and write topics/tags into the manifest."""
        results = tag_chats(
            self.manifest,
            llm_call,
            only_untagged=only_untagged,
            snippet_chars=snippet_chars,
            on_progress=on_progress,
            paths=paths,
        )
        self.manifest.save()
        return results
