from __future__ import annotations

from pathlib import Path
from typing import Sequence

import chromadb

from . import embeddings as emb
from .parser import DialogBlock


class VectorStore:
    """Per-model Chroma collection with idempotent upsert and dedup helpers."""

    def __init__(self, db_dir: Path, embedding_key: str):
        self.embedding_key = embedding_key
        db_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(db_dir))
        self.collection = self.client.get_or_create_collection(
            name=emb.collection_name(embedding_key),
            metadata={"hnsw:space": "cosine"},
        )

    # --- dedup / manifest checks -------------------------------------------------

    def file_already_ingested(self, file_path: str, file_hash: str) -> bool:
        got = self.collection.get(
            where={"$and": [{"file_path": file_path}, {"file_hash": file_hash}]},
            limit=1,
        )
        return bool(got.get("ids"))

    def content_exists(self, content_hash: str) -> bool:
        got = self.collection.get(where={"content_hash": content_hash}, limit=1)
        return bool(got.get("ids"))

    def delete_stale_for_file(self, file_path: str, current_hash: str) -> None:
        self.collection.delete(
            where={
                "$and": [
                    {"file_path": file_path},
                    {"file_hash": {"$ne": current_hash}},
                ]
            }
        )

    # --- writes ------------------------------------------------------------------

    def upsert_blocks(self, blocks: Sequence[DialogBlock]) -> int:
        if not blocks:
            return 0
        ids = [b.block_id for b in blocks]
        docs = [b.cleaned_content for b in blocks]
        metas = [
            {
                "chat_id": b.chat_id,
                "chunk_index": b.chunk_index,
                "content_hash": b.content_hash,
                "file_path": b.file_path,
                "file_hash": b.file_hash,
            }
            for b in blocks
        ]
        embs = emb.embed_passages(self.embedding_key, docs)
        self.collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        return len(ids)

    # --- reads -------------------------------------------------------------------

    def query(self, text: str, top_k: int) -> dict:
        q = emb.embed_query(self.embedding_key, text)
        return self.collection.query(query_embeddings=[q], n_results=top_k)

    def count(self) -> int:
        return self.collection.count()
