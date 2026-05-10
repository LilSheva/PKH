from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class EmbedderSpec:
    key: str
    hf_name: str
    dim: int
    max_seq_len: int
    query_prefix: str = ""
    passage_prefix: str = ""


REGISTRY: dict[str, EmbedderSpec] = {
    "bge-m3": EmbedderSpec(
        key="bge-m3",
        hf_name="BAAI/bge-m3",
        dim=1024,
        max_seq_len=8192,
    ),
    "qwen3": EmbedderSpec(
        key="qwen3",
        hf_name="Qwen/Qwen3-Embedding-0.6B",
        dim=1024,
        max_seq_len=32768,
    ),
    "e5-instruct": EmbedderSpec(
        key="e5-instruct",
        hf_name="intfloat/multilingual-e5-large-instruct",
        dim=1024,
        max_seq_len=512,
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
}


def get_spec(key: str) -> EmbedderSpec:
    if key not in REGISTRY:
        raise ValueError(
            f"Unknown embedding model: {key!r}. Available: {sorted(REGISTRY)}"
        )
    return REGISTRY[key]


@lru_cache(maxsize=3)
def get_model(key: str):
    spec = get_spec(key)
    from sentence_transformers import SentenceTransformer
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    return SentenceTransformer(spec.hf_name, device=device)


def embed_passages(key: str, texts: list[str], batch_size: int = 8) -> list[list[float]]:
    if not texts:
        return []
    spec = get_spec(key)
    model = get_model(key)
    if spec.passage_prefix:
        texts = [spec.passage_prefix + t for t in texts]
    arr = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
        batch_size=batch_size,
    )
    return arr.tolist()


def embed_query(key: str, text: str) -> list[float]:
    spec = get_spec(key)
    model = get_model(key)
    if spec.query_prefix:
        text = spec.query_prefix + text
    arr = model.encode(
        [text],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return arr[0].tolist()


def collection_name(key: str) -> str:
    return f"pkh_{key}"
