from functools import lru_cache
import hashlib
import math
import re

from src.config import get_settings

EMBEDDING_DIMENSION = 384


@lru_cache
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().EMBEDDING_MODEL_NAME)


def _hash_embedding(text: str) -> list[float]:
    """Create a small, deterministic lexical embedding for memory-constrained hosts."""
    vector = [0.0] * EMBEDDING_DIMENSION
    for token in re.findall(r"\b[\w-]+\b", text.lower()):
        index = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=4).digest(), "big") % EMBEDDING_DIMENSION
        vector[index] += 1.0
    magnitude = math.sqrt(sum(value * value for value in vector))
    return [value / magnitude for value in vector] if magnitude else vector


def embed_text(text: str) -> list[float]:
    if get_settings().EMBEDDING_BACKEND == "hashing":
        return _hash_embedding(text)
    return _model().encode(text, normalize_embeddings=True).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    if get_settings().EMBEDDING_BACKEND == "hashing":
        return [_hash_embedding(text) for text in texts]
    return _model().encode(texts, normalize_embeddings=True).tolist()
