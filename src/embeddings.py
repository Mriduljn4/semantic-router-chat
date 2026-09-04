from functools import lru_cache
import hashlib
import math
import re

from google import genai
from google.genai import types

from src.config import get_settings

EMBEDDING_DIMENSION = 384


@lru_cache
def _model():
    """Load and cache the local sentence-transformer only when that backend is selected."""
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


@lru_cache
def _gemini_client() -> genai.Client:
    """Create one cached Gemini API client for API-hosted embeddings."""
    return genai.Client(api_key=get_settings().GEMINI_API_KEY)


def _gemini_embeddings(texts: list[str], task_type: str) -> list[list[float]]:
    """Request task-aware Gemini vectors at a storage-efficient dimension."""
    response = _gemini_client().models.embed_content(
        model=get_settings().GEMINI_EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=768),
    )
    return [embedding.values for embedding in response.embeddings]


def _use_gemini_embeddings() -> bool:
    """Use Gemini only when its backend is selected and credentials are available."""
    settings = get_settings()
    return settings.EMBEDDING_BACKEND == "gemini" and bool(settings.GEMINI_API_KEY)


def embed_query(text: str) -> list[float]:
    """Embed a user request using query semantics appropriate for retrieval."""
    backend = get_settings().EMBEDDING_BACKEND
    if backend == "hashing":
        return _hash_embedding(text)
    if backend == "gemini" and _use_gemini_embeddings():
        return _gemini_embeddings([text], "RETRIEVAL_QUERY")[0]
    if backend == "gemini":
        return _hash_embedding(text)
    try:
        return _model().encode(text, normalize_embeddings=True).tolist()
    except (ModuleNotFoundError, ImportError):
        return _hash_embedding(text)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed stored profiles and RAG documents with document retrieval semantics."""
    backend = get_settings().EMBEDDING_BACKEND
    if backend == "hashing":
        return [_hash_embedding(text) for text in texts]
    if backend == "gemini" and _use_gemini_embeddings():
        return _gemini_embeddings(texts, "RETRIEVAL_DOCUMENT")
    if backend == "gemini":
        return [_hash_embedding(text) for text in texts]
    try:
        return _model().encode(texts, normalize_embeddings=True).tolist()
    except (ModuleNotFoundError, ImportError):
        return [_hash_embedding(text) for text in texts]


# Backward-compatible aliases for callers that do not need explicit task types.
embed_text = embed_query
embed_batch = embed_documents
