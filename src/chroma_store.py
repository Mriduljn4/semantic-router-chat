from functools import lru_cache

import chromadb

from src.config import get_settings


@lru_cache
def get_client() -> chromadb.PersistentClient:
    """Return the process-wide Chroma client backed by the configured local path."""
    return chromadb.PersistentClient(path=str(get_settings().chroma_path))


def _collection(name: str):
    """Get a cosine-distance collection, creating it when local storage is empty."""
    return get_client().get_or_create_collection(name, metadata={"hnsw:space": "cosine"})


def get_capabilities_collection():
    """Return examples used by the semantic intent classifier."""
    return _collection("agent_capabilities")


def get_research_docs_collection():
    """Return local documents used to ground research-specialist responses."""
    return _collection("research_docs")
