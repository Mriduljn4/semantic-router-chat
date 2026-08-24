from functools import lru_cache

import chromadb

from src.config import get_settings


@lru_cache
def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(get_settings().chroma_path))


def _collection(name: str):
    return get_client().get_or_create_collection(name, metadata={"hnsw:space": "cosine"})


def get_capabilities_collection():
    return _collection("agent_capabilities")


def get_research_docs_collection():
    return _collection("research_docs")
