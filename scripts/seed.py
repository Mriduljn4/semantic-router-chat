from src.capability_profiles import CAPABILITY_PROFILES
from src.chroma_store import get_capabilities_collection, get_research_docs_collection
from src.embeddings import embed_documents
from src.research_docs import RESEARCH_DOCS


def _clear(collection) -> None:
    """Delete current records so repeated seeding is deterministic and idempotent."""
    ids = collection.get(include=[])["ids"]
    if ids:
        collection.delete(ids=ids)


def seed() -> None:
    """Embed and persist capability profiles plus research documents in Chroma."""
    capabilities = get_capabilities_collection()
    research_docs = get_research_docs_collection()
    _clear(capabilities)
    _clear(research_docs)
    phrases = [(agent, phrase) for agent, values in CAPABILITY_PROFILES.items() for phrase in values]
    documents = [phrase for _, phrase in phrases]
    # Stored metadata lets the router aggregate nearest profiles by specialist.
    capabilities.upsert(
        ids=[f"cap-{index}" for index in range(len(documents))], documents=documents,
        embeddings=embed_documents(documents), metadatas=[{"agent": agent} for agent, _ in phrases],
    )
    research_docs.upsert(
        ids=[f"doc-{index}" for index in range(len(RESEARCH_DOCS))], documents=RESEARCH_DOCS,
        embeddings=embed_documents(RESEARCH_DOCS),
    )
    print(f"Seeded {len(documents)} capabilities and {len(RESEARCH_DOCS)} research documents.")


if __name__ == "__main__":
    seed()
