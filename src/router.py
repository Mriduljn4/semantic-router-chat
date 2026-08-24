from dataclasses import dataclass
from typing import Literal

from src.chroma_store import get_capabilities_collection
from src.config import get_settings
from src.embeddings import embed_query
from src.intent_classifier import classify_intent

AGENTS = ("research", "coding", "data")


@dataclass
class RoutingDecision:
    """The Groq-selected specialist and informational semantic similarity scores."""

    routed_agent: Literal["research", "coding", "data"]
    router_scores: dict[str, float]
    classifier_used: Literal["groq"]


def route(query: str) -> RoutingDecision:
    """Classify with Groq and calculate Chroma similarity scores for visibility.

    Chroma returns cosine distances for the nearest capability examples. Because
    the collection uses cosine space, $1 - distance$ is the cosine similarity.
    Profiles are grouped by agent and each agent receives the mean similarity.
    These scores are displayed in the UI but never select the specialist.
    """
    # Intent selection is intentionally Groq-only. Do not silently change the
    # selected agent when the classifier is unavailable.
    routed_agent = classify_intent(query)
    result = get_capabilities_collection().query(
        query_embeddings=[embed_query(query)], n_results=get_settings().ROUTER_TOP_K,
        include=["metadatas", "distances"],
    )
    grouped = {agent: [] for agent in AGENTS}
    for metadata, distance in zip(result["metadatas"][0], result["distances"][0]):
        # Convert Chroma cosine distance into a higher-is-better similarity score.
        grouped[metadata["agent"]].append(1 - float(distance))
    scores = {agent: (sum(values) / len(values) if values else 0.0) for agent, values in grouped.items()}
    return RoutingDecision(routed_agent, scores, "groq")
