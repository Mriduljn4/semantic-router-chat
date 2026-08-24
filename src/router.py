from dataclasses import dataclass
from typing import Literal

from src.chroma_store import get_capabilities_collection
from src.config import get_settings
from src.embeddings import embed_query

AGENTS = ("research", "coding", "data")


@dataclass
class RoutingDecision:
    """The specialist selected by semantic similarity and its per-agent scores."""

    routed_agent: Literal["research", "coding", "data"]
    router_scores: dict[str, float]


def route(query: str) -> RoutingDecision:
    """Classify a query by averaging nearest capability-profile similarities.

    Chroma returns cosine distances for the nearest capability examples. Because
    the collection uses cosine space, $1 - distance$ is the cosine similarity.
    Profiles are grouped by agent and each agent receives the mean similarity of
    its retrieved profiles; the highest mean selects the specialist.
    """
    result = get_capabilities_collection().query(
        query_embeddings=[embed_query(query)], n_results=get_settings().ROUTER_TOP_K,
        include=["metadatas", "distances"],
    )
    grouped = {agent: [] for agent in AGENTS}
    for metadata, distance in zip(result["metadatas"][0], result["distances"][0]):
        # Convert Chroma cosine distance into a higher-is-better similarity score.
        grouped[metadata["agent"]].append(1 - float(distance))
    scores = {agent: (sum(values) / len(values) if values else 0.0) for agent, values in grouped.items()}
    # A future confidence threshold can route uncertain queries to a generalist.
    best = max(scores, key=scores.get)
    return RoutingDecision(best, scores)  # type: ignore[arg-type]
