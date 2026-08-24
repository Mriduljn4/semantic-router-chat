from dataclasses import dataclass
import logging
from typing import Literal

from src.chroma_store import get_capabilities_collection
from src.config import get_settings
from src.embeddings import embed_query
from src.intent_classifier import classify_intent

logger = logging.getLogger(__name__)

AGENTS = ("research", "coding", "data")


@dataclass
class RoutingDecision:
    """The specialist selected by semantic similarity and its per-agent scores."""

    routed_agent: Literal["research", "coding", "data"]
    router_scores: dict[str, float]
    classifier_used: Literal["groq", "semantic"]


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
    semantic_best = max(scores, key=scores.get)
    try:
        return RoutingDecision(classify_intent(query), scores, "groq")
    except Exception:
        # Chroma remains a reliable intent fallback if Groq is unavailable.
        logger.warning("Groq intent classifier failed; using semantic routing.", exc_info=True)
        return RoutingDecision(semantic_best, scores, "semantic")  # type: ignore[arg-type]
