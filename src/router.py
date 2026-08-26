from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

from src.chroma_store import get_capabilities_collection
from src.config import get_settings
from src.embeddings import embed_query
from src.intent_classifier import classify_intent


AgentName = Literal["general", "research", "coding", "data"]
ProviderName = Literal["groq", "nvidia"]

AGENTS: tuple[AgentName, ...] = (
    "general",
    "research",
    "coding",
    "data",
)


@dataclass
class RoutingDecision:
    """Selected specialist and informational capability similarity scores."""

    routed_agent: AgentName
    router_scores: dict[str, float]
    classifier_used: ProviderName


def route(query: str) -> RoutingDecision:
    """
    Classify intent with Groq and use NVIDIA only as a fallback.

    Chroma scores are informational only. They are displayed in the UI but do
    not select the specialist because semantic similarity alone is unreliable
    for short, conversational, or follow-up messages.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        classification_future = executor.submit(classify_intent, query)

        result = get_capabilities_collection().query(
            query_embeddings=[embed_query(query)],
            n_results=get_settings().ROUTER_TOP_K,
            include=["metadatas", "distances"],
        )

        classification = classification_future.result()

    grouped_scores: dict[str, list[float]] = {
        agent: []
        for agent in AGENTS
    }

    metadatas = result.get("metadatas", [[]])
    distances = result.get("distances", [[]])

    for metadata, distance in zip(metadatas[0], distances[0]):
        agent_name = metadata.get("agent")

        if agent_name not in grouped_scores:
            continue

        grouped_scores[agent_name].append(
            1 - float(distance)
        )

    router_scores = {
        agent: (
            sum(scores) / len(scores)
            if scores
            else 0.0
        )
        for agent, scores in grouped_scores.items()
    }

    return RoutingDecision(
        routed_agent=classification.intent,
        router_scores=router_scores,
        classifier_used=classification.provider_used,
    )