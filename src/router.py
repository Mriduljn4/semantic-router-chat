from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Literal

from src.chroma_store import get_capabilities_collection
from src.config import get_settings
from src.embeddings import embed_query
from src.intent_classifier import IntentResult, classify_intent, heuristic_intent


AgentName = Literal["general", "research", "coding", "data"]
ClassifierName = Literal["rules", "groq", "openrouter"]

AGENTS: tuple[AgentName, ...] = (
    "general",
    "research",
    "coding",
    "data",
)

_MAX_TRACKED_CONVERSATIONS = 1_000
_conversation_agents: OrderedDict[str, AgentName] = OrderedDict()
_conversation_agents_lock = Lock()


def _previous_agent(conversation_id: str | None) -> AgentName | None:
    """Return and refresh the most recent specialist for a conversation."""
    if conversation_id is None:
        return None

    with _conversation_agents_lock:
        agent = _conversation_agents.get(conversation_id)
        if agent is not None:
            _conversation_agents.move_to_end(conversation_id)
        return agent


def _remember_agent(conversation_id: str | None, agent: AgentName) -> None:
    """Store bounded in-process routing state for ambiguous follow-ups."""
    if conversation_id is None:
        return

    with _conversation_agents_lock:
        _conversation_agents[conversation_id] = agent
        _conversation_agents.move_to_end(conversation_id)
        while len(_conversation_agents) > _MAX_TRACKED_CONVERSATIONS:
            _conversation_agents.popitem(last=False)


@dataclass
class RoutingDecision:
    """Selected specialist and informational capability similarity scores."""

    routed_agent: AgentName
    router_scores: dict[str, float]
    classifier_used: ClassifierName


def route(query: str, conversation_id: str | None = None) -> RoutingDecision:
    """
    Classify intent using heuristic rules or Groq LLM.

    Chroma scores are informational only. They are displayed in the UI but do
    not select the specialist because semantic similarity alone is unreliable
    for short, conversational, or follow-up messages.
    """
    heuristic = heuristic_intent(query)
    previous_agent = _previous_agent(conversation_id)
    inherited_classification = (
        IntentResult(intent=previous_agent, provider_used="rules")
        if heuristic is None and previous_agent is not None
        else None
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        classification_future = (
            None
            if inherited_classification is not None
            else executor.submit(classify_intent, query)
        )

        result = get_capabilities_collection().query(
            query_embeddings=[embed_query(query)],
            n_results=get_settings().ROUTER_TOP_K,
            include=["metadatas", "distances"],
        )

        classification = (
            inherited_classification
            if inherited_classification is not None
            else classification_future.result()  # type: ignore[union-attr]
        )

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

    decision = RoutingDecision(
        routed_agent=classification.intent,
        router_scores=router_scores,
        classifier_used=classification.provider_used,
    )
    _remember_agent(conversation_id, decision.routed_agent)
    return decision