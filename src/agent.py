from dataclasses import dataclass
from functools import lru_cache
import logging
from typing import Literal

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage

from src.chroma_store import get_research_docs_collection
from src.config import get_settings
from src.embeddings import embed_query
from src.llm import LLMProviderError, get_model

logger = logging.getLogger(__name__)

PROMPTS = {
    "research": "You are a research expert. Use only the supplied context for factual claims and state when it is insufficient.",
    "coding": "You are a coding expert. Give practical, correct programming guidance and concise examples when useful.",
    "data": "You are a data expert. Help with SQL, analysis, transformations, metrics, and visualization reasoning.",
}


@dataclass
class AgentResult:
    answer: str
    provider_used: Literal["groq", "gemini"]
    context: list[str]


@lru_cache
def get_agent(agent_name: str, provider: Literal["groq", "gemini"]):
    """Build a LangChain v1 specialist agent for the chosen provider."""
    return create_agent(
        model=get_model(provider),
        tools=[],
        system_prompt=PROMPTS[agent_name],
        name=f"{agent_name}_{provider}",
    )


def _message_text(message: BaseMessage) -> str:
    content = message.content
    return content if isinstance(content, str) else str(content)


def _invoke_agent(agent, prompt: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    return _message_text(result["messages"][-1])


def _provider_error_reason(error: Exception) -> str:
    message = str(error).lower()
    if "model" in message and ("not found" in message or "not available" in message):
        return "model_unavailable"
    if any(term in message for term in ("api key", "api_key", "unauthorized", "authentication", "permission denied")):
        return "authentication_failed"
    if any(term in message for term in ("quota", "rate limit", "resource exhausted")):
        return "quota_or_rate_limit"
    return "provider_error"


def run_agent(agent_name: str, query: str) -> AgentResult:
    context: list[str] = []
    prompt = query
    if agent_name == "research":
        result = get_research_docs_collection().query(
            query_embeddings=[embed_query(query)],
            n_results=get_settings().ROUTER_TOP_K,
            include=["documents"],
        )
        documents = result.get("documents", [])
        context = documents[0] if documents else []
        context_text = "\n".join(context)
        prompt = f"Context:\n{context_text}\n\nQuestion: {query}"

    primary = get_settings().LLM_PRIMARY_PROVIDER
    providers: tuple[Literal["groq", "gemini"], Literal["groq", "gemini"]]
    providers = ("gemini", "groq") if primary == "gemini" else ("groq", "gemini")

    try:
        answer = _invoke_agent(get_agent(agent_name, providers[0]), prompt)
        return AgentResult(answer, providers[0], context)
    except Exception:
        logger.warning("Primary %s agent failed; trying %s fallback.", providers[0], providers[1], exc_info=True)
        try:
            answer = _invoke_agent(get_agent(agent_name, providers[1]), prompt)
            return AgentResult(answer, providers[1], context)
        except Exception as error:
            logger.error("Fallback %s agent failed.", providers[1], exc_info=True)
            raise LLMProviderError(
                "Both configured LLM providers failed.",
                reason=_provider_error_reason(error),
            ) from error
