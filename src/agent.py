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

    try:
        answer = _invoke_agent(get_agent(agent_name, "groq"), prompt)
        return AgentResult(answer, "groq", context)
    except Exception:
        logger.warning("Groq agent failed; trying Gemini fallback.", exc_info=True)
        try:
            answer = _invoke_agent(get_agent(agent_name, "gemini"), prompt)
            return AgentResult(answer, "gemini", context)
        except Exception as error:
            logger.error("Gemini fallback agent failed.", exc_info=True)
            raise LLMProviderError("Both configured LLM providers failed.") from error
