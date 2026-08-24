from dataclasses import dataclass
from functools import lru_cache
import logging
from typing import AsyncIterator, Literal

from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk, BaseMessage

from src.chroma_store import get_research_docs_collection
from src.config import get_settings
from src.embeddings import embed_query
from src.guardrails import sanitize_output
from src.llm import LLMProviderError, get_model
from src.web_search import web_search

logger = logging.getLogger(__name__)

ANSWER_FORMAT = """Write complete, explanatory answers for a chat interface:
- Start with a direct answer, then explain the important reasoning, concepts, and trade-offs.
- For conceptual questions, cover what it is, how it works, why it matters, and a practical example when useful.
- For implementation questions, provide an actionable approach, explain the key decisions, and include a working example when appropriate.
- Use short Markdown headings, bullets, and numbered steps when they improve clarity.
- Use fenced code blocks with a language label for code or SQL.
- Keep paragraphs readable, but do not make an answer so brief that it skips the explanation. Only give a short answer when the user explicitly asks for one.
- Do not include citations, source lists, links, or reference sections unless the user explicitly asks for them."""

PROMPTS = {
    "research": """You are a research expert. Use supplied local context for factual claims and state when it is insufficient.
For timely or missing information, use the web_search tool. Do not claim information you cannot support.""",
    "coding": "You are a coding expert. Give practical, correct programming guidance, explain why the solution works, and include complete examples when useful.",
    "data": "You are a data expert. Help with SQL, analysis, transformations, metrics, and visualization reasoning.",
}


@dataclass
class AgentResult:
    """Normalized output returned by a specialist agent invocation."""

    answer: str
    provider_used: Literal["groq", "gemini"]
    context: list[str]
    tools_used: list[str]


@lru_cache
def get_agent(agent_name: str, provider: Literal["groq", "gemini"]):
    """Build a LangChain v1 specialist agent for the chosen provider."""
    return create_agent(
        model=get_model(provider),
        # Only Research can access the internet; Coding and Data remain tool-free.
        tools=[web_search] if agent_name == "research" else [],
        system_prompt=f"{PROMPTS[agent_name]}\n\n{ANSWER_FORMAT}",
        name=f"{agent_name}_{provider}",
    )


def _message_text(message: BaseMessage) -> str:
    """Extract displayable text from plain and Gemini-style structured messages."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_blocks = [
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
        ]
        if text_blocks:
            return "\n".join(text_blocks)
    return str(content)


def _invoke_agent(agent, prompt: str) -> tuple[str, list[str]]:
    """Invoke a LangChain agent and return final text plus any tools it called."""
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    tools_used = list(
        dict.fromkeys(
            message.name
            for message in result["messages"]
            if getattr(message, "type", "") == "tool" and getattr(message, "name", None)
        )
    )
    return _message_text(result["messages"][-1]), tools_used


def _research_prompt(query: str) -> tuple[str, list[str]]:
    """Build the Research prompt with local RAG context when required."""
    result = get_research_docs_collection().query(
        query_embeddings=[embed_query(query)],
        n_results=get_settings().ROUTER_TOP_K,
        include=["documents"],
    )
    documents = result.get("documents", [])
    context = documents[0] if documents else []
    return f"Context:\n{'\n\n'.join(context)}\n\nQuestion: {query}", context


def _provider_order() -> tuple[Literal["groq", "gemini"], Literal["groq", "gemini"]]:
    """Return the configured answer provider followed by its single fallback."""
    return ("gemini", "groq") if get_settings().LLM_PRIMARY_PROVIDER == "gemini" else ("groq", "gemini")


async def astream_agent(agent_name: str, query: str) -> AsyncIterator[dict[str, str]]:
    """Yield actual provider tokens and tool activity from a specialist agent."""
    prompt = query
    if agent_name == "research":
        import asyncio

        # Chroma is synchronous; keep retrieval off the event loop.
        prompt, _ = await asyncio.to_thread(_research_prompt, query)

    providers = _provider_order()
    for index, provider in enumerate(providers):
        emitted_token = False
        try:
            agent = get_agent(agent_name, provider)
            async for message, _metadata in agent.astream(
                {"messages": [{"role": "user", "content": prompt}]},
                stream_mode="messages",
            ):
                if getattr(message, "type", "") == "tool" and getattr(message, "name", None):
                    yield {"type": "tool", "name": message.name, "provider": provider}
                if isinstance(message, AIMessageChunk):
                    text = _message_text(message)
                    if text:
                        emitted_token = True
                        yield {"type": "token", "text": text, "provider": provider}
            return
        except Exception as error:
            # A fallback is safe only before output begins, avoiding mixed answers.
            if emitted_token or index == len(providers) - 1:
                raise LLMProviderError(
                    "Both configured LLM providers failed.",
                    reason=_provider_error_reason(error),
                    attempts={provider: _safe_provider_failure(error)},
                ) from error
            logger.warning("Streaming %s agent failed before output; trying fallback.", provider, exc_info=True)


def _provider_error_reason(error: Exception) -> str:
    """Map provider exceptions to safe, actionable categories for API clients."""
    message = str(error).lower()
    if "model" in message and ("not found" in message or "not available" in message):
        return "model_unavailable"
    if any(term in message for term in ("api key", "api_key", "unauthorized", "authentication", "permission denied")):
        return "authentication_failed"
    if any(term in message for term in ("quota", "rate limit", "resource exhausted")):
        return "quota_or_rate_limit"
    return "provider_error"


def _safe_provider_failure(error: Exception) -> str:
    """Return diagnostics safe to share with API clients; never include provider secrets."""
    return f"{_provider_error_reason(error)} ({type(error).__name__})"


def run_agent(agent_name: str, query: str) -> AgentResult:
    """Run a specialist, adding RAG context for research and provider fallback.

    Research first retrieves local documents with the same embedding backend used
    for ingestion. Coding and data specialists receive the query directly.
    """
    context: list[str] = []
    prompt = query
    if agent_name == "research":
        # RAG is restricted to the research specialist to avoid irrelevant context.
        prompt, context = _research_prompt(query)

    providers = _provider_order()

    try:
        answer, tools_used = _invoke_agent(get_agent(agent_name, providers[0]), prompt)
        return AgentResult(sanitize_output(answer), providers[0], context, tools_used)
    except Exception as primary_error:
        # Only the alternate provider is tried; no unbounded retry loop is used.
        logger.warning("Primary %s agent failed; trying %s fallback.", providers[0], providers[1], exc_info=True)
        try:
            answer, tools_used = _invoke_agent(get_agent(agent_name, providers[1]), prompt)
            return AgentResult(sanitize_output(answer), providers[1], context, tools_used)
        except Exception as error:
            logger.error("Fallback %s agent failed.", providers[1], exc_info=True)
            attempts = {
                providers[0]: _safe_provider_failure(primary_error),
                providers[1]: _safe_provider_failure(error),
            }
            raise LLMProviderError(
                "Both configured LLM providers failed.",
                reason=_provider_error_reason(error),
                attempts=attempts,
            ) from error
