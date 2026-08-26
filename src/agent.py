from dataclasses import dataclass
from functools import lru_cache
import logging
import re
from typing import AsyncIterator, Literal

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage

from src.chroma_store import get_research_docs_collection
from src.config import get_settings
from src.embeddings import embed_query
from src.guardrails import sanitize_output
from src.llm import LLMProviderError, get_model
from src.web_search import web_search

logger = logging.getLogger(__name__)

ANSWER_FORMAT = """Produce a useful, professional answer for a chat interface.

Response rules:
- Answer the user's actual question immediately. Do not restate the question or describe your role.
- Match the depth of the response to the complexity of the request.
- Prefer concise, information-dense explanations over filler or repetition.
- Use short Markdown headings, bullets, numbered steps, or tables only when they improve readability.
- Define important technical terms the first time they appear when the user may not know them.
- Clearly distinguish facts, assumptions, recommendations, and uncertainty when relevant.
- Never invent facts, APIs, configuration values, libraries, files, results, benchmarks, or system behavior.
- If required information is missing, state the assumption or ask for the minimum clarification needed.
- Do not add unrelated code, SQL, architecture diagrams, dashboards, data analysis, implementation details, or examples.
- Provide examples, checklists, calculations, or next steps only when they directly help answer the user's request.
- For code, configuration, commands, and SQL, use fenced code blocks with the correct language label.
- Code should be complete enough to run when presented as an implementation solution.
- Explain non-obvious implementation choices briefly.
- For production-oriented solutions, consider validation, error handling, security, observability, performance, maintainability, and testing when relevant.
- Do not over-engineer simple requests.
- For troubleshooting, separate the likely root cause from possible causes and provide diagnostic steps in priority order.
- Do not claim that something was tested, executed, verified, deployed, or measured unless it actually was.
- Do not include citations, source lists, links, or reference sections unless the user explicitly asks for them.
"""


PROMPTS = {
    "research": """You are a research specialist.

Deliver clear, fact-focused explanations, comparisons, investigations, and recommendations.

Guidelines:
- Identify the user's actual decision or question before presenting information.
- Prioritize authoritative and primary sources when external information is available.
- For time-sensitive information such as prices, releases, policies, announcements, dates, leadership, or current product capabilities, prefer current web evidence.
- Clearly separate established facts from interpretation, inference, and recommendation.
- Compare alternatives using the criteria that matter to the user's question rather than listing features without context.
- Highlight important trade-offs, limitations, risks, and uncertainty.
- Do not manufacture missing information. If evidence is insufficient, say so.
- For technical research, distinguish official capabilities from community practices or assumptions.
- End with a concise practical conclusion when the research supports one.

""",

    "coding": """You are a senior software engineer.

Provide correct, secure, maintainable, readable, and practical implementation guidance.

Guidelines:
- First identify the likely goal, constraints, existing stack, and expected behavior from the user's request.
- Preserve the user's stated programming language, framework, APIs, architecture, and dependencies unless there is a strong reason to recommend a change.
- Recommend the simplest sound solution before considering more complex alternatives.
- Explain why the approach works and mention meaningful trade-offs.
- Never invent project files, APIs, methods, dependencies, environment variables, configuration values, or runtime behavior.
- When information about an existing codebase is missing, make the smallest explicit assumption necessary.
- Use meaningful names, type hints, clear structure, and concise comments.
- For reusable production code, prefer small focused functions/classes with clear responsibilities.
- Include validation and error handling appropriate to the failure modes.
- Consider security implications such as secrets, input validation, injection risks, authentication, authorization, and unsafe deserialization when relevant.
- Consider logging, configuration management, retries, timeouts, idempotency, and observability when relevant to production systems.
- Avoid unnecessary abstractions, frameworks, or dependencies.
- For debugging:
  1. identify the most likely root cause,
  2. explain why it occurs,
  3. provide ordered diagnostic steps,
  4. show the smallest safe fix,
  5. mention verification or regression tests.
- Do not claim code has been executed or tested unless it actually has.

""",

    "data": """You are a senior data engineering and analytics specialist.

Help users design reliable data transformations, pipelines, SQL, metrics, data models, and data-quality processes.

Guidelines:
- Identify the business definition, source data, grain, time window, filters, and expected output before proposing a transformation.
- Prevent common data problems such as duplicate joins, incorrect aggregation grain, null handling errors, denominator errors, timezone issues, late-arriving data, schema drift, and inconsistent business definitions.
- Prefer readable, maintainable, portable SQL unless a specific database or warehouse is named.
- State database- or platform-specific assumptions when they materially affect the solution.
- For Python data pipelines, prefer modular, reusable functions with:
  - type hints,
  - clear docstrings,
  - meaningful names,
  - validation,
  - structured error handling,
  - logging where appropriate,
  - configuration separated from business logic.
- For production pipelines, consider:
  - idempotency,
  - incremental processing,
  - retries,
  - checkpointing,
  - schema evolution,
  - data-quality checks,
  - observability,
  - lineage,
  - performance,
  - partitioning,
  - failure recovery.
- When designing transformations, explicitly consider the input and output grain.
- For metrics, explain the numerator, denominator, population, filters, and aggregation level when ambiguity exists.
- For analysis, explain what the result means, what could bias it, and the most useful validation or follow-up check.
- Do not invent schemas, columns, business rules, source-system behavior, or data-quality results.
- Use small illustrative examples only when they materially improve understanding.
- Do not over-engineer a simple transformation.

""",
}


@dataclass
class AgentResult:
    """Normalized output returned by a specialist agent invocation."""

    answer: str
    provider_used: Literal["nvidia","groq"]
    context: list[str]
    tools_used: list[str]


@lru_cache
def get_agent(agent_name: str, provider: Literal["nvidia","groq"]):
    """Build a LangChain v1 specialist agent for the chosen provider."""
    return create_agent(
        model=get_model(provider),
        # ``None`` is important: an empty list is still bound as tool calling by
        # some providers, while this app pre-fetches web context separately.
        tools=None,
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


def _provider_order() -> tuple[Literal["nvidia", "groq"], Literal["nvidia", "groq"]]:
    """Return the configured answer provider followed by its single fallback."""
    return ("nvidia", "groq") if get_settings().LLM_PRIMARY_PROVIDER == "nvidia" else ("groq", "nvidia")


_QUERY_STOP_WORDS = {
    "a", "an", "and", "are", "be", "can", "do", "does", "explain", "for", "how",
    "i", "is", "it", "of", "on", "the", "to", "was", "what", "with", "you",
}

_FRESHNESS_TERMS = {
    "announcement", "announced", "current", "latest", "new", "news", "recent",
    "release", "released", "today", "update", "updated", "yesterday",
}


def _local_context_covers_query(query: str, documents: list[str]) -> bool:
    """Return whether meaningful query terms occur in the local RAG documents."""
    query_terms = {
        term for term in re.findall(r"[a-z0-9]+", query.lower())
        if len(term) > 2 and term not in _QUERY_STOP_WORDS
    }
    if not query_terms:
        return True
    corpus = " ".join(documents).lower()
    return any(term in corpus for term in query_terms)


def _requires_fresh_web_search(query: str) -> bool:
    """Identify questions whose answer may have changed after model training."""
    terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    return bool(terms & _FRESHNESS_TERMS)


def _research_prompt(query: str) -> tuple[str, list[str], list[str]]:
    """Build a grounded Research prompt and search when local RAG lacks coverage."""
    result = get_research_docs_collection().query(
        query_embeddings=[embed_query(query)],
        n_results=get_settings().ROUTER_TOP_K,
        include=["documents"],
    )
    documents = result.get("documents", [])
    local_context = documents[0] if documents else []
    web_context = ""
    web_search_status = "not needed"
    tools_used: list[str] = []

    local_context_is_relevant = _local_context_covers_query(query, local_context)
    needs_web_context = _requires_fresh_web_search(query) or not local_context_is_relevant
    if needs_web_context:
        try:
            web_context = web_search.invoke({"query": query})
            web_search_status = "available"
            if web_context != "No web results found.":
                tools_used.append("web_search")
        except Exception:
            # The agent still has the web-search tool and can retry if needed.
            web_search_status = "unavailable"
            logger.warning("Automatic web search failed; continuing with available context.", exc_info=True)

    prompt = f"""Local context (may be unrelated):
{'\n\n'.join(local_context) or 'No local context found.'}

Web-search status: {web_search_status}

Web context (use when provided):
{web_context or 'No web context was pre-fetched.'}

Question: {query}"""
    if needs_web_context and not web_context:
        prompt += """

Important: This question needs current or externally verified information, but web search is unavailable. Do not provide potentially stale model knowledge as current news. Briefly state that current information could not be verified and ask the user to retry after web search is configured."""
    return prompt, local_context, tools_used


async def astream_agent(agent_name: str, query: str) -> AsyncIterator[dict[str, object]]:
    """Yield actual model tokens, using fallback only before output begins.

    The regular LangChain agent graph is used for complete responses. Streaming
    uses the provider model directly because the specialists are tool-free and
    this preserves token arrival instead of buffering a completed answer.
    """
    import asyncio

    prompt = query
    tools_used: list[str] = []
    if agent_name == "research":
        yield {"type": "status", "message": "Searching research sources…"}
        prompt, _context, tools_used = await asyncio.to_thread(_research_prompt, query)

    system_prompt = f"{PROMPTS[agent_name]}\n\n{ANSWER_FORMAT}"
    providers = _provider_order()
    for index, provider in enumerate(providers):
        emitted_token = False
        try:
            async for message in get_model(provider).astream(
                [("system", system_prompt), ("human", prompt)]
            ):
                text = _message_text(message)
                if not text:
                    continue
                if not emitted_token:
                    emitted_token = True
                    yield {"type": "status", "message": "Generating answer…"}
                    yield {"type": "start", "provider": provider, "tools_used": tools_used}
                yield {"type": "token", "text": text}
            if emitted_token:
                return
            raise RuntimeError("Provider returned an empty response.")
        except Exception as error:
            if emitted_token or index == len(providers) - 1:
                raise LLMProviderError(
                    "Both configured LLM providers failed.",
                    reason=_provider_error_reason(error),
                    attempts={provider: _safe_provider_failure(error)},
                ) from error
            logger.warning("Streaming %s failed before output; trying fallback.", provider, exc_info=True)


def _provider_error_reason(error: Exception) -> str:
    """Map provider exceptions to safe, actionable categories for API clients."""
    message = str(error).lower()
    if "model" in message and ("not found" in message or "not available" in message or "does not exist" in message):
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
    prefetched_tools: list[str] = []
    prompt = query
    if agent_name == "research":
        # RAG is restricted to the research specialist to avoid irrelevant context.
        prompt, context, prefetched_tools = _research_prompt(query)

    providers = _provider_order()

    try:
        answer, tools_used = _invoke_agent(get_agent(agent_name, providers[0]), prompt)
        return AgentResult(sanitize_output(answer), providers[0], context, list(dict.fromkeys(prefetched_tools + tools_used)))
    except Exception as primary_error:
        # Only the alternate provider is tried; no unbounded retry loop is used.
        logger.warning("Primary %s agent failed; trying %s fallback.", providers[0], providers[1], exc_info=True)
        try:
            answer, tools_used = _invoke_agent(get_agent(agent_name, providers[1]), prompt)
            return AgentResult(sanitize_output(answer), providers[1], context, list(dict.fromkeys(prefetched_tools + tools_used)))
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
