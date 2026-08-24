import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent import _message_text
from src.config import get_settings
from src.llm import get_model

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """Rewrite the user's request for semantic routing and retrieval.
Preserve the user's intent, facts, language, and constraints. Expand only obvious shorthand
or ambiguity. Do not answer the request, add facts, or mention this instruction. Return only
the rewritten request in one concise sentence."""


def rewrite_query(query: str) -> str:
    """Normalize a query before intent classification, falling back safely to the original."""
    settings = get_settings()
    if not settings.QUERY_REWRITING_ENABLED:
        return query

    provider = "gemini" if settings.LLM_PRIMARY_PROVIDER == "gemini" else "groq"
    try:
        response = get_model(provider).invoke(
            [SystemMessage(REWRITE_PROMPT), HumanMessage(query)]
        )
        rewritten = _message_text(response).strip()
        return rewritten or query
    except Exception:
        logger.warning("Query rewriting failed; using the original query.", exc_info=True)
        return query