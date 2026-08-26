"""Tavily-backed internet search used by the Research specialist."""

from functools import lru_cache
from time import monotonic

from langchain.tools import tool
from tavily import TavilyClient

from src.config import get_settings

_search_cache_created: dict[str, float] = {}


@lru_cache(maxsize=128)
def _cached_search(query: str) -> str:
    """Fetch and cache one normalized Tavily query."""
    api_key = get_settings().TAVILY_API_KEY
    if not api_key:
        raise RuntimeError("Tavily search is not configured.")

    response = TavilyClient(api_key=api_key).search(
        query=query,
        search_depth="basic",
        max_results=5,
        include_answer="basic",
    )
    results = response.get("results", [])
    if not results:
        return "No web results found."

    answer = response.get("answer", "")
    formatted = [
        f"[{index}] {item.get('title', 'Untitled')}\nURL: {item.get('url', '')}\n{item.get('content', '')}"
        for index, item in enumerate(results, start=1)
    ]
    output = "\n\n".join(formatted)
    if answer:
        output = f"Tavily summary:\n{answer}\n\nSources:\n{output}"
    return output


@tool
def web_search(query: str) -> str:
    """Search Tavily and cache identical normalized queries for a short period."""
    normalized_query = " ".join(query.lower().split())
    now = monotonic()
    cached_at = _search_cache_created.get(normalized_query)
    if cached_at is not None and now - cached_at >= get_settings().TAVILY_CACHE_TTL_SECONDS:
        _cached_search.cache_clear()
        _search_cache_created.clear()

    result = _cached_search(normalized_query)
    _search_cache_created[normalized_query] = now
    return result