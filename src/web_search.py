"""Tavily-backed internet search used by the Research specialist."""

from langchain.tools import tool
from tavily import TavilyClient

from src.config import get_settings


@tool
def web_search(query: str) -> str:
    """Search the public web with Tavily and return concise factual results."""
    api_key = get_settings().TAVILY_API_KEY
    if not api_key:
        raise RuntimeError("Tavily search is not configured.")

    response = TavilyClient(api_key=api_key).search(
        query=query,
        search_depth="basic",
        max_results=5,
        include_answer="basic",
    )
    answer = response.get("answer", "")
    results = response.get("results", [])
    formatted = [
        f"[{index}] {item.get('title', 'Untitled')}\nURL: {item.get('url', '')}\n{item.get('content', '')}"
        for index, item in enumerate(results, start=1)
    ]
    output = "\n\n".join(formatted)
    if answer:
        output = f"Tavily summary:\n{answer}\n\nSources:\n{output}"
    return output or "No web results found."