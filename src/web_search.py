"""Internet search tool exposed only to the Research specialist."""

from ddgs import DDGS
from langchain.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the public web for recent factual information and return concise source results."""
    results = DDGS().text(query, max_results=5)
    formatted = [
        f"[{index}] {item.get('title', 'Untitled')}\nURL: {item.get('href', '')}\n{item.get('body', '')}"
        for index, item in enumerate(results, start=1)
    ]
    return "\n\n".join(formatted) if formatted else "No web results found."