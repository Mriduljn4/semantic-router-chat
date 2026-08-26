"""Small regression catalog for routing, web-search use, and answer quality."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EvaluationCase:
    """Expected outcome for one end-to-end router evaluation."""

    query: str
    expected_agent: Literal["research", "coding", "data"]
    expects_web_search: bool
    quality_checks: tuple[str, ...]


EVALUATION_CASES = (
    EvaluationCase("Explain retrieval-augmented generation.", "research", False, ("retriev", "generation")),
    EvaluationCase("What are the latest developments involving Virat Kohli?", "research", True, ("kohli",)),
    EvaluationCase("Write a FastAPI endpoint with Pydantic validation.", "coding", False, ("fastapi", "pydantic")),
    EvaluationCase("Write SQL for monthly active users.", "data", False, ("select", "month")),
)