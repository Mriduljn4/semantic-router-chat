"""Fast Groq intent classification with structured output."""

from functools import lru_cache
from typing import Literal

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from src.config import get_settings

Intent = Literal["research", "coding", "data"]


class IntentClassification(BaseModel):
    """Validated specialist category returned by the intent classifier."""

    agent: Intent = Field(description="The best specialist for the user request.")


CLASSIFIER_PROMPT = """Classify the request into exactly one specialist.

Routing rules, in priority order:
- research: default for general knowledge, factual questions, named people, companies, products, events, history, news, explanations, comparisons, and investigations. Examples: "Virat Kohli", "What is Amazon Bedrock?", "Compare AWS and Azure".
- coding: only when the user explicitly asks about programming, source code, debugging, implementation, software architecture, tests, or developer tooling.
- data: only when the user explicitly asks to write or analyze SQL, datasets, metrics, dashboards, transformations, visualizations, statistics, or data pipelines.

Do not infer a data task merely because a factual answer could contain statistics or a table. Select the user's primary intent. Do not answer the request."""


@lru_cache
def get_intent_classifier() -> ChatGroq:
    """Create the dedicated Groq model used only for intent selection."""
    settings = get_settings()
    return ChatGroq(
        model=settings.GROQ_INTENT_MODEL,
        api_key=settings.GROQ_API_KEY,
        temperature=0.3,
    )


def classify_intent(query: str) -> Intent:
    """Use the dedicated Groq model to select a specialist for a user query."""
    result = get_intent_classifier().with_structured_output(IntentClassification).invoke(
        [("system", CLASSIFIER_PROMPT), ("human", query)]
    )
    return result.agent if isinstance(result, IntentClassification) else IntentClassification.model_validate(result).agent