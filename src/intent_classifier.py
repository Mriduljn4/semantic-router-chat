"""Fast Groq intent classification with structured output."""

from typing import Literal

from pydantic import BaseModel, Field

from src.config import get_settings
from src.llm import get_model

Intent = Literal["research", "coding", "data"]


class IntentClassification(BaseModel):
    """Validated specialist category returned by the intent classifier."""

    agent: Intent = Field(description="The best specialist for the user request.")


CLASSIFIER_PROMPT = """Classify the request into exactly one specialist.
- research: explanations, comparisons, investigations, or factual research
- coding: software design, debugging, implementation, tests, and developer tools
- data: SQL, metrics, analytics, transformations, dashboards, and visualization
Select the user's primary intent. Do not answer the request."""


def classify_intent(query: str) -> Intent:
    """Use Groq structured output to select a specialist for a user query."""
    settings = get_settings()
    model = get_model("groq")
    if settings.GROQ_INTENT_MODEL != settings.GROQ_MODEL:
        model = model.bind(model=settings.GROQ_INTENT_MODEL)
    result = model.with_structured_output(IntentClassification).invoke(
        [("system", CLASSIFIER_PROMPT), ("human", query)]
    )
    return result.agent if isinstance(result, IntentClassification) else IntentClassification.model_validate(result).agent