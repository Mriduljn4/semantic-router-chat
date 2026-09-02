"""Intent classification with deterministic heuristics and LLM fallback."""

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field

from src.config import get_settings
from src.llm import LLMProviderError, get_model


Intent = Literal["general", "research", "coding", "data"]
ClassifierName = Literal["rules", "groq", "openrouter"]
ProviderName = Literal["groq", "openrouter"]


@dataclass(frozen=True)
class IntentResult:
    """Validated intent selection plus the provider that produced it."""

    intent: Intent
    provider_used: ClassifierName


class IntentClassification(BaseModel):
    """Structured specialist category returned by an intent model."""

    agent: Intent = Field(
        description="The best specialist for the user request.",
    )


CLASSIFIER_PROMPT = """You are an expert request router.
Your only job is to classify the user's request into exactly one of the four intents below.

Allowed intents:
- general: Greetings, introductions, casual conversation, thank-you messages, simple memory-based follow-ups, and messages that do NOT require research, code, SQL, or data-pipeline work.
- research: Factual explanations, information about named people/companies/products, history, current events/news, comparisons, deep-dive investigations, and recommendations requiring external knowledge.
- coding: Programming, source code creation/modification, debugging, APIs, software architecture, testing, deployment, and developer tooling.
- data: SQL queries, dataset manipulation, analytics, metrics definitions, dashboards, data transformations, statistics, visualizations, data engineering, ETL, and data pipelines.

Routing examples:
- "Hi" -> general
- "My name is Mridul" -> general
- "What is Amazon Bedrock?" -> research
- "Compare AWS and Azure" -> research
- "Fix this FastAPI error" -> coding
- "Write a SQL query for monthly revenue" -> data

Rules:
- You must return exactly one intent string.
- Do not answer the user's question.
- Do not add any explanations, introductory text, or markdown formatting.
- Only output the exact name of the intent.
"""


def heuristic_intent(query: str) -> Intent | None:
    """Return an intent only when explicit lexical rules are unambiguous."""
    normalized = " ".join(query.casefold().split())
    if not normalized:
        return "general"

    if re.fullmatch(
        r"(?:hi|hello|hey|thanks|thank you|good (?:morning|afternoon|evening)"
        r"|how are you|who am i|what is my name)[.!? ]*",
        normalized,
    ) or re.match(r"^my name is\b", normalized):
        return "general"

    # Explicit SQL and analytics language wins over generic words such as
    # "write", "function", or "pipeline".
    if re.search(
        r"\b(?:sql|dataset|dataframe|pandas|analytics|metric|dashboard|etl|"
        r"data pipeline|data visualization|group by|window function|monthly active users)\b",
        normalized,
    ):
        return "data"

    if re.search(
        r"\b(?:python|javascript|typescript|java|c#|fastapi|flask|react|pydantic|"
        r"sqlalchemy|httpx|npm|git|docker|kubernetes)\b",
        normalized,
    ) or re.search(
        r"\b(?:write|create|build|generate|implement|fix|debug|refactor|test)\b"
        r".*\b(?:code|function|class|method|api|endpoint|route|script|component|"
        r"handler|service|bug|error|exception|test)\b",
        normalized,
    ):
        return "coding"

    if re.search(
        r"^(?:what|who|when|where|why|how)\b|\b(?:explain|compare|latest|news|"
        r"history|research|overview|difference between|recommend)\b",
        normalized,
    ):
        return "research"

    # Ambiguous and conversational follow-ups are delegated to the model,
    # which can use the conversation history maintained by the specialist.
    return None


@lru_cache
def get_intent_model(provider: ProviderName):
    """Return the configured provider model for intent classification."""
    return get_model(provider)


def _validate_intent_result(result: object) -> Intent:
    """Validate structured or text provider output as one allowed intent."""
    if isinstance(result, IntentClassification):
        return result.agent

    if isinstance(result, dict):
        return IntentClassification.model_validate(result).agent

    content = getattr(result, "content", result)

    if isinstance(content, str):
        normalized = content.strip().lower()

        if normalized in {"general", "research", "coding", "data"}:
            return normalized  # type: ignore[return-value]

        return IntentClassification.model_validate_json(content).agent

    return IntentClassification.model_validate(content).agent


def _classify_with_provider(
    provider: ProviderName,
    query: str,
) -> Intent:
    """Classify one query using a provider with Pydantic structured output."""
    model = get_intent_model(provider)

    structured_model = model.with_structured_output(IntentClassification)

    result = structured_model.invoke(
        [
            ("system", CLASSIFIER_PROMPT),
            ("human", query),
        ]
    )

    return _validate_intent_result(result)


def classify_intent(query: str) -> IntentResult:
    """Use a deterministic short-circuit for common prompts before LLM fallback."""
    heuristic = heuristic_intent(query)
    if heuristic is not None:
        return IntentResult(intent=heuristic, provider_used="rules")

    providers: tuple[ProviderName, ...] = ("groq",)
    failures: dict[str, str] = {}

    for provider in providers:
        try:
            intent = _classify_with_provider(provider, query)

            return IntentResult(
                intent=intent,
                provider_used=provider,
            )

        except Exception as error:
            failures[provider] = f"{type(error).__name__}: {error}"

    raise LLMProviderError(
        "Both intent-classification providers failed.",
        reason="provider_error",
        attempts=failures,
    )