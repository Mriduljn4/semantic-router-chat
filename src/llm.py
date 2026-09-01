from functools import lru_cache
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from src.config import get_settings


class LLMProviderError(RuntimeError):
    """Raised when neither configured LLM provider can generate a response."""

    def __init__(
        self,
        message: str,
        reason: str = "provider_error",
        attempts: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.attempts = attempts or {}


@lru_cache
def get_model(provider: Literal["groq", "nvidia"]) -> BaseChatModel:
    """Return a cached LangChain v1 chat model for a configured provider."""
    settings = get_settings()
    if provider == "nvidia":
        return ChatNVIDIA(
            model=settings.NVIDIA_MODEL,
            api_key=settings.NVIDIA_API_KEY,
            temperature=0.3,
        )
    if provider == "groq":
        return ChatGroq(
            model=settings.GROQ_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=0.3,
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")