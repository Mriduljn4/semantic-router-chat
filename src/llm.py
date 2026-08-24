from functools import lru_cache
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from src.config import get_settings


class LLMProviderError(RuntimeError):
    """Raised when neither configured LLM provider can generate a response."""


@lru_cache
def get_model(provider: Literal["groq", "gemini"]) -> BaseChatModel:
    """Return a cached LangChain v1 chat model for a configured provider."""
    settings = get_settings()
    if provider == "groq":
        return ChatGroq(
            model=settings.GROQ_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=0,
        )
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        api_key=settings.GEMINI_API_KEY,
        temperature=0,
    )
