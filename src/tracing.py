import os

from src.config import get_settings


def configure_tracing() -> None:
    settings = get_settings()
    values = {
        "LANGSMITH_TRACING": settings.LANGSMITH_TRACING or settings.LANGCHAIN_TRACING_V2,
        "LANGSMITH_API_KEY": settings.LANGSMITH_API_KEY or settings.LANGCHAIN_API_KEY,
        "LANGSMITH_PROJECT": settings.LANGSMITH_PROJECT or settings.LANGCHAIN_PROJECT,
    }
    for name, value in values.items():
        if value:
            os.environ[name] = value
