import os

from src.config import get_settings


def configure_tracing() -> None:
    settings = get_settings()
    for name in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"):
        value = getattr(settings, name)
        if value:
            os.environ[name] = value
