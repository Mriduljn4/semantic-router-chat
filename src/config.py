from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent / ".env",
        extra="ignore",
    )

    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "groq/compound-mini"
    GROQ_INTENT_MODEL: str = "qwen/qwen3.6-27b"
    INTENT_CLASSIFIER_PROVIDER: str = "groq"
    TAVILY_API_KEY: str = ""
    TAVILY_CACHE_TTL_SECONDS: int = 600
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.5-flash"
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "openai/gpt-4o-mini"
    LLM_PRIMARY_PROVIDER: str = "openrouter"  # Options: "groq", "openrouter"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_BACKEND: str = "sentence-transformers"
    CHROMA_PERSIST_DIR: str = "./chroma_data"
    ROUTER_TOP_K: int = 6
    LANGSMITH_TRACING: str = "true"
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "agents-ai-semantic-router"
    LANGCHAIN_TRACING_V2: str = "true"
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = ""

    @property
    def chroma_path(self) -> Path:
        return Path(self.CHROMA_PERSIST_DIR)


@lru_cache
def get_settings() -> Settings:
    return Settings()
