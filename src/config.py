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
    NVIDIA_API_KEY: str = "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"
    NVIDIA_MODEL: str = "nvidia/nemotron-3.5-lightning-30b-a3b"
    LLM_PRIMARY_PROVIDER: str = "nvidia"  # Options: "groq", "nvidia"
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
