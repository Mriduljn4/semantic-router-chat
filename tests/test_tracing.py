import os
from types import SimpleNamespace
from unittest.mock import patch

from src.tracing import configure_tracing


def test_configure_tracing_supports_legacy_langchain_settings():
    settings = SimpleNamespace(
        LANGSMITH_TRACING="",
        LANGSMITH_API_KEY="",
        LANGSMITH_PROJECT="",
        LANGCHAIN_TRACING_V2="true",
        LANGCHAIN_API_KEY="legacy-key",
        LANGCHAIN_PROJECT="legacy-project",
    )
    with patch("src.tracing.get_settings", return_value=settings), patch.dict(os.environ, {}, clear=True):
        configure_tracing()
        assert os.environ["LANGSMITH_TRACING"] == "true"
        assert os.environ["LANGSMITH_API_KEY"] == "legacy-key"
        assert os.environ["LANGSMITH_PROJECT"] == "legacy-project"