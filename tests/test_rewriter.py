from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage

from src.rewriter import rewrite_query


def test_rewrite_query_returns_model_normalization():
    model = Mock()
    model.invoke.return_value = AIMessage(content="Explain retrieval-augmented generation (RAG).")
    settings = SimpleNamespace(QUERY_REWRITING_ENABLED=True, LLM_PRIMARY_PROVIDER="gemini")
    with patch("src.rewriter.get_settings", return_value=settings), patch(
        "src.rewriter.get_model", return_value=model
    ) as get_model:
        result = rewrite_query("rag?")
    assert result == "Explain retrieval-augmented generation (RAG)."
    get_model.assert_called_once_with("gemini")


def test_rewrite_query_falls_back_to_original_on_provider_error():
    settings = SimpleNamespace(QUERY_REWRITING_ENABLED=True, LLM_PRIMARY_PROVIDER="gemini")
    with patch("src.rewriter.get_settings", return_value=settings), patch(
        "src.rewriter.get_model", side_effect=RuntimeError("provider unavailable")
    ):
        assert rewrite_query("rag?") == "rag?"