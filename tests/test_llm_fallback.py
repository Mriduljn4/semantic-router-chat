from unittest.mock import patch

import pytest

from src.agent import run_agent
from src.llm import LLMProviderError


def test_groq_response_is_used():
    with patch("src.agent.get_agent", return_value=object()) as get_agent, patch(
        "src.agent._invoke_agent", return_value="primary"
    ) as invoke_agent:
        response = run_agent("coding", "hello")
    assert response.answer == "primary"
    assert response.provider_used == "groq"
    get_agent.assert_called_once_with("coding", "groq")
    invoke_agent.assert_called_once()


def test_gemini_is_used_when_groq_fails():
    with patch("src.agent.get_agent", side_effect=[object(), object()]), patch(
        "src.agent._invoke_agent", side_effect=[RuntimeError("rate limited"), "fallback"]
    ):
        response = run_agent("coding", "hello")
    assert response.answer == "fallback"
    assert response.provider_used == "gemini"


def test_provider_error_is_raised_when_both_providers_fail():
    with patch("src.agent.get_agent", side_effect=[object(), object()]), patch(
        "src.agent._invoke_agent", side_effect=[RuntimeError("groq failed"), RuntimeError("gemini failed")]
    ):
        with pytest.raises(LLMProviderError, match="Both configured LLM providers failed") as error:
            run_agent("coding", "hello")
    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "gemini failed"
