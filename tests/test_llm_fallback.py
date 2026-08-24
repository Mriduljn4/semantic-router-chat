from unittest.mock import patch
from types import SimpleNamespace

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


def test_groq_is_used_as_fallback_when_gemini_is_primary_and_fails():
    with patch("src.agent.get_settings", return_value=SimpleNamespace(LLM_PRIMARY_PROVIDER="gemini")), patch(
        "src.agent.get_agent", side_effect=[object(), object()]
    ) as get_agent, patch("src.agent._invoke_agent", side_effect=[RuntimeError("gemini failed"), "fallback"]):
        response = run_agent("coding", "hello")
    assert response.answer == "fallback"
    assert response.provider_used == "groq"
    assert get_agent.call_args_list[0].args == ("coding", "gemini")
    assert get_agent.call_args_list[1].args == ("coding", "groq")


def test_provider_error_is_raised_when_both_providers_fail():
    with patch("src.agent.get_agent", side_effect=[object(), object()]), patch(
        "src.agent._invoke_agent", side_effect=[RuntimeError("groq failed"), RuntimeError("gemini failed")]
    ):
        with pytest.raises(LLMProviderError, match="Both configured LLM providers failed") as error:
            run_agent("coding", "hello")
    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "gemini failed"
    assert error.value.attempts == {
        "groq": "provider_error (RuntimeError)",
        "gemini": "provider_error (RuntimeError)",
    }


def test_provider_error_classifies_missing_model():
    with patch("src.agent.get_agent", side_effect=[object(), object()]), patch(
        "src.agent._invoke_agent", side_effect=[RuntimeError("primary failed"), RuntimeError("Model not found")]
    ):
        with pytest.raises(LLMProviderError) as error:
            run_agent("coding", "hello")
    assert error.value.reason == "model_unavailable"
