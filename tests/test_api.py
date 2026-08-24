from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api import app
from src.llm import LLMProviderError


def test_health_returns_ok():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_ui_is_served():
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert "Semantic Router" in response.text


def test_query_returns_agent_response():
    expected = {
        "answer": "Use a list comprehension.",
        "routed_agent": "coding",
        "router_scores": {"research": 0.1, "coding": 0.9, "data": 0.2},
        "llm_provider_used": "groq",
    }
    with patch("src.api.run_query", return_value=expected) as run_query:
        response = TestClient(app).post("/query", json={"query": "  Help with Python  "})
    assert response.status_code == 200
    assert response.json() == expected
    run_query.assert_called_once_with("Help with Python")


def test_query_rejects_blank_input():
    response = TestClient(app).post("/query", json={"query": "   "})
    assert response.status_code == 422


def test_query_returns_service_unavailable_when_providers_fail():
    with patch("src.api.run_query", side_effect=LLMProviderError("providers failed")):
        response = TestClient(app).post("/query", json={"query": "Explain RAG"})
    assert response.status_code == 503
    assert response.json()["detail"] == {
        "message": "Language model service is unavailable.",
        "reason": "provider_error",
        "attempts": {},
    }
