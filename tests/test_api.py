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
        "rewritten_query": "Help with Python",
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


def test_query_stream_returns_status_and_answer_events():
    expected = {
        "answer": "RAG retrieves relevant documents.",
        "routed_agent": "research",
        "router_scores": {"research": 0.9, "coding": 0.1, "data": 0.0},
        "llm_provider_used": "gemini",
        "rewritten_query": "What is RAG?",
    }
    with patch("src.api.run_query", return_value=expected):
        response = TestClient(app).post("/query/stream", json={"query": "What is RAG?"})
    assert response.status_code == 200
    assert "event: status" in response.text
    assert "event: answer_start" in response.text
    assert "event: answer_chunk" in response.text
    assert "event: answer_complete" in response.text
    assert '"routed_agent": "research"' in response.text
