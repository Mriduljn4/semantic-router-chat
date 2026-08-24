import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.seed import seed
from src.agent import run_agent
from src.router import route

CASES = [
    ("Explain retrieval augmented generation", "research"),
    ("Compare two technical approaches", "research"),
    ("Summarize current research findings", "research"),
    ("Investigate the limitations of vector databases", "research"),
    ("Give an evidence grounded overview", "research"),
    ("Debug this Python exception", "coding"),
    ("Implement a REST API endpoint", "coding"),
    ("Write unit tests for a function", "coding"),
    ("Refactor this JavaScript code", "coding"),
    ("Review my code for bugs", "coding"),
    ("Write a SQL query for sales", "data"),
    ("Calculate a conversion rate", "data"),
    ("Clean duplicate rows in a dataset", "data"),
    ("Create a visualization for revenue", "data"),
    ("Perform exploratory data analysis", "data"),
]


def test_router_accuracy():
    lookup = {index: agent for index, (_, agent) in enumerate(CASES)}

    class Collection:
        def query(self, query_embeddings, **_):
            agent = lookup[int(query_embeddings[0][0])]
            return {"metadatas": [[{"agent": agent}]], "distances": [[0.05]]}

    with patch("src.router.embed_query", side_effect=lambda query: [CASES.index(next(case for case in CASES if case[0] == query))]), patch(
        "src.router.get_capabilities_collection", return_value=Collection()
    ), patch("src.router.get_settings", return_value=SimpleNamespace(ROUTER_TOP_K=1)):
        passed = sum(route(query).routed_agent == expected for query, expected in CASES)
    assert passed == len(CASES), f"router accuracy: {passed}/{len(CASES)}"


@pytest.mark.integration
@pytest.mark.skipif(
    not (os.getenv("GROQ_API_KEY") and os.getenv("RUN_INTEGRATION_TESTS") == "true"),
    reason="requires GROQ_API_KEY and RUN_INTEGRATION_TESTS=true",
)
def test_research_rag_quality():
    from deepeval import evaluate
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.models.base_model import DeepEvalBaseLLM
    from deepeval.test_case import LLMTestCase
    from src.agent import _message_text
    from src.llm import get_model

    seed()

    class GroqJudge(DeepEvalBaseLLM):
        def load_model(self):
            return None

        def generate(self, prompt: str, schema=None):
            return _message_text(get_model("groq").invoke(prompt))

        async def a_generate(self, prompt: str, schema=None):
            return self.generate(prompt, schema)

        def get_model_name(self):
            return "groq"

    cases = []
    for question in ("What is RAG?", "Why is chunking useful in RAG?"):
        result = run_agent("research", question)
        cases.append(LLMTestCase(input=question, actual_output=result.answer, retrieval_context=result.context))
    judge = GroqJudge()
    evaluate(cases, [AnswerRelevancyMetric(threshold=0.5, model=judge), FaithfulnessMetric(threshold=0.5, model=judge)])
