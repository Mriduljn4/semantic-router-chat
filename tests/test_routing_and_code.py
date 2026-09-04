import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

from src.intent_classifier import classify_intent, heuristic_intent
from src.agent import run_agent
from src.graph import run_query_async
from src.router import route


class RoutingAndCodingTests(unittest.TestCase):
    def test_heuristic_classification_for_clear_code_requests(self):
        self.assertEqual(heuristic_intent("Write a Python function to parse CSV"), "coding")
        self.assertEqual(heuristic_intent("Explain retrieval-augmented generation"), "research")
        self.assertEqual(heuristic_intent("Write SQL for monthly active users"), "data")

    def test_short_and_natural_requests_do_not_need_prompt_engineering(self):
        cases = {
            "Can you help me fix this Python error?": "coding",
            "I am getting a FastAPI exception": "coding",
            "Build a React login component": "coding",
            "Debug this API handler": "coding",
            "How to design a restapi": "coding",
            "Create a dashboard for conversion metrics": "data",
            "Compare AWS and Azure": "research",
            "hello": "general",
        }

        for query, expected in cases.items():
            with self.subTest(query=query):
                self.assertEqual(heuristic_intent(query), expected)

    def test_rules_use_word_boundaries_and_defer_ambiguous_followups(self):
        self.assertEqual(heuristic_intent("What is the capital of France?"), "research")
        self.assertIsNone(heuristic_intent("continue with the previous option"))

    def test_classify_intent_handles_direct_requirements_without_extra_prompting(self):
        result = classify_intent("Create a FastAPI endpoint with Pydantic validation.")
        self.assertEqual(result.intent, "coding")
        self.assertEqual(result.provider_used, "rules")

    def test_non_streaming_graph_receives_conversation_id(self):
        state = {
            "answer": "done",
            "routed_agent": "coding",
            "router_scores": {},
            "intent_classifier": "rules",
            "llm_provider_used": "openrouter",
            "tools_used": [],
        }

        with patch("src.graph.graph.ainvoke", new=AsyncMock(return_value=state)) as invoke:
            result = asyncio.run(run_query_async("fix it", "conversation-42"))

        invoke.assert_awaited_once_with(
            {"query": "fix it", "conversation_id": "conversation-42"}
        )
        self.assertEqual(result["answer"], "done")

    def test_ambiguous_followup_reuses_conversation_specialist(self):
        collection = Mock()
        collection.query.return_value = {"metadatas": [[]], "distances": [[]]}

        with (
            patch("src.router.get_capabilities_collection", return_value=collection),
            patch("src.router.embed_query", return_value=[0.0]),
        ):
            first = route("Build a Python API", "followup-conversation")
            followup = route("make it async", "followup-conversation")

        self.assertEqual(first.routed_agent, "coding")
        self.assertEqual(followup.routed_agent, "coding")
        self.assertEqual(followup.classifier_used, "rules")

    def test_code_agent_returns_actual_implementation(self):
        result = run_agent(
            "coding",
            "Write a Python function that reads a CSV file and returns a list of dicts.",
            "test-session",
        )
        self.assertIn("def ", result.answer)
        self.assertIn("csv", result.answer.lower())


if __name__ == "__main__":
    unittest.main()
