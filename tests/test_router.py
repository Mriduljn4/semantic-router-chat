from unittest.mock import Mock, patch

from src.router import route


def test_route_chooses_highest_mean_similarity():
    collection = Mock()
    collection.query.return_value = {
        "metadatas": [[{"agent": "coding"}, {"agent": "coding"}, {"agent": "research"}]],
        "distances": [[0.10, 0.30, 0.40]],
    }
    with patch("src.router.get_capabilities_collection", return_value=collection), patch(
        "src.router.embed_query", return_value=[0.1]
    ):
        decision = route("help me fix a Python exception")
    assert decision.routed_agent == "coding"
    assert decision.router_scores["coding"] == 0.8
    assert decision.router_scores["research"] == 0.6
    assert decision.router_scores["data"] == 0.0
