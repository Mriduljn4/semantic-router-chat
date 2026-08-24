from unittest.mock import patch

from src.embeddings import embed_documents, embed_query


def test_hashing_embeddings_are_deterministic_and_normalized():
    with patch("src.embeddings.get_settings") as settings:
        settings.return_value.EMBEDDING_BACKEND = "hashing"
        first = embed_query("Design a REST API")
        second = embed_query("Design a REST API")
        batch = embed_documents(["Design a REST API", "Write a SQL query"])
    assert first == second
    assert len(first) == 384
    assert round(sum(value * value for value in first), 6) == 1.0
    assert batch[0] == first


def test_gemini_embeddings_use_retrieval_task_types():
    with patch("src.embeddings.get_settings") as settings, patch(
        "src.embeddings._gemini_embeddings", return_value=[[0.1, 0.2]]
    ) as generate:
        settings.return_value.EMBEDDING_BACKEND = "gemini"
        assert embed_query("Find a document") == [0.1, 0.2]
        assert embed_documents(["Document text"]) == [[0.1, 0.2]]
    assert generate.call_args_list[0].args == (["Find a document"], "RETRIEVAL_QUERY")
    assert generate.call_args_list[1].args == (["Document text"], "RETRIEVAL_DOCUMENT")