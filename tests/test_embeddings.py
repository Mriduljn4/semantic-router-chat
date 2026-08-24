from unittest.mock import patch

from src.embeddings import embed_batch, embed_text


def test_hashing_embeddings_are_deterministic_and_normalized():
    with patch("src.embeddings.get_settings") as settings:
        settings.return_value.EMBEDDING_BACKEND = "hashing"
        first = embed_text("Design a REST API")
        second = embed_text("Design a REST API")
        batch = embed_batch(["Design a REST API", "Write a SQL query"])
    assert first == second
    assert len(first) == 384
    assert round(sum(value * value for value in first), 6) == 1.0
    assert batch[0] == first