RESEARCH_DOCS = [
    "Retrieval-augmented generation, or RAG, supplements a language model prompt with documents retrieved from an external knowledge source.",
    "A typical RAG pipeline embeds documents and queries into the same vector space, then retrieves the nearest document chunks for a query.",
    "Chunking divides source material into smaller passages. Good chunk boundaries preserve enough context for a retrieved passage to be useful.",
    "Embedding models map text to vectors. Similar meanings are represented by nearby vectors, enabling semantic retrieval beyond keyword matching.",
    "Vector databases store embeddings and metadata and commonly rank retrieval results with cosine similarity or an equivalent distance measure.",
    "Grounding an answer in retrieved context can reduce unsupported claims, but retrieval quality limits what the model can accurately answer.",
    "RAG systems are evaluated for retrieval relevance, answer relevance, and faithfulness: whether answer claims are supported by retrieved context.",
    "A RAG prompt should clearly separate instructions, user question, and retrieved context so the model can cite or rely on the intended evidence.",
]
