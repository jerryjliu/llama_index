"""Embedding utils for gpt index."""

from typing import List
from openai.embeddings_utils import get_embedding, cosine_similarity

SIMILARITY_MODE = "similarity"
TEXT_SEARCH_MODE = "text_search"

TEXT_SIMILARITY_DAVINCI = "text-similarity-davinci-001"
TEXT_SEARCH_DAVINCI_QUERY = "text-search-davinci-query-001"
TEXT_SEARCH_DAVINCI_DOC = "text-search-davinci-doc-001"


def get_query_embedding(
    query: str, mode: str = TEXT_SEARCH_MODE
) -> List[float]:
    """Get query embedding."""
    if mode == SIMILARITY_MODE:
        engine = TEXT_SIMILARITY_DAVINCI
    elif mode == TEXT_SEARCH_MODE:
        engine = TEXT_SEARCH_DAVINCI_QUERY
    return get_embedding(query, engine=engine)


def get_text_embedding(
    text: str, mode: str = TEXT_SEARCH_MODE
) -> List[float]:
    """Get text embedding."""
    if mode == SIMILARITY_MODE:
        engine = TEXT_SIMILARITY_DAVINCI
    elif mode == TEXT_SEARCH_MODE:
        engine = TEXT_SEARCH_DAVINCI_DOC
    return get_embedding(text, engine=engine)


def get_query_text_embedding_similarity(
    query: str,
    text: str,
    mode: str = TEXT_SEARCH_MODE,
) -> float:
    """Get similarity between query and text."""
    if mode == SIMILARITY_MODE:
        query_engine = TEXT_SIMILARITY_DAVINCI
        doc_engine = TEXT_SIMILARITY_DAVINCI
    elif mode == TEXT_SEARCH_MODE:
        query_engine = TEXT_SEARCH_DAVINCI_QUERY
        doc_engine = TEXT_SEARCH_DAVINCI_DOC
    query_embedding = get_embedding(query, engine=query_engine)
    text_embedding = get_embedding(text, engine=doc_engine)
    return cosine_similarity(query_embedding, text_embedding)


def get_embedding_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Get similarity between two embeddings."""
    return cosine_similarity(embedding1, embedding2)


def save_embeddings(embeddings: List[List[float]], file_path: str) -> None:
    """Save embeddings to file."""
    with open(file_path, "w") as f:
        for embedding in embeddings:
            f.write(",".join([str(x) for x in embedding]))
            f.write("\n")


def load_embeddings(file_path: str) -> List[List[float]]:
    """Load embeddings from file."""
    embeddings = []
    with open(file_path, "r") as f:
        for line in f:
            embedding = [float(x) for x in line.strip().split(",")]
            embeddings.append(embedding)
    return embeddings
