"""
rag/retriever.py

Thin wrapper around Chroma's retriever interface, kept as its own
module so retrieval strategy (k, search type, score threshold) can
be tuned in one place.
"""

from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever


def get_retriever(
    vectorstore: Chroma,
    k: int = 4,
    search_type: str = "similarity",
) -> VectorStoreRetriever:
    """
    Build a retriever from a Chroma vector store.

    Args:
        vectorstore: A built/loaded Chroma instance.
        k: Number of chunks to retrieve per query.
        search_type: "similarity" | "mmr" | "similarity_score_threshold".

    Returns:
        VectorStoreRetriever
    """
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )
