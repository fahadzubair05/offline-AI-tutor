"""
rag/retriever.py

Thin wrapper around Chroma's retriever interface, kept as its own
module so retrieval strategy (k, search type, score threshold) can
be tuned in one place.

Also provides MultiPDFRetriever, which queries several PDF collections
at once and merges/tags the results, so you can chat across multiple
ingested PDFs without picking just one.
"""

from typing import Dict, List

from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever


def get_retriever(
    vectorstore: Chroma,
    k: int = 4,
    search_type: str = "similarity",
) -> VectorStoreRetriever:
    """
    Build a retriever from a single Chroma vector store (one PDF).

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


class MultiPDFRetriever(BaseRetriever):
    """
    Retriever that queries multiple PDF collections and merges the results.

    Each returned Document gets a 'source_pdf' metadata field set to the
    collection name it came from, so downstream prompts/answers can say
    which PDF a piece of context is from.
    """

    vectorstores: Dict[str, Chroma]
    """Mapping of collection_name -> Chroma vectorstore."""

    k_per_pdf: int = 3
    """How many chunks to pull from EACH PDF per query."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        all_docs: List[Document] = []

        for pdf_name, store in self.vectorstores.items():
            retriever = store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.k_per_pdf},
            )
            docs = retriever.invoke(query)
            for doc in docs:
                # Copy so we don't mutate the original cached Document.
                tagged = Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "source_pdf": pdf_name},
                )
                all_docs.append(tagged)

        return all_docs


def get_multi_retriever(
    vectorstores: Dict[str, Chroma],
    k_per_pdf: int = 3,
) -> MultiPDFRetriever:
    """
    Build a retriever that searches across multiple ingested PDFs at once.

    Args:
        vectorstores: Mapping of collection_name -> Chroma instance,
            e.g. the result of rag.vectorstore.load_all_vectorstores().
        k_per_pdf: How many chunks to retrieve from each individual PDF.
            (Total context size = k_per_pdf * number of PDFs.)

    Returns:
        MultiPDFRetriever
    """
    return MultiPDFRetriever(vectorstores=vectorstores, k_per_pdf=k_per_pdf)