"""
rag/splitter.py

Splits loaded PDF Documents into smaller overlapping chunks suitable
for embedding and retrieval.
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Split documents into chunks.

    Args:
        documents: List of Document objects (e.g. from pdf_loader.load_pdf).
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap between consecutive chunks, helps preserve
            context across chunk boundaries.

    Returns:
        List[Document] — the chunked documents, each retaining the
        original page metadata plus a 'chunk_id'.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks
