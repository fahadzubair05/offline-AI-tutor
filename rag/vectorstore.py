"""
rag/vectorstore.py

Builds and loads a local Chroma vector store using Ollama embeddings.
Everything here runs fully offline once the Ollama embedding model
has been pulled (e.g. `ollama pull nomic-embed-text`).
"""

import shutil
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

DEFAULT_PERSIST_DIR = "data/chroma_db"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"


def get_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> OllamaEmbeddings:
    """Return the Ollama embedding function. Requires `ollama serve` running locally."""
    return OllamaEmbeddings(model=model_name)


def build_vectorstore(
    chunks: List[Document],
    persist_dir: str = DEFAULT_PERSIST_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    collection_name: str = "pdf_collection",
    overwrite: bool = True,
) -> Chroma:
    """
    Embed chunks and persist them to a local Chroma DB.

    Args:
        chunks: Chunked Document objects (from rag.splitter.split_documents).
        persist_dir: Folder to persist the Chroma DB to.
        embedding_model: Ollama embedding model name.
        collection_name: Chroma collection name — keeping it fixed per-PDF
            (or per-session) means "answer only from this PDF".
        overwrite: If True, wipes any existing DB at persist_dir first,
            so old PDFs' chunks don't leak into new answers.

    Returns:
        Chroma vector store instance.
    """
    persist_path = Path(persist_dir)

    if overwrite and persist_path.exists():
        shutil.rmtree(persist_path)

    persist_path.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings(embedding_model)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_path),
        collection_name=collection_name,
    )

    return vectorstore


def load_vectorstore(
    persist_dir: str = DEFAULT_PERSIST_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    collection_name: str = "pdf_collection",
) -> Optional[Chroma]:
    """Load a previously persisted Chroma DB, if one exists."""
    persist_path = Path(persist_dir)

    if not persist_path.exists():
        return None

    embeddings = get_embeddings(embedding_model)

    return Chroma(
        persist_directory=str(persist_path),
        embedding_function=embeddings,
        collection_name=collection_name,
    )
