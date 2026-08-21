"""
rag/vectorstore.py

Builds and loads a local Chroma vector store using Ollama embeddings.
Everything here runs fully offline once the Ollama embedding model
has been pulled (e.g. `ollama pull nomic-embed-text`).

Each ingested PDF gets its own Chroma *collection* (all sharing one
persist_dir on disk), so multiple PDFs can live side by side and you
choose which one to query at ask/chat time.
"""

import re
import shutil
from pathlib import Path
from typing import List, Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

DEFAULT_PERSIST_DIR = "data/chroma_db"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"


def sanitize_collection_name(pdf_filename: str) -> str:
    """
    Turn a PDF filename into a valid, stable Chroma collection name.

    Chroma collection name rules: 3-63 chars, only [a-zA-Z0-9._-],
    must start and end with an alphanumeric character.

    e.g. "my resume.pdf" -> "my_resume"
         "Q3 Report (final).pdf" -> "q3_report_final"
    """
    name = Path(pdf_filename).stem.lower()
    name = re.sub(r"[^a-z0-9._-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_.-")

    if len(name) < 3:
        name = f"pdf_{name}"
    name = name[:63]
    # Ensure it still ends alphanumeric after truncation
    name = name.rstrip("_.-") or "pdf_doc"

    return name


def get_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> OllamaEmbeddings:
    """Return the Ollama embedding function. Requires `ollama serve` running locally."""
    return OllamaEmbeddings(model=model_name)


def build_vectorstore(
    chunks: List[Document],
    collection_name: str,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    overwrite: bool = True,
) -> Chroma:
    """
    Embed chunks and persist them to a local Chroma DB, under their own
    collection so they don't mix with other ingested PDFs.

    Args:
        chunks: Chunked Document objects (from rag.splitter.split_documents).
        collection_name: Unique name for this PDF's collection (see
            sanitize_collection_name). Re-ingesting the same PDF re-uses
            (and by default replaces) its own collection only — other
            PDFs' collections are untouched.
        persist_dir: Folder to persist the Chroma DB to. Shared across
            all PDFs.
        embedding_model: Ollama embedding model name.
        overwrite: If True, deletes any existing collection with this
            same name before writing (so re-ingesting a PDF replaces
            its old chunks instead of duplicating them).

    Returns:
        Chroma vector store instance for this collection.
    """
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    if overwrite:
        client = chromadb.PersistentClient(path=str(persist_path))
        existing = {c.name for c in client.list_collections()}
        if collection_name in existing:
            client.delete_collection(collection_name)

    embeddings = get_embeddings(embedding_model)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_path),
        collection_name=collection_name,
    )

    return vectorstore


def load_vectorstore(
    collection_name: str,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Optional[Chroma]:
    """Load a previously persisted collection for one PDF, if it exists."""
    persist_path = Path(persist_dir)

    if not persist_path.exists():
        return None

    if collection_name not in list_collections(persist_dir):
        return None

    embeddings = get_embeddings(embedding_model)

    return Chroma(
        persist_directory=str(persist_path),
        embedding_function=embeddings,
        collection_name=collection_name,
    )


def load_all_vectorstores(
    persist_dir: str = DEFAULT_PERSIST_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> dict:
    """
    Load every ingested PDF's collection at once.

    Returns:
        Dict mapping collection_name -> Chroma instance. Empty dict if
        nothing has been ingested yet.
    """
    names = list_collections(persist_dir)
    embeddings = get_embeddings(embedding_model)

    stores = {}
    for name in names:
        stores[name] = Chroma(
            persist_directory=str(Path(persist_dir)),
            embedding_function=embeddings,
            collection_name=name,
        )
    return stores


def get_all_documents(
    collection_name: str,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> List[Document]:
    """
    Fetch every chunk of a single ingested PDF, in document order.

    Unlike retrieval (which finds chunks similar to a query), this reads
    the whole document — used for summarization, where "summarize this"
    has no strong semantic match to any one chunk and similarity search
    alone would return weak/irrelevant context.
    """
    vectorstore = load_vectorstore(
        collection_name=collection_name,
        persist_dir=persist_dir,
        embedding_model=embedding_model,
    )
    if vectorstore is None:
        return []

    raw = vectorstore.get(include=["documents", "metadatas"])
    documents = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw.get("documents", []), raw.get("metadatas", []))
    ]
    documents.sort(key=lambda d: (d.metadata.get("page", 0), d.metadata.get("chunk_id", 0)))
    return documents


def list_collections(persist_dir: str = DEFAULT_PERSIST_DIR) -> List[str]:
    """List the names of all ingested-PDF collections currently on disk."""
    persist_path = Path(persist_dir)
    if not persist_path.exists():
        return []

    client = chromadb.PersistentClient(path=str(persist_path))
    return sorted(c.name for c in client.list_collections())


def delete_collection(
    collection_name: str,
    persist_dir: str = DEFAULT_PERSIST_DIR,
) -> bool:
    """Delete a single PDF's collection. Returns True if it existed and was deleted."""
    persist_path = Path(persist_dir)
    if not persist_path.exists():
        return False

    client = chromadb.PersistentClient(path=str(persist_path))
    existing = {c.name for c in client.list_collections()}
    if collection_name not in existing:
        return False

    client.delete_collection(collection_name)
    return True


def wipe_all(persist_dir: str = DEFAULT_PERSIST_DIR) -> None:
    """Delete the entire vector store directory (all PDFs, all collections)."""
    persist_path = Path(persist_dir)
    if persist_path.exists():
        shutil.rmtree(persist_path)