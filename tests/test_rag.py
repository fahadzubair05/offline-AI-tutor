"""
tests/test_rag.py

Basic unit tests. Run with: pytest tests/
Note: vectorstore/llm tests are skipped by default since they require
a running local Ollama instance — run manually if you have it set up.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document
from rag.splitter import split_documents


def test_split_documents_basic():
    long_text = "This is a sentence. " * 200  # ~4000 chars
    docs = [Document(page_content=long_text, metadata={"page": 0})]

    chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 600  # allow a little slack
        assert chunk.metadata["page"] == 0
        assert "chunk_id" in chunk.metadata


def test_split_documents_preserves_metadata():
    docs = [
        Document(page_content="Short text.", metadata={"page": 1, "source_file": "a.pdf"})
    ]
    chunks = split_documents(docs)

    assert len(chunks) == 1
    assert chunks[0].metadata["source_file"] == "a.pdf"


def test_split_documents_empty_input():
    assert split_documents([]) == []
