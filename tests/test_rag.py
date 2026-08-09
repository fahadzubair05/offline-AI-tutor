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
from rag.vectorstore import sanitize_collection_name


def test_sanitize_collection_name_basic():
    assert sanitize_collection_name("my resume.pdf") == "my_resume"


def test_sanitize_collection_name_special_chars():
    assert sanitize_collection_name("Q3 Report (final).pdf") == "q3_report_final"


def test_sanitize_collection_name_short_name_padded():
    # Chroma requires >= 3 chars; short/odd names should still be valid
    result = sanitize_collection_name("a.pdf")
    assert len(result) >= 3


def test_sanitize_collection_name_is_stable():
    # Same filename should always produce the same collection name
    assert sanitize_collection_name("notes.pdf") == sanitize_collection_name("notes.pdf")


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