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
import rag.subjects as subjects
from llm.chain import summarize_chunks, is_summary_request


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeSummarizeLLM:
    """Mimics ChatOllama's .invoke() interface for summarization tests."""

    def __init__(self):
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        text = str(prompt)
        if "partial summaries" in text:
            return _FakeResponse("FINAL COMBINED SUMMARY")
        if "Summarize the following excerpt" in text:
            return _FakeResponse("PARTIAL SUMMARY")
        return _FakeResponse("normal answer")


def test_is_summary_request_detects_summary_phrasing():
    assert is_summary_request("Can you summarize this document?")
    assert is_summary_request("give me a tl;dr")
    assert is_summary_request("What are the key points?")


def test_is_summary_request_ignores_normal_questions():
    assert not is_summary_request("What is the capital of France?")


def test_summarize_chunks_single_batch_skips_combine_step():
    chunks = [Document(page_content="Short text.", metadata={"page": 0, "chunk_id": 0})]
    llm = _FakeSummarizeLLM()

    result = summarize_chunks(chunks, llm, max_chars_per_batch=6000)

    assert result == "PARTIAL SUMMARY"
    assert len(llm.calls) == 1  # no combine call needed for a single batch


def test_summarize_chunks_multi_batch_uses_combine_step():
    chunks = [
        Document(page_content="x" * 5000, metadata={"page": i, "chunk_id": 0}) for i in range(3)
    ]
    llm = _FakeSummarizeLLM()

    result = summarize_chunks(chunks, llm, max_chars_per_batch=6000)

    assert result == "FINAL COMBINED SUMMARY"
    assert len(llm.calls) > 1  # multiple batch summaries + one combine call


def test_summarize_chunks_empty_input():
    result = summarize_chunks([], _FakeSummarizeLLM())
    assert "no content" in result.lower()


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


def test_build_collection_name_avoids_cross_subject_collision(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    cn_a = subjects.build_collection_name("Biology 101", "notes.pdf")
    cn_b = subjects.build_collection_name("Q3 Finance", "notes.pdf")

    assert cn_a != cn_b


def test_subject_registry_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    name = subjects.register_subject("  Biology 101  ")
    assert name == "Biology 101"
    assert "Biology 101" in subjects.list_subjects()

    cn = subjects.build_collection_name("Biology 101", "Chapter 1.pdf")
    subjects.add_pdf_to_subject("Biology 101", cn, "Chapter 1.pdf")

    pdfs = subjects.get_subject_pdfs("Biology 101")
    assert pdfs == {cn: "Chapter 1.pdf"}
    assert subjects.find_subject_for_pdf(cn) == "Biology 101"

    subjects.remove_pdf_from_subject("Biology 101", cn)
    assert subjects.get_subject_pdfs("Biology 101") == {}

    removed = subjects.delete_subject("Biology 101")
    assert removed == []
    assert "Biology 101" not in subjects.list_subjects()