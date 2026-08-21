"""
llm/chain.py

Builds the RAG chain: retriever -> prompt -> llm -> string answer.
The prompt is written to strictly ground the model in the retrieved
context and to refuse when the answer isn't present. Works the same
whether the retriever covers one PDF or several (MultiPDFRetriever) —
when multiple PDFs are involved, each chunk is labeled with its source
PDF so the model can say which document an answer came from.
"""

from typing import List, Union

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableSequence
from langchain_core.vectorstores import VectorStoreRetriever

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY \
the provided context, which comes from one or more PDF documents.

Rules:
- Only use information found in the context below.
- If the answer is not present in the context, respond exactly with:
  "I couldn't find that in the document(s)."
- Do not use outside knowledge, even if you know the answer.
- Keep answers concise.
- Each context chunk is labeled with its source PDF and page number.
  When you use information from a chunk, mention which PDF it came from
  (e.g. "According to resume.pdf, ...") and the page number if useful.
- If chunks from different PDFs disagree or are unrelated, address them
  separately rather than blending them into one claim.

Context:
{context}
"""

USER_PROMPT = "Question: {question}"


def _format_docs(docs: List[Document]) -> str:
    """Combine retrieved chunks into a single context string, labeled by source PDF and page."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source_pdf") or doc.metadata.get("source_file", "document")
        page = doc.metadata.get("page", "unknown")
        formatted.append(f"[Source: {source} | Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain(
    retriever: Union[VectorStoreRetriever, BaseRetriever], llm
) -> RunnableSequence:
    """
    Build an LCEL chain: takes a question string, returns an answer string,
    grounded strictly in the retrieved chunks. Works with a single-PDF
    retriever or a MultiPDFRetriever transparently.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ]
    )

    def retrieve_and_format(question: str) -> dict:
        docs = retriever.invoke(question)
        return {"context": _format_docs(docs), "question": question}

    chain = retrieve_and_format | prompt | llm | StrOutputParser()
    return chain


def answer_with_sources(retriever, llm, question: str) -> dict:
    """
    Same as build_rag_chain, but also returns the raw source chunks
    (including which PDF each came from) so the app layer can display
    "sources used" alongside the answer.
    """
    docs = retriever.invoke(question)
    context = _format_docs(docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ]
    )

    messages = prompt.invoke({"context": context, "question": question})
    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "sources": [
            {
                "pdf": d.metadata.get("source_pdf") or d.metadata.get("source_file", "document"),
                "page": d.metadata.get("page", "unknown"),
                "text": d.page_content[:200],
            }
            for d in docs
        ],
    }


def stream_rag_answer(retriever, llm, question: str):
    """
    Same retrieval + grounding as answer_with_sources, but streams the
    answer token-by-token instead of waiting for the full response.
    This makes responses feel much faster since text appears immediately
    instead of after the whole generation finishes.

    Returns:
        (token_generator, docs) — iterate token_generator (yields string
        chunks) to display the streaming answer; docs are the retrieved
        source chunks, available immediately (retrieval itself isn't slow).
    """
    docs = retriever.invoke(question)
    context = _format_docs(docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ]
    )
    messages = prompt.invoke({"context": context, "question": question})

    def token_generator():
        for chunk in llm.stream(messages):
            if chunk.content:
                yield chunk.content

    return token_generator(), docs


def docs_to_sources(docs: List[Document]) -> List[dict]:
    """Convert raw retrieved Documents into the {pdf, page, text} shape used by the UI."""
    return [
        {
            "pdf": d.metadata.get("source_pdf") or d.metadata.get("source_file", "document"),
            "page": d.metadata.get("page", "unknown"),
            "text": d.page_content[:200],
        }
        for d in docs
    ]


# ---------------------------------------------------------------------------
# Whole-document summarization
#
# Summarization is NOT a retrieval task — a query like "summarize this"
# has no strong semantic match to any single chunk, so similarity search
# returns weak/irrelevant context and the strict grounding prompt (rightly)
# refuses to answer. Summarization instead reads through ALL of a
# document's chunks directly, using a map-reduce approach so it scales
# to documents longer than the model's context window:
#   1. Split chunks into batches that comfortably fit in context.
#   2. Summarize each batch independently ("map").
#   3. Combine the batch summaries into one final summary ("reduce").
# ---------------------------------------------------------------------------

SUMMARIZE_BATCH_PROMPT = """Summarize the following excerpt from a document. \
Capture the key points, facts, and structure concisely. Do not add \
information that isn't in the excerpt.

Excerpt:
{text}
"""

SUMMARIZE_COMBINE_PROMPT = """You were given partial summaries of consecutive \
sections of the same document. Combine them into a single, coherent, \
well-organized summary of the WHOLE document. Remove redundancy between \
sections, but don't drop distinct points. Do not add information that \
isn't in the partial summaries.

Partial summaries:
{text}
"""


def _batch_chunks(chunks: List[Document], max_chars_per_batch: int = 6000) -> List[List[Document]]:
    """Group chunks into batches that stay under a rough character budget per LLM call."""
    batches: List[List[Document]] = []
    current: List[Document] = []
    current_len = 0

    for chunk in chunks:
        chunk_len = len(chunk.page_content)
        if current and current_len + chunk_len > max_chars_per_batch:
            batches.append(current)
            current = []
            current_len = 0
        current.append(chunk)
        current_len += chunk_len

    if current:
        batches.append(current)

    return batches


def summarize_chunks(
    chunks: List[Document],
    llm,
    max_chars_per_batch: int = 12000,
    progress_callback=None,
) -> str:
    """
    Summarize an entire document (given as its list of chunks) via map-reduce.
    Reads through everything rather than relying on similarity search, so it
    works even for "summarize this" style questions that retrieval alone
    can't answer well.

    This makes one real LLM call per batch, plus one combine call if there's
    more than one batch — for a long document that's several sequential
    calls, which is inherently slower than a single retrieval-based answer.
    max_chars_per_batch controls that tradeoff: larger batches mean fewer
    (but longer) LLM calls.

    progress_callback, if given, is called with a short status string before
    each LLM call (e.g. "Summarizing section 2 of 5...") so calling UI code
    can show live progress instead of one long silent wait.
    """
    if not chunks:
        return "There's no content to summarize — this PDF appears to be empty or wasn't ingested correctly."

    # Keep chunks in document order for a coherent summary.
    ordered = sorted(chunks, key=lambda d: (d.metadata.get("page", 0), d.metadata.get("chunk_id", 0)))
    batches = _batch_chunks(ordered, max_chars_per_batch=max_chars_per_batch)

    if progress_callback and len(batches) > 1:
        progress_callback(f"Document split into {len(batches)} section(s) to read through...")

    partial_summaries = []
    for i, batch in enumerate(batches, start=1):
        if progress_callback:
            label = f"Summarizing section {i} of {len(batches)}..." if len(batches) > 1 else "Summarizing..."
            progress_callback(label)

        text = "\n\n".join(c.page_content for c in batch)
        prompt = SUMMARIZE_BATCH_PROMPT.format(text=text)
        response = llm.invoke(prompt)
        partial_summaries.append(response.content)

    if len(partial_summaries) == 1:
        return partial_summaries[0]

    if progress_callback:
        progress_callback("Combining section summaries into one...")

    combined_text = "\n\n---\n\n".join(partial_summaries)
    final_prompt = SUMMARIZE_COMBINE_PROMPT.format(text=combined_text)
    final_response = llm.invoke(final_prompt)
    return final_response.content


SUMMARY_KEYWORDS = (
    "summarize",
    "summarise",
    "summary",
    "tl;dr",
    "tldr",
    "give me an overview",
    "what is this document about",
    "what is this pdf about",
    "main points",
    "key points",
    "key takeaways",
)


def is_summary_request(question: str) -> bool:
    """Heuristic check for whether a question is asking for a whole-document summary."""
    q = question.lower()
    return any(keyword in q for keyword in SUMMARY_KEYWORDS)