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