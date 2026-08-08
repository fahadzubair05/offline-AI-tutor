"""
llm/chain.py

Builds the RAG chain: retriever -> prompt -> llm -> string answer.
The prompt is written to strictly ground the model in the retrieved
PDF context and to refuse when the answer isn't present.
"""

from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.vectorstores import VectorStoreRetriever

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY \
the provided context, which comes from a single PDF document.

Rules:
- Only use information found in the context below.
- If the answer is not present in the context, respond exactly with:
  "I couldn't find that in the document."
- Do not use outside knowledge, even if you know the answer.
- Keep answers concise and cite the page number(s) when available.

Context:
{context}
"""

USER_PROMPT = "Question: {question}"


def _format_docs(docs: List[Document]) -> str:
    """Combine retrieved chunks into a single context string, with page numbers."""
    formatted = []
    for doc in docs:
        page = doc.metadata.get("page", "unknown")
        formatted.append(f"[Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain(retriever: VectorStoreRetriever, llm) -> RunnableSequence:
    """
    Build an LCEL chain: takes a question string, returns an answer string,
    grounded strictly in the retrieved PDF chunks.
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


def answer_with_sources(retriever: VectorStoreRetriever, llm, question: str) -> dict:
    """
    Same as build_rag_chain, but also returns the raw source chunks so the
    app layer can display "sources used" alongside the answer.
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
            {"page": d.metadata.get("page", "unknown"), "text": d.page_content[:200]}
            for d in docs
        ],
    }
