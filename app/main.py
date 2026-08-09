"""
app/main.py

CLI entry point.

Each ingested PDF gets its own named collection in the vector store.
By default (no --pdf given), questions are answered by searching ACROSS
ALL ingested PDFs at once — the answer will say which PDF each piece of
info came from. Use --pdf to scope to one or a few specific PDFs.

Usage:
    python -m app.main ingest data/pdfs/mybook.pdf
    python -m app.main list

    # Search across every ingested PDF:
    python -m app.main ask "What is chapter 2 about?"
    python -m app.main chat

    # Scope to specific PDF(s):
    python -m app.main ask "What is chapter 2 about?" --pdf mybook
    python -m app.main ask "Compare these" --pdf resume --pdf q3_report
    python -m app.main chat --pdf mybook

    python -m app.main delete mybook

Run from the project root (offline-ai-tutor/) so relative imports resolve.
"""

import argparse
import sys
from pathlib import Path

# Allow running as `python app/main.py` as well as `python -m app.main`
sys.path.append(str(Path(__file__).resolve().parent.parent))

from pdf.pdf_loader import load_pdf
from rag.splitter import split_documents
from rag.vectorstore import (
    build_vectorstore,
    load_vectorstore,
    load_all_vectorstores,
    list_collections,
    delete_collection,
    sanitize_collection_name,
    DEFAULT_PERSIST_DIR,
)
from rag.retriever import get_retriever, get_multi_retriever
from llm.llm_ollama import get_llm
from llm.chain import build_rag_chain


def ingest(pdf_path: str) -> None:
    collection_name = sanitize_collection_name(Path(pdf_path).name)

    print(f"Loading PDF: {pdf_path}")
    documents = load_pdf(pdf_path)
    print(f"  -> {len(documents)} pages loaded")

    print("Splitting into chunks...")
    chunks = split_documents(documents)
    print(f"  -> {len(chunks)} chunks created")

    print(f"Embedding + persisting to Chroma as collection '{collection_name}'...")
    build_vectorstore(chunks, collection_name=collection_name, overwrite=True)
    print(f"Done. Vector store saved to '{DEFAULT_PERSIST_DIR}'.")
    print(f"Query it with: python -m app.main ask \"...\" --pdf {collection_name}")
    print("Or just run 'ask'/'chat' with no --pdf to search all ingested PDFs.")


def list_pdfs() -> None:
    collections = list_collections()
    if not collections:
        print("No PDFs ingested yet. Run 'ingest <path_to_pdf>' first.")
        return
    print("Ingested PDFs (collection names):")
    for name in collections:
        print(f"  - {name}")


def _resolve_collection_names(pdf_args: list) -> list:
    """
    Resolve --pdf argument(s) to actual collection names.
    Accepts raw collection names or filename-like input for each.
    """
    collections = list_collections()
    if not collections:
        print("No ingested PDF found. Run 'ingest <path_to_pdf>' first.")
        sys.exit(1)

    resolved = []
    for pdf_arg in pdf_args:
        candidate = pdf_arg if pdf_arg in collections else sanitize_collection_name(pdf_arg)
        if candidate not in collections:
            print(f"No ingested PDF named '{pdf_arg}'. Available:")
            for name in collections:
                print(f"  - {name}")
            sys.exit(1)
        resolved.append(candidate)

    return resolved


def _build_chain_for(pdf_args: list):
    """
    Build a RAG chain scoped to:
      - one PDF, if exactly one --pdf was given (or only one PDF exists)
      - several PDFs, if multiple --pdf values were given
      - ALL ingested PDFs, if no --pdf was given at all

    Returns (chain, description_string_for_display).
    """
    llm = get_llm()

    if pdf_args:
        collection_names = _resolve_collection_names(pdf_args)
    else:
        collection_names = list_collections()
        if not collection_names:
            print("No ingested PDF found. Run 'ingest <path_to_pdf>' first.")
            sys.exit(1)

    if len(collection_names) == 1:
        vectorstore = load_vectorstore(collection_name=collection_names[0])
        retriever = get_retriever(vectorstore)
        description = collection_names[0]
    else:
        all_stores = load_all_vectorstores()
        selected_stores = {name: all_stores[name] for name in collection_names}
        retriever = get_multi_retriever(selected_stores)
        description = f"{len(collection_names)} PDFs: " + ", ".join(collection_names)

    chain = build_rag_chain(retriever, llm)
    return chain, description


def ask(question: str, pdf_args: list) -> None:
    chain, description = _build_chain_for(pdf_args)
    answer = chain.invoke(question)
    print(f"\n[Searching: {description}]")
    print(f"Q: {question}\nA: {answer}\n")


def chat(pdf_args: list) -> None:
    chain, description = _build_chain_for(pdf_args)
    print(f"Chatting with: {description}. Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        answer = chain.invoke(question)
        print(f"Bot: {answer}\n")


def delete(pdf_arg: str) -> None:
    collection_name = _resolve_collection_names([pdf_arg])[0]
    if delete_collection(collection_name):
        print(f"Deleted collection '{collection_name}'.")
    else:
        print(f"No collection named '{collection_name}' found.")


def main():
    parser = argparse.ArgumentParser(description="Offline PDF RAG tutor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a PDF into the vector store")
    ingest_parser.add_argument("pdf_path", help="Path to the PDF file")

    subparsers.add_parser("list", help="List all ingested PDFs")

    ask_parser = subparsers.add_parser("ask", help="Ask a single question")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument(
        "--pdf",
        action="append",
        default=[],
        help="Which ingested PDF to query (collection name). Repeat to "
        "include several. Omit entirely to search ALL ingested PDFs.",
    )

    chat_parser = subparsers.add_parser("chat", help="Interactive Q&A loop")
    chat_parser.add_argument(
        "--pdf",
        action="append",
        default=[],
        help="Which ingested PDF to chat with (collection name). Repeat to "
        "include several. Omit entirely to chat across ALL ingested PDFs.",
    )

    delete_parser = subparsers.add_parser("delete", help="Delete an ingested PDF's data")
    delete_parser.add_argument("pdf", help="Which ingested PDF to delete (collection name)")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest(args.pdf_path)
    elif args.command == "list":
        list_pdfs()
    elif args.command == "ask":
        ask(args.question, args.pdf)
    elif args.command == "chat":
        chat(args.pdf)
    elif args.command == "delete":
        delete(args.pdf)


if __name__ == "__main__":
    main()