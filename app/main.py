"""
app/main.py

CLI entry point.

Usage:
    python -m app.main ingest data/pdfs/mybook.pdf
    python -m app.main ask "What is the main topic of chapter 2?"
    python -m app.main chat        # interactive Q&A loop

Run from the project root (offline-ai-tutor/) so relative imports resolve.
"""

import argparse
import sys
from pathlib import Path

# Allow running as `python app/main.py` as well as `python -m app.main`
sys.path.append(str(Path(__file__).resolve().parent.parent))

from pdf.pdf_loader import load_pdf
from rag.splitter import split_documents
from rag.vectorstore import build_vectorstore, load_vectorstore, DEFAULT_PERSIST_DIR
from rag.retriever import get_retriever
from llm.llm_ollama import get_llm
from llm.chain import build_rag_chain


def ingest(pdf_path: str) -> None:
    print(f"Loading PDF: {pdf_path}")
    documents = load_pdf(pdf_path)
    print(f"  -> {len(documents)} pages loaded")

    print("Splitting into chunks...")
    chunks = split_documents(documents)
    print(f"  -> {len(chunks)} chunks created")

    print("Embedding + persisting to Chroma (this may take a moment)...")
    build_vectorstore(chunks, overwrite=True)
    print(f"Done. Vector store saved to '{DEFAULT_PERSIST_DIR}'.")
    print("You can now run: python -m app.main chat")


def _load_chain():
    vectorstore = load_vectorstore()
    if vectorstore is None:
        print("No ingested PDF found. Run 'ingest <path_to_pdf>' first.")
        sys.exit(1)

    retriever = get_retriever(vectorstore)
    llm = get_llm()
    return build_rag_chain(retriever, llm)


def ask(question: str) -> None:
    chain = _load_chain()
    answer = chain.invoke(question)
    print(f"\nQ: {question}\nA: {answer}\n")


def chat() -> None:
    chain = _load_chain()
    print("Ask questions about the ingested PDF. Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        answer = chain.invoke(question)
        print(f"Bot: {answer}\n")


def main():
    parser = argparse.ArgumentParser(description="Offline PDF RAG tutor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a PDF into the vector store")
    ingest_parser.add_argument("pdf_path", help="Path to the PDF file")

    ask_parser = subparsers.add_parser("ask", help="Ask a single question")
    ask_parser.add_argument("question", help="Question to ask")

    subparsers.add_parser("chat", help="Interactive Q&A loop")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest(args.pdf_path)
    elif args.command == "ask":
        ask(args.question)
    elif args.command == "chat":
        chat()


if __name__ == "__main__":
    main()
