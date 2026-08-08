"""
pdf/pdf_loader.py

Responsible for loading a PDF file from disk into LangChain Document objects.
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(pdf_path: str) -> List[Document]:
    """
    Load a single PDF file and return a list of Document objects
    (one Document per page, with page metadata attached).

    Args:
        pdf_path: Path to the .pdf file.

    Returns:
        List[Document]

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file is not a .pdf
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    loader = PyPDFLoader(str(path))
    documents = loader.load()

    # Attach the source filename cleanly (PyPDFLoader already sets 'source'
    # to the full path — this keeps a friendlier version too).
    for doc in documents:
        doc.metadata["source_file"] = path.name

    return documents


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python pdf_loader.py <path_to_pdf>")
        sys.exit(1)

    docs = load_pdf(sys.argv[1])
    print(f"Loaded {len(docs)} pages from {sys.argv[1]}")
    print("--- First page preview ---")
    print(docs[0].page_content[:500])
