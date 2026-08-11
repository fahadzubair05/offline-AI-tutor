"""
rag/subjects.py

A lightweight JSON-backed registry that groups ingested PDFs by "subject"
(e.g. "Biology 101", "Q3 Finance"). This is purely an organizational layer
on top of the existing per-PDF Chroma collections in rag/vectorstore.py —
Chroma itself has no concept of subjects; this module is the source of
truth for "which PDFs belong to which subject".

Registry file layout (data/subjects.json):
{
  "Biology 101": {
    "biology_101_chapter_1": "Chapter 1.pdf",
    "biology_101_chapter_2": "Chapter 2.pdf"
  },
  "Q3 Finance": {
    "q3_finance_report": "report.pdf"
  }
}
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from rag.vectorstore import sanitize_collection_name

REGISTRY_PATH = Path("data/subjects.json")


# ---------------------------------------------------------------------------
# Low-level registry I/O
# ---------------------------------------------------------------------------

def _load_registry() -> Dict[str, Dict[str, str]]:
    if not REGISTRY_PATH.exists():
        return {}
    try:
        return json.loads(REGISTRY_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_registry(registry: Dict[str, Dict[str, str]]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2))


# ---------------------------------------------------------------------------
# Subject naming
# ---------------------------------------------------------------------------

def normalize_subject_name(subject: str) -> str:
    """Trim/collapse whitespace in a subject's display name (e.g. 'Biology 101')."""
    return re.sub(r"\s+", " ", subject.strip())


def _subject_slug(subject: str) -> str:
    """Filesystem/collection-safe slug used as a prefix for that subject's PDFs."""
    slug = re.sub(r"[^a-z0-9]+", "_", subject.lower()).strip("_")
    return slug or "subject"


def build_collection_name(subject: str, pdf_filename: str) -> str:
    """
    Build a globally-unique Chroma collection name for a PDF ingested
    under a specific subject, e.g. ("Biology 101", "Chapter 1.pdf")
    -> "biology_101_chapter_1". Prefixing with the subject avoids two
    subjects' same-named PDFs colliding in the vector store.
    """
    prefixed = f"{_subject_slug(subject)}_{pdf_filename}"
    return sanitize_collection_name(prefixed)


# ---------------------------------------------------------------------------
# Subject-level operations
# ---------------------------------------------------------------------------

def list_subjects() -> List[str]:
    return sorted(_load_registry().keys())


def subject_exists(subject: str) -> bool:
    return subject in _load_registry()


def register_subject(subject: str) -> str:
    """Register a new subject (idempotent if it already exists). Returns the normalized name."""
    name = normalize_subject_name(subject)
    if not name:
        raise ValueError("Subject name cannot be empty.")

    registry = _load_registry()
    if name not in registry:
        registry[name] = {}
        _save_registry(registry)
    return name


def delete_subject(subject: str) -> List[str]:
    """
    Remove a subject from the registry entirely.

    Returns the list of Chroma collection names that were under it, so
    the caller can also delete those collections from the vector store
    (this module doesn't touch Chroma directly, to keep it decoupled).
    """
    registry = _load_registry()
    pdfs = registry.pop(subject, {})
    _save_registry(registry)
    return list(pdfs.keys())


# ---------------------------------------------------------------------------
# PDF-level operations (within a subject)
# ---------------------------------------------------------------------------

def get_subject_pdfs(subject: str) -> Dict[str, str]:
    """Return {collection_name: original_filename} for all PDFs under a subject."""
    return _load_registry().get(subject, {})


def add_pdf_to_subject(subject: str, collection_name: str, original_filename: str) -> None:
    registry = _load_registry()
    registry.setdefault(subject, {})
    registry[subject][collection_name] = original_filename
    _save_registry(registry)


def remove_pdf_from_subject(subject: str, collection_name: str) -> None:
    registry = _load_registry()
    if subject in registry and collection_name in registry[subject]:
        del registry[subject][collection_name]
        _save_registry(registry)


def find_subject_for_pdf(collection_name: str) -> Optional[str]:
    """Reverse lookup: which subject (if any) a given collection belongs to."""
    registry = _load_registry()
    for subject, pdfs in registry.items():
        if collection_name in pdfs:
            return subject
    return None
