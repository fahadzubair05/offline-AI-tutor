"""
app/streamlit_app.py

Streamlit GUI for the offline PDF RAG tutor, organized by SUBJECT.

Flow:
  1. Home screen: register a new subject, or click into an existing one.
  2. Subject screen: ingest PDFs (scoped to that subject only) and ask
     questions (searches only that subject's PDFs, never other subjects').

Reuses the same pdf/rag/llm modules as the CLI (app/main.py) — no core
RAG logic is duplicated here; rag/subjects.py just adds an organizational
layer of "which PDFs belong to which subject" on top of the existing
per-PDF Chroma collections.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Allow imports from the project root regardless of where streamlit is launched from
sys.path.append(str(Path(__file__).resolve().parent.parent))

from pdf.pdf_loader import load_pdf
from rag.splitter import split_documents
from rag.vectorstore import (
    build_vectorstore,
    load_vectorstore,
    load_all_vectorstores,
    list_collections,
    delete_collection,
    get_all_documents,
)
from rag.subjects import (
    list_subjects,
    register_subject,
    delete_subject,
    get_subject_pdfs,
    add_pdf_to_subject,
    remove_pdf_from_subject,
    build_collection_name,
)
from rag.retriever import get_retriever, get_multi_retriever
from llm.llm_ollama import get_llm, DEFAULT_MODEL
from llm.chain import (
    stream_rag_answer,
    docs_to_sources,
    summarize_chunks,
    is_summary_request,
)

PDF_UPLOAD_DIR = Path("data/pdfs")

st.set_page_config(page_title="Offline AI Tutor", page_icon="📚", layout="wide")


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _cached_llm(model_name: str):
    return get_llm(model_name=model_name)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

def _init_state():
    st.session_state.setdefault("page", "home")
    st.session_state.setdefault("current_subject", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("selected_pdfs", None)


def _go_home():
    st.session_state.page = "home"
    st.session_state.current_subject = None
    st.session_state.messages = []
    st.session_state.selected_pdfs = None


def _open_subject(subject: str):
    st.session_state.page = "subject"
    st.session_state.current_subject = subject
    st.session_state.messages = []
    st.session_state.selected_pdfs = None


# ---------------------------------------------------------------------------
# HOME SCREEN — register / browse subjects
# ---------------------------------------------------------------------------

def render_home():
    st.title("📚 Offline AI Tutor")
    st.caption("Register a subject, then keep all of its PDFs together in one place.")

    with st.form("register_subject_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        new_subject = col1.text_input(
            "Register a new subject", placeholder="e.g. Biology 101", label_visibility="collapsed"
        )
        submitted = col2.form_submit_button("➕ Register", use_container_width=True)

        if submitted:
            if not new_subject.strip():
                st.error("Subject name can't be empty.")
            else:
                name = register_subject(new_subject)
                st.success(f"Registered '{name}'.")
                st.rerun()

    st.divider()
    st.subheader("Your subjects")

    subjects = list_subjects()
    if not subjects:
        st.info("No subjects registered yet. Add one above to get started.")
        return

    cols_per_row = 3
    for i in range(0, len(subjects), cols_per_row):
        row = subjects[i : i + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, subject in zip(cols, row):
            pdf_count = len(get_subject_pdfs(subject))
            with col:
                with st.container(border=True):
                    st.markdown(f"### 📁 {subject}")
                    st.caption(f"{pdf_count} PDF(s)")
                    if st.button("Open", key=f"open_{subject}", use_container_width=True):
                        _open_subject(subject)
                        st.rerun()
                    if st.button("🗑️ Delete subject", key=f"delsub_{subject}", use_container_width=True):
                        collection_names = delete_subject(subject)
                        for name in collection_names:
                            delete_collection(name)
                        st.rerun()


# ---------------------------------------------------------------------------
# SUBJECT SCREEN — ingest PDFs (scoped) + ask questions (scoped)
# ---------------------------------------------------------------------------

def render_subject_sidebar(subject: str):
    st.sidebar.title(f"📁 {subject}")
    if st.sidebar.button("← Back to subjects"):
        _go_home()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("1. Ingest a PDF")

    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{subject}",
    )

    if st.sidebar.button(
        "Ingest uploaded PDF(s)", type="primary", disabled=not uploaded_files
    ):
        PDF_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        for uploaded_file in uploaded_files:
            _ingest_uploaded_file(subject, uploaded_file)
        st.sidebar.success("Ingestion complete.")
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader(f"PDFs in '{subject}'")

    subject_pdfs = get_subject_pdfs(subject)  # {collection_name: filename}
    if not subject_pdfs:
        st.sidebar.info("No PDFs ingested for this subject yet.")
    else:
        for collection_name, filename in subject_pdfs.items():
            col1, col2 = st.sidebar.columns([4, 1])
            col1.write(f"📄 {filename}")
            if col2.button("🗑️", key=f"del_{collection_name}", help=f"Delete '{filename}'"):
                delete_collection(collection_name)
                remove_pdf_from_subject(subject, collection_name)
                if st.session_state.selected_pdfs and collection_name in st.session_state.selected_pdfs:
                    st.session_state.selected_pdfs.remove(collection_name)
                st.rerun()

    return subject_pdfs


def _ingest_uploaded_file(subject: str, uploaded_file) -> None:
    """Save an uploaded file to disk, run it through the RAG pipeline, and register it under the subject."""
    dest_path = PDF_UPLOAD_DIR / uploaded_file.name
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    collection_name = build_collection_name(subject, uploaded_file.name)

    with st.sidebar.status(f"Ingesting '{uploaded_file.name}'...", expanded=False) as status:
        status.write("Loading PDF pages...")
        documents = load_pdf(str(dest_path))

        status.write(f"Splitting {len(documents)} pages into chunks...")
        chunks = split_documents(documents)

        status.write(f"Embedding {len(chunks)} chunks (this can take a bit)...")
        build_vectorstore(chunks, collection_name=collection_name, overwrite=True)

        add_pdf_to_subject(subject, collection_name, uploaded_file.name)

        status.update(label=f"'{uploaded_file.name}' ingested ✅", state="complete")


def render_pdf_selector(subject: str, subject_pdfs: dict) -> list:
    """subject_pdfs: {collection_name: filename}. Returns the selected collection names."""
    st.subheader("Ask a question")

    collection_names = list(subject_pdfs.keys())
    name_by_collection = subject_pdfs

    options = ["All PDFs in this subject"] + collection_names
    default = st.session_state.selected_pdfs or collection_names

    selection = st.multiselect(
        "Search within:",
        options=options,
        default=default if default else ["All PDFs in this subject"],
        format_func=lambda c: c if c == "All PDFs in this subject" else name_by_collection.get(c, c),
        help="Leave 'All PDFs in this subject' selected to search every PDF in "
        f"'{subject}' at once, or pick specific ones to narrow it down.",
    )

    if not selection or "All PDFs in this subject" in selection:
        resolved = collection_names
    else:
        resolved = [c for c in selection if c in collection_names]

    st.session_state.selected_pdfs = resolved
    return resolved


def _build_retriever(selected_pdfs: list):
    if len(selected_pdfs) == 1:
        vectorstore = load_vectorstore(collection_name=selected_pdfs[0])
        return get_retriever(vectorstore)

    all_stores = load_all_vectorstores()
    scoped_stores = {name: all_stores[name] for name in selected_pdfs if name in all_stores}
    return get_multi_retriever(scoped_stores)


def _summarize_selected(selected_pdfs: list, name_by_collection: dict, status) -> str:
    """
    Summarize the selected PDF(s) by reading through their full content
    (map-reduce), not by similarity search — see llm/chain.py for why.
    Reports live progress to the given st.status container so the person
    can see it's actively working through the document, not stalled.
    """
    llm = _cached_llm(DEFAULT_MODEL)

    if len(selected_pdfs) == 1:
        chunks = get_all_documents(selected_pdfs[0])
        return summarize_chunks(chunks, llm, progress_callback=status.write)

    parts = []
    for collection_name in selected_pdfs:
        display_name = name_by_collection.get(collection_name, collection_name)
        status.write(f"Reading '{display_name}'...")
        chunks = get_all_documents(collection_name)
        summary = summarize_chunks(chunks, llm, progress_callback=status.write)
        parts.append(f"**{display_name}**\n\n{summary}")

    return "\n\n---\n\n".join(parts)


def _process_question(question: str, selected_pdfs: list, name_by_collection: dict):
    """Shared handling for both typed questions and the Summarize button."""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if is_summary_request(question):
            with st.status("Reading through the document(s)...", expanded=True) as status:
                answer_text = _summarize_selected(selected_pdfs, name_by_collection, status)
                status.update(label="Summary ready ✅", state="complete")
            st.markdown(answer_text)
            sources = []
        else:
            retriever = _build_retriever(selected_pdfs)
            llm = _cached_llm(DEFAULT_MODEL)
            token_stream, docs = stream_rag_answer(retriever, llm, question)
            answer_text = st.write_stream(token_stream)
            sources = docs_to_sources(docs)
            if sources:
                with st.expander("Sources used"):
                    for src in sources:
                        display_name = name_by_collection.get(src["pdf"], src["pdf"])
                        st.markdown(f"**{display_name}** — page {src['page']}")
                        st.caption(src["text"] + "...")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer_text, "sources": sources}
    )


def render_chat(subject: str, selected_pdfs: list, name_by_collection: dict):
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources used"):
                    for src in msg["sources"]:
                        display_name = name_by_collection.get(src["pdf"], src["pdf"])
                        st.markdown(f"**{display_name}** — page {src['page']}")
                        st.caption(src["text"] + "...")

    col1, col2 = st.columns([1, 5])
    summarize_clicked = col1.button(
        "📝 Summarize", disabled=not selected_pdfs, use_container_width=True
    )

    question = col2.chat_input(
        f"Ask something about your PDFs in '{subject}'..."
        if selected_pdfs
        else "Ingest a PDF into this subject first to start asking questions"
    )

    if summarize_clicked:
        if not selected_pdfs:
            st.warning("No PDFs available in this subject. Ingest one from the sidebar first.")
        else:
            _process_question(
                "Please provide a summary of the document(s).", selected_pdfs, name_by_collection
            )
            st.rerun()

    if question:
        if not selected_pdfs:
            st.warning("No PDFs available in this subject. Ingest one from the sidebar first.")
        else:
            _process_question(question, selected_pdfs, name_by_collection)


def render_clear_chat_button():
    if st.session_state.messages:
        if st.button("🧹 Clear chat"):
            st.session_state.messages = []
            st.rerun()


def render_subject_page(subject: str):
    subject_pdfs = render_subject_sidebar(subject)  # {collection_name: filename}

    st.title(f"📁 {subject}")
    st.caption("Ask questions grounded strictly in this subject's PDFs — nothing leaves your machine.")

    if not subject_pdfs:
        st.info("👈 Upload and ingest a PDF from the sidebar to get started.")
        return

    selected_pdfs = render_pdf_selector(subject, subject_pdfs)
    render_clear_chat_button()
    render_chat(subject, selected_pdfs, subject_pdfs)


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------

def main():
    _init_state()

    if st.session_state.page == "subject" and st.session_state.current_subject:
        render_subject_page(st.session_state.current_subject)
    else:
        render_home()


if __name__ == "__main__":
    main()