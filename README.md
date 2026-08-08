# Offline AI Tutor

A fully offline RAG (Retrieval-Augmented Generation) system that ingests a single
PDF and answers questions **only** using that PDF's content, via local Ollama
models (no internet / API calls required at runtime).

## How it works

```
PDF file --> pdf/pdf_loader.py --> rag/splitter.py --> rag/vectorstore.py (Chroma)
                                                              |
                                                              v
question --> rag/retriever.py --> llm/chain.py --> llm/llm_ollama.py --> answer
```

- **pdf/** — loads a PDF into LangChain `Document` objects (one per page).
- **rag/** — chunks text, embeds it with a local Ollama embedding model, and
  stores/retrieves it via Chroma (persisted to `data/chroma_db/`).
- **llm/** — wraps the local Ollama chat model and defines the prompt that
  forces answers to come only from retrieved PDF context.
- **app/** — CLI entry point tying it all together.
- **tests/** — unit tests (pytest).

## Prerequisites

1. Install [Ollama](https://ollama.com) and make sure it's running:
   ```bash
   ollama serve
   ```
2. Pull a chat model and an embedding model:
   ```bash
   ollama pull llama3.1
   ollama pull nomic-embed-text
   ```
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run everything from the project root.

**1. Ingest a PDF** (place it in `data/pdfs/` first, or point to any path):

```bash
python -m app.main ingest data/pdfs/mybook.pdf
```

**2. Ask a single question:**

```bash
python -m app.main ask "What is the main argument of chapter 3?"
```

**3. Or start an interactive chat:**

```bash
python -m app.main chat
```

Type `exit` to quit the chat loop.

## Notes

- Each `ingest` call **overwrites** the previous vector store, so the system
  only ever answers from the most recently ingested PDF. If you want multiple
  PDFs available at once, give each one a distinct `collection_name` in
  `rag/vectorstore.py` and adjust `app/main.py` to select one.
- If a question can't be answered from the PDF, the model is instructed to
  say so rather than guess.
- Everything (embeddings + generation) runs locally through Ollama — no data
  leaves your machine.
