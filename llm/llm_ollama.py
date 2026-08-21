"""
llm/llm_ollama.py

Wraps the local Ollama chat model. Requires Ollama running locally
(`ollama serve`) with a model pulled, e.g. `ollama pull llama3.2`.
"""

from langchain_ollama import ChatOllama

DEFAULT_MODEL = "llama3.2:1b"

# Ollama silently caps every model at a 2048-token context window unless
# told otherwise — regardless of what the model itself actually supports.
# That's tight enough that a summarization batch + system prompt can bump
# into it, forcing extra internal reprocessing. Setting this explicitly
# also keeps it CONSTANT across calls, avoiding reload overhead from
# Ollama re-initializing the model when context settings change between
# requests.
DEFAULT_NUM_CTX = 4096

# By default, Ollama unloads a model from memory after just 5 minutes of
# inactivity. If questions come more than 5 minutes apart, every single
# one pays the full cost of reloading the model from disk (multiple GB)
# before it can even start generating — often the single biggest source
# of "why is this so slow" for local RAG apps. Keeping the model resident
# for the whole session (or effectively indefinitely) avoids that entirely
# after the very first request. "-1" means "keep loaded until Ollama
# restarts" (unload it yourself with `ollama stop <model>` if needed).
DEFAULT_KEEP_ALIVE = -1


def get_llm(
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    num_ctx: int = DEFAULT_NUM_CTX,
    keep_alive: int = DEFAULT_KEEP_ALIVE,
) -> ChatOllama:
    """
    Return a local ChatOllama instance.

    temperature=0.0 keeps answers deterministic/grounded, which matters
    for a "only answer from this PDF" system.
    num_ctx sets the context window explicitly (see note above on why
    this matters for both correctness and speed).
    keep_alive keeps the model loaded in memory between requests instead
    of letting Ollama unload it after its default 5-minute idle timeout
    (see note above — this is often the biggest real speed win available).
    """
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        num_ctx=num_ctx,
        keep_alive=keep_alive,
    )