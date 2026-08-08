"""
llm/llm_ollama.py

Wraps the local Ollama chat model. Requires Ollama running locally
(`ollama serve`) with a model pulled, e.g. `ollama pull llama3.1`.
"""

from langchain_ollama import ChatOllama

DEFAULT_MODEL = "llama3.2:3b"


def get_llm(model_name: str = DEFAULT_MODEL, temperature: float = 0.0) -> ChatOllama:
    return ChatOllama(model=model_name, temperature=temperature)
