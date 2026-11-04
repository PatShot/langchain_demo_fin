"""A LangChain connector for a local Ollama server.

Ollama runs models on your own machine and serves them over HTTP, by default
at http://localhost:11434. `ChatOllama` (from the `langchain-ollama` package)
turns that server into a normal LangChain chat model, so the agents can use it
exactly like any other model: tool calling and token usage included.

Start the server yourself before running the demo, for example:
    ollama serve            # if it is not already running as a service
    ollama list             # see which models you already have

Docs: https://docs.langchain.com/oss/python/integrations/chat/ollama
"""

import os

from langchain_ollama import ChatOllama


def build_ollama_model() -> ChatOllama:
    """Return a chat model that talks to the Ollama server."""
    return ChatOllama(
        # Must be a model you already have - see `ollama list`. Pick one that
        # supports tool calling, or the agents cannot save files.
        model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.2,
        num_predict=1024,  # Ollama's name for "max tokens to generate"
    )
