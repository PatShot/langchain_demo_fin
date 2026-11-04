"""A LangChain connector for a local llama.cpp server (`llama-server`).

`llama-server` serves a model over an OpenAI-compatible API
(`/v1/chat/completions`). Because it speaks the same language as OpenAI, we do
not need a special llama.cpp package: LangChain's `ChatOpenAI` works once it is
pointed at the local server instead of api.openai.com.

Start the server yourself before running the demo, for example:
    llama-server -m path/to/model.gguf --port 9931

Tool calling needs the server's Jinja chat template, which is on by default
(`--jinja`).

Docs: https://docs.langchain.com/oss/python/integrations/chat/openai
      https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
"""

import os

from langchain_openai import ChatOpenAI


def build_llamacpp_model() -> ChatOpenAI:
    """Return a chat model that talks to the llama.cpp server."""
    return ChatOpenAI(
        # llama-server's default port is 9931. Change it here if you used --port.
        base_url=os.getenv("LLAMACPP_BASE_URL", "http://localhost:9931/v1"),
        # The OpenAI client insists on a key, but llama-server does not check
        # it unless you started it with --api-key. Any text will do.
        api_key=os.getenv("LLAMACPP_API_KEY", "not-needed"),
        # llama-server serves the one model it was started with, so this name
        # is mostly a label. It is still sent with every request.
        model=os.getenv("LLAMACPP_MODEL", "local-model"),
        temperature=0.2,
        max_tokens=1024,
    )
