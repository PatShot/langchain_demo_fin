# Summary
This is the completed state of a Langchain Demonstration. 

This demonstration must start with `patshot/langchain_demo_start`. Starting from that there are steps to get to this final state.

The demonstration shows how the an Agent is created based on the python library `langchain`, and `langgraph`, with observability coming from `langsmith`.

# Specifics
1. python
2. langchain
3. langgraph
4. langsmith
5. Local model servers: Ollama and llama.cpp (`llama-server`)
6. AWS Bedrock (optional)

# Quickstart

```bash
cp .env.example .env     # then pick LLM_PROVIDER and a model name
uv sync                  # install dependencies

uv run main.py           # chat with the LangChain agent
uv run main.py langgraph # chat with the LangGraph agent
```

Run the commands from the project root, because the agents write their files
into `./output/`.

Both agents answer questions and can save an answer to a file. Try:

```
you > Explain what a LangGraph node is, and save the answer to notes.md
```

After every reply the running token count for the conversation is printed:

```
[tokens used: 1483 (input 1402, output 81, model calls 2)]
```

# What to look at

| File | What it shows |
| --- | --- |
| `src/models/llm_models.py` | Picks the one shared chat model from `LLM_PROVIDER` |
| `src/models/ollama_model.py` | Connector for a local Ollama server (`ChatOllama`) |
| `src/models/llamacpp_model.py` | Connector for a local `llama-server` (`ChatOpenAI` with a local `base_url`) |
| `src/tools/file_tools.py` | The `write_answer` tool that saves files to `./output/` |
| `src/chat_agent/langchain_agent.py` | A LangChain agent built with `create_agent` - the loop comes for free |
| `src/chat_agent/langgraph_agent.py` | The same agent built by hand as a LangGraph `StateGraph`, so you can see the loop |
| `src/conversation.py` | Chat history plus the running token count, shared by both agents |
| `src/utils/token_counter.py` | How token usage is read off each reply and added up |
| `src/utils/observability.py` | How LangSmith tracing is switched on |
| `main.py` | The command line chat loop |

The two agents do the same job on purpose. Read `langchain_agent.py` first to
see how little code a working agent needs, then read `langgraph_agent.py` to
see what `create_agent` is doing underneath:

```
START -> call_model -> did the model ask for a tool?
                         yes -> tools -> call_model (loop)
                         no  -> END
```

# Configuration

Everything is read from `.env` (see `.env.example`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | `ollama` | `ollama`, `llamacpp` or `bedrock` |
| `OLLAMA_MODEL` | `llama3.1` | A model you already have (`ollama list`) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Where the Ollama server listens |
| `LLAMACPP_BASE_URL` | `http://localhost:9931/v1` | Where `llama-server` listens (its default port is 9931) |
| `LLAMACPP_MODEL` | `local-model` | Label sent to `llama-server`, which serves the model it was started with |
| `BEDROCK_MODEL_ID` | `anthropic.claude-haiku-5-5` | Only with `LLM_PROVIDER=bedrock`, plus AWS credentials |
| `LANGSMITH_TRACING` | off | Set to `true` to trace every run to LangSmith |
| `LANGSMITH_API_KEY` | - | Needed when tracing is on |
| `LANGSMITH_PROJECT` | - | Which LangSmith project to send traces to |

# Running a local model

The demo does not start servers or download models for you. Start one yourself:

```bash
# Ollama
ollama serve                 # skip if it already runs as a service
ollama list                  # pick a model that supports tool calling
LLM_PROVIDER=ollama OLLAMA_MODEL=<name> uv run main.py

# llama.cpp
llama-server -m path/to/model.gguf --port 9931
LLM_PROVIDER=llamacpp uv run main.py
```

Ollama has its own LangChain package, `langchain-ollama`. `llama-server`
speaks the OpenAI API, so the demo reuses `ChatOpenAI` from `langchain-openai`
and only changes `base_url`. To save files the agent needs a model that
supports tool calling. Small models sometimes answer without calling the tool.

# Tracing with LangSmith

Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY=...` in `.env`, then run an
agent and open [smith.langchain.com](https://smith.langchain.com). Each run shows
the full trace - every model call, every tool call and the tokens each one used.
No code changes are needed; LangChain picks up the environment variables itself.
