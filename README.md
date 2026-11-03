# Summary
This is the completed state of a Langchain Demonstration. 

This demonstration must start with `patshot/langchain_demo_start`. Starting from that there are steps to get to this final state.

The demonstration shows how the an Agent is created based on the python library `langchain`, and `langgraph`, with observability coming from `langsmith`.

# Specifics
1. python
2. langchain
3. langgraph
4. langsmith
5. AWS Bedrock
6. Other LLM API based inferences.

# Quickstart

```bash
cp .env.example .env     # then fill in your AWS Bedrock credentials
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
| `src/models/llm_models.py` | One shared chat model, running on AWS Bedrock |
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

| Variable | Required | Purpose |
| --- | --- | --- |
| `AWS_BEARER_TOKEN_BEDROCK` | yes* | Bedrock API key |
| `AWS_REGION` | no | Bedrock region, defaults to `us-east-1` |
| `BEDROCK_MODEL_ID` | no | Defaults to `anthropic.claude-haiku-5-5` |
| `LANGSMITH_TRACING` | no | Set to `true` to trace every run to LangSmith |
| `LANGSMITH_API_KEY` | no | Needed when tracing is on |
| `LANGSMITH_PROJECT` | no | Which LangSmith project to send traces to |

\* Standard AWS credentials (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`) work
instead of the bearer token.

Your AWS account also needs **model access** enabled for the model you pick, in
the Bedrock console under *Model access*. Without it every call fails with
`Error 002: Access to Bedrock models is not allowed for this account`. To see
which model ids your account and region offer:

```bash
uv run python -c "import boto3;print([m['modelId'] for m in \
  boto3.client('bedrock').list_foundation_models()['modelSummaries']])"
```

# Tracing with LangSmith

Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY=...` in `.env`, then run an
agent and open [smith.langchain.com](https://smith.langchain.com). Each run shows
the full trace - every model call, every tool call and the tokens each one used.
No code changes are needed; LangChain picks up the environment variables itself.
