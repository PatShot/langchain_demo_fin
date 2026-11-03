"""Chat with either demo agent from the terminal.

    uv run main.py              # the LangChain agent
    uv run main.py langgraph    # the LangGraph agent

Both agents can answer questions and, when you ask them to, write the answer
to a file in ./output/. The token count for the conversation is printed after
every reply.
"""

import sys

from src.chat_agent.langchain_agent import build_langchain_agent
from src.chat_agent.langgraph_agent import build_langgraph_agent
from src.conversation import Conversation
from src.utils.observability import tracing_status

AGENTS = {
    "langchain": build_langchain_agent,
    "langgraph": build_langgraph_agent,
}


def main():
    choice = sys.argv[1].lower() if len(sys.argv) > 1 else "langchain"
    if choice not in AGENTS:
        print(f"Unknown agent '{choice}'. Pick one of: {', '.join(AGENTS)}")
        raise SystemExit(1)

    conversation = Conversation(AGENTS[choice](), name=choice)

    print(f"The {choice} agent is ready. {tracing_status()}")
    print("Ask a question, or try: 'Explain LangGraph and save it to notes.md'")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            question = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        try:
            answer = conversation.ask(question)
        except Exception as error:
            # Usually a missing AWS credential or a model you cannot access.
            print(f"\nThe call to Bedrock failed: {error}")
            print("Check the AWS settings in your .env file, then try again.\n")
            continue

        print(f"\n{choice} > {answer}\n")
        print(f"[{conversation.tokens.summary()}]\n")

    print(f"Conversation over. {conversation.tokens.summary()}")


if __name__ == '__main__':
    main()
