"""Runs an agent and keeps track of the history and the token count.

The LangChain agent and the LangGraph agent both take the same input
(`{"messages": [...]}`) and return the same output, so this one class can drive
either of them.
"""

from langchain_core.messages import BaseMessage, HumanMessage

from src.utils.token_counter import TokenCounter


class Conversation:
    """One conversation with one agent."""

    def __init__(self, agent, name: str) -> None:
        self.agent = agent
        self.name = name
        self.messages: list[BaseMessage] = []
        self.tokens = TokenCounter()

    def ask(self, question: str) -> str:
        """Send a question, return the answer, and update the token count."""
        self.messages.append(HumanMessage(question))

        # The API has no memory of its own, so we send the whole history every
        # turn and get the history back with the new messages appended.
        result = self.agent.invoke({"messages": self.messages})
        history = result["messages"]

        # Only the messages added by this turn are new, so only those are
        # counted - otherwise we would count the same tokens again every turn.
        self.tokens.add(history[len(self.messages) :])
        self.messages = history

        return self._final_answer()

    def _final_answer(self) -> str:
        """The text of the last reply from the model."""
        for message in reversed(self.messages):
            if message.type == "ai" and message.text:
                return message.text
        return "(no answer)"
