"""A running count of the tokens one conversation has used.

Every reply from the model carries a `usage_metadata` dict that says how many
tokens the call cost. We add those numbers up as the conversation goes on.
"""

from dataclasses import dataclass

from langchain_core.messages import BaseMessage


@dataclass
class TokenCounter:
    """Totals for a single conversation."""

    input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, messages: list[BaseMessage]) -> None:
        """Add the usage reported by any model replies in `messages`.

        Messages the model did not produce (your questions, tool results) have
        no usage to report, so they are skipped.
        """
        for message in messages:
            usage = getattr(message, "usage_metadata", None)
            if not usage:
                continue
            self.input_tokens += usage.get("input_tokens", 0)
            self.output_tokens += usage.get("output_tokens", 0)
            self.model_calls += 1

    def summary(self) -> str:
        return (
            f"tokens used: {self.total_tokens} "
            f"(input {self.input_tokens}, output {self.output_tokens}, "
            f"model calls {self.model_calls})"
        )
