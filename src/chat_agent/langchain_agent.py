"""A basic agent built with LangChain's `create_agent` helper.

`create_agent` hands you a finished tool-calling loop: the model decides when
to call a tool, LangChain runs it, feeds the result back, and repeats until the
model has a final answer. Compare this with `langgraph_agent.py`, which builds
the same loop by hand.
"""

from langchain.agents import create_agent

from src.models.llm_models import llm_main_model
from src.tools.file_tools import write_answer

SYSTEM_PROMPT = (
    "You are a helpful assistant in a LangChain demo. "
    "Answer questions clearly and briefly. "
    "When the user asks you to save, write or export an answer, call the "
    "write_answer tool and then tell the user which file you wrote."
)


def build_langchain_agent():
    """Return an agent that takes {"messages": [...]} and replies."""
    return create_agent(
        model=llm_main_model,
        tools=[write_answer],
        system_prompt=SYSTEM_PROMPT,
    )
