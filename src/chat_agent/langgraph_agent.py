"""The same agent, built by hand with LangGraph so the loop is visible.

The graph has two nodes and one decision:

    START -> call_model -> did the model ask for a tool?
                             yes -> tools -> call_model (loop)
                             no  -> END

`call_model` asks the model what to do; `tools` runs whatever it asked for.
"""

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.models.llm_models import llm_main_model
from src.tools.file_tools import write_answer

SYSTEM_PROMPT = (
    "You are a helpful assistant in a LangGraph demo. "
    "Answer questions clearly and briefly. "
    "When the user asks you to save, write or export an answer, call the "
    "write_answer tool and then tell the user which file you wrote."
)

TOOLS = [write_answer]

# `bind_tools` is what tells the model which tools exist, so it can ask for
# them by name. Without it the model has no idea it can write files.
model_with_tools = llm_main_model.bind_tools(TOOLS)


def call_model(state: MessagesState) -> dict:
    """Ask the model what to do next, given the conversation so far."""
    answer = model_with_tools.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
    # Returning a message appends it to state["messages"] - that is what the
    # `MessagesState` schema does for us.
    return {"messages": [answer]}


def build_langgraph_agent():
    """Build the graph and compile it into a runnable agent."""
    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "call_model")
    # `tools_condition` reads the model's last message: it routes to the "tools"
    # node if a tool was requested, and to END if the answer is final.
    builder.add_conditional_edges("call_model", tools_condition, ["tools", END])
    # After a tool runs, go back to the model so it can use the result.
    builder.add_edge("tools", "call_model")

    # No checkpointer is needed here because `Conversation` keeps the history
    # and passes it in every turn. LangGraph checkpointers, which store the
    # history for you, are a good next thing to explore.
    return builder.compile()
