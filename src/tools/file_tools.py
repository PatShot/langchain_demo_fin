"""Tools the agents are allowed to call.

A "tool" is just a Python function with a docstring. LangChain turns the
docstring and the argument names into a description the model can read, so the
model knows when and how to call it.
"""

from pathlib import Path

from langchain_core.tools import tool

OUTPUT_DIR = Path("output")


@tool
def write_answer(filename: str, content: str) -> str:
    """Save an answer to a file in the ./output/ folder.

    Use this whenever the user asks for an answer to be written, saved or
    exported to a file.

    Args:
        filename: Name of the file to write, for example "answer.md".
        content: The full text to write into the file.
    """
    # Keep only the file name, so the agent cannot write outside ./output/.
    name = Path(filename).name
    if name in {"", ".", ".."}:
        name = "answer.md"

    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / name
    path.write_text(content, encoding="utf-8")
    return f"Saved {len(content)} characters to {path}."
