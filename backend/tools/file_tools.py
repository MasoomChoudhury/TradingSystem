"""
File operation tools for the LangGraph agent.
"""
import os
from typing import Annotated
from langchain_core.tools import tool

STRATEGIES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "strategies")

# Ensure strategies directory exists
os.makedirs(STRATEGIES_DIR, exist_ok=True)


@tool
def list_files() -> str:
    """List all strategy files in the strategies directory."""
    try:
        files = [f for f in os.listdir(STRATEGIES_DIR) if f.endswith('.py')]
        if not files:
            return "No strategy files found. You can create one using write_file."
        return "Strategy files:\n" + "\n".join(f"- {f}" for f in files)
    except Exception as e:
        return f"Error listing files: {e}"


@tool
def read_file(filename: Annotated[str, "Name of the file to read (e.g., 'my_strategy.py')"]) -> str:
    """Read the contents of a strategy file."""
    try:
        # Security check
        if ".." in filename or "/" in filename:
            return "Error: Invalid filename. Cannot use path separators."
        
        filepath = os.path.join(STRATEGIES_DIR, filename)
        if not os.path.exists(filepath):
            return f"Error: File '{filename}' not found."
        
        with open(filepath, 'r') as f:
            content = f.read()
        return f"Contents of {filename}:\n```python\n{content}\n```"
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def write_file(
    filename: Annotated[str, "Name of the file to write (e.g., 'my_strategy.py')"],
    content: Annotated[str, "Python code content to write to the file"]
) -> str:
    """Write or update a strategy file with the given content."""
    try:
        # Security check
        if ".." in filename or "/" in filename:
            return "Error: Invalid filename. Cannot use path separators."
        
        if not filename.endswith('.py'):
            filename = filename + '.py'
        
        filepath = os.path.join(STRATEGIES_DIR, filename)
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        return f"Successfully wrote {len(content)} characters to '{filename}'"
    except Exception as e:
        return f"Error writing file: {e}"
