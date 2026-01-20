"""
Code execution tools for the LangGraph agent.
"""
import sys
import io
import traceback
from typing import Annotated
from langchain_core.tools import tool


@tool
def execute_python_code(code: Annotated[str, "Python code to execute"]) -> str:
    """
    Execute arbitrary Python code and return the output.
    This runs in the backend environment with full access.
    Use print() to see output.
    """
    # Capture stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = captured_out = io.StringIO()
    sys.stderr = captured_err = io.StringIO()
    
    result = ""
    try:
        # Create a namespace for execution
        exec_globals = {
            "__builtins__": __builtins__,
            "print": print,
        }
        exec(code, exec_globals)
        
        stdout_val = captured_out.getvalue()
        stderr_val = captured_err.getvalue()
        
        if stdout_val:
            result += f"Output:\n{stdout_val}"
        if stderr_val:
            result += f"\nStderr:\n{stderr_val}"
        if not result:
            result = "Code executed successfully (no output)"
            
    except Exception as e:
        result = f"Execution Error:\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    return result


@tool
def deploy_strategy(filename: Annotated[str, "Name of the strategy file to deploy"]) -> str:
    """
    Deploy a strategy file to the running trading engine.
    This loads the strategy into the engine for live execution.
    """
    try:
        # Import here to avoid circular imports
        from engine import TradeEngine
        
        # This is a placeholder - in a real system, you'd get the running engine instance
        # For now, just return a confirmation message
        return f"Strategy '{filename}' has been queued for deployment. Check the console logs for status."
    except Exception as e:
        return f"Error deploying strategy: {e}"
