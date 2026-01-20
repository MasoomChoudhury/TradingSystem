"""
OpenAlgo Strategy Tools for LangGraph Agents

Provides webhook-based strategy order execution for automated trading.
"""
import os
import json
import logging
from typing import Annotated, Optional
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Global strategy client instance
_strategy_client = None
_strategy_logs = []


def get_strategy_client():
    """Get or create the Strategy client singleton."""
    global _strategy_client
    
    if _strategy_client is None:
        try:
            from openalgo import Strategy
            
            host_url = os.environ.get("OPENALGO_HOST", "http://127.0.0.1:5000")
            webhook_id = os.environ.get("OPENALGO_WEBHOOK_ID", "")
            
            if not webhook_id:
                logger.warning("OPENALGO_WEBHOOK_ID not set. Strategy orders will fail.")
                return None
            
            _strategy_client = Strategy(
                host_url=host_url,
                webhook_id=webhook_id
            )
            logger.info(f"OpenAlgo Strategy client initialized. Host: {host_url}")
        except ImportError:
            logger.error("OpenAlgo library not installed. Run: pip install openalgo")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Strategy client: {e}")
            raise
    
    return _strategy_client


def add_strategy_log(action: str, symbol: str, position_size: int, response: dict):
    """Add a strategy execution log entry."""
    import datetime
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "action": action,
        "symbol": symbol,
        "position_size": position_size,
        "response": response
    }
    _strategy_logs.append(log_entry)
    # Keep last 50 logs
    if len(_strategy_logs) > 50:
        _strategy_logs.pop(0)


def get_strategy_logs(limit: int = 20):
    """Get recent strategy execution logs."""
    return _strategy_logs[-limit:]


# =============================================================================
# STRATEGY TOOLS
# =============================================================================

@tool
def openalgo_strategy_order(
    symbol: Annotated[str, "Trading symbol (e.g., 'RELIANCE', 'INFY', 'SBIN')"],
    action: Annotated[str, "Order action: 'BUY' or 'SELL'"],
    position_size: Annotated[int, "Position size (>0 to enter, 0 to close position)"]
) -> str:
    """
    Execute a strategy order via OpenAlgo webhook.
    
    This is the primary way to place orders through the Strategy Management Module.
    The order will be processed according to the strategy mode configured in OpenAlgo.
    
    Args:
        symbol: Trading symbol (e.g., 'RELIANCE', 'NIFTY', 'BANKNIFTY')
        action: 'BUY' for long entry/close short, 'SELL' for short entry/close long
        position_size: Quantity (>0 to enter position, 0 to close existing position)
    
    Returns:
        JSON with order execution result.
    
    Examples:
        - Long entry: symbol='RELIANCE', action='BUY', position_size=1
        - Short entry: symbol='RELIANCE', action='SELL', position_size=1
        - Close long: symbol='RELIANCE', action='SELL', position_size=0
        - Close short: symbol='RELIANCE', action='BUY', position_size=0
    """
    try:
        client = get_strategy_client()
        if client is None:
            return json.dumps({
                "error": "Strategy client not initialized. Set OPENALGO_WEBHOOK_ID env var.",
                "symbol": symbol,
                "action": action
            })
        
        # Validate action
        action = action.upper()
        if action not in ["BUY", "SELL"]:
            return json.dumps({"error": f"Invalid action: {action}. Must be 'BUY' or 'SELL'."})
        
        # Execute strategy order
        response = client.strategyorder(symbol, action, position_size)
        
        # Log the execution
        add_strategy_log(action, symbol, position_size, response)
        
        result = {
            "status": "success",
            "symbol": symbol,
            "action": action,
            "position_size": position_size,
            "response": response
        }
        
        logger.info(f"Strategy order executed: {symbol} {action} qty={position_size}")
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_result = {
            "status": "error",
            "symbol": symbol,
            "action": action,
            "position_size": position_size,
            "error": str(e)
        }
        add_strategy_log(action, symbol, position_size, error_result)
        logger.error(f"Strategy order failed: {e}")
        return json.dumps(error_result, indent=2)


@tool
def openalgo_close_position(
    symbol: Annotated[str, "Trading symbol to close position for"],
    current_side: Annotated[str, "Current position side: 'LONG' or 'SHORT'"]
) -> str:
    """
    Close an existing position for a symbol.
    
    Automatically determines the correct action based on current position side.
    
    Args:
        symbol: Trading symbol
        current_side: 'LONG' if currently long, 'SHORT' if currently short
    
    Returns:
        JSON with close order result.
    """
    try:
        client = get_strategy_client()
        if client is None:
            return json.dumps({"error": "Strategy client not initialized."})
        
        # Determine action to close position
        current_side = current_side.upper()
        if current_side == "LONG":
            action = "SELL"
        elif current_side == "SHORT":
            action = "BUY"
        else:
            return json.dumps({"error": f"Invalid side: {current_side}. Must be 'LONG' or 'SHORT'."})
        
        # Close position (position_size=0)
        response = client.strategyorder(symbol, action, 0)
        
        add_strategy_log(f"CLOSE_{current_side}", symbol, 0, response)
        
        result = {
            "status": "success",
            "symbol": symbol,
            "action": f"CLOSE {current_side}",
            "response": response
        }
        
        logger.info(f"Position closed: {symbol} was {current_side}")
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Close position failed: {e}")
        return json.dumps({"status": "error", "error": str(e)})


@tool
def openalgo_get_strategy_logs(
    limit: Annotated[int, "Number of recent logs to retrieve"] = 20
) -> str:
    """
    Get recent strategy order execution logs.
    
    Useful for reviewing order history and debugging.
    
    Args:
        limit: Number of logs to return (default 20)
    
    Returns:
        JSON array of recent strategy executions.
    """
    logs = get_strategy_logs(limit)
    return json.dumps({"logs": logs, "count": len(logs)}, indent=2)


# =============================================================================
# Export all strategy tools
# =============================================================================

OPENALGO_STRATEGY_TOOLS = [
    openalgo_strategy_order,
    openalgo_close_position,
    openalgo_get_strategy_logs,
]

__all__ = [
    "OPENALGO_STRATEGY_TOOLS",
    "openalgo_strategy_order",
    "openalgo_close_position",
    "openalgo_get_strategy_logs",
    "get_strategy_logs",
]
