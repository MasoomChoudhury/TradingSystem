"""
Strategy Management Tools for LangGraph Agents

Tools to create, manage, and execute strategies with automatic webhook handling.
"""
import json
import logging
from typing import Annotated
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def get_registry():
    """Get strategy registry."""
    from strategy_registry import get_strategy_registry
    return get_strategy_registry()


def get_strategy_client(webhook_id: str):
    """Get a Strategy client for a specific webhook."""
    import os
    from openalgo import Strategy
    
    host_url = os.environ.get("OPENALGO_HOST", "http://127.0.0.1:5000")
    return Strategy(host_url=host_url, webhook_id=webhook_id)


# =============================================================================
# STRATEGY MANAGEMENT TOOLS
# =============================================================================

@tool
def create_strategy(
    name: Annotated[str, "Unique strategy name (e.g., 'NIFTY_Scalper')"],
    mode: Annotated[str, "Mode: LONG_ONLY, SHORT_ONLY, or BOTH"] = "BOTH",
    exchange: Annotated[str, "Default exchange: NSE, NFO, etc."] = "NSE",
    symbols: Annotated[str, "Comma-separated symbols (e.g., 'RELIANCE,INFY')"] = "",
    description: Annotated[str, "Strategy description"] = "",
    webhook_id: Annotated[str, "Webhook ID from OpenAlgo (get from dashboard)"] = ""
) -> str:
    """
    Create or update a strategy in the registry.
    
    After creating a strategy here, you need to:
    1. Create the same strategy in OpenAlgo dashboard (Strategy Management)
    2. Get the webhook_id from OpenAlgo and call set_strategy_webhook
    
    Args:
        name: Unique strategy identifier
        mode: LONG_ONLY, SHORT_ONLY, or BOTH
        exchange: Default exchange for orders
        symbols: Comma-separated list of symbols
        description: Strategy description
        webhook_id: OpenAlgo webhook ID (can be set later)
    
    Returns:
        JSON with creation result.
    """
    try:
        registry = get_registry()
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
        
        result = registry.create_strategy(
            name=name,
            webhook_id=webhook_id if webhook_id else None,
            mode=mode.upper(),
            exchange=exchange.upper(),
            symbols=symbol_list,
            description=description
        )
        
        if not webhook_id:
            result["note"] = "Strategy created. Use set_strategy_webhook to add the webhook ID from OpenAlgo dashboard."
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to create strategy: {str(e)}"})


@tool
def set_strategy_webhook(
    name: Annotated[str, "Strategy name"],
    webhook_id: Annotated[str, "Webhook ID from OpenAlgo dashboard"]
) -> str:
    """
    Set or update the webhook ID for a strategy.
    
    Get the webhook ID from OpenAlgo dashboard:
    1. Go to Strategy Management
    2. Create/select strategy
    3. Click "View Strategy" 
    4. Copy the webhook ID
    
    Args:
        name: Strategy name (must exist in registry)
        webhook_id: The webhook ID from OpenAlgo
    
    Returns:
        JSON confirmation.
    """
    try:
        registry = get_registry()
        result = registry.set_webhook_id(name, webhook_id)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def list_strategies(
    active_only: Annotated[bool, "Only show active strategies"] = True
) -> str:
    """
    List all strategies in the registry.
    
    Returns:
        JSON list of strategies with their configurations.
    """
    try:
        registry = get_registry()
        strategies = registry.list_strategies(active_only=active_only)
        return json.dumps({
            "count": len(strategies),
            "strategies": strategies
        }, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_strategy(
    name: Annotated[str, "Strategy name"]
) -> str:
    """
    Get details of a specific strategy.
    
    Returns:
        JSON with strategy configuration and webhook status.
    """
    try:
        registry = get_registry()
        strategy = registry.get_strategy(name)
        if strategy:
            strategy["webhook_configured"] = bool(strategy.get("webhook_id"))
            return json.dumps(strategy, indent=2, default=str)
        else:
            return json.dumps({"error": f"Strategy '{name}' not found"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def execute_strategy_order(
    strategy_name: Annotated[str, "Strategy name from registry"],
    symbol: Annotated[str, "Trading symbol"],
    action: Annotated[str, "BUY or SELL"],
    position_size: Annotated[int, "Position size (>0 to enter, 0 to close)"]
) -> str:
    """
    Execute an order through a registered strategy's webhook.
    
    This uses the webhook ID stored in the registry for the strategy.
    
    Args:
        strategy_name: Name of strategy in registry
        symbol: Trading symbol
        action: BUY or SELL
        position_size: Quantity (>0 to enter, 0 to close)
    
    Returns:
        JSON with order result.
    """
    try:
        registry = get_registry()
        strategy = registry.get_strategy(strategy_name)
        
        if not strategy:
            return json.dumps({"error": f"Strategy '{strategy_name}' not found in registry"})
        
        webhook_id = strategy.get("webhook_id")
        if not webhook_id:
            return json.dumps({
                "error": f"No webhook ID configured for strategy '{strategy_name}'",
                "hint": "Use set_strategy_webhook to add the webhook ID"
            })
        
        # Get strategy client and execute
        client = get_strategy_client(webhook_id)
        response = client.strategyorder(symbol, action.upper(), position_size)
        
        # Log the order
        registry.log_order(strategy_name, symbol, action, position_size, response)
        
        result = {
            "status": "success",
            "strategy": strategy_name,
            "symbol": symbol,
            "action": action,
            "position_size": position_size,
            "response": response
        }
        
        logger.info(f"Strategy order: {strategy_name} {action} {symbol} qty={position_size}")
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Order execution failed: {str(e)}"})


@tool
def get_strategy_order_history(
    strategy_name: Annotated[str, "Strategy name (optional, leave empty for all)"] = "",
    limit: Annotated[int, "Number of orders to retrieve"] = 50
) -> str:
    """
    Get order history for a strategy or all strategies.
    
    Returns:
        JSON list of orders.
    """
    try:
        registry = get_registry()
        history = registry.get_order_history(
            strategy_name if strategy_name else None,
            limit
        )
        return json.dumps({
            "count": len(history),
            "orders": history
        }, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# Export tools
# =============================================================================

STRATEGY_MANAGEMENT_TOOLS = [
    create_strategy,
    set_strategy_webhook,
    list_strategies,
    get_strategy,
    execute_strategy_order,
    get_strategy_order_history,
]

__all__ = [
    "STRATEGY_MANAGEMENT_TOOLS",
    "create_strategy",
    "set_strategy_webhook",
    "list_strategies",
    "get_strategy",
    "execute_strategy_order",
    "get_strategy_order_history",
]
