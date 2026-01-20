"""
OpenAlgo Orders Tools for LangGraph Agents

Provides order placement and management tools - the core trading functionality.
"""
import os
import json
import logging
from typing import Annotated, Optional
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Order execution logs
_order_logs = []


def get_openalgo_client():
    """Get the OpenAlgo API client."""
    from .openalgo_tools import get_openalgo_client as get_client
    return get_client()


def add_order_log(action: str, symbol: str, order_type: str, response: dict):
    """Add an order execution log entry."""
    import datetime
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "action": action,
        "symbol": symbol,
        "order_type": order_type,
        "response": response
    }
    _order_logs.append(log_entry)
    if len(_order_logs) > 100:
        _order_logs.pop(0)


def get_order_logs(limit: int = 30):
    """Get recent order execution logs."""
    return _order_logs[-limit:]


# =============================================================================
# ORDER PLACEMENT TOOLS
# =============================================================================

@tool
def openalgo_place_order(
    symbol: Annotated[str, "Trading symbol (e.g., 'RELIANCE', 'INFY')"],
    exchange: Annotated[str, "Exchange: NSE, BSE, NFO, BFO, CDS, MCX"],
    action: Annotated[str, "Order action: BUY or SELL"],
    quantity: Annotated[int, "Number of shares/lots"],
    price_type: Annotated[str, "Price type: MARKET, LIMIT, SL, SL-M"],
    product: Annotated[str, "Product: CNC (delivery), MIS (intraday), NRML (F&O)"],
    price: Annotated[str, "Price for LIMIT orders (use '0' for MARKET)"] = "0",
    trigger_price: Annotated[str, "Trigger price for SL/SL-M orders"] = "0"
) -> str:
    """
    Place a regular order on the exchange.
    
    Args:
        symbol: Trading symbol
        exchange: NSE, BSE, NFO, BFO, CDS, MCX
        action: BUY or SELL
        quantity: Number of shares/lots
        price_type: MARKET, LIMIT, SL, SL-M
        product: CNC (delivery), MIS (intraday), NRML (F&O carry)
        price: Required for LIMIT orders (default "0" for MARKET)
        trigger_price: Required for SL/SL-M orders
    
    Returns:
        JSON with order ID and status.
    """
    try:
        client = get_openalgo_client()
        
        result = client.placeorder(
            symbol=symbol,
            exchange=exchange.upper(),
            action=action.upper(),
            quantity=quantity,
            price_type=price_type.upper(),
            product=product.upper(),
            price=price,
            trigger_price=trigger_price
        )
        
        add_order_log(action, symbol, "REGULAR", result)
        logger.info(f"Order placed: {action} {quantity} {symbol} @ {price_type}")
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error = {"error": f"Order failed: {str(e)}"}
        add_order_log(action, symbol, "REGULAR_FAILED", error)
        logger.error(f"Order failed: {e}")
        return json.dumps(error)


@tool
def openalgo_place_smart_order(
    symbol: Annotated[str, "Trading symbol"],
    exchange: Annotated[str, "Exchange: NSE, BSE, NFO, etc."],
    action: Annotated[str, "BUY or SELL"],
    quantity: Annotated[int, "Number of shares/lots"],
    position_size: Annotated[int, "Target position size"],
    price_type: Annotated[str, "MARKET, LIMIT, SL, SL-M"],
    product: Annotated[str, "CNC, MIS, or NRML"],
    price: Annotated[str, "Price for LIMIT orders"] = "0"
) -> str:
    """
    Place a smart order with position sizing logic.
    
    Automatically adjusts order quantity based on current position and target size.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange code
        action: BUY or SELL
        quantity: Base quantity
        position_size: Target position size
        price_type: Order price type
        product: Product type
        price: Price for LIMIT orders
    
    Returns:
        JSON with order result.
    """
    try:
        client = get_openalgo_client()
        
        result = client.placesmartorder(
            symbol=symbol,
            exchange=exchange.upper(),
            action=action.upper(),
            quantity=quantity,
            position_size=position_size,
            price_type=price_type.upper(),
            product=product.upper(),
            price=price
        )
        
        add_order_log(action, symbol, "SMART", result)
        logger.info(f"Smart order: {action} {symbol} target_pos={position_size}")
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error = {"error": f"Smart order failed: {str(e)}"}
        add_order_log(action, symbol, "SMART_FAILED", error)
        return json.dumps(error)


@tool
def openalgo_basket_order(
    orders_json: Annotated[str, "JSON array of order objects"]
) -> str:
    """
    Place multiple orders simultaneously as a basket.
    
    Each order object should have: symbol, exchange, action, quantity, pricetype, product
    
    Example:
        [{"symbol": "RELIANCE", "exchange": "NSE", "action": "BUY",
          "quantity": 1, "pricetype": "MARKET", "product": "MIS"}]
    
    Returns:
        JSON with basket order results.
    """
    try:
        orders = json.loads(orders_json)
        
        if not isinstance(orders, list):
            return json.dumps({"error": "orders must be a JSON array"})
        
        client = get_openalgo_client()
        result = client.basketorder(orders=orders)
        
        add_order_log("BASKET", f"{len(orders)} orders", "BASKET", result)
        logger.info(f"Basket order placed: {len(orders)} orders")
        return json.dumps(result, indent=2)
        
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"Basket order failed: {str(e)}"})


@tool
def openalgo_split_order(
    symbol: Annotated[str, "Trading symbol"],
    exchange: Annotated[str, "Exchange code"],
    action: Annotated[str, "BUY or SELL"],
    quantity: Annotated[int, "Total quantity to split"],
    splitsize: Annotated[int, "Size of each split order"],
    price_type: Annotated[str, "MARKET, LIMIT, etc."],
    product: Annotated[str, "CNC, MIS, NRML"]
) -> str:
    """
    Split a large order into smaller orders.
    
    Useful for avoiding market impact with large orders.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange code
        action: BUY or SELL
        quantity: Total quantity (e.g., 105)
        splitsize: Size of each order (e.g., 20 → creates 6 orders: 5x20 + 1x5)
        price_type: Order price type
        product: Product type
    
    Returns:
        JSON with split order results.
    """
    try:
        client = get_openalgo_client()
        
        result = client.splitorder(
            symbol=symbol,
            exchange=exchange.upper(),
            action=action.upper(),
            quantity=quantity,
            splitsize=splitsize,
            price_type=price_type.upper(),
            product=product.upper()
        )
        
        num_orders = (quantity + splitsize - 1) // splitsize
        add_order_log(action, symbol, f"SPLIT_{num_orders}", result)
        logger.info(f"Split order: {quantity} into {splitsize} chunks")
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Split order failed: {str(e)}"})


# =============================================================================
# ORDER MANAGEMENT TOOLS
# =============================================================================

@tool
def openalgo_order_status(
    order_id: Annotated[str, "Order ID to check"],
    strategy: Annotated[str, "Strategy name"] = ""
) -> str:
    """
    Check status of a specific order.
    
    Returns:
        JSON with order status details.
    """
    try:
        client = get_openalgo_client()
        result = client.orderstatus(order_id=order_id, strategy=strategy)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Status check failed: {str(e)}"})


@tool
def openalgo_open_position(
    symbol: Annotated[str, "Trading symbol"],
    exchange: Annotated[str, "Exchange code"],
    product: Annotated[str, "Product: CNC, MIS, NRML"]
) -> str:
    """
    Get current open position for a specific symbol.
    
    Returns:
        JSON with position details (quantity, average price, P&L).
    """
    try:
        client = get_openalgo_client()
        result = client.openposition(
            symbol=symbol,
            exchange=exchange.upper(),
            product=product.upper()
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Position check failed: {str(e)}"})


@tool
def openalgo_modify_order(
    order_id: Annotated[str, "Order ID to modify"],
    symbol: Annotated[str, "Trading symbol"],
    exchange: Annotated[str, "Exchange code"],
    action: Annotated[str, "BUY or SELL"],
    quantity: Annotated[int, "New quantity"],
    price: Annotated[str, "New price"],
    product: Annotated[str, "Product type"],
    price_type: Annotated[str, "New price type"]
) -> str:
    """
    Modify an existing pending order.
    
    Can change quantity, price, and price type.
    
    Returns:
        JSON with modification result.
    """
    try:
        client = get_openalgo_client()
        
        result = client.modifyorder(
            order_id=order_id,
            symbol=symbol,
            action=action.upper(),
            exchange=exchange.upper(),
            quantity=quantity,
            price=price,
            product=product.upper(),
            price_type=price_type.upper()
        )
        
        add_order_log("MODIFY", symbol, order_id, result)
        logger.info(f"Order modified: {order_id}")
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Modify failed: {str(e)}"})


@tool
def openalgo_cancel_order(
    order_id: Annotated[str, "Order ID to cancel"]
) -> str:
    """
    Cancel a specific pending order.
    
    Returns:
        JSON with cancellation result.
    """
    try:
        client = get_openalgo_client()
        result = client.cancelorder(order_id=order_id)
        
        add_order_log("CANCEL", order_id, "CANCEL", result)
        logger.info(f"Order cancelled: {order_id}")
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Cancel failed: {str(e)}"})


@tool
def openalgo_cancel_all_orders() -> str:
    """
    Cancel ALL open/pending orders.
    
    WARNING: This will cancel every pending order in the account!
    
    Returns:
        JSON with cancellation results.
    """
    try:
        client = get_openalgo_client()
        result = client.cancelallorder()
        
        add_order_log("CANCEL_ALL", "ALL", "CANCEL_ALL", result)
        logger.warning("All orders cancelled!")
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Cancel all failed: {str(e)}"})


@tool
def openalgo_close_all_positions() -> str:
    """
    Close ALL open positions.
    
    WARNING: This will close every open position in the account!
    
    Returns:
        JSON with close results.
    """
    try:
        client = get_openalgo_client()
        result = client.closeposition()
        
        add_order_log("CLOSE_ALL", "ALL", "CLOSE_POSITIONS", result)
        logger.warning("All positions closed!")
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Close positions failed: {str(e)}"})


@tool
def openalgo_get_order_logs(
    limit: Annotated[int, "Number of recent logs"] = 30
) -> str:
    """
    Get recent order execution logs.
    
    Returns:
        JSON array of recent order activity.
    """
    logs = get_order_logs(limit)
    return json.dumps({"logs": logs, "count": len(logs)}, indent=2)


# =============================================================================
# Export all order tools
# =============================================================================

OPENALGO_ORDER_TOOLS = [
    openalgo_place_order,
    openalgo_place_smart_order,
    openalgo_basket_order,
    openalgo_split_order,
    openalgo_order_status,
    openalgo_open_position,
    openalgo_modify_order,
    openalgo_cancel_order,
    openalgo_cancel_all_orders,
    openalgo_close_all_positions,
    openalgo_get_order_logs,
]

__all__ = [
    "OPENALGO_ORDER_TOOLS",
    "openalgo_place_order",
    "openalgo_place_smart_order",
    "openalgo_basket_order",
    "openalgo_split_order",
    "openalgo_order_status",
    "openalgo_open_position",
    "openalgo_modify_order",
    "openalgo_cancel_order",
    "openalgo_cancel_all_orders",
    "openalgo_close_all_positions",
    "openalgo_get_order_logs",
    "get_order_logs",
]
