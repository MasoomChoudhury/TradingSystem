"""
OpenAlgo Tools for LangGraph Agents

Provides trading tools that integrate with OpenAlgo API.
"""
import os
import json
import logging
from typing import Annotated, Optional, List, Dict, Any
from langchain_core.tools import tool

# Configure logging
logger = logging.getLogger(__name__)

# Global client instance (initialized on first use)
_openalgo_client = None
_ws_logs: List[Dict[str, Any]] = []


def get_openalgo_client():
    """Get or create the OpenAlgo client singleton."""
    global _openalgo_client
    
    if _openalgo_client is None:
        try:
            from openalgo import api
            
            api_key = os.environ.get("OPENALGO_API_KEY", "")
            host = os.environ.get("OPENALGO_HOST", "http://127.0.0.1:5000")
            
            _openalgo_client = api(
                api_key=api_key,
                host=host
            )
            logger.info(f"OpenAlgo client initialized. Host: {host}")
        except ImportError:
            logger.error("OpenAlgo library not installed. Run: pip install openalgo")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize OpenAlgo client: {e}")
            raise
    
    return _openalgo_client


def add_ws_log(category: str, message: str, data: dict = None):
    """Add a WebSocket log entry."""
    import datetime
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "category": category,  # WS, AUTH, SUB, UNSUB, LTP, QUOTE, DEPTH, ERROR
        "message": message,
        "data": data
    }
    _ws_logs.append(log_entry)
    # Keep last 100 logs
    if len(_ws_logs) > 100:
        _ws_logs.pop(0)


def get_ws_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """Get recent WebSocket logs."""
    return _ws_logs[-limit:]


def clear_ws_logs():
    """Clear all WebSocket logs."""
    global _ws_logs
    _ws_logs = []


# =============================================================================
# WebSocket Connection Tools
# =============================================================================

@tool
def openalgo_connect() -> str:
    """
    Connect to OpenAlgo WebSocket server.
    Must be called before subscribing to market data feeds.
    
    Returns:
        Status message indicating connection result.
    """
    try:
        client = get_openalgo_client()
        client.connect()
        add_ws_log("WS", "Connected to WebSocket server")
        return "Successfully connected to OpenAlgo WebSocket server."
    except Exception as e:
        add_ws_log("ERROR", f"Connection failed: {str(e)}")
        return f"Failed to connect: {str(e)}"


@tool
def openalgo_disconnect() -> str:
    """
    Disconnect from OpenAlgo WebSocket server.
    Call this when done with market data feeds.
    
    Returns:
        Status message indicating disconnection result.
    """
    try:
        client = get_openalgo_client()
        client.disconnect()
        add_ws_log("WS", "Disconnected from WebSocket server")
        return "Successfully disconnected from OpenAlgo WebSocket server."
    except Exception as e:
        add_ws_log("ERROR", f"Disconnection failed: {str(e)}")
        return f"Failed to disconnect: {str(e)}"


# =============================================================================
# Market Data Subscription Tools
# =============================================================================

@tool
def openalgo_subscribe_ltp(
    instruments: Annotated[str, "JSON array of instruments, e.g., [{\"exchange\":\"NSE\",\"symbol\":\"RELIANCE\"}]"]
) -> str:
    """
    Subscribe to Last Traded Price (LTP) updates for given instruments.
    Provides real-time price updates with minimal data.
    
    Args:
        instruments: JSON string array of {exchange, symbol} objects.
        
    Returns:
        Subscription status.
    """
    try:
        client = get_openalgo_client()
        instrument_list = json.loads(instruments)
        
        def on_ltp_update(data):
            add_ws_log("LTP", f"{data.get('symbol')}: {data.get('data', {}).get('ltp')}", data)
        
        client.subscribe_ltp(instrument_list, on_data_received=on_ltp_update)
        add_ws_log("SUB", f"Subscribed to LTP for {len(instrument_list)} instruments", {"instruments": instrument_list})
        return f"Subscribed to LTP updates for {len(instrument_list)} instruments."
    except json.JSONDecodeError as e:
        return f"Invalid JSON format: {str(e)}"
    except Exception as e:
        add_ws_log("ERROR", f"LTP subscription failed: {str(e)}")
        return f"Subscription failed: {str(e)}"


@tool
def openalgo_subscribe_quote(
    instruments: Annotated[str, "JSON array of instruments, e.g., [{\"exchange\":\"NSE\",\"symbol\":\"RELIANCE\"}]"]
) -> str:
    """
    Subscribe to Quote updates (OHLC + LTP) for given instruments.
    Provides more detail than LTP including Open, High, Low, Close.
    
    Args:
        instruments: JSON string array of {exchange, symbol} objects.
        
    Returns:
        Subscription status.
    """
    try:
        client = get_openalgo_client()
        instrument_list = json.loads(instruments)
        
        def on_quote_update(data):
            quote = data.get('data', {})
            msg = f"{data.get('symbol')}: LTP={quote.get('ltp')} O={quote.get('open')} H={quote.get('high')} L={quote.get('low')}"
            add_ws_log("QUOTE", msg, data)
        
        client.subscribe_quote(instrument_list, on_data_received=on_quote_update)
        add_ws_log("SUB", f"Subscribed to Quotes for {len(instrument_list)} instruments", {"instruments": instrument_list})
        return f"Subscribed to Quote updates for {len(instrument_list)} instruments."
    except json.JSONDecodeError as e:
        return f"Invalid JSON format: {str(e)}"
    except Exception as e:
        add_ws_log("ERROR", f"Quote subscription failed: {str(e)}")
        return f"Subscription failed: {str(e)}"


@tool
def openalgo_subscribe_depth(
    instruments: Annotated[str, "JSON array of instruments, e.g., [{\"exchange\":\"NSE\",\"symbol\":\"RELIANCE\"}]"]
) -> str:
    """
    Subscribe to Market Depth (order book) updates for given instruments.
    Provides bid/ask levels with quantities.
    
    Args:
        instruments: JSON string array of {exchange, symbol} objects.
        
    Returns:
        Subscription status.
    """
    try:
        client = get_openalgo_client()
        instrument_list = json.loads(instruments)
        
        def on_depth_update(data):
            add_ws_log("DEPTH", f"{data.get('symbol')} depth updated", data)
        
        client.subscribe_depth(instrument_list, on_data_received=on_depth_update)
        add_ws_log("SUB", f"Subscribed to Depth for {len(instrument_list)} instruments", {"instruments": instrument_list})
        return f"Subscribed to Market Depth updates for {len(instrument_list)} instruments."
    except json.JSONDecodeError as e:
        return f"Invalid JSON format: {str(e)}"
    except Exception as e:
        add_ws_log("ERROR", f"Depth subscription failed: {str(e)}")
        return f"Subscription failed: {str(e)}"


# =============================================================================
# Market Data Retrieval Tools
# =============================================================================

@tool
def openalgo_get_ltp() -> str:
    """
    Get cached Last Traded Price data for all subscribed instruments.
    
    Returns:
        JSON string with LTP data organized by exchange and symbol.
    """
    try:
        client = get_openalgo_client()
        ltp_data = client.get_ltp()
        return json.dumps(ltp_data, indent=2)
    except Exception as e:
        return f"Failed to get LTP data: {str(e)}"


@tool
def openalgo_get_quotes() -> str:
    """
    Get cached Quote data (OHLC + LTP) for all subscribed instruments.
    
    Returns:
        JSON string with quote data including open, high, low, close, ltp.
    """
    try:
        client = get_openalgo_client()
        quotes_data = client.get_quotes()
        return json.dumps(quotes_data, indent=2)
    except Exception as e:
        return f"Failed to get quote data: {str(e)}"


@tool
def openalgo_get_depth() -> str:
    """
    Get cached Market Depth data for all subscribed instruments.
    
    Returns:
        JSON string with order book data including bid/ask levels.
    """
    try:
        client = get_openalgo_client()
        depth_data = client.get_depth()
        return json.dumps(depth_data, indent=2)
    except Exception as e:
        return f"Failed to get depth data: {str(e)}"


# =============================================================================
# Unsubscription Tools
# =============================================================================

@tool
def openalgo_unsubscribe_ltp(
    instruments: Annotated[str, "JSON array of instruments to unsubscribe"]
) -> str:
    """
    Unsubscribe from LTP updates for given instruments.
    
    Args:
        instruments: JSON string array of {exchange, symbol} objects.
        
    Returns:
        Unsubscription status.
    """
    try:
        client = get_openalgo_client()
        instrument_list = json.loads(instruments)
        client.unsubscribe_ltp(instrument_list)
        add_ws_log("UNSUB", f"Unsubscribed from LTP for {len(instrument_list)} instruments")
        return f"Unsubscribed from LTP updates for {len(instrument_list)} instruments."
    except Exception as e:
        return f"Unsubscription failed: {str(e)}"


@tool
def openalgo_unsubscribe_quote(
    instruments: Annotated[str, "JSON array of instruments to unsubscribe"]
) -> str:
    """
    Unsubscribe from Quote updates for given instruments.
    """
    try:
        client = get_openalgo_client()
        instrument_list = json.loads(instruments)
        client.unsubscribe_quote(instrument_list)
        add_ws_log("UNSUB", f"Unsubscribed from Quotes for {len(instrument_list)} instruments")
        return f"Unsubscribed from Quote updates for {len(instrument_list)} instruments."
    except Exception as e:
        return f"Unsubscription failed: {str(e)}"


@tool
def openalgo_unsubscribe_depth(
    instruments: Annotated[str, "JSON array of instruments to unsubscribe"]
) -> str:
    """
    Unsubscribe from Market Depth updates for given instruments.
    """
    try:
        client = get_openalgo_client()
        instrument_list = json.loads(instruments)
        client.unsubscribe_depth(instrument_list)
        add_ws_log("UNSUB", f"Unsubscribed from Depth for {len(instrument_list)} instruments")
        return f"Unsubscribed from Market Depth updates for {len(instrument_list)} instruments."
    except Exception as e:
        return f"Unsubscription failed: {str(e)}"


# =============================================================================
# WebSocket Logs Tool (for debugging)
# =============================================================================

@tool
def openalgo_get_ws_logs(
    limit: Annotated[int, "Number of recent logs to retrieve"] = 20
) -> str:
    """
    Get recent WebSocket activity logs.
    Useful for debugging connection and subscription issues.
    
    Args:
        limit: Number of recent logs to return (default 20).
        
    Returns:
        JSON string with recent WebSocket logs.
    """
    logs = get_ws_logs(limit)
    return json.dumps(logs, indent=2)


# =============================================================================
# Export all tools
# =============================================================================

OPENALGO_WEBSOCKET_TOOLS = [
    openalgo_connect,
    openalgo_disconnect,
    openalgo_subscribe_ltp,
    openalgo_subscribe_quote,
    openalgo_subscribe_depth,
    openalgo_get_ltp,
    openalgo_get_quotes,
    openalgo_get_depth,
    openalgo_unsubscribe_ltp,
    openalgo_unsubscribe_quote,
    openalgo_unsubscribe_depth,
    openalgo_get_ws_logs,
]

# All tools (will grow as more phases are added)
ALL_OPENALGO_TOOLS = OPENALGO_WEBSOCKET_TOOLS
