"""
OpenAlgo Market Data Tools for LangGraph Agents

Provides REST market data tools: quotes, depth, history, symbol search, expiry.
"""
import os
import json
import logging
from typing import Annotated, Optional
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def get_openalgo_client():
    """Get the OpenAlgo API client."""
    from .openalgo_tools import get_openalgo_client as get_client
    return get_client()


# =============================================================================
# MARKET DATA TOOLS
# =============================================================================

@tool
def openalgo_get_quotes(
    symbol: Annotated[str, "Trading symbol (e.g., 'RELIANCE')"],
    exchange: Annotated[str, "Exchange: NSE, BSE, NFO, MCX, etc."]
) -> str:
    """
    Get real-time quotes for a symbol via REST API.
    
    Returns bid/ask, LTP, volume, and other quote data.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange code
    
    Returns:
        JSON with quote data.
    """
    try:
        client = get_openalgo_client()
        result = client.quotes(symbol=symbol, exchange=exchange.upper())
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Quotes failed: {str(e)}"})


@tool
def openalgo_get_market_depth(
    symbol: Annotated[str, "Trading symbol"],
    exchange: Annotated[str, "Exchange code"]
) -> str:
    """
    Get market depth (order book) with top 5 bids/asks.
    
    Returns:
        JSON with bid and ask levels.
    """
    try:
        client = get_openalgo_client()
        result = client.depth(symbol=symbol, exchange=exchange.upper())
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Depth failed: {str(e)}"})


@tool
def openalgo_get_history(
    symbol: Annotated[str, "Trading symbol"],
    exchange: Annotated[str, "Exchange code"],
    interval: Annotated[str, "Time interval: 1m, 5m, 15m, 30m, 60m, D"],
    start_date: Annotated[str, "Start date: YYYY-MM-DD"],
    end_date: Annotated[str, "End date: YYYY-MM-DD"]
) -> str:
    """
    Get historical OHLC price data.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange code
        interval: 1m, 2m, 3m, 5m, 10m, 15m, 30m, 60m, D
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        JSON with OHLC data (last 50 bars).
    """
    try:
        client = get_openalgo_client()
        df = client.history(
            symbol=symbol,
            exchange=exchange.upper(),
            interval=interval,
            start_date=start_date,
            end_date=end_date
        )
        
        # Convert DataFrame to JSON (last 50 rows)
        if hasattr(df, 'tail'):
            data = df.tail(50).reset_index().to_dict(orient='records')
            return json.dumps({
                "status": "success",
                "symbol": symbol,
                "interval": interval,
                "count": len(data),
                "data": data
            }, indent=2, default=str)
        else:
            return json.dumps(df, indent=2, default=str)
            
    except Exception as e:
        return json.dumps({"error": f"History failed: {str(e)}"})


@tool
def openalgo_get_intervals() -> str:
    """
    Get supported time intervals for historical data.
    
    Returns:
        JSON with available intervals (seconds, minutes, days, etc.).
    """
    try:
        client = get_openalgo_client()
        result = client.intervals()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Intervals failed: {str(e)}"})


@tool
def openalgo_get_symbol_info(
    symbol: Annotated[str, "Symbol to lookup (e.g., 'NIFTY24APR25FUT')"],
    exchange: Annotated[str, "Exchange code"]
) -> str:
    """
    Get detailed symbol information.
    
    Returns lotsize, token, expiry, instrumenttype, tick_size, etc.
    
    Args:
        symbol: Full symbol name
        exchange: Exchange code
    
    Returns:
        JSON with symbol details.
    """
    try:
        client = get_openalgo_client()
        result = client.symbol(symbol=symbol, exchange=exchange.upper())
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Symbol lookup failed: {str(e)}"})


@tool
def openalgo_search_symbols(
    query: Annotated[str, "Search query (e.g., 'NIFTY', 'RELIANCE')"],
    exchange: Annotated[str, "Optional exchange filter"] = ""
) -> str:
    """
    Search for symbols across exchanges.
    
    Args:
        query: Search term
        exchange: Optional filter (NSE, NFO, BSE, BFO, MCX, CDS, BCD, NCDEX)
    
    Returns:
        JSON list of matching symbols with details.
    """
    try:
        client = get_openalgo_client()
        if exchange:
            result = client.search(query=query, exchange=exchange.upper())
        else:
            result = client.search(query=query)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Search failed: {str(e)}"})


@tool
def openalgo_get_expiry(
    symbol: Annotated[str, "Underlying symbol (e.g., 'NIFTY', 'BANKNIFTY')"],
    exchange: Annotated[str, "Exchange (NFO, BFO, MCX)"],
    instrumenttype: Annotated[str, "Type: 'futures' or 'options'"]
) -> str:
    """
    Get expiry dates for futures or options.
    
    Args:
        symbol: Underlying symbol
        exchange: Exchange code
        instrumenttype: 'futures' or 'options'
    
    Returns:
        JSON list of expiry dates.
    """
    try:
        client = get_openalgo_client()
        result = client.expiry(
            symbol=symbol,
            exchange=exchange.upper(),
            instrumenttype=instrumenttype.lower()
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Expiry failed: {str(e)}"})


# =============================================================================
# Export all market data tools
# =============================================================================

OPENALGO_MARKETDATA_TOOLS = [
    openalgo_get_quotes,
    openalgo_get_market_depth,
    openalgo_get_history,
    openalgo_get_intervals,
    openalgo_get_symbol_info,
    openalgo_search_symbols,
    openalgo_get_expiry,
]

__all__ = [
    "OPENALGO_MARKETDATA_TOOLS",
    "openalgo_get_quotes",
    "openalgo_get_market_depth",
    "openalgo_get_history",
    "openalgo_get_intervals",
    "openalgo_get_symbol_info",
    "openalgo_search_symbols",
    "openalgo_get_expiry",
]
