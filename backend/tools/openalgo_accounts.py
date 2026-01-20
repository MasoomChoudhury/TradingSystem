"""
OpenAlgo Accounts Tools for LangGraph Agents

Provides account management tools: funds, positions, holdings, margin.
"""
import os
import json
import logging
from typing import Annotated, List, Dict, Any
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Reuse the api client from openalgo_tools
def get_openalgo_client():
    """Get the OpenAlgo API client."""
    from .openalgo_tools import get_openalgo_client as get_client
    return get_client()


# =============================================================================
# ACCOUNT TOOLS
# =============================================================================

@tool
def openalgo_get_funds() -> str:
    """
    Get available funds and margin details of the trading account.
    
    Returns:
        JSON with availablecash, collateral, m2mrealized, m2munrealized, utiliseddebits.
    """
    try:
        client = get_openalgo_client()
        result = client.funds()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to get funds: {str(e)}"})


@tool
def openalgo_get_orderbook() -> str:
    """
    Get orderbook details with order statistics.
    
    Returns:
        JSON with list of orders and statistics (total buy/sell, completed/open/rejected).
    """
    try:
        client = get_openalgo_client()
        result = client.orderbook()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to get orderbook: {str(e)}"})


@tool
def openalgo_get_tradebook() -> str:
    """
    Get executed trades from tradebook.
    
    Returns:
        JSON with list of trades (symbol, action, quantity, avgprice, tradevalue, timestamp).
    """
    try:
        client = get_openalgo_client()
        result = client.tradebook()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to get tradebook: {str(e)}"})


@tool
def openalgo_get_positions() -> str:
    """
    Get current open positions across all segments.
    
    Returns:
        JSON with list of positions (symbol, exchange, product, quantity, avgprice).
    """
    try:
        client = get_openalgo_client()
        result = client.positionbook()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to get positions: {str(e)}"})


@tool
def openalgo_get_holdings() -> str:
    """
    Get stock holdings with P&L details.
    
    Returns:
        JSON with holdings list and statistics (totalholdingvalue, totalinvvalue, totalpnlpercentage).
    """
    try:
        client = get_openalgo_client()
        result = client.holdings()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to get holdings: {str(e)}"})


@tool
def openalgo_analyzer_status() -> str:
    """
    Get analyzer status - whether in analyze (simulated) or live mode.
    
    Returns:
        JSON with analyze_mode (bool), mode (live/analyze), total_logs.
    """
    try:
        client = get_openalgo_client()
        result = client.analyzerstatus()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to get analyzer status: {str(e)}"})


@tool
def openalgo_analyzer_toggle(
    mode: Annotated[bool, "True for analyze mode (simulated), False for live mode"]
) -> str:
    """
    Toggle between analyze (simulated) and live trading modes.
    
    WARNING: Setting mode=False enables LIVE trading with real money!
    
    Args:
        mode: True = analyze mode (simulated orders), False = live mode (real orders)
    
    Returns:
        JSON with mode status after toggle.
    """
    try:
        client = get_openalgo_client()
        result = client.analyzertoggle(mode=mode)
        mode_str = "ANALYZE (simulated)" if mode else "LIVE (real orders)"
        logger.info(f"Analyzer toggled to: {mode_str}")
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to toggle analyzer: {str(e)}"})


@tool
def openalgo_calculate_margin(
    positions_json: Annotated[str, "JSON array of position objects for margin calculation"]
) -> str:
    """
    Calculate margin requirements for positions (max 50).
    
    Each position object should have:
    - symbol: Trading symbol (e.g., "SBIN", "NIFTY30DEC25FUT")
    - exchange: NSE, BSE, NFO, BFO, CDS, MCX
    - action: BUY or SELL
    - product: CNC (delivery), MIS (intraday), NRML (F&O)
    - pricetype: MARKET, LIMIT, SL, SL-M
    - quantity: Number of shares/lots
    - price: Required for LIMIT orders (use "0" for MARKET)
    
    Example:
        [{"symbol": "SBIN", "exchange": "NSE", "action": "BUY", 
          "product": "MIS", "pricetype": "LIMIT", "quantity": "10", "price": "750.50"}]
    
    Returns:
        JSON with total_margin_required, span_margin, exposure_margin.
    """
    try:
        positions = json.loads(positions_json)
        
        if not isinstance(positions, list):
            return json.dumps({"error": "positions must be a JSON array"})
        
        if len(positions) > 50:
            return json.dumps({"error": "Maximum 50 positions allowed per request"})
        
        client = get_openalgo_client()
        result = client.margin(positions=positions)
        return json.dumps(result, indent=2)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"Failed to calculate margin: {str(e)}"})


# =============================================================================
# Export all account tools
# =============================================================================

OPENALGO_ACCOUNT_TOOLS = [
    openalgo_get_funds,
    openalgo_get_orderbook,
    openalgo_get_tradebook,
    openalgo_get_positions,
    openalgo_get_holdings,
    openalgo_analyzer_status,
    openalgo_analyzer_toggle,
    openalgo_calculate_margin,
]

__all__ = [
    "OPENALGO_ACCOUNT_TOOLS",
    "openalgo_get_funds",
    "openalgo_get_orderbook",
    "openalgo_get_tradebook",
    "openalgo_get_positions",
    "openalgo_get_holdings",
    "openalgo_analyzer_status",
    "openalgo_analyzer_toggle",
    "openalgo_calculate_margin",
]
