"""
OpenAlgo Options Tools for LangGraph Agents

Provides options trading tools: Greeks calculation, symbol lookup, smart orders.
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
# OPTIONS TOOLS
# =============================================================================

@tool
def openalgo_option_greeks(
    symbol: Annotated[str, "Option symbol (e.g., 'NIFTY28NOV2526000CE')"],
    exchange: Annotated[str, "Exchange (NFO, BFO, MCX)"],
    interest_rate: Annotated[float, "Interest rate (RBI repo rate)"] = 6.5,
    underlying_symbol: Annotated[str, "Optional underlying symbol for futures"] = "",
    underlying_exchange: Annotated[str, "Optional underlying exchange"] = "",
    expiry_time: Annotated[str, "Optional expiry time (e.g., '19:00' for MCX)"] = ""
) -> str:
    """
    Calculate Option Greeks using Black-Scholes Model.
    
    Returns Delta, Gamma, Theta, Vega, Rho, and Implied Volatility.
    Requires mibian library: pip install mibian
    
    Args:
        symbol: Full option symbol
        exchange: NFO, BFO, MCX
        interest_rate: Current interest rate (default RBI repo rate 6.5%)
        underlying_symbol: Use futures as underlying (for arbitrage)
        underlying_exchange: Exchange of underlying futures
        expiry_time: Custom expiry time for MCX options
    
    Returns:
        JSON with Greeks: delta, gamma, theta, vega, rho, implied_volatility.
    """
    try:
        client = get_openalgo_client()
        
        kwargs = {
            "symbol": symbol,
            "exchange": exchange.upper(),
            "interest_rate": interest_rate
        }
        
        if underlying_symbol:
            kwargs["underlying_symbol"] = underlying_symbol
        if underlying_exchange:
            kwargs["underlying_exchange"] = underlying_exchange.upper()
        if expiry_time:
            kwargs["expiry_time"] = expiry_time
        
        result = client.optiongreeks(**kwargs)
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Greeks calculation failed: {str(e)}"})


@tool
def openalgo_option_symbol(
    underlying: Annotated[str, "Underlying symbol (e.g., 'NIFTY', 'BANKNIFTY')"],
    exchange: Annotated[str, "Exchange (NSE_INDEX, NFO, etc.)"],
    expiry_date: Annotated[str, "Expiry date (e.g., '28NOV24')"],
    strike_int: Annotated[int, "Strike interval (e.g., 50 for NIFTY, 100 for BANKNIFTY)"],
    offset: Annotated[str, "Offset: ATM, ITM1-ITM50, OTM1-OTM50"],
    option_type: Annotated[str, "Option type: CE (Call) or PE (Put)"]
) -> str:
    """
    Get option symbol details without placing an order.
    
    Useful for finding the correct option symbol based on underlying and offset.
    
    Args:
        underlying: Underlying symbol or futures symbol
        exchange: NSE_INDEX, NFO, BSE_INDEX, BFO, MCX
        expiry_date: Format like '28NOV24'
        strike_int: Strike interval (50 for NIFTY, 100 for BANKNIFTY)
        offset: ATM, ITM1-50, OTM1-50
        option_type: CE (Call) or PE (Put)
    
    Returns:
        JSON with resolved symbol, lotsize, tick_size, underlying_ltp.
    """
    try:
        client = get_openalgo_client()
        
        result = client.optionsymbol(
            underlying=underlying,
            exchange=exchange.upper(),
            expiry_date=expiry_date,
            strike_int=strike_int,
            offset=offset.upper(),
            option_type=option_type.upper()
        )
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Symbol lookup failed: {str(e)}"})


@tool
def openalgo_option_order(
    strategy: Annotated[str, "Strategy name for tracking"],
    underlying: Annotated[str, "Underlying symbol (e.g., 'NIFTY')"],
    exchange: Annotated[str, "Exchange (NSE_INDEX, NFO, etc.)"],
    expiry_date: Annotated[str, "Expiry date (e.g., '28NOV24')"],
    strike_int: Annotated[int, "Strike interval"],
    offset: Annotated[str, "Offset: ATM, ITM1-50, OTM1-50"],
    option_type: Annotated[str, "CE (Call) or PE (Put)"],
    action: Annotated[str, "BUY or SELL"],
    quantity: Annotated[int, "Quantity (should be multiple of lot size)"],
    price_type: Annotated[str, "MARKET, LIMIT, SL, SL-M"] = "MARKET",
    product: Annotated[str, "MIS (intraday) or NRML (carry)"] = "MIS",
    price: Annotated[str, "Price for LIMIT orders"] = "0",
    trigger_price: Annotated[str, "Trigger price for SL orders"] = "0"
) -> str:
    """
    Place an option order with auto-resolved symbol.
    
    Automatically resolves the option symbol based on underlying and offset.
    
    Args:
        strategy: Strategy name for order tracking
        underlying: Underlying symbol (NIFTY, BANKNIFTY) or futures symbol
        exchange: NSE_INDEX for index, NFO for futures
        expiry_date: Expiry in format '28NOV24'
        strike_int: Strike interval (50, 100, etc.)
        offset: ATM, ITM1-50, OTM1-50
        option_type: CE or PE
        action: BUY or SELL
        quantity: Order quantity
        price_type: MARKET, LIMIT, SL, SL-M
        product: MIS or NRML
        price: Required for LIMIT orders
        trigger_price: Required for SL orders
    
    Returns:
        JSON with orderid, resolved symbol, and order details.
    """
    try:
        client = get_openalgo_client()
        
        kwargs = {
            "strategy": strategy,
            "underlying": underlying,
            "exchange": exchange.upper(),
            "expiry_date": expiry_date,
            "strike_int": strike_int,
            "offset": offset.upper(),
            "option_type": option_type.upper(),
            "action": action.upper(),
            "quantity": quantity,
            "price_type": price_type.upper(),
            "product": product.upper()
        }
        
        if price and price != "0":
            kwargs["price"] = price
        if trigger_price and trigger_price != "0":
            kwargs["trigger_price"] = trigger_price
        
        result = client.optionsorder(**kwargs)
        
        logger.info(f"Option order: {action} {offset} {option_type} on {underlying}")
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Option order failed: {str(e)}"})


@tool
def openalgo_build_iron_condor(
    underlying: Annotated[str, "Underlying symbol (e.g., 'NIFTY')"],
    exchange: Annotated[str, "Exchange (NSE_INDEX)"],
    expiry_date: Annotated[str, "Expiry date"],
    strike_int: Annotated[int, "Strike interval"],
    quantity: Annotated[int, "Lot quantity"],
    sell_offset: Annotated[str, "Offset for sold options (e.g., 'OTM1')"] = "OTM1",
    buy_offset: Annotated[str, "Offset for bought options (e.g., 'OTM3')"] = "OTM3",
    product: Annotated[str, "MIS or NRML"] = "MIS"
) -> str:
    """
    Build an Iron Condor strategy (4 legs).
    
    Sells OTM calls and puts at one strike, buys further OTM at another.
    
    Args:
        underlying: Underlying symbol
        exchange: Exchange code
        expiry_date: Expiry date
        strike_int: Strike interval
        quantity: Lot quantity
        sell_offset: Offset for sold options (default OTM1)
        buy_offset: Offset for bought options (default OTM3)
        product: MIS or NRML
    
    Returns:
        JSON with all 4 order results.
    """
    try:
        client = get_openalgo_client()
        
        common = {
            "strategy": "iron_condor",
            "underlying": underlying,
            "exchange": exchange.upper(),
            "expiry_date": expiry_date,
            "strike_int": strike_int,
            "quantity": quantity,
            "price_type": "MARKET",
            "product": product.upper()
        }
        
        results = {"legs": []}
        
        # Leg 1: Sell OTM Call
        leg1 = client.optionsorder(
            offset=sell_offset.upper(), option_type="CE", action="SELL", **common
        )
        results["legs"].append({"leg": 1, "type": f"SELL {sell_offset} CE", "result": leg1})
        
        # Leg 2: Sell OTM Put
        leg2 = client.optionsorder(
            offset=sell_offset.upper(), option_type="PE", action="SELL", **common
        )
        results["legs"].append({"leg": 2, "type": f"SELL {sell_offset} PE", "result": leg2})
        
        # Leg 3: Buy OTM Call (hedge)
        leg3 = client.optionsorder(
            offset=buy_offset.upper(), option_type="CE", action="BUY", **common
        )
        results["legs"].append({"leg": 3, "type": f"BUY {buy_offset} CE", "result": leg3})
        
        # Leg 4: Buy OTM Put (hedge)
        leg4 = client.optionsorder(
            offset=buy_offset.upper(), option_type="PE", action="BUY", **common
        )
        results["legs"].append({"leg": 4, "type": f"BUY {buy_offset} PE", "result": leg4})
        
        results["status"] = "success"
        results["strategy"] = "Iron Condor"
        
        logger.info(f"Iron Condor built on {underlying}")
        return json.dumps(results, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Iron Condor failed: {str(e)}"})


# =============================================================================
# Export all options tools
# =============================================================================

OPENALGO_OPTIONS_TOOLS = [
    openalgo_option_greeks,
    openalgo_option_symbol,
    openalgo_option_order,
    openalgo_build_iron_condor,
]

__all__ = [
    "OPENALGO_OPTIONS_TOOLS",
    "openalgo_option_greeks",
    "openalgo_option_symbol",
    "openalgo_option_order",
    "openalgo_build_iron_condor",
]
