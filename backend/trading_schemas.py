"""
Trading Schemas & Validators - Production Robustness

Strict schemas with enums for:
- Order types, sides, products, exchanges
- Symbol validation, quantity precision
- Plan IDs and client order IDs for idempotency

Fail-fast validation before any tool execution.
"""
import re
import hashlib
from enum import Enum
from typing import Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import json


# =============================================================================
# ENUMS (Strict Type Safety)
# =============================================================================

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"  # Stop Loss
    SL_M = "SL-M"  # Stop Loss Market


class Product(str, Enum):
    CNC = "CNC"  # Cash and Carry (delivery)
    MIS = "MIS"  # Margin Intraday Square-off
    NRML = "NRML"  # Normal (F&O carry forward)


class Exchange(str, Enum):
    NSE = "NSE"
    BSE = "BSE"
    NFO = "NFO"
    BFO = "BFO"
    MCX = "MCX"
    NSE_INDEX = "NSE_INDEX"
    BSE_INDEX = "BSE_INDEX"


class TimeInForce(str, Enum):
    DAY = "DAY"
    IOC = "IOC"  # Immediate or Cancel
    GTC = "GTC"  # Good Till Cancelled


class TradeIntent(str, Enum):
    INFO = "info"
    ANALYSIS = "analysis"
    TRADE = "trade"
    DEPLOY = "deploy"


class PlanStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# VALIDATION FUNCTIONS (Fail Fast)
# =============================================================================

def validate_order_side(side: str) -> OrderSide:
    """Validate and return OrderSide enum."""
    side_upper = side.upper()
    if side_upper not in [e.value for e in OrderSide]:
        raise ValueError(f"Invalid order side: '{side}'. Must be BUY or SELL")
    return OrderSide(side_upper)


def validate_order_type(order_type: str) -> OrderType:
    """Validate and return OrderType enum."""
    order_type_upper = order_type.upper()
    if order_type_upper not in [e.value for e in OrderType]:
        raise ValueError(f"Invalid order type: '{order_type}'. Must be MARKET, LIMIT, SL, or SL-M")
    return OrderType(order_type_upper)


def validate_product(product: str) -> Product:
    """Validate and return Product enum."""
    product_upper = product.upper()
    if product_upper not in [e.value for e in Product]:
        raise ValueError(f"Invalid product: '{product}'. Must be CNC, MIS, or NRML")
    return Product(product_upper)


def validate_exchange(exchange: str) -> Exchange:
    """Validate and return Exchange enum."""
    exchange_upper = exchange.upper()
    if exchange_upper not in [e.value for e in Exchange]:
        raise ValueError(f"Invalid exchange: '{exchange}'. Must be NSE, BSE, NFO, BFO, MCX, etc.")
    return Exchange(exchange_upper)


def validate_symbol_format(symbol: str, exchange: Exchange) -> str:
    """
    Validate symbol format based on exchange.
    Returns cleaned symbol or raises ValueError.
    """
    if not symbol or not symbol.strip():
        raise ValueError("Symbol cannot be empty")
    
    symbol = symbol.strip().upper()
    
    # Basic format checks
    if len(symbol) > 50:
        raise ValueError(f"Symbol too long: {len(symbol)} chars (max 50)")
    
    if not re.match(r'^[A-Z0-9\-_]+$', symbol):
        raise ValueError(f"Invalid symbol format: '{symbol}'. Only A-Z, 0-9, -, _ allowed")
    
    # Exchange-specific validation
    if exchange in [Exchange.NFO, Exchange.BFO, Exchange.MCX]:
        # F&O symbols should have expiry and strike info
        # e.g., NIFTY28NOV2524000CE, BANKNIFTY28NOV2550000PE
        pass  # More specific validation can be added
    
    return symbol


def validate_quantity(quantity: int, lot_size: int = 1) -> int:
    """
    Validate quantity is positive and respects lot size.
    Returns validated quantity or raises ValueError.
    """
    if quantity <= 0:
        raise ValueError(f"Quantity must be positive, got: {quantity}")
    
    if lot_size > 1 and quantity % lot_size != 0:
        raise ValueError(f"Quantity {quantity} must be multiple of lot size {lot_size}")
    
    return quantity


def validate_price(price: float, price_type: OrderType) -> float:
    """
    Validate price based on order type.
    """
    if price_type == OrderType.MARKET:
        return 0.0  # Market orders don't need price
    
    if price <= 0:
        raise ValueError(f"Limit/SL order requires positive price, got: {price}")
    
    return price


def validate_price_tick(price: float, tick_size: float = 0.05) -> float:
    """
    Validate price respects tick size.
    Rounds to nearest tick if needed.
    """
    if price <= 0:
        return price
    
    if tick_size <= 0:
        return price
    
    # Round to tick size
    rounded = round(price / tick_size) * tick_size
    return round(rounded, 2)


# =============================================================================
# TRADE PLAN SCHEMA (With Plan ID)
# =============================================================================

@dataclass
class TradePlan:
    """
    Strict trade plan with plan_id for idempotency.
    All orders derived from this plan use the same plan_id.
    """
    symbol: str
    exchange: Exchange
    side: OrderSide
    quantity: int
    order_type: OrderType
    product: Product
    price: float = 0.0
    trigger_price: float = 0.0
    time_in_force: TimeInForce = TimeInForce.DAY
    
    # Idempotency
    plan_id: str = field(default_factory=lambda: "")
    
    # Metadata
    strategy_name: str = ""
    intent: TradeIntent = TradeIntent.TRADE
    status: PlanStatus = PlanStatus.PENDING
    
    # Validation results
    lot_size: int = 1
    tick_size: float = 0.05
    symbol_token: str = ""
    
    # Approval
    approval_token: Optional[str] = None
    approved_at: Optional[str] = None
    
    def __post_init__(self):
        # Generate plan_id if not provided
        if not self.plan_id:
            data = f"{self.symbol}:{self.side.value}:{self.quantity}:{datetime.now().isoformat()}"
            self.plan_id = hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def generate_client_order_id(self, sequence: int = 1) -> str:
        """Generate unique client order ID for this plan."""
        return f"{self.plan_id}-{sequence}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "plan_id": self.plan_id,
            "symbol": self.symbol,
            "exchange": self.exchange.value if isinstance(self.exchange, Enum) else self.exchange,
            "side": self.side.value if isinstance(self.side, Enum) else self.side,
            "quantity": self.quantity,
            "order_type": self.order_type.value if isinstance(self.order_type, Enum) else self.order_type,
            "product": self.product.value if isinstance(self.product, Enum) else self.product,
            "price": self.price,
            "trigger_price": self.trigger_price,
            "time_in_force": self.time_in_force.value if isinstance(self.time_in_force, Enum) else self.time_in_force,
            "strategy_name": self.strategy_name,
            "intent": self.intent.value if isinstance(self.intent, Enum) else self.intent,
            "status": self.status.value if isinstance(self.status, Enum) else self.status,
            "lot_size": self.lot_size,
            "tick_size": self.tick_size,
            "symbol_token": self.symbol_token,
            "approval_token": self.approval_token,
            "approved_at": self.approved_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TradePlan':
        """Create TradePlan from dictionary."""
        return cls(
            symbol=data.get("symbol", ""),
            exchange=validate_exchange(data.get("exchange", "NSE")),
            side=validate_order_side(data.get("side", data.get("action", "BUY"))),
            quantity=data.get("quantity", 0),
            order_type=validate_order_type(data.get("order_type", data.get("price_type", "MARKET"))),
            product=validate_product(data.get("product", "MIS")),
            price=float(data.get("price", 0)),
            trigger_price=float(data.get("trigger_price", 0)),
            plan_id=data.get("plan_id", ""),
            strategy_name=data.get("strategy_name", ""),
            lot_size=data.get("lot_size", 1),
            tick_size=data.get("tick_size", 0.05),
            symbol_token=data.get("symbol_token", ""),
            approval_token=data.get("approval_token"),
        )


# =============================================================================
# FULL VALIDATION PIPELINE
# =============================================================================

def validate_trade_plan(data: dict) -> tuple[bool, TradePlan | None, str]:
    """
    Full validation of a trade plan.
    Returns (is_valid, TradePlan object, error message)
    """
    errors = []
    
    # Required fields
    required = ["symbol", "exchange", "quantity"]
    for field in required:
        if field not in data or not data[field]:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        return False, None, "; ".join(errors)
    
    try:
        # Validate each field
        side = validate_order_side(data.get("side", data.get("action", "BUY")))
        order_type = validate_order_type(data.get("order_type", data.get("price_type", "MARKET")))
        product = validate_product(data.get("product", "MIS"))
        exchange = validate_exchange(data.get("exchange", "NSE"))
        
        symbol = validate_symbol_format(data.get("symbol", ""), exchange)
        quantity = validate_quantity(data.get("quantity", 0), data.get("lot_size", 1))
        price = validate_price(float(data.get("price", 0)), order_type)
        
        # Create validated plan
        plan = TradePlan(
            symbol=symbol,
            exchange=exchange,
            side=side,
            quantity=quantity,
            order_type=order_type,
            product=product,
            price=price,
            trigger_price=float(data.get("trigger_price", 0)),
            plan_id=data.get("plan_id", ""),
            strategy_name=data.get("strategy_name", ""),
            lot_size=data.get("lot_size", 1),
            tick_size=data.get("tick_size", 0.05),
            symbol_token=data.get("symbol_token", ""),
        )
        
        return True, plan, "Validation passed"
        
    except ValueError as e:
        return False, None, str(e)
    except Exception as e:
        return False, None, f"Validation error: {str(e)}"


# =============================================================================
# DYNAMIC TOOL SELECTION
# =============================================================================

TOOL_KEYWORDS = {
    # Market Data
    "price": ["get_quotes", "get_ltp"],
    "quote": ["get_quotes"],
    "history": ["get_history"],
    "historical": ["get_history"],
    "search": ["search_symbols"],
    "symbol": ["get_symbol_info", "search_symbols"],
    "expiry": ["get_expiry_dates"],
    "depth": ["get_market_depth"],
    
    # Indicators
    "indicator": ["calculate_indicator", "list_indicators", "validate_indicator"],
    "rsi": ["calculate_indicator"],
    "macd": ["calculate_indicator"],
    "ema": ["calculate_indicator"],
    "sma": ["calculate_indicator"],
    "bollinger": ["calculate_indicator"],
    "atr": ["calculate_indicator"],
    
    # Options
    "option": ["get_option_greeks", "get_option_symbol"],
    "greeks": ["get_option_greeks"],
    "delta": ["get_option_greeks"],
    "theta": ["get_option_greeks"],
    "iv": ["get_option_greeks"],
    "strike": ["get_option_symbol"],
    "atm": ["get_option_symbol"],
    "otm": ["get_option_symbol"],
    "itm": ["get_option_symbol"],
    
    # Accounts
    "fund": ["get_funds"],
    "balance": ["get_funds"],
    "position": ["get_positions", "get_open_position"],
    "holding": ["get_holdings"],
    "order": ["get_orderbook", "get_order_status"],
    "trade": ["get_tradebook"],
    "margin": ["calculate_margin"],
    
    # Execution
    "buy": ["place_order"],
    "sell": ["place_order"],
    "place": ["place_order", "place_smart_order"],
    "modify": ["modify_order"],
    "cancel": ["cancel_order"],
    "basket": ["place_basket_order"],
    "split": ["place_split_order"],
    
    # Emergency
    "emergency": ["emergency_cancel_all_orders", "emergency_close_all_positions"],
    "circuit": ["trip_circuit_breaker", "reset_circuit_breaker", "get_circuit_breaker_status"],
    "close all": ["emergency_close_all_positions"],
    "cancel all": ["emergency_cancel_all_orders"],

    # Fundamentals
    "fundamental": ["fundamental_analysis"],
    "thesis": ["fundamental_analysis"],
    "research": ["fundamental_analysis"],
    "balance sheet": ["fundamental_analysis"],
    "earnings": ["fundamental_analysis"],
    "revenue": ["fundamental_analysis"],
    "profit": ["fundamental_analysis"],
}


def select_relevant_tools(query: str, available_tools: List[str], top_k: int = 5) -> List[str]:
    """
    Select top-k relevant tools based on query keywords.
    BigTool-style dynamic tool selection.
    """
    query_lower = query.lower()
    tool_scores = {tool: 0 for tool in available_tools}
    
    # Score tools based on keyword matches
    for keyword, tools in TOOL_KEYWORDS.items():
        if keyword in query_lower:
            for tool in tools:
                if tool in tool_scores:
                    tool_scores[tool] += 1
    
    # Sort by score (descending) and return top-k
    sorted_tools = sorted(tool_scores.items(), key=lambda x: x[1], reverse=True)
    selected = [tool for tool, score in sorted_tools[:top_k] if score > 0]
    
    # If no matches, return first top_k tools as fallback
    if not selected:
        selected = available_tools[:top_k]
    
    return selected


def get_worker_for_intent(intent: str) -> str:
    """Map intent to appropriate worker."""
    intent_map = {
        "price": "market_data",
        "quote": "market_data",
        "history": "market_data",
        "search": "market_data",
        "indicator": "indicators",
        "rsi": "indicators",
        "macd": "indicators",
        "option": "options",
        "greeks": "options",
        "fund": "accounts",
        "position": "accounts",
        "fund": "accounts",
        "position": "accounts",
        "margin": "accounts",
        "fundamental": "fundamentals",
        "thesis": "fundamentals",
        "research": "fundamentals",
        "balance": "fundamentals",
        "earnings": "fundamentals",
        "revenue": "fundamentals",
    }
    
    intent_lower = intent.lower()
    for keyword, worker in intent_map.items():
        if keyword in intent_lower:
            return worker
    
    return "market_data"  # Default


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "OrderSide",
    "OrderType", 
    "Product",
    "Exchange",
    "TimeInForce",
    "TradeIntent",
    "PlanStatus",
    # Validators
    "validate_order_side",
    "validate_order_type",
    "validate_product",
    "validate_exchange",
    "validate_symbol_format",
    "validate_quantity",
    "validate_price",
    "validate_price_tick",
    "validate_trade_plan",
    # Schema
    "TradePlan",
    # Dynamic selection
    "select_relevant_tools",
    "get_worker_for_intent",
]
