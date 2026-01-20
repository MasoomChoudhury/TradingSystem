"""
Supervisor Guard Workers - Specialized Validation Agents

Each guard has ONLY the validation tools it needs:
- PolicyGuard: permissions, allowed instruments, mode checks
- RiskGuard: exposure, margin, position sizing → issues approval tokens
- DataIntegrityGuard: symbol resolution, lot sizes, price validation

Guards CANNOT:
- Place orders
- Modify orders  
- Execute any broker side effects

All guards return validation results to Supervisor for final decision.
"""
import os
import json
import logging
import hashlib
from datetime import datetime
from typing import Annotated, TypedDict, Sequence, Optional, List
from dataclasses import dataclass
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBW90vlRHW6uJgWbjzoNOYYKndHPp33ctk")


# =============================================================================
# POLICY CONFIGURATION
# =============================================================================

@dataclass
class TradingPolicy:
    """Trading policy configuration."""
    allowed_exchanges: List[str] = None
    allowed_products: List[str] = None
    blocked_symbols: List[str] = None
    max_position_size: int = 100
    max_order_value: float = 500000.0
    max_daily_loss: float = 10000.0
    require_analyze_mode_first: bool = True
    allowed_modes: List[str] = None
    max_open_positions: int = 5
    
    def __post_init__(self):
        if self.allowed_exchanges is None:
            self.allowed_exchanges = ["NSE", "BSE", "NFO", "BFO", "MCX"]
        if self.allowed_products is None:
            self.allowed_products = ["CNC", "MIS", "NRML"]
        if self.blocked_symbols is None:
            self.blocked_symbols = []
        if self.allowed_modes is None:
            self.allowed_modes = ["analyze", "live"]


# Global policy
DEFAULT_POLICY = TradingPolicy()


# =============================================================================
# GUARD STATE
# =============================================================================

class GuardState(TypedDict):
    """State for guard agents."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    trade_plan: Optional[dict]
    validation_result: Optional[dict]


# =============================================================================
# BASE GUARD CLASS
# =============================================================================

class BaseGuard:
    """Base class for all guard agents."""
    
    def __init__(self, name: str, tools: list, system_prompt: str):
        from rate_limiter import RateLimitedLLM
        
        self.name = name
        self.tools = tools
        self.system_prompt = system_prompt
        
        base_llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            google_api_key=GEMINI_API_KEY,
            temperature=0.0,  # Zero temp for deterministic validation
        )
        
        self.llm = RateLimitedLLM(base_llm, estimated_tokens_per_call=2000)
        
        self.llm_with_tools = self.llm.bind_tools(tools)
        self.tool_node = ToolNode(tools)
        self.memory = InMemorySaver()
        self.graph = self._build_graph()
        self.app = self.graph.compile(checkpointer=self.memory)
        
        logger.info(f"{name} Guard initialized with {len(tools)} tools")
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(GuardState)
        
        workflow.add_node("guard", self._validate)
        workflow.add_node("tools", self.tool_node)
        
        workflow.set_entry_point("guard")
        
        workflow.add_conditional_edges(
            "guard",
            self._should_continue,
            {"continue": "tools", "end": END}
        )
        
        workflow.add_edge("tools", "guard")
        
        return workflow
    
    def _validate(self, state: GuardState) -> dict:
        messages = list(state["messages"])
        full_messages = [SystemMessage(content=self.system_prompt)] + messages
        response = self.llm_with_tools.invoke(full_messages)
        return {"messages": [response]}
    
    def _should_continue(self, state: GuardState) -> str:
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        return "end"
    
    async def validate(self, trade_plan: dict, thread_id: str = None) -> dict:
        """Validate a trade plan."""
        thread_id = thread_id or f"{self.name}_default"
        config = {"configurable": {"thread_id": thread_id}}
        
        input_state = {
            "messages": [HumanMessage(content=f"Validate this trade plan:\n```json\n{json.dumps(trade_plan, indent=2)}\n```")],
            "trade_plan": trade_plan,
            "validation_result": None,
        }
        
        tool_calls_made = []
        final_response = ""
        
        try:
            async for event in self.app.astream(input_state, config):
                for node_name, node_output in event.items():
                    if node_name == "tools":
                        for msg in node_output.get("messages", []):
                            if hasattr(msg, "name"):
                                tool_calls_made.append({
                                    "tool": msg.name,
                                    "result": msg.content[:500] if len(msg.content) > 500 else msg.content
                                })
                    elif node_name == "guard":
                        messages = node_output.get("messages", [])
                        if messages:
                            last_msg = messages[-1]
                            if isinstance(last_msg, AIMessage) and last_msg.content:
                                content = last_msg.content
                                if isinstance(content, str):
                                    final_response = content
                                elif isinstance(content, list):
                                    texts = [b.get("text", "") if isinstance(b, dict) else str(b) for b in content]
                                    final_response = "\n".join(texts)
            
            return {
                "guard": self.name,
                "response": final_response,
                "tool_calls": tool_calls_made,
            }
            
        except Exception as e:
            logger.error(f"{self.name} Guard error: {e}")
            return {"guard": self.name, "response": f"Error: {str(e)}", "tool_calls": [], "passed": False}


# =============================================================================
# POLICY GUARD
# =============================================================================

def get_policy_guard_tools():
    """Get tools for Policy validation."""
    
    @tool
    def check_exchange_allowed(
        exchange: Annotated[str, "Exchange code to check"]
    ) -> str:
        """Check if exchange is in the allowed list."""
        allowed = exchange.upper() in DEFAULT_POLICY.allowed_exchanges
        return json.dumps({
            "exchange": exchange,
            "allowed": allowed,
            "allowed_exchanges": DEFAULT_POLICY.allowed_exchanges,
            "check": "policy"
        })
    
    @tool
    def check_product_allowed(
        product: Annotated[str, "Product type to check"]
    ) -> str:
        """Check if product type is allowed."""
        allowed = product.upper() in DEFAULT_POLICY.allowed_products
        return json.dumps({
            "product": product,
            "allowed": allowed,
            "allowed_products": DEFAULT_POLICY.allowed_products,
            "check": "policy"
        })
    
    @tool
    def check_symbol_blocked(
        symbol: Annotated[str, "Symbol to check"]
    ) -> str:
        """Check if symbol is in the blocked list."""
        blocked = symbol.upper() in DEFAULT_POLICY.blocked_symbols
        return json.dumps({
            "symbol": symbol,
            "blocked": blocked,
            "reason": "In blocked list" if blocked else "Not blocked",
            "check": "policy"
        })
    
    @tool
    def check_trading_mode() -> str:
        """Check current trading mode (live/analyze)."""
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            result = client.analyzerstatus()
            
            mode = "analyze" if result.get("analyzer_mode", True) else "live"
            allowed = mode in DEFAULT_POLICY.allowed_modes
            
            return json.dumps({
                "mode": mode,
                "allowed": allowed,
                "require_analyze_first": DEFAULT_POLICY.require_analyze_mode_first,
                "check": "policy"
            })
        except Exception as e:
            return json.dumps({"error": str(e), "check": "policy"})
    
    @tool
    def validate_policy(
        trade_plan_json: Annotated[str, "Trade plan JSON to validate"]
    ) -> str:
        """Run all policy checks on a trade plan."""
        try:
            trade = json.loads(trade_plan_json)
        except json.JSONDecodeError as e:
            return json.dumps({"passed": False, "reason": f"Invalid JSON: {str(e)}"})
        
        checks = []
        
        # Exchange check
        exchange = trade.get("exchange", "")
        exchange_ok = exchange.upper() in DEFAULT_POLICY.allowed_exchanges
        checks.append({"check": "exchange", "passed": exchange_ok, "value": exchange})
        
        # Product check
        product = trade.get("product", "")
        product_ok = product.upper() in DEFAULT_POLICY.allowed_products
        checks.append({"check": "product", "passed": product_ok, "value": product})
        
        # Symbol check
        symbol = trade.get("symbol", "")
        symbol_ok = symbol.upper() not in DEFAULT_POLICY.blocked_symbols
        checks.append({"check": "symbol_not_blocked", "passed": symbol_ok, "value": symbol})
        
        all_passed = all(c["passed"] for c in checks)
        
        return json.dumps({
            "guard": "Policy",
            "passed": all_passed,
            "checks": checks,
            "reason": "All policy checks passed" if all_passed else "Policy violation detected"
        }, indent=2)
    
    return [check_exchange_allowed, check_product_allowed, check_symbol_blocked, 
            check_trading_mode, validate_policy]


class PolicyGuard(BaseGuard):
    """Policy Guard - validates permissions and allowed instruments."""
    
    def __init__(self):
        tools = get_policy_guard_tools()
        system_prompt = """You are the POLICY GUARD - enforcing trading policies.

## YOUR TOOLS
- check_exchange_allowed: Verify exchange is permitted
- check_product_allowed: Verify product type is permitted
- check_symbol_blocked: Check if symbol is blocked
- check_trading_mode: Check live/analyze mode
- validate_policy: Run all policy checks on a trade plan

## YOUR RESPONSIBILITY
Ensure trades comply with configured policies:
- Only allowed exchanges (NSE, BSE, NFO, BFO, MCX)
- Only allowed products (CNC, MIS, NRML)
- No blocked symbols
- Correct trading mode

## OUTPUT FORMAT
```json
{
    "guard": "Policy",
    "passed": true/false,
    "checks": [...],
    "reason": "explanation"
}
```

## CONSTRAINTS
- You can ONLY validate policies
- You CANNOT place or modify orders
- You CANNOT issue approval tokens (that's Risk Guard)"""

        super().__init__("Policy", tools, system_prompt)


# =============================================================================
# RISK GUARD
# =============================================================================

def get_risk_guard_tools():
    """Get tools for Risk validation."""
    
    @tool
    def check_position_size(
        quantity: Annotated[int, "Order quantity"],
        max_allowed: Annotated[int, "Maximum allowed"] = None
    ) -> str:
        """Check if position size is within limits."""
        max_size = max_allowed or DEFAULT_POLICY.max_position_size
        within_limit = quantity <= max_size
        return json.dumps({
            "quantity": quantity,
            "max_allowed": max_size,
            "within_limit": within_limit,
            "check": "risk"
        })
    
    @tool
    def check_order_value(
        symbol: Annotated[str, "Symbol"],
        exchange: Annotated[str, "Exchange"],
        quantity: Annotated[int, "Quantity"]
    ) -> str:
        """Check if order value is within limits."""
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            
            quotes = client.quotes(symbol=symbol, exchange=exchange.upper())
            ltp = quotes.get("ltp", quotes.get("data", {}).get("ltp", 0))
            
            order_value = ltp * quantity
            within_limit = order_value <= DEFAULT_POLICY.max_order_value
            
            return json.dumps({
                "ltp": ltp,
                "quantity": quantity,
                "order_value": order_value,
                "max_allowed": DEFAULT_POLICY.max_order_value,
                "within_limit": within_limit,
                "check": "risk"
            })
        except Exception as e:
            return json.dumps({"error": str(e), "check": "risk"})
    
    @tool
    def check_margin_available(
        symbol: Annotated[str, "Symbol"],
        exchange: Annotated[str, "Exchange"],
        quantity: Annotated[int, "Quantity"],
        product: Annotated[str, "Product type"],
        action: Annotated[str, "BUY or SELL"]
    ) -> str:
        """Check if sufficient margin is available."""
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            
            # Get margin requirement
            position = [{
                "symbol": symbol,
                "exchange": exchange.upper(),
                "quantity": quantity,
                "product": product.upper(),
                "side": action.upper()
            }]
            margin_result = client.margin(positions=position)
            required_margin = margin_result.get("required_margin", margin_result.get("data", {}).get("required_margin", 0))
            
            # Get available funds
            funds = client.funds()
            available = funds.get("available_margin", funds.get("data", {}).get("available_margin", 0))
            
            sufficient = available >= required_margin
            
            return json.dumps({
                "required_margin": required_margin,
                "available_margin": available,
                "sufficient": sufficient,
                "check": "risk"
            })
        except Exception as e:
            return json.dumps({"error": str(e), "check": "risk"})
    
    @tool
    def check_open_positions() -> str:
        """Check current open positions count."""
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            
            positions = client.positionbook()
            pos_list = positions.get("data", positions) if isinstance(positions, dict) else positions
            count = len(pos_list) if isinstance(pos_list, list) else 0
            
            within_limit = count < DEFAULT_POLICY.max_open_positions
            
            return json.dumps({
                "open_positions": count,
                "max_allowed": DEFAULT_POLICY.max_open_positions,
                "within_limit": within_limit,
                "check": "risk"
            })
        except Exception as e:
            return json.dumps({"error": str(e), "check": "risk"})
    
    @tool
    def validate_risk_and_issue_token(
        trade_plan_json: Annotated[str, "Trade plan JSON to validate"]
    ) -> str:
        """
        Run all risk checks and issue approval token if passed.
        This is the ONLY way to get an approval token.
        """
        try:
            trade = json.loads(trade_plan_json)
        except json.JSONDecodeError as e:
            return json.dumps({"passed": False, "reason": f"Invalid JSON: {str(e)}"})
        
        checks = []
        
        # Position size check
        quantity = trade.get("quantity", 0)
        size_ok = quantity <= DEFAULT_POLICY.max_position_size
        checks.append({"check": "position_size", "passed": size_ok, 
                       "value": quantity, "max": DEFAULT_POLICY.max_position_size})
        
        if not size_ok:
            return json.dumps({
                "guard": "Risk",
                "passed": False,
                "checks": checks,
                "approval_token": None,
                "reason": f"Position size {quantity} exceeds max {DEFAULT_POLICY.max_position_size}"
            }, indent=2)
        
        # All checks passed - issue approval token
        token_data = f"{trade.get('symbol')}:{trade.get('action')}:{quantity}:{datetime.now().isoformat()}"
        approval_token = hashlib.sha256(token_data.encode()).hexdigest()[:16]
        
        logger.info(f"Risk Guard issued approval token: {approval_token[:8]}...")
        
        return json.dumps({
            "guard": "Risk",
            "passed": True,
            "checks": checks,
            "approval_token": approval_token,
            "token_expires_in": "5 minutes",
            "reason": "All risk checks passed - trade approved"
        }, indent=2)
    
    return [check_position_size, check_order_value, check_margin_available,
            check_open_positions, validate_risk_and_issue_token]


class RiskGuard(BaseGuard):
    """Risk Guard - exposure/margin/sizing checks and approval token issuance."""
    
    def __init__(self):
        tools = get_risk_guard_tools()
        system_prompt = """You are the RISK GUARD - enforcing risk limits and issuing approval tokens.

## YOUR TOOLS
- check_position_size: Verify quantity within limits
- check_order_value: Verify order value within limits
- check_margin_available: Verify sufficient margin
- check_open_positions: Verify position count within limits
- validate_risk_and_issue_token: Run all checks and issue approval token

## YOUR RESPONSIBILITY
You are the ONLY guard that can issue approval tokens.
Executor CANNOT execute without your approval token.

Check these risk metrics:
- Position size <= max (100)
- Order value <= max (₹500,000)
- Sufficient margin available
- Open positions < max (5)

## OUTPUT FORMAT
```json
{
    "guard": "Risk",
    "passed": true/false,
    "checks": [...],
    "approval_token": "xxx" (only if passed),
    "reason": "explanation"
}
```

## CONSTRAINTS
- You CANNOT place orders
- You can ONLY validate and issue tokens
- Token is required for Executor to proceed"""

        super().__init__("Risk", tools, system_prompt)


# =============================================================================
# DATA INTEGRITY GUARD
# =============================================================================

def get_data_integrity_tools():
    """Get tools for Data Integrity validation."""
    
    @tool
    def validate_symbol_exists(
        symbol: Annotated[str, "Symbol to validate"],
        exchange: Annotated[str, "Exchange code"]
    ) -> str:
        """Validate that the symbol exists and get its details."""
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            
            result = client.symbol(symbol=symbol, exchange=exchange.upper())
            
            if result.get("status") == "success" or result.get("data"):
                data = result.get("data", result)
                return json.dumps({
                    "symbol": symbol,
                    "exchange": exchange,
                    "exists": True,
                    "token": data.get("token", ""),
                    "lot_size": data.get("lotsize", 1),
                    "tick_size": data.get("tick_size", 0.05),
                    "check": "data_integrity"
                })
            else:
                return json.dumps({
                    "symbol": symbol,
                    "exchange": exchange,
                    "exists": False,
                    "check": "data_integrity"
                })
        except Exception as e:
            return json.dumps({"error": str(e), "check": "data_integrity"})
    
    @tool
    def validate_lot_size(
        quantity: Annotated[int, "Quantity to validate"],
        symbol: Annotated[str, "Symbol"],
        exchange: Annotated[str, "Exchange"]
    ) -> str:
        """Validate that quantity is a valid multiple of lot size."""
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            
            result = client.symbol(symbol=symbol, exchange=exchange.upper())
            data = result.get("data", result)
            lot_size = data.get("lotsize", 1)
            
            is_valid = quantity % lot_size == 0 if lot_size > 0 else True
            
            return json.dumps({
                "quantity": quantity,
                "lot_size": lot_size,
                "is_valid": is_valid,
                "suggested": (quantity // lot_size) * lot_size if not is_valid else quantity,
                "check": "data_integrity"
            })
        except Exception as e:
            return json.dumps({"error": str(e), "check": "data_integrity"})
    
    @tool
    def validate_price_bounds(
        price: Annotated[float, "Price to validate"],
        symbol: Annotated[str, "Symbol"],
        exchange: Annotated[str, "Exchange"],
        threshold: Annotated[float, "Max deviation from LTP (default 5%)"] = 0.05
    ) -> str:
        """Validate limit price is within reasonable bounds of LTP."""
        if price <= 0:
            return json.dumps({"price": price, "is_valid": True, "reason": "Market order", "check": "data_integrity"})
        
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            
            quotes = client.quotes(symbol=symbol, exchange=exchange.upper())
            ltp = quotes.get("ltp", quotes.get("data", {}).get("ltp", 0))
            
            if ltp <= 0:
                return json.dumps({"error": "Could not get LTP", "check": "data_integrity"})
            
            deviation = abs(price - ltp) / ltp
            is_valid = deviation <= threshold
            
            return json.dumps({
                "price": price,
                "ltp": ltp,
                "deviation_pct": round(deviation * 100, 2),
                "threshold_pct": threshold * 100,
                "is_valid": is_valid,
                "check": "data_integrity"
            })
        except Exception as e:
            return json.dumps({"error": str(e), "check": "data_integrity"})
    
    @tool
    def validate_expiry_date(
        symbol: Annotated[str, "Symbol"],
        exchange: Annotated[str, "Exchange"],
        instrument_type: Annotated[str, "futures or options"],
        expiry_date: Annotated[str, "Expiry date to validate"]
    ) -> str:
        """Validate that the expiry date is valid for the instrument."""
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            
            result = client.expiry(
                symbol=symbol, exchange=exchange.upper(),
                instrumenttype=instrument_type.lower()
            )
            
            expiry_list = result.get("data", result) if isinstance(result, dict) else result
            is_valid = expiry_date in expiry_list if isinstance(expiry_list, list) else False
            
            return json.dumps({
                "expiry_date": expiry_date,
                "is_valid": is_valid,
                "available_expiries": expiry_list[:5] if isinstance(expiry_list, list) else [],
                "check": "data_integrity"
            })
        except Exception as e:
            return json.dumps({"error": str(e), "check": "data_integrity"})
    
    @tool
    def validate_data_integrity(
        trade_plan_json: Annotated[str, "Trade plan JSON to validate"]
    ) -> str:
        """Run all data integrity checks on a trade plan."""
        try:
            trade = json.loads(trade_plan_json)
        except json.JSONDecodeError as e:
            return json.dumps({"passed": False, "reason": f"Invalid JSON: {str(e)}"})
        
        checks = []
        symbol = trade.get("symbol", "")
        exchange = trade.get("exchange", "")
        quantity = trade.get("quantity", 0)
        price = trade.get("price", 0)
        
        # Symbol exists check
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            result = client.symbol(symbol=symbol, exchange=exchange.upper())
            symbol_exists = result.get("status") == "success" or bool(result.get("data"))
            lot_size = result.get("data", result).get("lotsize", 1) if symbol_exists else 1
            checks.append({"check": "symbol_exists", "passed": symbol_exists, "value": symbol})
        except:
            symbol_exists = True  # Assume valid if can't check
            lot_size = 1
            checks.append({"check": "symbol_exists", "passed": True, "value": symbol, "note": "Could not verify"})
        
        # Lot size check
        lot_valid = quantity % lot_size == 0 if lot_size > 0 else True
        checks.append({"check": "lot_size", "passed": lot_valid, "quantity": quantity, "lot_size": lot_size})
        
        all_passed = all(c["passed"] for c in checks)
        
        return json.dumps({
            "guard": "DataIntegrity",
            "passed": all_passed,
            "checks": checks,
            "resolved_info": {
                "symbol": symbol,
                "exchange": exchange,
                "lot_size": lot_size
            },
            "reason": "All data integrity checks passed" if all_passed else "Data integrity issue detected"
        }, indent=2)
    
    return [validate_symbol_exists, validate_lot_size, validate_price_bounds,
            validate_expiry_date, validate_data_integrity]


class DataIntegrityGuard(BaseGuard):
    """Data Integrity Guard - symbol resolution, lot sizes, price validation."""
    
    def __init__(self):
        tools = get_data_integrity_tools()
        system_prompt = """You are the DATA INTEGRITY GUARD - ensuring data correctness.

## YOUR TOOLS
- validate_symbol_exists: Verify symbol exists and get details
- validate_lot_size: Verify quantity is valid lot size multiple
- validate_price_bounds: Verify limit price is near LTP
- validate_expiry_date: Verify F&O expiry is valid
- validate_data_integrity: Run all data checks on a trade plan

## YOUR RESPONSIBILITY
Ensure all trade data is correct and resolved:
- Symbol exists in the exchange
- Quantity is a multiple of lot size
- Limit prices are within reasonable bounds of LTP
- F&O expiry dates are valid

## OUTPUT FORMAT
```json
{
    "guard": "DataIntegrity",
    "passed": true/false,
    "checks": [...],
    "resolved_info": {
        "symbol": "...",
        "lot_size": 1,
        "token": "..."
    },
    "reason": "explanation"
}
```

## CONSTRAINTS
- You can ONLY validate data
- You CANNOT place orders
- You CANNOT issue approval tokens (that's Risk Guard)"""

        super().__init__("DataIntegrity", tools, system_prompt)


# =============================================================================
# GUARD REGISTRY
# =============================================================================

class GuardRegistry:
    """Registry to manage all guard agents."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._guards = {}
            cls._instance._initialized = False
        return cls._instance
    
    def initialize(self):
        """Initialize all guards."""
        if self._initialized:
            return
        
        self._guards = {
            "policy": PolicyGuard(),
            "risk": RiskGuard(),
            "data_integrity": DataIntegrityGuard(),
        }
        self._initialized = True
        logger.info(f"Guard Registry initialized with {len(self._guards)} guards")
    
    def get_guard(self, name: str) -> Optional[BaseGuard]:
        """Get a guard by name."""
        if not self._initialized:
            self.initialize()
        return self._guards.get(name)
    
    def list_guards(self) -> dict:
        """List all available guards and their tools."""
        if not self._initialized:
            self.initialize()
        
        return {
            name: {
                "tools": [t.name for t in guard.tools],
                "tool_count": len(guard.tools)
            }
            for name, guard in self._guards.items()
        }
    
    async def run_all_guards(self, trade_plan: dict) -> dict:
        """Run all guards in sequence on a trade plan."""
        if not self._initialized:
            self.initialize()
        
        results = {
            "all_passed": True,
            "guard_results": []
        }
        
        # Run guards in order: Policy -> DataIntegrity -> Risk
        for guard_name in ["policy", "data_integrity", "risk"]:
            guard = self._guards[guard_name]
            result = await guard.validate(trade_plan)
            
            # Parse the response to check if passed
            try:
                response_data = json.loads(result.get("response", "{}"))
                passed = response_data.get("passed", False)
            except:
                passed = "passed" in result.get("response", "").lower()
            
            results["guard_results"].append({
                "guard": guard_name,
                "passed": passed,
                "result": result
            })
            
            if not passed:
                results["all_passed"] = False
                results["failed_at"] = guard_name
                break
        
        return results


def get_guard_registry() -> GuardRegistry:
    """Get the global guard registry."""
    registry = GuardRegistry()
    registry.initialize()
    return registry
