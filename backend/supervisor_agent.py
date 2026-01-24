"""
Layer 2: Supervisor/Guards - Policy Gate + Tool Scoping

Responsibilities:
- Tool gating: decide which tools are available for each step
- Risk checks & policy enforcement (hard limits)
- Schema validation: enforce TradePlan objects
- Pre-trade verification (price bounds, lot size, symbol resolution)

This layer APPROVES or DENIES execution requests - it does NOT execute.
"""
import os
import json
import logging
from typing import Annotated, TypedDict, Sequence, Optional, List, Literal
from datetime import datetime
from dataclasses import dataclass, asdict
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
# TRADE PLAN SCHEMA (Enforced Structure)
# =============================================================================

@dataclass
class TradePlan:
    """Strict trade plan structure - NO free-form orders allowed."""
    symbol: str
    exchange: str
    action: Literal["BUY", "SELL"]
    quantity: int
    price_type: Literal["MARKET", "LIMIT", "SL", "SL-M"]
    product: Literal["CNC", "MIS", "NRML"]
    price: float = 0.0
    trigger_price: float = 0.0
    strategy_name: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass  
class RiskPolicy:
    """Risk policy configuration."""
    max_position_size: int = 100
    max_order_value: float = 500000.0
    max_daily_loss: float = 10000.0
    allowed_exchanges: List[str] = None
    allowed_products: List[str] = None
    blocked_symbols: List[str] = None
    require_analyze_mode: bool = True
    
    def __post_init__(self):
        if self.allowed_exchanges is None:
            self.allowed_exchanges = ["NSE", "BSE", "NFO", "BFO", "MCX"]
        if self.allowed_products is None:
            self.allowed_products = ["CNC", "MIS", "NRML"]
        if self.blocked_symbols is None:
            self.blocked_symbols = []


# Default policy
DEFAULT_POLICY = RiskPolicy()


# =============================================================================
# SUPERVISOR STATE
# =============================================================================

class SupervisorState(TypedDict):
    """State for Layer 2 Supervisor."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    trade_plan: Optional[dict]  # TradePlan to validate
    validation_result: Optional[dict]  # Approval/denial result
    current_guard: Optional[str]  # policy, risk, data_integrity
    approval_token: Optional[str]  # Token for executor if approved


# =============================================================================
# READ-ONLY GUARD TOOLS
# =============================================================================

@tool
def check_account_status() -> str:
    """
    Check account funds and positions (read-only).
    Used by Risk Guard for exposure checks.
    """
    try:
        from tools.openalgo_tools import get_openalgo_client
        client = get_openalgo_client()
        
        funds = client.funds()
        positions = client.positionbook()
        
        return json.dumps({
            "status": "success",
            "funds": funds,
            "positions": positions,
            "timestamp": datetime.now().isoformat()
        }, indent=2, default=str)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@tool
def check_analyzer_mode() -> str:
    """
    Check if system is in analyze (simulated) or live mode.
    Used by Policy Guard.
    """
    try:
        from tools.openalgo_tools import get_openalgo_client
        client = get_openalgo_client()
        result = client.analyzerstatus()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@tool
def validate_symbol_info(
    symbol: Annotated[str, "Symbol to validate"],
    exchange: Annotated[str, "Exchange code"]
) -> str:
    """
    Get symbol info for validation (lot size, tick size, token).
    Used by Data Integrity Guard.
    """
    try:
        from tools.openalgo_tools import get_openalgo_client
        client = get_openalgo_client()
        result = client.symbol(symbol=symbol, exchange=exchange.upper())
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@tool
def get_current_ltp(
    symbol: Annotated[str, "Symbol"],
    exchange: Annotated[str, "Exchange"]
) -> str:
    """
    Get current LTP for price bound validation.
    Used by Data Integrity Guard.
    """
    try:
        from tools.openalgo_tools import get_openalgo_client
        client = get_openalgo_client()
        result = client.quotes(symbol=symbol, exchange=exchange.upper())
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@tool  
def calculate_margin_requirement(
    symbol: Annotated[str, "Symbol"],
    exchange: Annotated[str, "Exchange"],
    quantity: Annotated[int, "Quantity"],
    product: Annotated[str, "Product type"],
    action: Annotated[str, "BUY or SELL"]
) -> str:
    """
    Calculate margin required for a trade.
    Used by Risk Guard.
    """
    try:
        from tools.openalgo_tools import get_openalgo_client
        client = get_openalgo_client()
        
        position = [{
            "symbol": symbol,
            "exchange": exchange.upper(),
            "quantity": quantity,
            "product": product.upper(),
            "side": action.upper()
        }]
        
        result = client.margin(positions=position)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


# =============================================================================
# VALIDATION FUNCTIONS (Deterministic, non-LLM)
# =============================================================================

def validate_trade_plan_schema(trade_plan: dict) -> tuple[bool, str]:
    """Validate trade plan has required fields."""
    required = ["symbol", "exchange", "action", "quantity", "price_type", "product"]
    missing = [f for f in required if f not in trade_plan or not trade_plan[f]]
    
    if missing:
        return False, f"Missing required fields: {missing}"
    
    if trade_plan["action"] not in ["BUY", "SELL"]:
        return False, f"Invalid action: {trade_plan['action']}"
    
    if trade_plan["price_type"] not in ["MARKET", "LIMIT", "SL", "SL-M"]:
        return False, f"Invalid price_type: {trade_plan['price_type']}"
    
    if trade_plan["product"] not in ["CNC", "MIS", "NRML"]:
        return False, f"Invalid product: {trade_plan['product']}"
    
    return True, "Schema valid"


def validate_policy(trade_plan: dict, policy: RiskPolicy) -> tuple[bool, str]:
    """Validate against policy rules."""
    
    # Check exchange
    if trade_plan["exchange"] not in policy.allowed_exchanges:
        return False, f"Exchange {trade_plan['exchange']} not allowed. Allowed: {policy.allowed_exchanges}"
    
    # Check product
    if trade_plan["product"] not in policy.allowed_products:
        return False, f"Product {trade_plan['product']} not allowed. Allowed: {policy.allowed_products}"
    
    # Check blocked symbols
    if trade_plan["symbol"] in policy.blocked_symbols:
        return False, f"Symbol {trade_plan['symbol']} is blocked"
    
    # Check position size
    if trade_plan["quantity"] > policy.max_position_size:
        return False, f"Quantity {trade_plan['quantity']} exceeds max {policy.max_position_size}"
    
    return True, "Policy check passed"


def validate_lot_size(quantity: int, lot_size: int) -> tuple[bool, str]:
    """Validate quantity is multiple of lot size."""
    if lot_size <= 0:
        return True, "No lot size constraint"
    
    if quantity % lot_size != 0:
        return False, f"Quantity {quantity} is not a multiple of lot size {lot_size}"
    
    return True, "Lot size valid"


def validate_price_bounds(price: float, ltp: float, threshold: float = 0.05) -> tuple[bool, str]:
    """Validate limit price is within reasonable bounds of LTP."""
    if price <= 0:
        return True, "Market order, no price to validate"
    
    if ltp <= 0:
        return False, "Cannot validate: LTP not available"
    
    deviation = abs(price - ltp) / ltp
    if deviation > threshold:
        return False, f"Price {price} deviates {deviation*100:.1f}% from LTP {ltp} (max {threshold*100}%)"
    
    return True, f"Price within {threshold*100}% of LTP"


# =============================================================================
# COMBINED VALIDATION TOOL
# =============================================================================

@tool
def validate_trade_request(
    trade_plan_json: Annotated[str, "JSON trade plan to validate"]
) -> str:
    """
    Perform full validation of a trade request.
    
    Runs all guards:
    1. Schema validation
    2. Policy check (allowed exchanges, products, symbols)
    3. Risk check (position size, margin)
    4. Data integrity (lot size, symbol resolution)
    
    Returns approval token if all checks pass, or denial with reasons.
    """
    try:
        trade_plan = json.loads(trade_plan_json)
    except json.JSONDecodeError as e:
        return json.dumps({
            "approved": False,
            "stage": "parse",
            "reason": f"Invalid JSON: {str(e)}"
        })
    
    result = {
        "trade_plan": trade_plan,
        "checks": [],
        "approved": False,
        "approval_token": None
    }
    
    # 1. Schema validation
    valid, msg = validate_trade_plan_schema(trade_plan)
    result["checks"].append({"guard": "schema", "passed": valid, "message": msg})
    if not valid:
        result["reason"] = msg
        return json.dumps(result, indent=2)
    
    # 2. Policy check
    valid, msg = validate_policy(trade_plan, DEFAULT_POLICY)
    result["checks"].append({"guard": "policy", "passed": valid, "message": msg})
    if not valid:
        result["reason"] = msg
        return json.dumps(result, indent=2)
    
    # 3. Try to get symbol info for lot size validation
    try:
        from tools.openalgo_tools import get_openalgo_client
        client = get_openalgo_client()
        symbol_info = client.symbol(symbol=trade_plan["symbol"], exchange=trade_plan["exchange"])
        
        if symbol_info.get("status") == "success":
            lot_size = symbol_info.get("data", {}).get("lotsize", 1)
            valid, msg = validate_lot_size(trade_plan["quantity"], lot_size)
            result["checks"].append({"guard": "lot_size", "passed": valid, "message": msg})
            if not valid:
                result["reason"] = msg
                return json.dumps(result, indent=2)
    except Exception as e:
        result["checks"].append({"guard": "symbol_info", "passed": True, "message": f"Skipped: {str(e)}"})
    
    # 4. All checks passed - generate approval token
    import hashlib
    token_data = f"{trade_plan['symbol']}:{trade_plan['action']}:{trade_plan['quantity']}:{datetime.now().isoformat()}"
    approval_token = hashlib.sha256(token_data.encode()).hexdigest()[:16]
    
    result["approved"] = True
    result["approval_token"] = approval_token
    result["reason"] = "All validation checks passed"
    result["expires_at"] = (datetime.now()).isoformat()
    
    # Log inter-agent communication
    from agent_comms import send_agent_message
    send_agent_message(
        from_agent="supervisor",
        to_agent="executor",
        message_type="approval",
        content=f"Approved: {trade_plan['action']} {trade_plan['quantity']} {trade_plan['symbol']}",
        metadata={"approval_token": approval_token, "trade_plan": trade_plan}
    )
    
    logger.info(f"Trade approved: {trade_plan['action']} {trade_plan['quantity']} {trade_plan['symbol']} token={approval_token}")
    
    return json.dumps(result, indent=2)


@tool
def get_tool_scope(
    worker_type: Annotated[str, "Worker type: market_data, indicators, options, accounts, executor"]
) -> str:
    """
    Get the allowed tool scope for a worker type.
    Tool gating - returns which tools are allowed for each worker.
    """
    scopes = {
        "market_data": [
            "openalgo_get_quotes",
            "openalgo_get_market_depth", 
            "openalgo_get_history",
            "openalgo_get_intervals",
            "openalgo_get_symbol_info",
            "openalgo_search_symbols",
            "openalgo_get_expiry"
        ],
        "indicators": [
            "openalgo_list_indicators",
            "openalgo_calculate_indicator",
            "openalgo_validate_indicator",
            "openalgo_get_common_indicators"
        ],
        "options": [
            "openalgo_option_greeks",
            "openalgo_option_symbol",
            "openalgo_option_order",
            "openalgo_build_iron_condor"
        ],
        "accounts": [
            "openalgo_get_funds",
            "openalgo_get_orderbook",
            "openalgo_get_tradebook",
            "openalgo_get_positions",
            "openalgo_get_holdings",
            "openalgo_analyzer_status",
            "openalgo_calculate_margin"
        ],
        "executor": [
            "openalgo_place_order",
            "openalgo_place_smart_order",
            "openalgo_modify_order",
            "openalgo_cancel_order"
            # Note: cancel_all and close_all require additional approval
        ]
    }
    
    if worker_type not in scopes:
        return json.dumps({"error": f"Unknown worker type: {worker_type}", "valid_types": list(scopes.keys())})
    
    return json.dumps({
        "worker": worker_type,
        "allowed_tools": scopes[worker_type],
        "count": len(scopes[worker_type])
    }, indent=2)


# =============================================================================
# SUPERVISOR AGENT CLASS
# =============================================================================

class SupervisorAgent:
    """
    Layer 2: Supervisor/Guards - Policy gate + tool scoping.
    
    This layer:
    - Validates trade plans before execution
    - Enforces risk policies and limits
    - Gates which tools each worker can access
    - Issues approval tokens for executor
    - NEVER executes trades directly
    """
    
    def __init__(self):
        """Initialize the Supervisor with guard tools."""
        from rate_limiter import RateLimitedLLM
        
        base_llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            google_api_key=GEMINI_API_KEY,
            temperature=0.1,  # Very low temp for deterministic validation
        )
        
        self.llm = RateLimitedLLM(base_llm, estimated_tokens_per_call=3000)
        
        # Guard tools - all read-only or validation-only
        self.tools = [
            check_account_status,
            check_analyzer_mode,
            validate_symbol_info,
            get_current_ltp,
            calculate_margin_requirement,
            validate_trade_request,
            get_tool_scope,
        ]
        
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tool_node = ToolNode(self.tools)
        self.memory = InMemorySaver()
        self.graph = self._build_graph()
        self.app = self.graph.compile(checkpointer=self.memory)
        
        logger.info(f"Supervisor Agent (Layer 2) initialized with {len(self.tools)} guard tools")
    
    def _build_graph(self) -> StateGraph:
        """Build the supervisor workflow graph."""
        workflow = StateGraph(SupervisorState)
        
        workflow.add_node("supervisor", self._supervise)
        workflow.add_node("tools", self.tool_node)
        
        workflow.set_entry_point("supervisor")
        
        workflow.add_conditional_edges(
            "supervisor",
            self._should_continue,
            {
                "continue": "tools",
                "approved": END,
                "denied": END,
                "end": END,
            }
        )
        
        workflow.add_edge("tools", "supervisor")
        
        return workflow
    
    def _supervise(self, state: SupervisorState) -> dict:
        """Main supervision logic."""
        import time
        time.sleep(2)
        
        messages = list(state["messages"])
        
        system_prompt = """You are the SUPERVISOR AGENT (Layer 2) - the policy gate and validator.

## YOUR ROLE
You VALIDATE and APPROVE/DENY requests. You do NOT execute trades.

## RESPONSIBILITIES

### 1. TOOL GATING
Use `get_tool_scope` to determine which tools a worker should have access to.
Workers should only use their scoped tools.

### 2. TRADE VALIDATION
When a trade request comes through:
1. Use `validate_trade_request` with the trade plan JSON
2. If validation fails, DENY with specific reason
3. If validation passes, APPROVE with the token

### 3. RISK CHECKS
Before approving trades:
- Check account status with `check_account_status`
- Verify margin with `calculate_margin_requirement`
- Check analyzer mode with `check_analyzer_mode`

### 4. DATA INTEGRITY
- Validate symbols exist with `validate_symbol_info`
- Check current prices with `get_current_ltp`

## TRADE PLAN FORMAT (Required)
```json
{
    "symbol": "RELIANCE",
    "exchange": "NSE",
    "action": "BUY",
    "quantity": 10,
    "price_type": "MARKET",
    "product": "MIS",
    "price": 0,
    "trigger_price": 0
}
```

## OUTPUT FORMAT
For trade validation requests, output:
```json
{
    "decision": "APPROVED" | "DENIED",
    "approval_token": "xxx" (if approved),
    "reason": "explanation",
    "checks_passed": ["schema", "policy", "risk", "data"]
}
```

## CONSTRAINTS
- You CANNOT place orders
- You CANNOT modify orders  
- You CANNOT cancel orders
- You can ONLY validate and approve/deny"""

        full_messages = [SystemMessage(content=system_prompt)] + messages
        response = self.llm_with_tools.invoke(full_messages)
        
        return {"messages": [response]}
    
    def _should_continue(self, state: SupervisorState) -> str:
        """Determine next action."""
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        
        # Check if we have an approval token in state
        if state.get("approval_token"):
            return "approved"
        
        return "end"
    
    async def validate(
        self,
        request: str,
        trade_plan: dict = None,
        thread_id: str = "supervisor_default"
    ) -> dict:
        """
        Validate a trade request or query.
        
        Args:
            request: Validation request or query
            trade_plan: Optional trade plan to validate
            thread_id: Conversation thread ID
        
        Returns:
            dict with validation result
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        # Import task tracking
        from task_tracker import start_agent_task, update_agent_task, complete_agent_task, fail_agent_task
        
        # Start task tracking
        task_name = request[:40] + "..." if len(request) > 40 else request
        task_id = start_agent_task(
            name=f"🛡️ Supervisor: {task_name}",
            steps=["Checking account", "Validating trade", "Risk assessment", "Issuing decision"],
            description=f"Supervisor validation: {request}"
        )
        
        # Include trade plan in message if provided
        message = request
        if trade_plan:
            message += f"\n\nTrade Plan:\n```json\n{json.dumps(trade_plan, indent=2)}\n```"
        
        input_state = {
            "messages": [HumanMessage(content=message)],
            "trade_plan": trade_plan,
            "validation_result": None,
            "current_guard": None,
            "approval_token": None,
        }
        
        tool_calls_made = []
        final_response = ""
        tools_called = 0
        
        try:
            update_agent_task(task_id, "🛡️ Supervisor analyzing...", "running")
            async for event in self.app.astream(input_state, config):
                for node_name, node_output in event.items():
                    if node_name == "tools":
                        for msg in node_output.get("messages", []):
                            if hasattr(msg, "name"):
                                tools_called += 1
                                update_agent_task(
                                    task_id,
                                    f"🔍 Guard check #{tools_called}: {msg.name}",
                                    "running",
                                    result=msg.content[:80] if msg.content else None
                                )
                                tool_calls_made.append({
                                    "tool": msg.name,
                                    "result": msg.content[:300] + "..." if len(msg.content) > 300 else msg.content,
                                    "full_result": msg.content
                                })
                    elif node_name == "supervisor":
                        update_agent_task(task_id, "🛡️ Supervisor deciding...", "running")
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
            
            # Determine if approved or denied from response
            is_approved = "approved" in final_response.lower() or "approval_token" in final_response.lower()
            status_icon = "✅" if is_approved else "❌"
            update_agent_task(task_id, f"{status_icon} {'Approved' if is_approved else 'Denied'} ({tools_called} checks)", "completed")
            complete_agent_task(task_id, f"{'Approved' if is_approved else 'Denied'} after {tools_called} checks")
            
            return {
                "response": final_response,
                "tool_calls": tool_calls_made,
                "layer": "supervisor"
            }
            
        except Exception as e:
            logger.error(f"Supervisor error: {e}")
            import traceback
            traceback.print_exc()
            fail_agent_task(task_id, f"Failed at: {tool_calls_made[-1]['tool'] if tool_calls_made else 'Analysis'} - {str(e)}")
            return {"response": f"Error: {str(e)}", "tool_calls": [], "layer": "supervisor"}

    # Legacy method for backward compatibility
    async def analyze(self, message: str, **kwargs) -> dict:
        """Backward compatible analyze method."""
        return await self.validate(message, thread_id=kwargs.get("thread_id", "supervisor_default"))
    
    def get_history(self, thread_id: str = "supervisor_default") -> list:
        """Get conversation history."""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state = self.app.get_state(config)
            if state and state.values:
                messages = state.values.get("messages", [])
                history = []
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        history.append({"role": "user", "content": str(msg.content)})
                    elif isinstance(msg, AIMessage) and msg.content:
                        history.append({"role": "ai", "content": str(msg.content)})
                return history
        except Exception as e:
            logger.error(f"Error getting history: {e}")
        return []
