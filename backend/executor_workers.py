"""
Executor Specialist Workers - Scoped Execution Tools

Each worker has ONLY the execution tools it needs:
- OrderExecutor: place_order, smart_order, basket_order, split_order
- OrderManager: modify_order, cancel_order, order_status
- EmergencyExecutor: cancel_all, close_all, circuit breaker

ALL execution workers require approval tokens from RiskGuard.
"""
import os
import json
import logging
import hashlib
import sqlite3
from datetime import datetime
from typing import Annotated, TypedDict, Sequence, Optional
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
EXECUTION_DB = os.path.join(os.path.dirname(__file__), "execution_log.db")


# =============================================================================
# EXECUTION LOG
# =============================================================================

def get_execution_log():
    """Get the execution log from executor_agent."""
    from executor_agent import get_execution_log
    return get_execution_log()


# =============================================================================
# EXECUTOR STATE
# =============================================================================

class ExecutorWorkerState(TypedDict):
    """State for executor workers."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    approval_token: Optional[str]
    execution_result: Optional[dict]


# =============================================================================
# BASE EXECUTOR WORKER CLASS
# =============================================================================

class BaseExecutorWorker:
    """Base class for all executor workers."""
    
    def __init__(self, name: str, tools: list, system_prompt: str):
        from rate_limiter import RateLimitedLLM
        
        self.name = name
        self.tools = tools
        self.system_prompt = system_prompt
        
        base_llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            google_api_key=GEMINI_API_KEY,
            temperature=0.0,  # Zero temp for deterministic execution
        )
        
        self.llm = RateLimitedLLM(base_llm, estimated_tokens_per_call=2000)
        
        self.llm_with_tools = self.llm.bind_tools(tools)
        self.tool_node = ToolNode(tools)
        self.memory = InMemorySaver()
        self.graph = self._build_graph()
        self.app = self.graph.compile(checkpointer=self.memory)
        
        logger.info(f"{name} Executor Worker initialized with {len(tools)} tools")
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(ExecutorWorkerState)
        
        workflow.add_node("executor_worker", self._execute)
        workflow.add_node("tools", self.tool_node)
        
        workflow.set_entry_point("executor_worker")
        
        workflow.add_conditional_edges(
            "executor_worker",
            self._should_continue,
            {"continue": "tools", "end": END}
        )
        
        workflow.add_edge("tools", "executor_worker")
        
        return workflow
    
    def _execute(self, state: ExecutorWorkerState) -> dict:
        messages = list(state["messages"])
        full_messages = [SystemMessage(content=self.system_prompt)] + messages
        response = self.llm_with_tools.invoke(full_messages)
        return {"messages": [response]}
    
    def _should_continue(self, state: ExecutorWorkerState) -> str:
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        return "end"
    
    async def execute(self, task: str, approval_token: str = None, thread_id: str = None) -> dict:
        """Execute a task."""
        thread_id = thread_id or f"{self.name}_default"
        config = {"configurable": {"thread_id": thread_id}}
        
        input_state = {
            "messages": [HumanMessage(content=task)],
            "approval_token": approval_token,
            "execution_result": None,
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
                    elif node_name == "executor_worker":
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
                "worker": self.name,
                "response": final_response,
                "tool_calls": tool_calls_made,
            }
            
        except Exception as e:
            logger.error(f"{self.name} Executor Worker error: {e}")
            return {"worker": self.name, "response": f"Error: {str(e)}", "tool_calls": []}


# =============================================================================
# ORDER EXECUTOR - Places new orders
# =============================================================================

def get_order_executor_tools():
    """Get tools for order placement."""
    
    @tool
    def place_order(
        symbol: Annotated[str, "Trading symbol"],
        exchange: Annotated[str, "Exchange code"],
        action: Annotated[str, "BUY or SELL"],
        quantity: Annotated[int, "Quantity"],
        price_type: Annotated[str, "MARKET, LIMIT, SL, SL-M"],
        product: Annotated[str, "CNC, MIS, NRML"],
        approval_token: Annotated[str, "Approval token from RiskGuard"],
        price: Annotated[str, "Price for LIMIT orders"] = "0",
        trigger_price: Annotated[str, "Trigger price for SL orders"] = "0"
    ) -> str:
        """
        Place a regular order. REQUIRES approval token.
        """
        exec_log = get_execution_log()
        
        # Check circuit breaker
        is_tripped, reason = exec_log.is_circuit_breaker_tripped()
        if is_tripped:
            return json.dumps({"status": "rejected", "reason": f"Circuit breaker tripped: {reason}"})
        
        if not approval_token:
            return json.dumps({"status": "rejected", "reason": "No approval token - must be approved by RiskGuard"})
        
        # Generate idempotency key
        idem_data = f"{symbol}:{action}:{quantity}:{approval_token}"
        idempotency_key = hashlib.sha256(idem_data.encode()).hexdigest()[:24]
        
        # Check for duplicate
        existing = exec_log.check_idempotency(idempotency_key)
        if existing:
            return json.dumps({
                "status": "duplicate",
                "reason": "Order already executed",
                "original_order_id": existing.get("order_id")
            })
        
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            
            result = client.placeorder(
                symbol=symbol, exchange=exchange.upper(),
                action=action.upper(), quantity=quantity,
                price_type=price_type.upper(), product=product.upper(),
                price=price, trigger_price=trigger_price
            )
            
            order_id = result.get("orderid", result.get("order_id", ""))
            trade = {"symbol": symbol, "exchange": exchange, "action": action, 
                     "quantity": quantity, "price_type": price_type, "product": product, "price": float(price or 0)}
            
            exec_log.log_execution(idempotency_key, approval_token, trade, order_id, 
                                   "executed" if order_id else "failed", result)
            
            logger.info(f"✅ Order placed: {action} {quantity} {symbol} ID={order_id}")
            return json.dumps({"status": "executed", "order_id": order_id, "result": result}, indent=2)
            
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})
    
    @tool
    def place_smart_order(
        symbol: Annotated[str, "Trading symbol"],
        exchange: Annotated[str, "Exchange code"],
        action: Annotated[str, "BUY or SELL"],
        position_size: Annotated[int, "Position size to achieve"],
        approval_token: Annotated[str, "Approval token from RiskGuard"],
        price_type: Annotated[str, "MARKET, LIMIT"] = "MARKET",
        product: Annotated[str, "CNC, MIS, NRML"] = "MIS"
    ) -> str:
        """
        Place a smart order with position sizing logic. REQUIRES approval token.
        """
        exec_log = get_execution_log()
        
        is_tripped, reason = exec_log.is_circuit_breaker_tripped()
        if is_tripped:
            return json.dumps({"status": "rejected", "reason": f"Circuit breaker tripped: {reason}"})
        
        if not approval_token:
            return json.dumps({"status": "rejected", "reason": "No approval token"})
        
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            
            result = client.placesmartorder(
                symbol=symbol, exchange=exchange.upper(),
                action=action.upper(), position_size=position_size,
                price_type=price_type.upper(), product=product.upper()
            )
            
            logger.info(f"✅ Smart order: {action} {position_size} {symbol}")
            return json.dumps({"status": "executed", "result": result}, indent=2)
            
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})
    
    @tool
    def place_basket_order(
        orders_json: Annotated[str, "JSON array of orders"],
        approval_token: Annotated[str, "Approval token from RiskGuard"]
    ) -> str:
        """
        Place multiple orders at once. REQUIRES approval token.
        """
        exec_log = get_execution_log()
        
        is_tripped, reason = exec_log.is_circuit_breaker_tripped()
        if is_tripped:
            return json.dumps({"status": "rejected", "reason": f"Circuit breaker tripped: {reason}"})
        
        if not approval_token:
            return json.dumps({"status": "rejected", "reason": "No approval token"})
        
        try:
            orders = json.loads(orders_json)
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            
            result = client.basketorder(orders=orders)
            
            logger.info(f"✅ Basket order: {len(orders)} orders")
            return json.dumps({"status": "executed", "order_count": len(orders), "result": result}, indent=2)
            
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})
    
    @tool
    def place_split_order(
        symbol: Annotated[str, "Trading symbol"],
        exchange: Annotated[str, "Exchange code"],
        action: Annotated[str, "BUY or SELL"],
        quantity: Annotated[int, "Total quantity"],
        split_quantity: Annotated[int, "Quantity per split"],
        approval_token: Annotated[str, "Approval token from RiskGuard"],
        price_type: Annotated[str, "MARKET, LIMIT"] = "MARKET",
        product: Annotated[str, "CNC, MIS, NRML"] = "MIS"
    ) -> str:
        """
        Split a large order into smaller chunks. REQUIRES approval token.
        """
        exec_log = get_execution_log()
        
        is_tripped, reason = exec_log.is_circuit_breaker_tripped()
        if is_tripped:
            return json.dumps({"status": "rejected", "reason": f"Circuit breaker tripped: {reason}"})
        
        if not approval_token:
            return json.dumps({"status": "rejected", "reason": "No approval token"})
        
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            
            result = client.splitorder(
                symbol=symbol, exchange=exchange.upper(),
                action=action.upper(), quantity=quantity,
                splitby=split_quantity, price_type=price_type.upper(),
                product=product.upper()
            )
            
            num_splits = (quantity + split_quantity - 1) // split_quantity
            logger.info(f"✅ Split order: {quantity} into {num_splits} parts of {split_quantity}")
            return json.dumps({"status": "executed", "splits": num_splits, "result": result}, indent=2)
            
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})
    
    return [place_order, place_smart_order, place_basket_order, place_split_order]


class OrderExecutor(BaseExecutorWorker):
    """Order Executor - places new orders ONLY."""
    
    def __init__(self):
        tools = get_order_executor_tools()
        system_prompt = """You are the ORDER EXECUTOR - specialized in placing new orders.

## YOUR TOOLS
- place_order: Place a regular order
- place_smart_order: Place order with position sizing
- place_basket_order: Place multiple orders at once
- place_split_order: Split large orders into chunks

## REQUIREMENTS
Every order MUST have a valid approval_token from RiskGuard.
Orders without tokens will be REJECTED.

## CONSTRAINTS
- You can ONLY place new orders
- You CANNOT modify existing orders (use OrderManager)
- You CANNOT cancel orders (use OrderManager)
- You CANNOT use emergency functions (use EmergencyExecutor)

## OUTPUT FORMAT
```json
{
    "status": "executed/rejected/error",
    "order_id": "xxx",
    "reason": "explanation"
}
```"""

        super().__init__("OrderExecutor", tools, system_prompt)


# =============================================================================
# ORDER MANAGER - Modifies/cancels existing orders
# =============================================================================

def get_order_manager_tools():
    """Get tools for order management."""
    
    @tool
    def get_order_status(
        order_id: Annotated[str, "Order ID to check"]
    ) -> str:
        """
        Get status of an order. No approval token needed for read-only.
        """
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            result = client.orderstatus(order_id=order_id)
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    @tool
    def modify_order(
        order_id: Annotated[str, "Order ID to modify"],
        symbol: Annotated[str, "Symbol"],
        exchange: Annotated[str, "Exchange"],
        action: Annotated[str, "BUY or SELL"],
        quantity: Annotated[int, "New quantity"],
        price: Annotated[str, "New price"],
        price_type: Annotated[str, "New price type"],
        product: Annotated[str, "Product type"],
        approval_token: Annotated[str, "Approval token from RiskGuard"]
    ) -> str:
        """
        Modify an existing order. REQUIRES approval token.
        """
        exec_log = get_execution_log()
        
        is_tripped, reason = exec_log.is_circuit_breaker_tripped()
        if is_tripped:
            return json.dumps({"status": "rejected", "reason": f"Circuit breaker tripped: {reason}"})
        
        if not approval_token:
            return json.dumps({"status": "rejected", "reason": "No approval token"})
        
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            
            result = client.modifyorder(
                order_id=order_id, symbol=symbol,
                exchange=exchange.upper(), action=action.upper(),
                quantity=quantity, price=price,
                price_type=price_type.upper(), product=product.upper()
            )
            
            logger.info(f"📝 Order modified: {order_id}")
            return json.dumps({"status": "modified", "order_id": order_id, "result": result}, indent=2)
            
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})
    
    @tool
    def cancel_order(
        order_id: Annotated[str, "Order ID to cancel"],
        approval_token: Annotated[str, "Approval token from RiskGuard"]
    ) -> str:
        """
        Cancel a specific order. REQUIRES approval token.
        """
        exec_log = get_execution_log()
        
        is_tripped, reason = exec_log.is_circuit_breaker_tripped()
        if is_tripped:
            return json.dumps({"status": "rejected", "reason": f"Circuit breaker tripped: {reason}"})
        
        if not approval_token:
            return json.dumps({"status": "rejected", "reason": "No approval token"})
        
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            result = client.cancelorder(order_id=order_id)
            
            logger.info(f"❌ Order cancelled: {order_id}")
            return json.dumps({"status": "cancelled", "order_id": order_id, "result": result}, indent=2)
            
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})
    
    @tool
    def get_open_position(
        symbol: Annotated[str, "Symbol"],
        exchange: Annotated[str, "Exchange"],
        product: Annotated[str, "Product type"]
    ) -> str:
        """
        Get current open position for a symbol. No approval needed for read-only.
        """
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            result = client.openposition(
                symbol=symbol, exchange=exchange.upper(),
                product=product.upper()
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    @tool
    def get_execution_history(
        limit: Annotated[int, "Number of recent executions"] = 20
    ) -> str:
        """
        Get recent execution history. No approval needed for read-only.
        """
        exec_log = get_execution_log()
        executions = exec_log.get_recent_executions(limit)
        return json.dumps({"count": len(executions), "executions": executions}, indent=2, default=str)
    
    return [get_order_status, modify_order, cancel_order, get_open_position, get_execution_history]


class OrderManager(BaseExecutorWorker):
    """Order Manager - manages existing orders."""
    
    def __init__(self):
        tools = get_order_manager_tools()
        system_prompt = """You are the ORDER MANAGER - managing existing orders.

## YOUR TOOLS
- get_order_status: Check order status (read-only, no token needed)
- modify_order: Modify pending order (REQUIRES token)
- cancel_order: Cancel specific order (REQUIRES token)
- get_open_position: Check position for symbol (read-only)
- get_execution_history: View recent executions (read-only)

## REQUIREMENTS
Modify and cancel operations REQUIRE approval_token from RiskGuard.
Read-only operations (status, position, history) do NOT need tokens.

## CONSTRAINTS
- You CANNOT place new orders (use OrderExecutor)
- You CANNOT cancel ALL orders (use EmergencyExecutor)
- You CANNOT close ALL positions (use EmergencyExecutor)

## OUTPUT FORMAT
```json
{
    "status": "modified/cancelled/error",
    "order_id": "xxx",
    "reason": "explanation"
}
```"""

        super().__init__("OrderManager", tools, system_prompt)


# =============================================================================
# EMERGENCY EXECUTOR - Emergency operations only
# =============================================================================

def get_emergency_executor_tools():
    """Get emergency execution tools."""
    
    @tool
    def emergency_cancel_all_orders() -> str:
        """
        EMERGENCY: Cancel ALL pending orders.
        Available even when circuit breaker is tripped.
        NO approval token required for emergencies.
        """
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            result = client.cancelallorder()
            
            logger.warning("🚨 EMERGENCY: All orders cancelled!")
            return json.dumps({"status": "success", "action": "cancel_all_orders", "result": result}, indent=2)
            
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})
    
    @tool
    def emergency_close_all_positions() -> str:
        """
        EMERGENCY: Close ALL open positions.
        Available even when circuit breaker is tripped.
        NO approval token required for emergencies.
        """
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            result = client.closeposition()
            
            logger.warning("🚨 EMERGENCY: All positions closed!")
            return json.dumps({"status": "success", "action": "close_all_positions", "result": result}, indent=2)
            
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})
    
    @tool
    def trip_circuit_breaker(
        reason: Annotated[str, "Reason for tripping the circuit breaker"]
    ) -> str:
        """
        Trip the circuit breaker to halt ALL normal trading.
        Only emergency operations will be allowed after this.
        """
        exec_log = get_execution_log()
        exec_log.trip_circuit_breaker(reason)
        
        logger.warning(f"🚨 Circuit breaker TRIPPED: {reason}")
        return json.dumps({
            "status": "circuit_breaker_tripped",
            "reason": reason,
            "allowed_actions": ["emergency_cancel_all_orders", "emergency_close_all_positions", "reset_circuit_breaker"]
        }, indent=2)
    
    @tool
    def reset_circuit_breaker() -> str:
        """
        Reset the circuit breaker to resume normal trading.
        Only do this after the emergency is resolved.
        """
        exec_log = get_execution_log()
        exec_log.reset_circuit_breaker()
        
        logger.info("✅ Circuit breaker RESET - normal trading resumed")
        return json.dumps({"status": "circuit_breaker_reset", "trading_enabled": True}, indent=2)
    
    @tool
    def get_circuit_breaker_status() -> str:
        """
        Check current circuit breaker status.
        """
        exec_log = get_execution_log()
        is_tripped, reason = exec_log.is_circuit_breaker_tripped()
        
        return json.dumps({
            "circuit_breaker_tripped": is_tripped,
            "reason": reason,
            "normal_trading_enabled": not is_tripped
        }, indent=2)
    
    return [emergency_cancel_all_orders, emergency_close_all_positions, 
            trip_circuit_breaker, reset_circuit_breaker, get_circuit_breaker_status]


class EmergencyExecutor(BaseExecutorWorker):
    """Emergency Executor - emergency operations ONLY."""
    
    def __init__(self):
        tools = get_emergency_executor_tools()
        system_prompt = """You are the EMERGENCY EXECUTOR - handling emergency situations.

## YOUR TOOLS
- emergency_cancel_all_orders: Cancel ALL pending orders immediately
- emergency_close_all_positions: Close ALL positions immediately
- trip_circuit_breaker: Halt all normal trading
- reset_circuit_breaker: Resume normal trading
- get_circuit_breaker_status: Check circuit breaker state

## WHEN TO USE
Use these tools ONLY in emergencies:
- Market flash crash
- Risk limits breached
- System malfunction
- Manual intervention required

## NO APPROVAL TOKEN REQUIRED
Emergency operations bypass normal approval flow for speed.

## CONSTRAINTS
- You CANNOT place new orders (use OrderExecutor)
- You CANNOT modify/cancel individual orders (use OrderManager)
- These are ONLY for system-wide emergency actions

## OUTPUT FORMAT
```json
{
    "status": "success/error",
    "action": "what was done",
    "reason": "explanation"
}
```

## ⚠️ WARNING
These actions affect ALL orders and positions!
Use with extreme caution!"""

        super().__init__("EmergencyExecutor", tools, system_prompt)


# =============================================================================
# EXECUTOR WORKER REGISTRY
# =============================================================================

class ExecutorWorkerRegistry:
    """Registry to manage all executor workers."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._workers = {}
            cls._instance._initialized = False
        return cls._instance
    
    def initialize(self):
        """Initialize all executor workers."""
        if self._initialized:
            return
        
        self._workers = {
            "order_executor": OrderExecutor(),
            "order_manager": OrderManager(),
            "emergency_executor": EmergencyExecutor(),
        }
        self._initialized = True
        logger.info(f"Executor Worker Registry initialized with {len(self._workers)} workers")
    
    def get_worker(self, name: str) -> Optional[BaseExecutorWorker]:
        """Get a worker by name."""
        if not self._initialized:
            self.initialize()
        return self._workers.get(name)
    
    def list_workers(self) -> dict:
        """List all available workers and their tools."""
        if not self._initialized:
            self.initialize()
        
        return {
            name: {
                "tools": [t.name for t in worker.tools],
                "tool_count": len(worker.tools)
            }
            for name, worker in self._workers.items()
        }


def get_executor_worker_registry() -> ExecutorWorkerRegistry:
    """Get the global executor worker registry."""
    registry = ExecutorWorkerRegistry()
    registry.initialize()
    return registry
