"""
Layer 3: Executor Agent - Broker Side Effects ONLY

This is the ONLY layer that can:
- Place orders
- Modify orders
- Cancel orders
- Close positions

Requirements:
- Must receive ApprovedTradePlan with valid approval token from Supervisor
- Idempotency keys prevent duplicate orders
- Emergency circuit breaker for risk breaches
- Immutable execution logs
"""
import os
import json
import logging
import hashlib
import sqlite3
from datetime import datetime
from typing import Annotated, TypedDict, Sequence, Optional, Literal
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
EXECUTION_DB = os.path.join(os.path.dirname(__file__), "execution_log.db")


# =============================================================================
# APPROVED TRADE PLAN (Required from Supervisor)
# =============================================================================

@dataclass
class ApprovedTradePlan:
    """Trade plan that has been approved by Supervisor."""
    symbol: str
    exchange: str
    action: Literal["BUY", "SELL"]
    quantity: int
    price_type: Literal["MARKET", "LIMIT", "SL", "SL-M"]
    product: Literal["CNC", "MIS", "NRML"]
    price: float
    trigger_price: float
    approval_token: str
    lot_size: int = 1
    symbol_token: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# EXECUTION LOG (Immutable)
# =============================================================================

class ExecutionLog:
    """Immutable execution log storage."""
    
    def __init__(self, db_path: str = EXECUTION_DB):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                idempotency_key TEXT UNIQUE,
                approval_token TEXT NOT NULL,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL,
                action TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price_type TEXT NOT NULL,
                product TEXT NOT NULL,
                price REAL,
                order_id TEXT,
                status TEXT NOT NULL,
                response TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS circuit_breaker (
                id INTEGER PRIMARY KEY,
                is_tripped BOOLEAN DEFAULT 0,
                reason TEXT,
                tripped_at TIMESTAMP,
                reset_at TIMESTAMP
            )
        """)
        # Initialize circuit breaker row
        conn.execute("INSERT OR IGNORE INTO circuit_breaker (id, is_tripped) VALUES (1, 0)")
        conn.commit()
        conn.close()
    
    def check_idempotency(self, key: str) -> Optional[dict]:
        """Check if this trade was already executed."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM executions WHERE idempotency_key = ?", (key,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None
    
    def log_execution(self, idempotency_key: str, approval_token: str, 
                      trade: dict, order_id: str, status: str, response: dict) -> None:
        """Log an execution (immutable insert)."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO executions 
                (idempotency_key, approval_token, symbol, exchange, action, quantity, 
                 price_type, product, price, order_id, status, response)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                idempotency_key, approval_token, trade["symbol"], trade["exchange"],
                trade["action"], trade["quantity"], trade["price_type"], trade["product"],
                trade.get("price", 0), order_id, status, json.dumps(response)
            ))
            conn.commit()
        except sqlite3.IntegrityError:
            logger.warning(f"Duplicate execution attempt: {idempotency_key}")
        finally:
            conn.close()
    
    def is_circuit_breaker_tripped(self) -> tuple[bool, str]:
        """Check if circuit breaker is tripped."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT is_tripped, reason FROM circuit_breaker WHERE id = 1").fetchone()
        conn.close()
        return (bool(row[0]), row[1]) if row else (False, None)
    
    def trip_circuit_breaker(self, reason: str) -> None:
        """Trip the circuit breaker."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE circuit_breaker SET is_tripped = 1, reason = ?, tripped_at = ? WHERE id = 1",
            (reason, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
        logger.warning(f"🚨 CIRCUIT BREAKER TRIPPED: {reason}")
    
    def reset_circuit_breaker(self) -> None:
        """Reset the circuit breaker."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE circuit_breaker SET is_tripped = 0, reason = NULL, reset_at = ? WHERE id = 1",
            (datetime.now().isoformat(),)
        )
        conn.commit()
        conn.close()
        logger.info("Circuit breaker reset")
    
    def get_recent_executions(self, limit: int = 20) -> list:
        """Get recent executions."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM executions ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]


# Global execution log
_execution_log = None

def get_execution_log() -> ExecutionLog:
    global _execution_log
    if _execution_log is None:
        _execution_log = ExecutionLog()
    return _execution_log


# =============================================================================
# EXECUTOR STATE
# =============================================================================

class ExecutorState(TypedDict):
    """State for Layer 3 Executor."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    approved_plan: Optional[dict]  # ApprovedTradePlan
    execution_result: Optional[dict]
    is_emergency: bool


# =============================================================================
# EXECUTION TOOLS (HIGH RISK - THE ONLY PLACE TRADES HAPPEN)
# =============================================================================

@tool
def execute_approved_order(
    trade_plan_json: Annotated[str, "JSON ApprovedTradePlan with approval_token"]
) -> str:
    """
    Execute an order that has been approved by Supervisor.
    
    REQUIRES:
    - Valid approval_token from Supervisor
    - Complete trade plan with symbol, exchange, action, quantity, etc.
    
    Will REJECT if:
    - No approval token
    - Circuit breaker is tripped
    - Duplicate execution (idempotency check)
    """
    exec_log = get_execution_log()
    
    # Check circuit breaker first
    is_tripped, reason = exec_log.is_circuit_breaker_tripped()
    if is_tripped:
        return json.dumps({
            "status": "rejected",
            "reason": f"Circuit breaker is tripped: {reason}",
            "allowed_actions": ["emergency_cancel_all", "emergency_close_all", "reset_circuit_breaker"]
        })
    
    try:
        trade = json.loads(trade_plan_json)
    except json.JSONDecodeError as e:
        return json.dumps({"status": "error", "reason": f"Invalid JSON: {str(e)}"})
    
    # Validate approval token
    approval_token = trade.get("approval_token")
    if not approval_token:
        return json.dumps({"status": "rejected", "reason": "No approval token - order must be approved by Supervisor first"})
    
    # Generate idempotency key
    idem_data = f"{trade['symbol']}:{trade['action']}:{trade['quantity']}:{approval_token}"
    idempotency_key = hashlib.sha256(idem_data.encode()).hexdigest()[:24]
    
    # Check for duplicate
    existing = exec_log.check_idempotency(idempotency_key)
    if existing:
        return json.dumps({
            "status": "duplicate",
            "reason": "This order was already executed",
            "original_order_id": existing.get("order_id"),
            "original_status": existing.get("status")
        })
    
    # Execute the order
    try:
        from tools.openalgo_tools import get_openalgo_client
        client = get_openalgo_client()
        
        result = client.placeorder(
            symbol=trade["symbol"],
            exchange=trade["exchange"],
            action=trade["action"],
            quantity=trade["quantity"],
            price_type=trade["price_type"],
            product=trade["product"],
            price=str(trade.get("price", 0)),
            trigger_price=str(trade.get("trigger_price", 0))
        )
        
        order_id = result.get("orderid", result.get("order_id", ""))
        status = "executed" if order_id else "failed"
        
        # Log execution
        exec_log.log_execution(idempotency_key, approval_token, trade, order_id, status, result)
        
        logger.info(f"✅ Order executed: {trade['action']} {trade['quantity']} {trade['symbol']} ID={order_id}")
        
        # Log inter-agent communication
        from agent_comms import send_agent_message
        send_agent_message(
            from_agent="executor",
            to_agent="supervisor",
            message_type="response",
            content=f"Executed: {trade['action']} {trade['quantity']} {trade['symbol']} ID={order_id}",
            metadata={"order_id": order_id, "status": status, "trade": trade}
        )
        
        return json.dumps({
            "status": status,
            "order_id": order_id,
            "symbol": trade["symbol"],
            "action": trade["action"],
            "quantity": trade["quantity"],
            "idempotency_key": idempotency_key,
            "response": result
        }, indent=2)
        
    except Exception as e:
        exec_log.log_execution(idempotency_key, approval_token, trade, "", "error", {"error": str(e)})
        return json.dumps({"status": "error", "reason": str(e)})


@tool
def check_order_status(
    order_id: Annotated[str, "Order ID to check"]
) -> str:
    """
    Check status of an executed order.
    """
    try:
        from tools.openalgo_tools import get_openalgo_client
        client = get_openalgo_client()
        result = client.orderstatus(order_id=order_id)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


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
    approval_token: Annotated[str, "Approval token from Supervisor"]
) -> str:
    """
    Modify an existing order. Requires approval token.
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
            order_id=order_id,
            symbol=symbol,
            action=action.upper(),
            exchange=exchange.upper(),
            quantity=quantity,
            price=price,
            product=product.upper(),
            price_type=price_type.upper()
        )
        
        logger.info(f"Order modified: {order_id}")
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@tool
def cancel_order(
    order_id: Annotated[str, "Order ID to cancel"],
    approval_token: Annotated[str, "Approval token from Supervisor"]
) -> str:
    """
    Cancel a specific order. Requires approval token.
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
        
        logger.info(f"Order cancelled: {order_id}")
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


# =============================================================================
# EMERGENCY TOOLS (Available even when circuit breaker is tripped)
# =============================================================================

@tool
def emergency_cancel_all() -> str:
    """
    EMERGENCY: Cancel all pending orders.
    Available even when circuit breaker is tripped.
    """
    exec_log = get_execution_log()
    
    try:
        from tools.openalgo_tools import get_openalgo_client
        client = get_openalgo_client()
        result = client.cancelallorder()
        
        logger.warning("🚨 EMERGENCY: All orders cancelled!")
        return json.dumps({"status": "success", "action": "cancel_all", "result": result}, indent=2)
        
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@tool
def emergency_close_all() -> str:
    """
    EMERGENCY: Close all positions.
    Available even when circuit breaker is tripped.
    """
    exec_log = get_execution_log()
    
    try:
        from tools.openalgo_tools import get_openalgo_client
        client = get_openalgo_client()
        result = client.closeposition()
        
        logger.warning("🚨 EMERGENCY: All positions closed!")
        return json.dumps({"status": "success", "action": "close_all", "result": result}, indent=2)
        
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@tool
def trip_circuit_breaker(
    reason: Annotated[str, "Reason for tripping the circuit breaker"]
) -> str:
    """
    Trip the circuit breaker to halt all trading.
    Only emergency actions will be allowed after this.
    """
    exec_log = get_execution_log()
    exec_log.trip_circuit_breaker(reason)
    
    return json.dumps({
        "status": "circuit_breaker_tripped",
        "reason": reason,
        "allowed_actions": ["emergency_cancel_all", "emergency_close_all", "reset_circuit_breaker"]
    })


@tool
def reset_circuit_breaker() -> str:
    """
    Reset the circuit breaker to resume normal trading.
    Should only be done after the risk issue is resolved.
    """
    exec_log = get_execution_log()
    exec_log.reset_circuit_breaker()
    
    return json.dumps({"status": "circuit_breaker_reset", "trading_enabled": True})


@tool
def get_execution_history(
    limit: Annotated[int, "Number of recent executions to retrieve"] = 20
) -> str:
    """
    Get recent execution history.
    """
    exec_log = get_execution_log()
    executions = exec_log.get_recent_executions(limit)
    return json.dumps({"count": len(executions), "executions": executions}, indent=2, default=str)


# =============================================================================
# EXECUTOR AGENT CLASS
# =============================================================================

class ExecutorAgent:
    """
    Layer 3: Executor - The ONLY layer that can execute broker operations.
    
    Requirements:
    - ApprovedTradePlan with valid approval token
    - Idempotency keys to prevent duplicates
    - Circuit breaker for emergencies
    """
    
    def __init__(self):
        """Initialize the Executor with execution tools."""
        from rate_limiter import RateLimitedLLM
        
        base_llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            google_api_key=GEMINI_API_KEY,
            temperature=0.0,  # Zero temp for deterministic execution
        )
        
        self.llm = RateLimitedLLM(base_llm, estimated_tokens_per_call=2000)
        
        # Execution tools - HIGH RISK
        self.tools = [
            execute_approved_order,
            check_order_status,
            modify_order,
            cancel_order,
            emergency_cancel_all,
            emergency_close_all,
            trip_circuit_breaker,
            reset_circuit_breaker,
            get_execution_history,
        ]
        
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tool_node = ToolNode(self.tools)
        self.memory = InMemorySaver()
        self.graph = self._build_graph()
        self.app = self.graph.compile(checkpointer=self.memory)
        
        logger.info(f"Executor Agent (Layer 3) initialized with {len(self.tools)} execution tools")
    
    def _build_graph(self) -> StateGraph:
        """Build the executor workflow graph."""
        workflow = StateGraph(ExecutorState)
        
        workflow.add_node("executor", self._execute)
        workflow.add_node("tools", self.tool_node)
        
        workflow.set_entry_point("executor")
        
        workflow.add_conditional_edges(
            "executor",
            self._should_continue,
            {
                "continue": "tools",
                "end": END,
            }
        )
        
        workflow.add_edge("tools", "executor")
        
        return workflow
    
    def _execute(self, state: ExecutorState) -> dict:
        """Main execution logic."""
        import time
        time.sleep(1)  # Minimal delay for execution
        
        messages = list(state["messages"])
        
        system_prompt = """You are the EXECUTOR AGENT (Layer 3) - the ONLY layer that can execute trades.

## YOUR ROLE
You EXECUTE orders that have been approved by Supervisor. You are the gateway to the broker.

## REQUIREMENTS FOR ORDER EXECUTION
Every order MUST have:
1. Valid `approval_token` from Supervisor
2. Complete trade details (symbol, exchange, action, quantity, price_type, product)
3. Resolved instrument info (lot_size, symbol_token) from earlier steps

## TOOLS AVAILABLE

### Normal Execution (requires approval token)
- `execute_approved_order`: Execute a Supervisor-approved order
- `check_order_status`: Check status of an order
- `modify_order`: Modify pending order (requires approval)
- `cancel_order`: Cancel specific order (requires approval)

### Emergency Actions (always available)
- `emergency_cancel_all`: Cancel ALL pending orders
- `emergency_close_all`: Close ALL positions
- `trip_circuit_breaker`: Halt all trading
- `reset_circuit_breaker`: Resume trading

### Monitoring
- `get_execution_history`: View recent executions

## CIRCUIT BREAKER
When tripped:
- Normal execution is BLOCKED
- Only emergency actions are allowed
- Must be reset to resume trading

## IDEMPOTENCY
Each order generates an idempotency key. Duplicate orders are automatically rejected.

## OUTPUT FORMAT
For execution results:
```json
{
    "executed": true/false,
    "order_id": "xxx",
    "status": "executed/rejected/duplicate/error",
    "reason": "explanation"
}
```

## CONSTRAINTS
- NEVER execute without approval token
- NEVER skip idempotency check
- ALWAYS respect circuit breaker state"""

        full_messages = [SystemMessage(content=system_prompt)] + messages
        response = self.llm_with_tools.invoke(full_messages)
        
        return {"messages": [response]}
    
    def _should_continue(self, state: ExecutorState) -> str:
        """Determine next action."""
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        
        return "end"
    
    async def chat(self, message: str, thread_id: str = "executor_default") -> dict:
        """Chat wrapper for execute method (for API compatibility)."""
        return await self.execute(request=message, thread_id=thread_id)
    
    async def execute(
        self,
        request: str,
        approved_plan: dict = None,
        is_emergency: bool = False,
        thread_id: str = "executor_default"
    ) -> dict:
        """
        Execute a trade request.
        
        Args:
            request: Execution request
            approved_plan: ApprovedTradePlan from Supervisor
            is_emergency: Whether this is an emergency action
            thread_id: Thread ID
        
        Returns:
            dict with execution result
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        # Import task tracking
        from task_tracker import start_agent_task, update_agent_task, complete_agent_task, fail_agent_task
        
        # Start task tracking
        task_name = request[:40] + "..." if len(request) > 40 else request
        task_type = "🚨 EMERGENCY" if is_emergency else "⚡ Execute"
        task_id = start_agent_task(
            name=f"{task_type}: {task_name}",
            steps=["Validating approval", "Executing order", "Confirming result"],
            description=f"Executor action: {request}"
        )
        
        message = request
        if approved_plan:
            message += f"\n\nApproved Plan:\n```json\n{json.dumps(approved_plan, indent=2)}\n```"
        
        input_state = {
            "messages": [HumanMessage(content=message)],
            "approved_plan": approved_plan,
            "execution_result": None,
            "is_emergency": is_emergency,
        }
        
        tool_calls_made = []
        final_response = ""
        tools_called = 0
        
        try:
            update_agent_task(task_id, f"⚡ Executor processing...", "running")
            async for event in self.app.astream(input_state, config):
                for node_name, node_output in event.items():
                    if node_name == "tools":
                        for msg in node_output.get("messages", []):
                            if hasattr(msg, "name"):
                                tools_called += 1
                                tool_name = msg.name
                                icon = "🚨" if "emergency" in tool_name.lower() else "⚡"
                                update_agent_task(
                                    task_id,
                                    f"{icon} Action #{tools_called}: {tool_name}",
                                    "running",
                                    result=msg.content[:80] if msg.content else None
                                )
                                tool_calls_made.append({
                                    "tool": msg.name,
                                    "result": msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
                                })
                    elif node_name == "executor":
                        update_agent_task(task_id, "⚡ Executor confirming...", "running")
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
            
            # Determine result from response
            is_success = "executed" in final_response.lower() or "success" in final_response.lower()
            status_icon = "✅" if is_success else "⚠️"
            update_agent_task(task_id, f"{status_icon} {'Executed' if is_success else 'Completed'} ({tools_called} actions)", "completed")
            complete_agent_task(task_id, f"{'Executed' if is_success else 'Completed'} with {tools_called} actions")
            
            return {
                "response": final_response,
                "tool_calls": tool_calls_made,
                "layer": "executor"
            }
            
        except Exception as e:
            logger.error(f"Executor error: {e}")
            import traceback
            traceback.print_exc()
            fail_agent_task(task_id, f"Failed at: {tool_calls_made[-1]['tool'] if tool_calls_made else 'Initialization'} - {str(e)}")
            return {"response": f"Error: {str(e)}", "tool_calls": [], "layer": "executor"}
    
    def get_status(self) -> dict:
        """Get executor status including circuit breaker state."""
        exec_log = get_execution_log()
        is_tripped, reason = exec_log.is_circuit_breaker_tripped()
        
        return {
            "circuit_breaker_tripped": is_tripped,
            "circuit_breaker_reason": reason,
            "recent_executions": len(exec_log.get_recent_executions(10))
        }
    
    def get_history(self, thread_id: str = "executor_default") -> list:
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
