"""
Layer 1: Orchestrator Agent - Intent → Plan → Route

Responsibilities:
- Interpret user intent (info, analysis, trade, deploy)
- Create structured TaskPlans
- Route to specialist workers (no direct execution)
- NO high-risk tools (no order placement, cancellation, deployment)

This layer is the entry point that coordinates the workflow.
"""
import os
import json
import logging
from typing import Annotated, TypedDict, Sequence, Optional, List, Literal
from datetime import datetime
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
from tools.openalgo_options import openalgo_get_option_chain
from tools.openalgo_accounts import openalgo_get_funds, openalgo_get_positions
from tools.openalgo_marketdata import (
    openalgo_get_quotes,
    openalgo_search_symbols,
    openalgo_get_market_depth,
    openalgo_get_history
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBW90vlRHW6uJgWbjzoNOYYKndHPp33ctk")


# =============================================================================
# ORCHESTRATOR STATE
# =============================================================================

class TaskPlan(TypedDict, total=False):
    """Structured plan created by orchestrator."""
    steps: List[dict]  # Ordered steps
    current_step: int
    status: str  # pending, in_progress, completed, failed


class OrchestratorState(TypedDict):
    """Extended state for orchestrator layer."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    intent: Optional[str]  # info, analysis, trade, deploy
    constraints: Optional[dict]  # risk budget, max positions, etc.
    plan: Optional[TaskPlan]
    next_worker: Optional[str]  # Which specialist to route to


# =============================================================================
# SAFE ORCHESTRATOR TOOLS (Read-only, no execution)
# =============================================================================

@tool
def create_task_plan(
    intent: Annotated[str, "User intent: 'info', 'analysis', 'trade', or 'deploy'"],
    steps: Annotated[str, "JSON array of step objects with 'worker' and 'action' fields"]
) -> str:
    """
    Create a structured task plan for execution.
    
    Each step should specify which worker handles it:
    - market_data: For symbol info, history, quotes, search, expiry
    - indicators: For technical indicator calculations
    - options: For options symbol lookup, greeks, strategies
    - accounts: For funds, positions, holdings, margin
    - supervisor: For risk validation and trade approval
    - executor: For actual order placement (requires supervisor approval)
    
    Example steps:
    [
        {"worker": "market_data", "action": "get_symbol_info", "params": {"symbol": "RELIANCE"}},
        {"worker": "indicators", "action": "calculate", "params": {"indicator": "RSI"}},
        {"worker": "supervisor", "action": "validate_trade"},
        {"worker": "executor", "action": "place_order"}
    ]
    """
    try:
        step_list = json.loads(steps)
        plan = {
            "intent": intent,
            "steps": step_list,
            "current_step": 0,
            "status": "pending",
            "created_at": datetime.now().isoformat()
        }
        return json.dumps({"status": "success", "plan": plan}, indent=2)
    except json.JSONDecodeError as e:
        return json.dumps({"status": "error", "error": f"Invalid JSON: {str(e)}"})


@tool
def route_to_worker(
    worker: Annotated[str, "Worker name: market_data, indicators, options, accounts, supervisor, executor"],
    action: Annotated[str, "Action to perform"],
    params: Annotated[str, "JSON parameters for the action"] = "{}"
) -> str:
    """
    Route a task to a specialist worker.
    
    Workers:
    - market_data: Symbol info, history, quotes, depth, search, expiry
    - indicators: Calculate technical indicators on OHLCV data
    - options: Option symbol lookup, greeks calculation, strategies
    - accounts: Check funds, positions, holdings, margin
    - supervisor: Validate trades, check risk limits, approve execution
    - executor: Place/modify/cancel orders (requires supervisor approval)
    
    Returns routing instruction for the workflow.
    """
    valid_workers = ["market_data", "indicators", "options", "accounts", "supervisor", "executor"]
    
    if worker not in valid_workers:
        return json.dumps({"error": f"Invalid worker: {worker}", "valid": valid_workers})
    
    try:
        params_dict = json.loads(params)
    except json.JSONDecodeError:
        params_dict = {}
    
    routing = {
        "worker": worker,
        "action": action,
        "params": params_dict,
        "routed_at": datetime.now().isoformat()
    }
    
    # Log inter-agent communication for supervisor/executor routing
    # Log inter-agent communication for all routing
    from agent_comms import send_agent_message
    send_agent_message(
        from_agent="orchestrator",
        to_agent=worker,
        message_type="request",
        content=f"Routing action: {action}",
        metadata=routing
    )
    
    return json.dumps({"status": "routed", "routing": routing}, indent=2)


@tool
def get_system_status() -> str:
    """
    Get current system status including available workers and their capabilities.
    Safe read-only operation.
    """
    status = {
        "workers": {
            "market_data": {
                "status": "available",
                "capabilities": ["get_quotes", "get_history", "search_symbols", "get_expiry", "get_symbol_info"]
            },
            "indicators": {
                "status": "available", 
                "capabilities": ["calculate_indicator", "list_indicators", "validate_indicator"]
            },
            "options": {
                "status": "available",
                "capabilities": ["option_greeks", "option_symbol", "option_order", "iron_condor"]
            },
            "accounts": {
                "status": "available",
                "capabilities": ["get_funds", "get_positions", "get_holdings", "calculate_margin"]
            },
            "supervisor": {
                "status": "available",
                "capabilities": ["validate_trade", "check_risk", "approve_execution"]
            },
            "executor": {
                "status": "available",
                "capabilities": ["place_order", "modify_order", "cancel_order"]
            }
        },
        "timestamp": datetime.now().isoformat()
    }
    return json.dumps(status, indent=2)


@tool
def list_registered_strategies() -> str:
    """
    List all registered strategies (read-only).
    Safe to call - does not execute anything.
    """
    try:
        from strategy_registry import get_strategy_registry
        registry = get_strategy_registry()
        strategies = registry.list_strategies(active_only=True)
        return json.dumps({
            "count": len(strategies),
            "strategies": [{"name": s["name"], "mode": s["mode"], "has_webhook": bool(s.get("webhook_id"))} 
                          for s in strategies]
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_strategy_config(
    name: Annotated[str, "Strategy name to lookup"]
) -> str:
    """
    Get configuration of a registered strategy (read-only).
    Does not execute any orders.
    """
    try:
        from strategy_registry import get_strategy_registry
        registry = get_strategy_registry()
        strategy = registry.get_strategy(name)
        if strategy:
            return json.dumps(strategy, indent=2, default=str)
        return json.dumps({"error": f"Strategy '{name}' not found"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def start_trading_session(
    market_report: Annotated[str, "Daily market report text with strategy details"]
) -> str:
    """
    Start a new daily trading session by parsing a market report.
    This initializes strategies, risk limits, and session state.
    """
    try:
        from trading_session import get_session_manager
        manager = get_session_manager()
        session = manager.start_session(market_report)
        return json.dumps({
            "status": "started",
            "session_id": session.session_id,
            "bias": session.market_bias.value,
            "active_strategy": session.strategies[session.active_strategy_index].name
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# ORCHESTRATOR AGENT CLASS
# =============================================================================

class OrchestratorAgent:
    """
    Layer 1: Orchestrator - Plans and routes, no broker side effects.
    
    This is the top-level coordinator that:
    - Interprets user intent
    - Creates task plans
    - Routes to specialist workers
    - NEVER directly executes trades
    """
    
    def __init__(self):
        """Initialize the Orchestrator with safe tools only."""
        from rate_limiter import RateLimitedLLM
        
        base_llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            google_api_key=GEMINI_API_KEY,
            temperature=0.3,  # Lower temp for more deterministic planning
        )
        
        # Wrap with rate limiter for throttling and retry
        self.llm = RateLimitedLLM(base_llm, estimated_tokens_per_call=3000)
        
        # SAFE TOOLS ONLY - no execution, no high-risk operations
        self.tools = [
            create_task_plan,
            route_to_worker,
            get_system_status,
            list_registered_strategies,
            get_system_status,
            list_registered_strategies,
            get_strategy_config,
            start_trading_session,
            openalgo_get_option_chain,
            openalgo_get_funds,
            openalgo_get_positions,
            openalgo_get_quotes,
            openalgo_search_symbols,
            openalgo_get_market_depth,
            openalgo_get_history,
        ]
        
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.memory = InMemorySaver()
        self.graph = self._build_graph()
        self.app = self.graph.compile(checkpointer=self.memory)
        
        logger.info(f"Orchestrator Agent initialized with {len(self.tools)} safe tools")
    
    def _build_graph(self) -> StateGraph:
        """Build the orchestrator workflow graph."""
        from langgraph.prebuilt import ToolNode
        
        workflow = StateGraph(OrchestratorState)
        
        # Nodes
        workflow.add_node("orchestrator", self._orchestrate)
        workflow.add_node("tools", ToolNode(self.tools))
        
        # Entry point
        workflow.set_entry_point("orchestrator")
        
        # Conditional routing
        workflow.add_conditional_edges(
            "orchestrator",
            self._should_continue,
            {
                "continue": "tools",
                "route_worker": END,  # Will be picked up by supervisor
                "end": END,
            }
        )
        
        workflow.add_edge("tools", "orchestrator")
        
        return workflow
    
    def _orchestrate(self, state: OrchestratorState) -> dict:
        """Main orchestration logic."""
        import time
        time.sleep(2)  # Rate limiting
        
        messages = list(state["messages"])
        
        system_prompt = """You are the ORCHESTRATOR AGENT (Layer 1) of a 3-layer trading system.

## YOUR ROLE
You PLAN and ROUTE tasks. You do NOT execute trades directly.

## WORKFLOW
1. Analyze user request to determine INTENT:
   - "info": User wants data (quotes, symbol info, history)
   - "analysis": User wants technical analysis or indicators
   - "trade": User wants to place/modify orders
   - "deploy": User wants to deploy or configure a strategy

2. Create a TASK PLAN with ordered steps

3. ROUTE to specialist workers:
   - market_data: For quotes, history, symbol search, expiry dates
   - indicators: For technical indicator calculations
   - options: For options greeks, symbol lookup, strategies
   - accounts: For funds, positions, holdings, margin check
   - supervisor: For trade validation and risk checks
   - executor: For actual order execution (ONLY after supervisor approval)

## CONSTRAINTS
- You CANNOT place orders directly
- You CANNOT cancel orders
- You CANNOT close positions
- You can ONLY plan and route

## EXAMPLES

User: "What's the current price of RELIANCE?"
→ Intent: info
→ Plan: [market_data.get_quotes(RELIANCE)]

User: "Calculate RSI for NIFTY with 14 period"
→ Intent: analysis  
→ Plan: [market_data.get_history(NIFTY) → indicators.calculate(RSI,14)]

User: "Buy 10 shares of INFY"
→ Intent: trade
→ Plan: [accounts.get_funds → supervisor.validate_trade → executor.place_order]

## IMPORTANT
1. **NO RESPONSE BUG:** If you PLAN a "market_data" or "info" step but only "route" it, the user sees NOTHING. You MUST execute the tool directly.
2. **DIRECT EXECUTION:** For `get_quotes`, `search_symbols`, `get_funds`, `option_chain`, etc., DO NOT route. CALL THE TOOL DIRECTLY in the same turn.
3. **ONLY ROUTE** for:
   - Complex strategy creation (Analysis) -> to Supervisor/Analyst
   - Placing/Modifying Orders (Trade) -> to Supervisor (then to Executor)
   - Configuration (Deploy)
   
**Example Correct Behavior:**
User: "Price of INFOSYS?"
CORRECT: Call tool `openalgo_get_quotes(symbol="INFY", exchange="NSE")` directly.
WRONG: `route_to_worker("market_data", ...)` <- THIS CAUSES THE BUG.
"""
        full_messages = [SystemMessage(content=system_prompt)] + messages
        response = self.llm_with_tools.invoke(full_messages)
        
        return {"messages": [response]}
    
    def _should_continue(self, state: OrchestratorState) -> str:
        """Determine next action based on state."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If LLM made tool calls, process them (Execute tool)
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        
        return "end"
    
    async def send_message(self, message: str, thread_id: str = "default") -> dict:
        """Send a message and get orchestrated response."""
        from task_tracker import start_agent_task, update_agent_task, complete_agent_task, fail_agent_task
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Start task tracking
        task_name = message[:50] + "..." if len(message) > 50 else message
        task_id = start_agent_task(
            name=f"Processing: {task_name}",
            steps=["Analyzing request", "Planning workflow", "Executing tools", "Generating response"],
            description=message
        )
        
        input_state = {
            "messages": [HumanMessage(content=message)],
            "intent": None,
            "constraints": None,
            "plan": None,
            "next_worker": None,
        }
        
        tool_calls_made = []
        final_response = ""
        tools_called = 0
        
        try:
            update_agent_task(task_id, "Analyzing request", "running")
            async for event in self.app.astream(input_state, config):
                for node_name, node_output in event.items():
                    if node_name == "tools":
                        for msg in node_output.get("messages", []):
                            if isinstance(msg, ToolMessage):
                                tools_called += 1
                                tool_name = msg.name
                                # Update task with current tool
                                update_agent_task(
                                    task_id, 
                                    f"🔧 Tool #{tools_called}: {tool_name}", 
                                    "running",
                                    result=msg.content[:100] if msg.content else None
                                )
                                tool_calls_made.append({
                                    "tool": msg.name,
                                    "result": msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
                                })
                    elif node_name == "orchestrator":
                        update_agent_task(task_id, "🤖 Orchestrator thinking...", "running")
                        messages = node_output.get("messages", [])
                        if messages:
                            last_msg = messages[-1]
                            if isinstance(last_msg, AIMessage) and last_msg.content:
                                content = last_msg.content
                                if isinstance(content, str):
                                    final_response = content
                                elif isinstance(content, list):
                                    texts = []
                                    for block in content:
                                        if isinstance(block, dict) and block.get("type") == "text":
                                            texts.append(block.get("text", ""))
                                        elif isinstance(block, str):
                                            texts.append(block)
                                    final_response = "\n".join(texts)
            
            update_agent_task(task_id, f"✅ Completed ({tools_called} tools used)", "completed")
            complete_agent_task(task_id, f"Response generated. Tools: {tools_called}")
            
            return {
                "response": final_response,
                "tool_calls": tool_calls_made,
                "layer": "orchestrator"
            }
            
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            import traceback
            traceback.print_exc()
            fail_agent_task(task_id, f"Failed at step: {tool_calls_made[-1]['tool'] if tool_calls_made else 'Analysis'} - {str(e)}")
            return {"response": f"Error: {str(e)}", "tool_calls": [], "layer": "orchestrator"}
    
    def get_history(self, thread_id: str = "default") -> list:
        """Get conversation history."""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state = self.app.get_state(config)
            if state and state.values:
                messages = state.values.get("messages", [])
                history = []
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        history.append({"role": "user", "content": msg.content})
                    elif isinstance(msg, AIMessage) and msg.content:
                        history.append({"role": "ai", "content": msg.content})
                return history
        except Exception as e:
            logger.error(f"Error getting history: {e}")
        return []
