"""
Trading Graph Architecture - Production Control Flow

Enforces:
1. Supervisor pattern - Orchestrator always regains control
2. Fixed execution path - Strategy → RiskGate → Executor (ONLY path to orders)
3. Emergency-only branch - Separate isolated path for emergencies
4. Context de-cluttering - Minimal handoff messages
5. Large outputs as artifacts - Summaries only in LLM context
"""
import os
import json
import logging
from typing import TypedDict, Annotated, Sequence, Optional, Literal
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# GRAPH NODE TYPES
# =============================================================================

class NodeType(str, Enum):
    ORCHESTRATOR = "orchestrator"
    SUPERVISOR = "supervisor"
    WORKER = "worker"
    GUARD = "guard"
    EXECUTOR = "executor"
    EMERGENCY = "emergency"


class ExecutionPath(str, Enum):
    """Fixed execution paths - no shortcuts allowed."""
    NORMAL = "orchestrator → supervisor → guards → executor"
    EMERGENCY = "orchestrator → supervisor → emergency_executor"
    QUERY = "orchestrator → workers"


# =============================================================================
# ARTIFACTS SYSTEM (Large outputs stay out of LLM context)
# =============================================================================

class ArtifactStore:
    """
    Store for large data artifacts.
    Keeps big JSON out of LLM message context.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._artifacts = {}
            cls._instance._summaries = {}
        return cls._instance
    
    def store(self, key: str, data: dict, summary: str = None) -> str:
        """Store artifact and return summary for LLM context."""
        self._artifacts[key] = {
            "data": data,
            "stored_at": datetime.now().isoformat(),
            "size_bytes": len(json.dumps(data))
        }
        
        # Generate summary if not provided
        if summary is None:
            summary = self._generate_summary(key, data)
        
        self._summaries[key] = summary
        return summary
    
    def _generate_summary(self, key: str, data: dict) -> str:
        """Generate concise summary for LLM context."""
        if isinstance(data, list):
            return f"[Artifact:{key}] List with {len(data)} items"
        elif isinstance(data, dict):
            if "data" in data:
                inner = data["data"]
                if isinstance(inner, list):
                    return f"[Artifact:{key}] {len(inner)} records"
                elif isinstance(inner, dict):
                    return f"[Artifact:{key}] Object with keys: {list(inner.keys())[:5]}"
            return f"[Artifact:{key}] Object with {len(data)} keys"
        return f"[Artifact:{key}] Stored"
    
    def get(self, key: str) -> Optional[dict]:
        """Retrieve full artifact data."""
        artifact = self._artifacts.get(key)
        return artifact["data"] if artifact else None
    
    def get_summary(self, key: str) -> str:
        """Get summary for LLM context."""
        return self._summaries.get(key, f"[Artifact:{key}] Not found")
    
    def list_artifacts(self) -> list:
        """List all stored artifacts."""
        return [
            {"key": k, "size_bytes": v["size_bytes"], "stored_at": v["stored_at"]}
            for k, v in self._artifacts.items()
        ]
    
    def clear(self):
        """Clear all artifacts."""
        self._artifacts.clear()
        self._summaries.clear()


def get_artifact_store() -> ArtifactStore:
    return ArtifactStore()


def store_large_output(key: str, data: dict, max_inline_size: int = 500) -> str:
    """
    Store data as artifact if large, otherwise return inline.
    Returns summary string for LLM context.
    """
    json_str = json.dumps(data)
    
    if len(json_str) <= max_inline_size:
        return json_str  # Small enough to inline
    
    # Store as artifact
    store = get_artifact_store()
    return store.store(key, data)


# =============================================================================
# GRAPH STATE
# =============================================================================

class TradingGraphState(TypedDict):
    """
    Unified state for the trading graph.
    Minimal context to reduce clutter.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Current request
    intent: Optional[str]  # query, analysis, trade, emergency
    request_id: str
    
    # Execution path tracking
    current_node: str
    path_history: list  # Track which nodes have been visited
    
    # Trade plan (if applicable)
    trade_plan: Optional[dict]
    approval_token: Optional[str]
    
    # Artifacts (references only, not full data)
    artifact_keys: list
    
    # Routing control
    next_node: Optional[str]
    return_to: Optional[str]  # Always return control to this node
    
    # Emergency flag
    is_emergency: bool
    
    # Results
    result: Optional[dict]
    error: Optional[str]


# =============================================================================
# PATH ENFORCEMENT
# =============================================================================

ALLOWED_TRANSITIONS = {
    # Orchestrator can go to workers or supervisor
    "orchestrator": ["market_data_worker", "indicators_worker", "options_worker", 
                     "accounts_worker", "fundamentals_worker", "technicals_worker", "news_worker", "sentiment_worker", "institutional_worker", "options_analytics_worker", "vol_surface_worker", "liquidity_worker", "correlation_worker", "chart_pattern_worker", "supervisor", "end"],
    
    # Workers MUST return to orchestrator
    "market_data_worker": ["orchestrator"],
    "indicators_worker": ["orchestrator"],
    "options_worker": ["orchestrator"],
    "options_worker": ["orchestrator"],
    "accounts_worker": ["orchestrator"],
    "fundamentals_worker": ["orchestrator"],
    "technicals_worker": ["orchestrator"],
    "news_worker": ["orchestrator"],
    "sentiment_worker": ["orchestrator"],
    "institutional_worker": ["orchestrator"],
    "options_analytics_worker": ["orchestrator"],
    "vol_surface_worker": ["orchestrator"],
    "liquidity_worker": ["orchestrator"],
    "correlation_worker": ["orchestrator"],
    "chart_pattern_worker": ["orchestrator"],
    
    # Supervisor can go to guards or emergency
    "supervisor": ["policy_guard", "risk_guard", "data_integrity_guard", 
                   "emergency_executor", "orchestrator"],
    
    # Guards MUST return to supervisor
    "policy_guard": ["supervisor"],
    "risk_guard": ["supervisor"],
    "data_integrity_guard": ["supervisor"],
    
    # ONLY supervisor can reach executor (after guards approve)
    # This is the FIXED EXECUTION PATH
    "supervisor_approved": ["order_executor", "order_manager"],
    
    # Executor workers return to supervisor
    "order_executor": ["supervisor"],
    "order_manager": ["supervisor"],
    
    # Emergency is isolated
    "emergency_executor": ["supervisor"],
}


def validate_transition(from_node: str, to_node: str) -> bool:
    """Validate that a transition is allowed."""
    allowed = ALLOWED_TRANSITIONS.get(from_node, [])
    return to_node in allowed


def get_execution_path(intent: str) -> ExecutionPath:
    """Get the required execution path for an intent."""
    if intent == "emergency":
        return ExecutionPath.EMERGENCY
    elif intent in ["trade", "execute", "order"]:
        return ExecutionPath.NORMAL
    else:
        return ExecutionPath.QUERY


# =============================================================================
# FIXED EXECUTION PATH (Strategy → Risk Gate → Executor)
# =============================================================================

@dataclass
class ExecutionCheckpoint:
    """Track progress through the fixed execution path."""
    plan_validated: bool = False
    policy_approved: bool = False
    data_integrity_passed: bool = False
    risk_approved: bool = False
    approval_token: str = ""
    executed: bool = False
    
    def can_proceed_to(self, stage: str) -> tuple[bool, str]:
        """Check if we can proceed to a stage."""
        if stage == "policy_guard":
            if not self.plan_validated:
                return False, "Trade plan not validated"
            return True, ""
        
        elif stage == "data_integrity_guard":
            if not self.policy_approved:
                return False, "Policy check not passed"
            return True, ""
        
        elif stage == "risk_guard":
            if not self.data_integrity_passed:
                return False, "Data integrity check not passed"
            return True, ""
        
        elif stage == "executor":
            if not self.risk_approved:
                return False, "Risk approval not obtained"
            if not self.approval_token:
                return False, "No approval token"
            return True, ""
        
        return True, ""
    
    def to_dict(self) -> dict:
        return {
            "plan_validated": self.plan_validated,
            "policy_approved": self.policy_approved,
            "data_integrity_passed": self.data_integrity_passed,
            "risk_approved": self.risk_approved,
            "has_approval_token": bool(self.approval_token),
            "executed": self.executed
        }


# =============================================================================
# CONTEXT DE-CLUTTERING
# =============================================================================

def create_minimal_handoff(
    from_node: str,
    to_node: str,
    data: dict = None,
    artifact_key: str = None
) -> dict:
    """
    Create minimal handoff message.
    Large data stored as artifact, only reference passed.
    """
    handoff = {
        "from": from_node,
        "to": to_node,
        "timestamp": datetime.now().isoformat()
    }
    
    if artifact_key:
        handoff["artifact"] = artifact_key
        handoff["summary"] = get_artifact_store().get_summary(artifact_key)
    elif data:
        # Check if data should be an artifact
        json_size = len(json.dumps(data))
        if json_size > 500:
            key = f"{from_node}_{to_node}_{datetime.now().strftime('%H%M%S')}"
            store_large_output(key, data)
            handoff["artifact"] = key
            handoff["summary"] = get_artifact_store().get_summary(key)
        else:
            handoff["data"] = data
    
    return handoff


def clean_context_messages(messages: list, max_messages: int = 10) -> list:
    """
    Clean message context to reduce clutter.
    Keep only essential messages.
    """
    if len(messages) <= max_messages:
        return messages
    
    # Keep system message (if any) + last N messages
    cleaned = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            cleaned.append(msg)
            break
    
    # Add recent messages
    cleaned.extend(messages[-max_messages:])
    
    return cleaned


# =============================================================================
# SUPERVISOR CONTROL PATTERN
# =============================================================================

class SupervisorControl:
    """
    Ensures supervisor always regains control.
    Workers cannot bypass supervisor.
    """
    
    def __init__(self):
        self.pending_returns = {}  # Track who needs to return
    
    def delegate(self, to_worker: str, request_id: str) -> dict:
        """Delegate to worker, expect return."""
        self.pending_returns[request_id] = {
            "worker": to_worker,
            "delegated_at": datetime.now().isoformat(),
            "returned": False
        }
        return {"delegated_to": to_worker, "request_id": request_id}
    
    def worker_return(self, request_id: str, result: dict) -> dict:
        """Worker returns control to supervisor."""
        if request_id in self.pending_returns:
            self.pending_returns[request_id]["returned"] = True
            self.pending_returns[request_id]["result"] = result
        return {"control_returned": True, "request_id": request_id}
    
    def get_pending(self) -> list:
        """Get workers that haven't returned."""
        return [
            k for k, v in self.pending_returns.items() 
            if not v.get("returned", False)
        ]


# =============================================================================
# EMERGENCY ISOLATION
# =============================================================================

class EmergencyBranch:
    """
    Isolated emergency execution path.
    CANNOT be reached from normal execution flow.
    """
    
    def __init__(self):
        self.is_active = False
        self.triggered_at = None
        self.reason = None
    
    def activate(self, reason: str) -> dict:
        """Activate emergency mode."""
        self.is_active = True
        self.triggered_at = datetime.now().isoformat()
        self.reason = reason
        logger.warning(f"🚨 EMERGENCY BRANCH ACTIVATED: {reason}")
        return {
            "emergency_active": True,
            "reason": reason,
            "allowed_actions": ["cancel_all", "close_all", "circuit_breaker"]
        }
    
    def deactivate(self) -> dict:
        """Deactivate emergency mode."""
        self.is_active = False
        logger.info("Emergency branch deactivated")
        return {"emergency_active": False}
    
    def check_access(self, from_node: str) -> tuple[bool, str]:
        """Check if node can access emergency functions."""
        # Only supervisor can access emergency
        if from_node not in ["supervisor", "orchestrator"]:
            return False, f"Node '{from_node}' cannot access emergency functions"
        return True, ""


# =============================================================================
# GRAPH BUILDER
# =============================================================================

def build_trading_graph() -> StateGraph:
    """
    Build the production trading graph with all safety constraints.
    """
    graph = StateGraph(TradingGraphState)
    
    # Define nodes
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("supervisor", supervisor_node)
    
    # Workers (query path)
    graph.add_node("market_data_worker", worker_node)
    graph.add_node("indicators_worker", worker_node)
    graph.add_node("options_worker", worker_node)
    graph.add_node("accounts_worker", worker_node)
    
    # Fundamentals Agent
    from fundamentals_agent import fundamentals_agent_node
    graph.add_node("fundamentals_worker", fundamentals_agent_node)
    
    # Technicals Agent
    from technicals_agent import technicals_agent_node
    graph.add_node("technicals_worker", technicals_agent_node)

    # News & Event Risk Agent
    from news_agent import news_agent_node
    graph.add_node("news_worker", news_agent_node)
    
    # Sentiment Agent
    from sentiment_agent import sentiment_agent_node
    graph.add_node("sentiment_worker", sentiment_agent_node)
    
    # Institutional Flow Agent
    from institutional_agent import institutional_agent_node
    graph.add_node("institutional_worker", institutional_agent_node)
    
    # Options Analytics Agent
    from options_agent import options_agent_node
    graph.add_node("options_analytics_worker", options_agent_node)
    
    # Vol Surface Agent
    from vol_surface_agent import vol_surface_agent_node
    graph.add_node("vol_surface_worker", vol_surface_agent_node)
    
    # Liquidity Constraints Agent
    from liquidity_agent import liquidity_agent_node
    graph.add_node("liquidity_worker", liquidity_agent_node)
    
    # Correlation Exposure Agent
    from correlation_agent import correlation_agent_node
    graph.add_node("correlation_worker", correlation_agent_node)
    
    # Chart Pattern Agent
    from chart_pattern_agent import chart_pattern_agent_node
    graph.add_node("chart_pattern_worker", chart_pattern_agent_node)
    
    # Guards (validation path)
    graph.add_node("policy_guard", guard_node)
    graph.add_node("data_integrity_guard", guard_node)
    graph.add_node("risk_guard", guard_node)
    
    # Executors (execution path - ONLY reachable after guards)
    graph.add_node("order_executor", executor_node)
    graph.add_node("order_manager", executor_node)
    
    # Emergency (isolated branch)
    graph.add_node("emergency_executor", emergency_node)
    
    # Set entry point
    graph.set_entry_point("orchestrator")
    
    # Routing
    graph.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "market_data_worker": "market_data_worker",
            "indicators_worker": "indicators_worker",
            "options_worker": "options_worker",
            "accounts_worker": "accounts_worker",
            "fundamentals_worker": "fundamentals_worker",
            "technicals_worker": "technicals_worker",
            "news_worker": "news_worker",
            "sentiment_worker": "sentiment_worker",
            "institutional_worker": "institutional_worker",
            "options_analytics_worker": "options_analytics_worker",
            "vol_surface_worker": "vol_surface_worker",
            "liquidity_worker": "liquidity_worker",
            "correlation_worker": "correlation_worker",
            "chart_pattern_worker": "chart_pattern_worker",
            "supervisor": "supervisor",
            "end": END
        }
    )
    
    # Workers return to orchestrator
    # Workers return to orchestrator
    for worker in ["market_data_worker", "indicators_worker", "options_worker", "accounts_worker", "fundamentals_worker", "technicals_worker", "news_worker", "sentiment_worker", "institutional_worker", "options_analytics_worker", "vol_surface_worker", "liquidity_worker", "correlation_worker", "chart_pattern_worker"]:
        graph.add_edge(worker, "orchestrator")
    
    # Supervisor routing
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "policy_guard": "policy_guard",
            "data_integrity_guard": "data_integrity_guard",
            "risk_guard": "risk_guard",
            "order_executor": "order_executor",
            "order_manager": "order_manager",
            "emergency_executor": "emergency_executor",
            "orchestrator": "orchestrator",
            "end": END
        }
    )
    
    # Guards return to supervisor
    graph.add_edge("policy_guard", "supervisor")
    graph.add_edge("data_integrity_guard", "supervisor")
    graph.add_edge("risk_guard", "supervisor")
    
    # Executors return to supervisor
    graph.add_edge("order_executor", "supervisor")
    graph.add_edge("order_manager", "supervisor")
    
    # Emergency returns to supervisor
    graph.add_edge("emergency_executor", "supervisor")
    
    return graph


# =============================================================================
# NODE IMPLEMENTATIONS (Stubs - integrate with actual agents)
# =============================================================================

def orchestrator_node(state: TradingGraphState) -> dict:
    """Orchestrator - routes and manages overall flow."""
    state["current_node"] = "orchestrator"
    state["path_history"].append("orchestrator")
    return state


def supervisor_node(state: TradingGraphState) -> dict:
    """Supervisor - manages validation and execution approval."""
    state["current_node"] = "supervisor"
    state["path_history"].append("supervisor")
    return state


def worker_node(state: TradingGraphState) -> dict:
    """Generic worker - handles queries, returns to orchestrator."""
    state["path_history"].append(state["current_node"])
    state["return_to"] = "orchestrator"  # Always return control
    return state


def guard_node(state: TradingGraphState) -> dict:
    """Guard - validates, returns to supervisor."""
    state["path_history"].append(state["current_node"])
    state["return_to"] = "supervisor"  # Always return control
    return state


def executor_node(state: TradingGraphState) -> dict:
    """Executor - executes orders, returns to supervisor."""
    state["path_history"].append(state["current_node"])
    state["return_to"] = "supervisor"
    return state


def emergency_node(state: TradingGraphState) -> dict:
    """Emergency - handles emergency actions, returns to supervisor."""
    state["path_history"].append("emergency_executor")
    state["is_emergency"] = True
    state["return_to"] = "supervisor"
    return state


def route_from_orchestrator(state: TradingGraphState) -> str:
    """Route from orchestrator based on intent."""
    intent = state.get("intent", "query")
    next_node = state.get("next_node")
    
    if next_node and validate_transition("orchestrator", next_node):
        return next_node
    
    if intent in ["trade", "execute", "order", "emergency"]:
        return "supervisor"
    elif "price" in str(state.get("messages", [])).lower():
        return "market_data_worker"
    elif "indicator" in str(state.get("messages", [])).lower():
        return "indicators_worker"
    elif any(k in str(state.get("messages", [])).lower() for k in ["fundamental", "analysis", "thesis", "research"]):
        return "fundamentals_worker"
    elif any(k in str(state.get("messages", [])).lower() for k in ["technical", "chart", "trend", "levels", "support", "resistance"]):
        return "technicals_worker"
    elif any(k in str(state.get("messages", [])).lower() for k in ["news", "event", "risk", "rumor", "headline"]):
        return "news_worker"
    elif any(k in str(state.get("messages", [])).lower() for k in ["sentiment", "crowd", "mood", "narrative", "social"]):
        return "sentiment_worker"
    elif any(k in str(state.get("messages", [])).lower() for k in ["institutional", "flow", "fii", "dii", "smart money", "delivery", "block deal"]):
        return "institutional_worker"
    elif any(k in str(state.get("messages", [])).lower() for k in ["options", "greeks", "volatility", "straddle", "strangle", "spread", "delta", "gamma", "theta"]):
        # Note: 'options_worker' is the data fetcher (chains). 'options_analytics_worker' is the quantitative analyst.
        # For now, if keywords imply analysis ("greeks", "volatility", "strategy"), route to analyst.
        # If keywords imply raw data ("chain", "price"), orchestrator might route to options_worker.
        # We can bias towards analysis here if ambiguous.
        if "data" not in str(state.get("messages", [])).lower() and "chain" not in str(state.get("messages", [])).lower():
            return "options_analytics_worker"
        return "options_worker" # Fallback to data fetcher if unsure
    elif any(k in str(state.get("messages", [])).lower() for k in ["surface", "skew", "term structure", "relative value", "arb", "mispricing", "volatility surface"]):
        return "vol_surface_worker"
    elif any(k in str(state.get("messages", [])).lower() for k in ["liquidity", "slippage", "execution", "tradeable", "market depth", "bid ask", "spread cost"]):
        return "liquidity_worker"
    elif any(k in str(state.get("messages", [])).lower() for k in ["risk", "correlation", "exposure", "portfolio impact", "concentration", "beta", "hedging"]):
        return "correlation_worker"
    elif any(k in str(state.get("messages", [])).lower() for k in ["pattern", "flag", "pennant", "head and shoulders", "triangle", "wedge", "harmonic", "candlestick", "reversal"]):
        # "trend" is handled by technicals, but specific patterns go here
        return "chart_pattern_worker"

    
    return "end"


def route_from_supervisor(state: TradingGraphState) -> str:
    """Route from supervisor based on approval status."""
    if state.get("is_emergency"):
        return "emergency_executor"
    
    next_node = state.get("next_node")
    if next_node and validate_transition("supervisor", next_node):
        return next_node
    
    # Check if we have approval to execute
    if state.get("approval_token"):
        return "order_executor"
    
    return "orchestrator"


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "NodeType",
    "ExecutionPath",
    "ArtifactStore",
    "get_artifact_store",
    "store_large_output",
    "TradingGraphState",
    "ALLOWED_TRANSITIONS",
    "validate_transition",
    "get_execution_path",
    "ExecutionCheckpoint",
    "create_minimal_handoff",
    "clean_context_messages",
    "SupervisorControl",
    "EmergencyBranch",
    "build_trading_graph",
]
