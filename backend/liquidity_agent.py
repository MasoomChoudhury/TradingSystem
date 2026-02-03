
import json
import logging
import os
import math
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

logger = logging.getLogger("liquidity_agent")

# =============================================================================
# INPUT SCHEMA
# =============================================================================

class LiquidityInput(BaseModel):
    instrument: Dict[str, Any]
    proposed_trade: Dict[str, Any] # direction, size, time_horizon
    market_snapshot: Dict[str, Any] # ltp, bids, asks, avg_volume
    order_book: Optional[Dict[str, Any]] = None
    short_constraints: Optional[Dict[str, Any]] = None
    margin_requirements: Optional[Dict[str, Any]] = None
    regulatory: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None

# =============================================================================
# STATE
# =============================================================================

class LiquidityAgentState(TypedDict):
    inputs: Dict[str, Any]
    validated: bool
    missing_data: List[str]
    
    # Internal processing
    liquidity_metrics: Dict[str, Any]
    cost_estimates: Dict[str, Any]
    
    # Output
    output: Dict[str, Any]

# =============================================================================
# NODES
# =============================================================================

def validate_inputs(state: LiquidityAgentState) -> Dict[str, Any]:
    """
    Node 1: Validate inputs.
    Require: LTP, proposed size, and some measure of volume (avg_volume or float).
    """
    inputs = state.get("inputs", {})
    missing = []
    
    snapshot = inputs.get("market_snapshot", {})
    trade = inputs.get("proposed_trade", {})
    
    if "ltp" not in snapshot:
        missing.append("market values (ltp)")
    if "size" not in trade:
        missing.append("proposed_trade.size")
    if "avg_volume_20d" not in snapshot and "free_float" not in snapshot:
        missing.append("volume_data (avg_vol or float)")
        
    if missing:
        return {
            "validated": False,
            "missing_data": missing,
            "output": {
                "tradeability_verdict": "insufficient_data",
                "data_gaps": missing,
                "recommended_next_data_to_fetch": ["Please provide LTP, Size, and Volume data."]
            }
        }
        
    return {"validated": True, "missing_data": []}

def liquidity_sizing(state: LiquidityAgentState) -> Dict[str, Any]:
    """
    Node 2: Calculate size vs ADV and impact.
    """
    inputs = state["inputs"]
    snapshot = inputs.get("market_snapshot", {})
    trade = inputs.get("proposed_trade", {})
    
    size = trade.get("size", 0)
    adv = snapshot.get("avg_volume_20d", 1) # Avoid div by zero
    float_shares = snapshot.get("free_float", 0)
    
    metrics = {
        "size_vs_adv_pct": (size / adv) * 100 if adv > 0 else 0,
        "size_vs_float_pct": (size / float_shares) * 100 if float_shares > 0 else 0,
        "is_large_trade": False
    }
    
    if metrics["size_vs_adv_pct"] > 5.0:
        metrics["is_large_trade"] = True
        
    return {"liquidity_metrics": metrics}

def spread_analysis(state: LiquidityAgentState) -> Dict[str, Any]:
    """
    Node 3: Analyze spread costs.
    """
    inputs = state["inputs"]
    snapshot = inputs.get("market_snapshot", {})
    
    bid = snapshot.get("bid", 0)
    ask = snapshot.get("ask", 0)
    ltp = snapshot.get("ltp", 1)
    
    cost_metrics = {
        "spread_pct": 0,
        "half_spread_cost_bps": 0
    }
    
    if bid > 0 and ask > 0:
        spread = ask - bid
        cost_metrics["spread_pct"] = (spread / ltp) * 100
        cost_metrics["half_spread_cost_bps"] = (spread / 2 / ltp) * 10000
        
    return {"cost_estimates": cost_metrics}

def execution_modeling(state: LiquidityAgentState) -> Dict[str, Any]:
    """
    Node 4: Simple slippage model (Square Root Law).
    """
    metrics = state.get("liquidity_metrics", {})
    pct_adv = metrics.get("size_vs_adv_pct", 0) / 100
    
    # Heuristic: Slippage ~ Volatility * sqrt(Size/ADV)
    # Assume generic daily vol of 2% for estimation if not provided
    est_volatility = 0.02 
    estimated_slippage = est_volatility * math.sqrt(pct_adv) if pct_adv > 0 else 0
    
    updated_estimates = state.get("cost_estimates", {})
    updated_estimates["est_slippage_pct"] = estimated_slippage * 100
    
    return {"cost_estimates": updated_estimates}

def generate_verdict(state: LiquidityAgentState) -> Dict[str, Any]:
    """
    Node 5: LLM Gatekeeper.
    """
    inputs = state["inputs"]
    liq_metrics = state.get("liquidity_metrics", {})
    costs = state.get("cost_estimates", {})
    
    from llm_client import run_llm
    
    llm_context = {
        "instrument": inputs.get("instrument"),
        "trade": inputs.get("proposed_trade"),
        "metrics": liq_metrics,
        "costs": costs,
        "constraints": inputs.get("short_constraints"),
        "regulatory": inputs.get("regulatory")
    }
    
    
    system_prompt = """You are LiquidityAgent, a high-frequency execution trader.
    Assess tradeability based on liquidity metrics and costs.
    
    OUTPUT JSON:
    {
        "tradeability_verdict": "GO|NO_GO|CAUTION",
        "estimated_cost_bps": 15,
        "max_size_suggested": 500,
        "reasoning": "..."
    }
    """

    user_prompt = f"""
    Assess the tradeability of this order.
    
    INPUT:
    {json.dumps(llm_context, default=str)}
    """
    
    try:
        response_text = run_llm(system_prompt, user_prompt)
        
        parsed = json.loads(response_text.replace("```json", "").replace("```", "").strip())
        return {"output": parsed}
        
    except Exception as e:
        logger.error(f"Liquidity Check Failed: {e}")
        return {
            "output": {
                "tradeability_verdict": "insufficient_data",
                "data_gaps": [f"Analysis Error: {str(e)}"]
            }
        }

# =============================================================================
# GRAPH
# =============================================================================

def build_liquidity_graph():
    graph = StateGraph(LiquidityAgentState)
    
    graph.add_node("validate_inputs", validate_inputs)
    graph.add_node("liquidity_sizing", liquidity_sizing)
    graph.add_node("spread_analysis", spread_analysis)
    graph.add_node("execution_modeling", execution_modeling)
    graph.add_node("generate_verdict", generate_verdict)
    
    graph.add_edge(START, "validate_inputs")
    
    def validation_router(state):
        if state.get("validated"):
            return "liquidity_sizing"
        return END
        
    graph.add_conditional_edges("validate_inputs", validation_router, {
        "liquidity_sizing": "liquidity_sizing",
        END: END
    })
    
    graph.add_edge("liquidity_sizing", "spread_analysis")
    graph.add_edge("spread_analysis", "execution_modeling")
    graph.add_edge("execution_modeling", "generate_verdict")
    graph.add_edge("generate_verdict", END)
    
    return graph.compile()

liquidity_agent_app = build_liquidity_graph()

def liquidity_agent_node(state: dict):
    """
    Wrapper for LangGraph integration.
    """
    inputs = state.get("liquidity_input", {})
    
    result = liquidity_agent_app.invoke({
        "inputs": inputs,
        "validated": False,
        "missing_data": [],
        "liquidity_metrics": {},
        "cost_estimates": {},
        "output": {}
    })
    
    return {"liquidity_output": result["output"]}
