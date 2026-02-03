
import json
import logging
import os
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

logger = logging.getLogger("options_agent")

# =============================================================================
# INPUT SCHEMA
# =============================================================================

class OptionsInput(BaseModel):
    instrument: Dict[str, Any] # symbol, exchange, spot, lot_size etc
    time_horizon: str
    view: Dict[str, Any] # direction, conviction
    options_chain: Dict[str, List[Dict[str, Any]]] # calls, puts
    market_data: Dict[str, Any] # atm_iv, risk_free_rate etc
    context: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, Any]] = None

# =============================================================================
# STATE
# =============================================================================

class OptionsAgentState(TypedDict):
    inputs: Dict[str, Any]
    validated: bool
    missing_data: List[str]
    
    # Internal processing
    exchange_normalized: Dict[str, Any]
    chain_metrics: Dict[str, Any]
    
    # Output
    output: Dict[str, Any]

# =============================================================================
# NODES
# =============================================================================

def validate_inputs(state: OptionsAgentState) -> Dict[str, Any]:
    """
    Node 1: Validate inputs.
    Require: spot price, at least some options chain data (calls/puts).
    """
    inputs = state.get("inputs", {})
    missing = []
    
    instrument = inputs.get("instrument", {})
    if "spot" not in instrument or not instrument["spot"]:
        missing.append("instrument.spot")
        
    chain = inputs.get("options_chain", {})
    calls = chain.get("calls", [])
    puts = chain.get("puts", [])
    
    if not calls and not puts:
        missing.append("options_chain(calls|puts)")
        
    if missing:
        return {
            "validated": False,
            "missing_data": missing,
            "output": {
                "overall_recommendation": "insufficient_data",
                "data_gaps": missing,
                "recommended_next_data_to_fetch": ["Please provide spot price and options chain data."]
            }
        }
        
    return {"validated": True, "missing_data": []}

def normalize_exchange(state: OptionsAgentState) -> Dict[str, Any]:
    """
    Node 2: Normalize exchange conventions.
    - Sets defaults for expiry format if missing.
    - Ensures lot size is present (default 1 if missing).
    """
    inputs = state["inputs"]
    instrument = inputs.get("instrument", {})
    
    normalized = instrument.copy()
    if "lot_size" not in normalized:
        normalized["lot_size"] = 1 # Default for US equity options often 100, but logic can handle multiplier later
    
    # Simple check for US vs India conventions could go here
    # For now, we pass through as is, assuming upstream standardization or LLM handling
    
    return {"exchange_normalized": normalized}

def chain_processing(state: OptionsAgentState) -> Dict[str, Any]:
    """
    Node 3: Compute chain metrics (IV skew, spread width).
    """
    inputs = state["inputs"]
    chain = inputs.get("options_chain", {})
    calls = chain.get("calls", [])
    puts = chain.get("puts", [])
    
    metrics = {
        "atm_iv": 0,
        "skew": "neutral",
        "avg_spread": 0
    }
    
    # Simple heuristic: Compute average spread % logic
    spreads = []
    for opt in calls + puts:
        bid = opt.get("bid", 0)
        ask = opt.get("ask", 0)
        ltp = opt.get("ltp", 0)
        if ask > 0 and bid > 0:
            spread_pct = (ask - bid) / ltp if ltp > 0 else 0
            spreads.append(spread_pct)
            
    if spreads:
        metrics["avg_spread"] = sum(spreads) / len(spreads)
        
    return {"chain_metrics": metrics}

def generate_options_brief(state: OptionsAgentState) -> Dict[str, Any]:
    """
    Node 4: LLM Analysis (Greeks, Structures, Stress Test).
    """
    inputs = state["inputs"]
    metrics = state.get("chain_metrics", {})
    normalized_instr = state.get("exchange_normalized", {})
    
    llm_context = {
        "instrument": normalized_instr,
        "time_horizon": inputs.get("time_horizon"),
        "view": inputs.get("view"),
        "chain_metrics": metrics,
        "market_data": inputs.get("market_data"),
        "context": inputs.get("context"),
        "constraints": inputs.get("constraints")
    }
    
    from llm_client import run_llm
    
    system_prompt = """You are OptionsAnalyticsAgent, a quantitative derivatives trader.
    Analyze the Greeks and Volatility surface.
    Recommend strategies (Spreads, Iron Condors, etc) based on view.
    
    OUTPUT JSON:
    {
        "overall_recommendation": "BUY_CALL|SELL_PUT|IRON_CONDOR|etc",
        "greeks_summary": "...",
        "strategy_idea": "..."
    }
    """
    
    user_prompt = f"Analyze options chain: {json.dumps(llm_context, default=str)}"

    try:
        response_text = run_llm(system_prompt, user_prompt)
        parsed = json.loads(response_text.replace("```json", "").replace("```", "").strip())
        return {"output": parsed}
    except Exception as e:
        logger.error(f"Options Analysis Failed: {e}")
        return {
            "output": {
                "overall_recommendation": "UNKNOWN",
                "strategy_idea": f"Error: {e}"
            }
        }

# =============================================================================
# GRAPH
# =============================================================================

def build_options_graph():
    graph = StateGraph(OptionsAgentState)
    
    graph.add_node("validate_inputs", validate_inputs)
    graph.add_node("normalize_exchange", normalize_exchange)
    graph.add_node("chain_processing", chain_processing)
    graph.add_node("generate_options_brief", generate_options_brief)
    
    graph.add_edge(START, "validate_inputs")
    
    def validation_router(state):
        if state.get("validated"):
            return "normalize_exchange"
        return END
        
    graph.add_conditional_edges("validate_inputs", validation_router, {
        "normalize_exchange": "normalize_exchange",
        END: END
    })
    
    graph.add_edge("normalize_exchange", "chain_processing")
    graph.add_edge("chain_processing", "generate_options_brief")
    graph.add_edge("generate_options_brief", END)
    
    return graph.compile()

options_agent_app = build_options_graph()

def options_agent_node(state: dict):
    """
    Wrapper for LangGraph integration.
    """
    inputs = state.get("options_input", {})
    
    result = options_agent_app.invoke({
        "inputs": inputs,
        "validated": False,
        "missing_data": [],
        "exchange_normalized": {},
        "chain_metrics": {},
        "output": {}
    })
    
    return {"options_output": result["output"]}
