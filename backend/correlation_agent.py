
import json
import logging
import os
import math
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

logger = logging.getLogger("correlation_agent")

# =============================================================================
# INPUT SCHEMA
# =============================================================================

class CorrelationInput(BaseModel):
    as_of: str
    portfolio: Dict[str, Any] # cash, total_aum, holdings
    proposed_trade: Dict[str, Any] # symbol, size, expected_weight_pct
    correlation_matrix: Dict[str, Any] # symbols, corr_30d, corr_252d
    market_regime: Optional[Dict[str, Any]] = None
    concentration_limits: Optional[Dict[str, Any]] = None
    stress_scenarios: Optional[Dict[str, Any]] = None

# =============================================================================
# STATE
# =============================================================================

class CorrelationAgentState(TypedDict):
    inputs: Dict[str, Any]
    validated: bool
    missing_data: List[str]
    
    # Internal processing
    post_trade_weights: Dict[str, float]
    correlation_metrics: Dict[str, Any]
    concentration_scores: Dict[str, Any]
    stress_results: Dict[str, Any]
    
    # Output
    output: Dict[str, Any]

# =============================================================================
# NODES
# =============================================================================

def validate_portfolio(state: CorrelationAgentState) -> Dict[str, Any]:
    """
    Node 1: Validate inputs.
    Require: Portfolio holdings, proposed trade, correlation matrix.
    """
    inputs = state.get("inputs", {})
    missing = []
    
    portfolio = inputs.get("portfolio", {})
    trade = inputs.get("proposed_trade", {})
    corr_matrix = inputs.get("correlation_matrix", {})
    
    if not portfolio.get("holdings"):
        missing.append("portfolio.holdings")
    if not trade.get("symbol"):
        missing.append("proposed_trade.symbol")
    if not corr_matrix.get("corr_30d"):
        # We can survive without matrix if we just do concentration checks, 
        # but for specific correlation agent, it's key.
        # We'll flag it but might allow partial checks.
        missing.append("correlation_matrix")
        
    if missing:
        return {
            "validated": False,
            "missing_data": missing,
            "output": {
                "portfolio_impact_verdict": "insufficient_data",
                "data_gaps": missing,
                "recommended_next_data_to_fetch": ["Please provide FULL portfolio and correlation matrix."]
            }
        }
        
    return {"validated": True, "missing_data": []}

def project_post_trade(state: CorrelationAgentState) -> Dict[str, Any]:
    """
    Node 2: Calculate new portfolio weights.
    """
    inputs = state["inputs"]
    portfolio = inputs.get("portfolio", {})
    trade = inputs.get("proposed_trade", {})
    
    current_aum = portfolio.get("total_aum", 1)
    # Usually size is value? If size is qty, we need price.
    # Input schema says 'proposed_size' and 'price' or 'expected_weight_pct'
    
    # We will prioritize expected_weight_pct if given
    trade_weight = trade.get("expected_weight_pct", 0) / 100.0
    
    # Simple re-weighting: Assume trade adds leverage or reallocates cash.
    # For simplicity, we just add this new position to the list of weights
    # In a real system, we'd adjust cash.
    
    holdings = portfolio.get("holdings", [])
    new_weights = {}
    
    for h in holdings:
        sym = h.get("symbol")
        w = h.get("weight_pct", 0) / 100.0
        new_weights[sym] = w
        
    # Add proposed
    sym_proposed = trade.get("symbol")
    if sym_proposed in new_weights:
        new_weights[sym_proposed] += trade_weight
    else:
        new_weights[sym_proposed] = trade_weight
        
    return {"post_trade_weights": new_weights}

def correlation_computation(state: CorrelationAgentState) -> Dict[str, Any]:
    """
    Node 3: Compute portfolio correlation metrics.
    """
    inputs = state["inputs"]
    corr_data = inputs.get("correlation_matrix", {})
    trade_sym = inputs.get("proposed_trade", {}).get("symbol")
    
    corr_matrix = corr_data.get("corr_30d", [])
    symbols = corr_data.get("symbols", [])
    
    metrics = {
        "avg_corr_to_portfolio": 0,
        "highest_corr_pair": {"symbol": "None", "corr": 0}
    }
    
    if trade_sym in symbols and corr_matrix:
        idx = symbols.index(trade_sym)
        # Calculate average correlation of this symbol to others
        row = corr_matrix[idx]
        if len(row) == len(symbols):
            # Exclude self (1.0)
            other_corrs = [c for i, c in enumerate(row) if i != idx]
            if other_corrs:
                metrics["avg_corr_to_portfolio"] = sum(other_corrs) / len(other_corrs)
                max_corr = max(other_corrs)
                max_idx = row.index(max_corr)
                metrics["highest_corr_pair"] = {"symbol": symbols[max_idx], "corr": max_corr}
                
    return {"correlation_metrics": metrics}

def concentration_monitoring(state: CorrelationAgentState) -> Dict[str, Any]:
    """
    Node 4: Check limits.
    """
    inputs = state["inputs"]
    limits = inputs.get("concentration_limits", {})
    weights = state.get("post_trade_weights", {})
    
    max_single = limits.get("max_single_name_pct", 10) / 100.0
    
    breaches = []
    
    for sym, w in weights.items():
        if w > max_single:
            breaches.append(f"{sym} weight {w*100:.1f}% > limit {max_single*100}%")
            
    return {"concentration_scores": {"breaches": breaches}}

def stress_testing(state: CorrelationAgentState) -> Dict[str, Any]:
    """
    Node 5: Simple stress test simulations.
    """
    # Logic placeholder: In real system, re-price portfolio under shocks.
    # Here we mock it or use simple beta assumption if available.
    return {"stress_results": {"status": "simulated_ok"}}

def generate_risk_verdict(state: CorrelationAgentState) -> Dict[str, Any]:
    """
    Node 6: LLM Risk Gatekeeper.
    """
    inputs = state["inputs"]
    weights = state.get("post_trade_weights", {})
    corr_metrics = state.get("correlation_metrics", {})
    concentration = state.get("concentration_scores", {})
    
    from llm_client import run_llm
    
    llm_context = {
        "proposed_trade": inputs.get("proposed_trade"),
        "post_trade_weights_top_10": dict(list(weights.items())[:10]),
        "analysis": {
            "correlation": corr_metrics,
            "concentration_breaches": concentration.get("breaches", []),
            "market_regime": inputs.get("market_regime")
        }
    }
    
    system_prompt = """You are CorrelationAgent, a portfolio risk manager.
    Assess marginal risk contribution of the proposed trade.
    
    OUTPUT JSON:
    {
        "portfolio_impact_verdict": "SAFE|WARNING|REJECT",
        "correlation_risk": "LOW|MED|HIGH",
        "concentration_risk": "LOW|MED|HIGH",
        "reasoning": "..."
    }
    """

    user_prompt = f"""
    Assess the portfolio impact of this trade.
    
    INPUT:
    {json.dumps(llm_context, default=str)}
    """
    
    try:
        response_text = run_llm(system_prompt, user_prompt)
        
        parsed = json.loads(response_text.replace("```json", "").replace("```", "").strip())
        return {"output": parsed}
        
    except Exception as e:
        logger.error(f"Risk Check Failed: {e}")
        return {
            "output": {
                "portfolio_impact_verdict": "insufficient_data",
                "data_gaps": [f"Analysis Error: {str(e)}"]
            }
        }

# =============================================================================
# GRAPH
# =============================================================================

def build_correlation_graph():
    graph = StateGraph(CorrelationAgentState)
    
    graph.add_node("validate_portfolio", validate_portfolio)
    graph.add_node("project_post_trade", project_post_trade)
    graph.add_node("correlation_computation", correlation_computation)
    graph.add_node("concentration_monitoring", concentration_monitoring)
    graph.add_node("stress_testing", stress_testing)
    graph.add_node("generate_risk_verdict", generate_risk_verdict)
    
    graph.add_edge(START, "validate_portfolio")
    
    def validation_router(state):
        if state.get("validated"):
            return "project_post_trade"
        return END
        
    graph.add_conditional_edges("validate_portfolio", validation_router, {
        "project_post_trade": "project_post_trade",
        END: END
    })
    
    graph.add_edge("project_post_trade", "correlation_computation")
    graph.add_edge("correlation_computation", "concentration_monitoring")
    graph.add_edge("concentration_monitoring", "stress_testing")
    graph.add_edge("stress_testing", "generate_risk_verdict")
    graph.add_edge("generate_risk_verdict", END)
    
    return graph.compile()

correlation_agent_app = build_correlation_graph()

def correlation_agent_node(state: dict):
    """
    Wrapper for LangGraph integration.
    """
    inputs = state.get("correlation_input", {})
    
    result = correlation_agent_app.invoke({
        "inputs": inputs,
        "validated": False,
        "missing_data": [],
        "post_trade_weights": {},
        "correlation_metrics": {},
        "concentration_scores": {},
        "stress_results": {},
        "output": {}
    })
    
    return {"correlation_output": result["output"]}
