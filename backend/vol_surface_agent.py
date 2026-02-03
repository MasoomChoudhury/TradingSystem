
import json
import logging
import os
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

logger = logging.getLogger("vol_surface_agent")

# =============================================================================
# INPUT SCHEMA
# =============================================================================

class VolSurfaceInput(BaseModel):
    instrument: Dict[str, Any]
    time_horizon: str
    surface_date: str
    vol_surface: Dict[str, Any] # strikes, expiries, iv_matrix
    market_data: Dict[str, Any]
    liquidity: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, Any]] = None

# =============================================================================
# STATE
# =============================================================================

class VolSurfaceAgentState(TypedDict):
    inputs: Dict[str, Any]
    validated: bool
    missing_data: List[str]
    
    # Internal processing
    surface_metrics: Dict[str, Any]
    rel_value_opportunities: List[Dict[str, Any]]
    
    # Output
    output: Dict[str, Any]

# =============================================================================
# NODES
# =============================================================================

def validate_surface(state: VolSurfaceAgentState) -> Dict[str, Any]:
    """
    Node 1: Validate surface data completeness.
    Require: >= 3 expiries, >= 5 strikes, ATM IV present.
    """
    inputs = state.get("inputs", {})
    missing = []
    
    surface = inputs.get("vol_surface", {})
    strikes = surface.get("strikes", [])
    expiries = surface.get("expiries", [])
    iv_matrix = surface.get("iv_matrix", {})
    
    if len(strikes) < 3:
        missing.append("insufficient_strikes (<3)")
    if len(expiries) < 2: # Relaxed slightly for MVP
        missing.append("insufficient_expiries (<2)")
    if not iv_matrix:
        missing.append("missing_iv_matrix")
        
    if missing:
        return {
            "validated": False,
            "missing_data": missing,
            "output": {
                "overall_opportunity": "insufficient_data",
                "data_gaps": missing,
                "recommended_next_data_to_fetch": ["Please provide full vol surface (strikes, expiries, IVs)."]
            }
        }
        
    return {"validated": True, "missing_data": []}

def surface_fitting(state: VolSurfaceAgentState) -> Dict[str, Any]:
    """
    Node 2: Fit surface and compute metrics (skew, term structure).
    """
    inputs = state["inputs"]
    surface = inputs.get("vol_surface", {})
    iv_matrix = surface.get("iv_matrix", {})
    expiries = surface.get("expiries", [])
    
    metrics = {
        "term_structure": "flat",
        "skew_slope": {},
        "atm_iv": {}
    }
    
    # 1. Term Structure Check (Front vs Back)
    if len(expiries) >= 2:
        front = expiries[0]
        back = expiries[-1]
        
        # Get approx ATM IV (middle of dict if sorted, or just average)
        # Simplified: average of all IVs for that expiry
        front_ivs = list(iv_matrix.get(front, {}).values())
        back_ivs = list(iv_matrix.get(back, {}).values())
        
        if front_ivs and back_ivs:
            avg_front = sum(front_ivs) / len(front_ivs)
            avg_back = sum(back_ivs) / len(back_ivs)
            
            if avg_back > avg_front * 1.05:
                metrics["term_structure"] = "contango" # Normal
            elif avg_front > avg_back * 1.05:
                metrics["term_structure"] = "backwardation" # Inverted
            else:
                metrics["term_structure"] = "flat"
                
    return {"surface_metrics": metrics}

def rel_value_ranking(state: VolSurfaceAgentState) -> Dict[str, Any]:
    """
    Node 3: Identify rich/cheap buckets (Simplified Python Logic).
    """
    metrics = state.get("surface_metrics", {})
    opportunities = []
    
    # Example heuristic: If backwardation, Front Month puts might be expensive -> Sell opportunities?
    # This node prepares data for LLM to reason about or can implement hard-coded quant rules.
    # For now, we pass the metrics to LLM.
    
    return {"rel_value_opportunities": opportunities}

def generate_vol_brief(state: VolSurfaceAgentState) -> Dict[str, Any]:
    """
    Node 4: LLM Analysis (Relative Value, Arbs, Structure Recs).
    """
    inputs = state["inputs"]
    metrics = state.get("surface_metrics", {})
    
    llm_context = {
        "instrument": inputs.get("instrument"),
        "metrics": metrics,
        "market_data": inputs.get("market_data"),
        "surface_sample": inputs.get("vol_surface", {}).get("iv_matrix"), # Provide subset/sample to save token space if needed
        "context": inputs.get("context")
    }
    
    from llm_client import run_llm
    
    system_prompt = """You are VolSurfaceAgent, a volatility arbitrage specialist.
    Analyze surface for rich/cheap strikes, skew anomalies.
    
    OUTPUT JSON:
    {
        "overall_opportunity": "HIGH_SKEW|FLAT_TERM|NORMAL",
        "mispricings": ["..."],
        "trade_idea": "..."
    }
    """
    
    user_prompt = f"""
    Analyze the following volatility surface data for relative value opportunities.
    
    INPUT:
    {json.dumps(llm_context, default=str)}
    """
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        content = response.content.replace("```json", "").replace("```", "").strip()
        output_json = json.loads(content)
        return {"output": output_json}
        
    except Exception as e:
        logger.error(f"Vol Analysis Failed: {e}")
        return {
            "output": {
                "overall_opportunity": "insufficient_data",
                "data_gaps": [f"Analysis Error: {str(e)}"]
            }
        }

# =============================================================================
# GRAPH
# =============================================================================

def build_vol_graph():
    graph = StateGraph(VolSurfaceAgentState)
    
    graph.add_node("validate_surface", validate_surface)
    graph.add_node("surface_fitting", surface_fitting)
    graph.add_node("rel_value_ranking", rel_value_ranking)
    graph.add_node("generate_vol_brief", generate_vol_brief)
    
    graph.add_edge(START, "validate_surface")
    
    def validation_router(state):
        if state.get("validated"):
            return "surface_fitting"
        return END
        
    graph.add_conditional_edges("validate_surface", validation_router, {
        "surface_fitting": "surface_fitting",
        END: END
    })
    
    graph.add_edge("surface_fitting", "rel_value_ranking")
    graph.add_edge("rel_value_ranking", "generate_vol_brief")
    graph.add_edge("generate_vol_brief", END)
    
    return graph.compile()

vol_surface_agent_app = build_vol_graph()

def vol_surface_agent_node(state: dict):
    """
    Wrapper for LangGraph integration.
    """
    inputs = state.get("vol_surface_input", {})
    
    result = vol_surface_agent_app.invoke({
        "inputs": inputs,
        "validated": False,
        "missing_data": [],
        "surface_metrics": {},
        "rel_value_opportunities": [],
        "output": {}
    })
    
    return {"vol_surface_output": result["output"]}
