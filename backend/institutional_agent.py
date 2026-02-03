
import json
import logging
import os
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

logger = logging.getLogger("institutional_agent")

# =============================================================================
# INPUT SCHEMA
# =============================================================================

class InstitutionalInput(BaseModel):
    instrument: Dict[str, str]
    as_of: str
    time_horizon: str
    lookback: Dict[str, Any]
    institutional_flows: Dict[str, Any] # fii_net, dii_net, trend
    delivery_data: Dict[str, Any] # daily_delivery, trend
    block_bulk_deals: List[Dict[str, Any]]
    options_flow: Optional[Dict[str, Any]] = None
    price_volume_context: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None

# =============================================================================
# STATE
# =============================================================================

class InstitutionalAgentState(TypedDict):
    inputs: Dict[str, Any]
    validated: bool
    missing_data: List[str]
    
    # Internal processing
    flow_metrics: Dict[str, Any]
    
    # Output
    output: Dict[str, Any]

# =============================================================================
# NODES
# =============================================================================

def validate_inputs(state: InstitutionalAgentState) -> Dict[str, Any]:
    """
    Node 1: Validate inputs.
    Require FII/DII net flows OR delivery data OR block deals.
    """
    inputs = state.get("inputs", {})
    missing = []
    
    if "instrument" not in inputs or "symbol" not in inputs["instrument"]:
        missing.append("instrument.symbol")
        
    flows = inputs.get("institutional_flows", {})
    delivery = inputs.get("delivery_data", {})
    deals = inputs.get("block_bulk_deals", [])
    
    has_flows = len(flows.get("fii_net", [])) > 0 or len(flows.get("dii_net", [])) > 0
    has_delivery = len(delivery.get("daily_delivery", [])) > 0
    has_deals = len(deals) > 0
    
    if not (has_flows or has_delivery or has_deals):
        missing.append("flow_data(fii_dii|delivery|deals)")
        
    if missing:
        return {
            "validated": False,
            "missing_data": missing,
            "output": {
                "smart_money_stance": "insufficient_data",
                "data_gaps": missing,
                "recommended_next_data_to_fetch": ["Please provide FII/DII flows, delivery data, or block deals."]
            }
        }
        
    return {"validated": True, "missing_data": []}

def analyze_flows(state: InstitutionalAgentState) -> Dict[str, Any]:
    """
    Node 2: Preprocess and compute flow metrics.
    - summarizes net flows
    - flags significant block deals
    - detects delivery spikes
    """
    inputs = state["inputs"]
    flows = inputs.get("institutional_flows", {})
    delivery = inputs.get("delivery_data", {})
    deals = inputs.get("block_bulk_deals", [])
    context = inputs.get("price_volume_context", {})
    float_shares = context.get("float_shares", 0)
    
    metrics = {
        "fii_trend": "neutral",
        "dii_trend": "neutral",
        "delivery_spike": False,
        "significant_deals": []
    }
    
    # Simple Python heuristic helpers
    # 1. Net Flow Trend
    fii_net = sum(item.get("net_qty", 0) for item in flows.get("fii_net", []))
    if fii_net > 0: metrics["fii_trend"] = "accumulating"
    elif fii_net < 0: metrics["fii_trend"] = "distributing"
    
    # 2. Delivery Spike
    avg_delivery = delivery.get("delivery_trend", {}).get("avg_5d_pct", 0)
    current_delivery = 0
    daily = delivery.get("daily_delivery", [])
    if daily:
        current_delivery = daily[0].get("delivery_pct", 0)
    
    if current_delivery > (avg_delivery * 1.2) and current_delivery > 0.5:
        metrics["delivery_spike"] = True
        
    # 3. Signals from Deaks
    for deal in deals:
        qty = deal.get("qty", 0)
        pct_float = (qty / float_shares) if float_shares else 0
        if pct_float > 0.005: # > 0.5% of float
            metrics["significant_deals"].append(deal)
            
    return {"flow_metrics": metrics}

def generate_flow_brief(state: InstitutionalAgentState) -> Dict[str, Any]:
    """
    Node 3: LLM Analysis.
    """
    inputs = state["inputs"]
    metrics = state.get("flow_metrics", {})
    
    llm_context = {
        "instrument": inputs.get("instrument"),
        "time_horizon": inputs.get("time_horizon"),
        "timestamp": inputs.get("as_of"),
        "inputs": inputs, # Give full raw data too
        "calculated_metrics": metrics
    }
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"output": {"smart_money_stance": "insufficient_data", "data_gaps": ["API Key Missing"]}}

    from llm_client import run_llm
    
    system_prompt = """You are InstitutionalFlowAgent tracking FII/DII flows to ride smart money.
    Analyze flow data, delivery % and timestamps.
    Detect: Accumulation, Distribution, Capitulation.
    
    OUTPUT JSON:
    {
        "smart_money_stance": "ACCUMULATING|DISTRIBUTING|NEUTRAL",
        "flow_strength": 0-100,
        "analysis": "..."
    }
    """
    
    user_prompt = f"Analyze these flows: {json.dumps(llm_context, default=str)}"

    try:
        response_text = run_llm(system_prompt, user_prompt)
        parsed = json.loads(response_text.replace("```json", "").replace("```", "").strip())
        return {"output": parsed}
    except Exception as e:
        logger.error(f"Flow Analysis Failed: {e}")
        return {
            "output": {
                "smart_money_stance": "insufficient_data",
                "data_gaps": [f"Analysis Error: {str(e)}"]
            }
        }

# =============================================================================
# GRAPH
# =============================================================================

def build_institutional_graph():
    graph = StateGraph(InstitutionalAgentState)
    
    graph.add_node("validate_inputs", validate_inputs)
    graph.add_node("analyze_flows", analyze_flows)
    graph.add_node("generate_flow_brief", generate_flow_brief)
    
    graph.add_edge(START, "validate_inputs")
    
    def validation_router(state):
        if state.get("validated"):
            return "analyze_flows"
        return END
        
    graph.add_conditional_edges("validate_inputs", validation_router, {
        "analyze_flows": "analyze_flows",
        END: END
    })
    
    graph.add_edge("analyze_flows", "generate_flow_brief")
    graph.add_edge("generate_flow_brief", END)
    
    return graph.compile()

institutional_agent_app = build_institutional_graph()

def institutional_agent_node(state: dict):
    """
    Wrapper for LangGraph integration.
    """
    inputs = state.get("institutional_input", {})
    
    result = institutional_agent_app.invoke({
        "inputs": inputs,
        "validated": False,
        "missing_data": [],
        "flow_metrics": {},
        "output": {}
    })
    
    return {"institutional_output": result["output"]}
