
import json
import logging
import os
import math
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

logger = logging.getLogger("chart_pattern_agent")

# =============================================================================
# INPUT SCHEMA
# =============================================================================

class ChartPatternInput(BaseModel):
    instrument: Dict[str, Any]
    analysis_window: Optional[Dict[str, Any]] = None # primary_candles, secondary_tf
    ohlcv: List[Dict[str, Any]] # ts, open, high, low, close, volume
    volume_profile: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    time_horizon: Optional[str] = "swing"

# =============================================================================
# STATE
# =============================================================================

class ChartPatternAgentState(TypedDict):
    inputs: Dict[str, Any]
    validated: bool
    missing_data: List[str]
    
    # Internal processing
    pivot_points: List[Dict[str, Any]]
    pattern_candidates: List[Dict[str, Any]] # Placeholder for Python-detected candidates
    
    # Output
    output: Dict[str, Any]

# =============================================================================
# NODES
# =============================================================================

def validate_chart_data(state: ChartPatternAgentState) -> Dict[str, Any]:
    """
    Node 1: Validate inputs.
    Require: Sufficient OHLCV data (>50 candles ideally).
    """
    inputs = state.get("inputs", {})
    missing = []
    
    ohlcv = inputs.get("ohlcv", [])
    
    if len(ohlcv) < 50:
        missing.append(f"Insufficient OHLCV data (got {len(ohlcv)}, need 50+)")
        
    # Check structure of first candle
    if ohlcv and not all(k in ohlcv[0] for k in ["open", "high", "low", "close"]):
         missing.append("Invalid OHLC structure")

    if missing:
        return {
            "validated": False,
            "missing_data": missing,
            "output": {
                "primary_pattern": "insufficient_data",
                "data_gaps": missing,
                "recommended_next_data_to_fetch": ["Fetch at least 100 candles of OHLCV."]
            }
        }
        
    return {"validated": True, "missing_data": []}

def pivot_detection(state: ChartPatternAgentState) -> Dict[str, Any]:
    """
    Node 2: Identify Pivot Highs and Lows (ZigZag / Fractal logic).
    Simple Python implementation to aid LLM.
    """
    ohlcv = state["inputs"].get("ohlcv", [])
    pivots = []
    
    # Simple 5-bar fractal detection
    # High: H[i-2] < H[i] > H[i+2] (and surrounding)
    # This is a simplified rolling check
    
    for i in range(2, len(ohlcv) - 2):
        candle = ohlcv[i]
        prev1 = ohlcv[i-1]
        prev2 = ohlcv[i-2]
        next1 = ohlcv[i+1]
        next2 = ohlcv[i+2]
        
        # Pivot High
        if (candle["high"] > prev1["high"] and 
            candle["high"] > prev2["high"] and 
            candle["high"] > next1["high"] and 
            candle["high"] > next2["high"]):
            pivots.append({"type": "high", "price": candle["high"], "ts": candle.get("ts"), "index": i})
            
        # Pivot Low
        if (candle["low"] < prev1["low"] and 
            candle["low"] < prev2["low"] and 
            candle["low"] < next1["low"] and 
            candle["low"] < next2["low"]):
            pivots.append({"type": "low", "price": candle["low"], "ts": candle.get("ts"), "index": i})
            
    return {"pivot_points": pivots}

def pattern_recognition(state: ChartPatternAgentState) -> Dict[str, Any]:
    """
    Node 3: Pattern Candidates (Placeholder).
    Real logic could find triangles by connecting pivot lines.
    For now, we leave the heavy lifting to the LLM "vision" on the pivots/OHLC data.
    """
    return {"pattern_candidates": []} 

def generate_pattern_brief(state: ChartPatternAgentState) -> Dict[str, Any]:
    """
    Node 4: LLM Chart Analyst.
    """
    inputs = state["inputs"]
    pivots = state.get("pivot_points", [])
    ohlcv = inputs.get("ohlcv", [])
    
    # Optimize context validation: only send last 50-100 candles to LLM if list is huge
    # to save tokens, or send condensed summary.
    # For this agent, we'll send the raw OHLCV of the last 50 candles + Pivot list.
    
    recent_ohlcv = ohlcv[-60:] if len(ohlcv) > 60 else ohlcv
    
    llm_context = {
        "instrument": inputs.get("instrument"),
        "recent_ohlcv_sample": recent_ohlcv,
        "detected_pivots": pivots[-10:] if len(pivots) > 10 else pivots, # Recent pivots
        "volume_profile": inputs.get("volume_profile"),
        "context": inputs.get("context")
    }
    
    api_key = os.environ.get("GEMINI_API_KEY")
    from llm_client import run_llm
    
    system_prompt = """You are ChartPatternAgent, a discretionary technical trader.
    Identify classical chart patterns in the OHLC data.
    
    OUTPUT JSON:
    {
        "primary_pattern": "FLAG|HEAD_AND_SHOULDERS|NONE",
        "confidence": 0-100,
        "action": "BUY_BREAKOUT|etc",
        "targets": [100, 105],
        "stop_loss": 95
    }
    """
    
    user_prompt = f"Analyze chart pattern: {json.dumps(llm_context)}"
    
    try:
        response_text = run_llm(system_prompt, user_prompt)
        
        parsed = json.loads(response_text.replace("```json", "").replace("```", "").strip())
        return {"output": parsed}
        
    except Exception as e:
        logger.error(f"Pattern Analysis Failed: {e}")
        return {
            "output": {
                "primary_pattern": "insufficient_data",
                "data_gaps": [f"Analysis Error: {str(e)}"]
            }
        }

# =============================================================================
# GRAPH
# =============================================================================

def build_chart_pattern_graph():
    graph = StateGraph(ChartPatternAgentState)
    
    graph.add_node("validate_chart_data", validate_chart_data)
    graph.add_node("pivot_detection", pivot_detection)
    graph.add_node("pattern_recognition", pattern_recognition)
    graph.add_node("generate_pattern_brief", generate_pattern_brief)
    
    graph.add_edge(START, "validate_chart_data")
    
    def validation_router(state):
        if state.get("validated"):
            return "pivot_detection"
        return END
        
    graph.add_conditional_edges("validate_chart_data", validation_router, {
        "pivot_detection": "pivot_detection",
        END: END
    })
    
    graph.add_edge("pivot_detection", "pattern_recognition")
    graph.add_edge("pattern_recognition", "generate_pattern_brief")
    graph.add_edge("generate_pattern_brief", END)
    
    return graph.compile()

chart_pattern_agent_app = build_chart_pattern_graph()

def chart_pattern_agent_node(state: dict):
    """
    Wrapper for LangGraph integration.
    """
    inputs = state.get("chart_pattern_input", {})
    
    result = chart_pattern_agent_app.invoke({
        "inputs": inputs,
        "validated": False,
        "missing_data": [],
        "pivot_points": [],
        "pattern_candidates": [],
        "output": {}
    })
    
    return {"chart_pattern_output": result["output"]}
