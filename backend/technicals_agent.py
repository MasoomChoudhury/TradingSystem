
import json
import logging
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
import os

logger = logging.getLogger("technicals_agent")

# =============================================================================
# SCHEMAS (Pydantic for strict typing)
# =============================================================================

class Instrument(BaseModel):
    symbol: str
    exchange: str = "NSE"
    type: Literal["equity", "index", "future", "option"] = "equity"
    underlying_symbol: Optional[str] = None

class MarketData(BaseModel):
    ohlcv: Dict[str, List[Dict[str, Any]]] # e.g. {"5m": [...], "1D": [...]}
    session_info: Dict[str, Any]

class TechnicalsInput(BaseModel):
    instrument: Instrument
    time_horizon: Literal["intraday", "swing", "3-6m", "6-12m"]
    timeframes: List[str]
    market_data: MarketData
    derived_features: Optional[Dict[str, Any]] = None
    options_context: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, Any]] = None

# =============================================================================
# STATE
# =============================================================================

class TechnicalsAgentState(TypedDict):
    inputs: Dict[str, Any]
    validated: bool
    missing_data: List[str]
    # Analysis context
    primary_timeframe: str
    secondary_timeframes: List[str]
    # Output storage
    output: Dict[str, Any]
    retries: int

# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are TechnicalsAgent, a senior technical analyst.
You must:
- Be regime-aware (trend vs range vs transition).
- Anchor every claim to observable evidence from OHLCV/derived indicators provided.
- Provide levels and invalidations that are precise and testable.
- Propose only setups that fit the regime and liquidity conditions.
- If data is missing, output stance=insufficient_data and ask for exact missing candles/timeframes.
Return ONLY valid JSON matching the output schema. No extra text."""

# =============================================================================
# NODES
# =============================================================================

def validate_inputs(state: TechnicalsAgentState) -> Dict[str, Any]:
    """
    Node 1: Validate inputs.
    Ensure minimal fields: instrument.symbol, time_horizon, at least one OHLCV.
    """
    inputs = state.get("inputs", {})
    missing = []
    
    # 1. Check top-level fields
    if "instrument" not in inputs:
        missing.append("instrument")
    if "time_horizon" not in inputs:
        missing.append("time_horizon")
    
    # 2. Check Market Data
    market_data = inputs.get("market_data", {})
    ohlcv_data = market_data.get("ohlcv", {})
    
    total_candles = sum(len(candles) for candles in ohlcv_data.values())
    if total_candles == 0:
        missing.append("market_data.ohlcv (empty)")
        
    if missing:
        logger.warning(f"Validation failed. Missing: {missing}")
        return {
            "validated": False, 
            "missing_data": missing,
            "output": {
                "stance": "insufficient_data",
                "data_gaps": missing,
                "recommended_next_data_to_fetch": ["Please provide instrument and OHLCV data."]
            }
        }
        
    return {"validated": True, "missing_data": []}

def choose_timeframes(state: TechnicalsAgentState) -> Dict[str, Any]:
    """
    Node 3: Select primary/secondary timeframes based on horizon.
    """
    horizon = state["inputs"].get("time_horizon", "swing")
    inputs = state["inputs"]
    available_tfs = list(inputs.get("market_data", {}).get("ohlcv", {}).keys())
    
    primary = ""
    secondaries = []
    
    if horizon == "intraday":
        # Prefer 5m or 15m
        if "5m" in available_tfs: primary = "5m"
        elif "15m" in available_tfs: primary = "15m"
        elif available_tfs: primary = available_tfs[0]
        
        if "1h" in available_tfs: secondaries.append("1h")
        if "1D" in available_tfs: secondaries.append("1D")
        
    elif horizon == "swing":
        # Prefer 1D
        if "1D" in available_tfs: primary = "1D"
        elif available_tfs: primary = available_tfs[0]
        
        if "1h" in available_tfs: secondaries.append("1h")
        if "1W" in available_tfs: secondaries.append("1W")
        
    else: # Long term
        if "1W" in available_tfs: primary = "1W"
        elif "1D" in available_tfs: primary = "1D"
        else: primary = str(available_tfs[0]) if available_tfs else ""
        
        if "1D" in available_tfs and primary != "1D": secondaries.append("1D")
        
    return {"primary_timeframe": primary, "secondary_timeframes": secondaries}

def generate_analysis(state: TechnicalsAgentState) -> Dict[str, Any]:
    """
    Node 4: LLM Analysis.
    """
    inputs = state["inputs"]
    
    # Initialize LLM
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"output": {"error": "API Key missing"}}
        
    system_prompt = """You are a Technical Analysis Expert.
    Analyze the provided market data and indicators.
    Produce a JSON output with trend analysis, key levels, and actionable signals.
    """
    
    brief_content = f"""
    Analyze the following market data and produce a structured technical brief.
    
    INPUT:
    {json.dumps(inputs, default=str)}
    
    FOCUS:
    Horizon: {inputs.get('time_horizon')}
    Primary Timeframe: {state.get('primary_timeframe')}
    """
    
    try:
        response_text = run_llm(system_prompt, brief_content)
        
        # Clean markdown
        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean_text)
        return {"output": parsed}
        
    except Exception as e:
        logger.error(f"LLM Analysis Failed: {str(e)}")
        return {
            "output": {
                "stance": "insufficient_data", 
                "data_gaps": ["LLM processing error"],
                "error": str(e)
            }
        }

# =============================================================================
# GRAPH
# =============================================================================

def build_technicals_graph():
    workflow = StateGraph(TechnicalsAgentState)
    
    # Add Nodes
    workflow.add_node("validate_inputs", validate_inputs)
    workflow.add_node("choose_timeframes", choose_timeframes)
    workflow.add_node("generate_analysis", generate_analysis)
    
    # Edges
    workflow.add_edge(START, "validate_inputs")
    
    def validation_router(state):
        if state.get("validated"):
            return "choose_timeframes"
        return END
        
    workflow.add_conditional_edges(
        "validate_inputs",
        validation_router,
        {
            "choose_timeframes": "choose_timeframes",
            END: END
        }
    )
    
    workflow.add_edge("choose_timeframes", "generate_analysis")
    workflow.add_edge("generate_analysis", END)
    
    return workflow.compile()

# Entry point for the graph worker
technicals_agent_app = build_technicals_graph()

def technicals_agent_node(state: dict):
    """
    Wrapper for LangGraph integration.
    """
    # Map incoming state to agent inputs
    # For now, we assume state has 'messages' and we parse the last message
    # Or strict inputs are passed.
    
    # Input adapter (simplified for now)
    inputs = state.get("technicals_input", {}) # Expect specific key or parse from tasks
    
    result = technicals_agent_app.invoke({
        "inputs": inputs,
        "validated": False,
        "missing_data": [],
        "primary_timeframe": "",
        "secondary_timeframes": [],
        "output": {},
        "retries": 0
    })
    
    return {"technicals_output": result["output"]}
