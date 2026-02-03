
import json
import logging
import os
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal
from datetime import datetime
from difflib import SequenceMatcher
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

logger = logging.getLogger("sentiment_agent")

# =============================================================================
# SCHEMAS
# =============================================================================

class SentimentInput(BaseModel):
    instrument: Dict[str, str]
    as_of: str
    time_horizon: str
    lookback: Dict[str, Any]
    sources: Dict[str, List[Dict[str, Any]]] # social, news, analyst, positioning_proxies
    context: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, Any]] = None

# =============================================================================
# STATE
# =============================================================================

class SentimentAgentState(TypedDict):
    inputs: Dict[str, Any]
    validated: bool
    missing_data: List[str]
    
    # Internal processing
    clean_items: List[Dict[str, Any]]
    
    # Output
    output: Dict[str, Any]

# =============================================================================
# NODES
# =============================================================================

def validate_inputs(state: SentimentAgentState) -> Dict[str, Any]:
    """
    Node 1: Validate inputs.
    """
    inputs = state.get("inputs", {})
    missing = []
    
    if "instrument" not in inputs or "symbol" not in inputs["instrument"]:
        missing.append("instrument.symbol")
    if "as_of" not in inputs:
        missing.append("as_of")
        
    sources = inputs.get("sources", {})
    has_data = any(len(sources.get(k, [])) > 0 for k in ["social", "news", "analyst"])
    
    if not has_data:
        missing.append("sources(social|news|analyst)")
        
    if missing:
        return {
            "validated": False, 
            "missing_data": missing,
            "output": {
                "overall_sentiment": "insufficient_data",
                "data_gaps": missing,
                "recommended_next_data_to_fetch": ["Please provide social, news, or analyst data."]
            }
        }
        
    return {"validated": True, "missing_data": []}

def clean_and_filter(state: SentimentAgentState) -> Dict[str, Any]:
    """
    Node 2: Deduplicate and filter items.
    """
    inputs = state["inputs"]
    sources = inputs.get("sources", {})
    
    all_items = []
    
    # Flatten sources
    for cat, items in sources.items():
        if isinstance(items, list):
            for item in items:
                item["_source_category"] = cat
                all_items.append(item)
                
    # Dedupe by text similarity
    unique_items = []
    seen_texts = []
    
    for item in all_items:
        text = item.get("text") or item.get("title") or item.get("summary") or ""
        if not text:
            continue
            
        # Check dupe
        is_dupe = False
        for seen in seen_texts:
            if SequenceMatcher(None, text, seen).ratio() > 0.85:
                is_dupe = True
                break
        
        if not is_dupe:
            unique_items.append(item)
            seen_texts.append(text)
            
    # Cap items to avoid token overload
    return {"clean_items": unique_items[:100]}

def generate_sentiment_brief(state: SentimentAgentState) -> Dict[str, Any]:
    """
    Node 3: LLM Sentiment Analysis.
    """
    inputs = state["inputs"]
    clean_items = state.get("clean_items", [])
    
    llm_context = {
        "instrument": inputs.get("instrument"),
        "time_horizon": inputs.get("time_horizon"),
        "timestamp": inputs.get("as_of"),
        "items": clean_items,
        "context": inputs.get("context")
    }
    
    from llm_client import run_llm
    
    system_prompt = """You are SentimentAgent, a senior buy-side sentiment and narrative analyst.
    Analyze social media and news for CROWD PSYCHOLOGY and NARRATIVES.
    Detect: Euphoria, Panic, Apathy, Disbelief.
    
    OUTPUT JSON:
    {
        "overall_sentiment": "BULLISH|BEARISH|NEUTRAL",
        "sentiment_score": -100 to 100,
        "key_narratives": ["..."],
        "summary": "..."
    }
    """
    
    user_prompt = f"Analyze this social data: {json.dumps(clean_items)}"

    try:
        response_text = run_llm(system_prompt, user_prompt)
        parsed = json.loads(response_text.replace("```json", "").replace("```", "").strip())
        return {"output": parsed}
    except Exception as e:
        logger.error(f"Sentiment Analysis Failed: {e}")
        return {
            "output": {
                "overall_sentiment": "UNKNOWN",
                "summary": f"Error: {e}"
            }
        }

# =============================================================================
# GRAPH
# =============================================================================

def build_sentiment_graph():
    graph = StateGraph(SentimentAgentState)
    
    graph.add_node("validate_inputs", validate_inputs)
    graph.add_node("clean_and_filter", clean_and_filter)
    graph.add_node("generate_sentiment_brief", generate_sentiment_brief)
    
    graph.add_edge(START, "validate_inputs")
    
    def validation_router(state):
        if state.get("validated"):
            return "clean_and_filter"
        return END
        
    graph.add_conditional_edges("validate_inputs", validation_router, {
        "clean_and_filter": "clean_and_filter",
        END: END
    })
    
    graph.add_edge("clean_and_filter", "generate_sentiment_brief")
    graph.add_edge("generate_sentiment_brief", END)
    
    return graph.compile()

sentiment_agent_app = build_sentiment_graph()

def sentiment_agent_node(state: dict):
    """
    Wrapper for LangGraph integration.
    """
    inputs = state.get("sentiment_input", {})
    
    result = sentiment_agent_app.invoke({
        "inputs": inputs,
        "validated": False,
        "missing_data": [],
        "clean_items": [],
        "output": {}
    })
    
    return {"sentiment_output": result["output"]}
