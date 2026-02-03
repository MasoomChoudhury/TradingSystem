
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

logger = logging.getLogger("news_agent")

# =============================================================================
# SCHEMAS
# =============================================================================

class Headline(BaseModel):
    id: str
    ts: str
    source: str
    title: str
    snippet: Optional[str] = ""
    url: Optional[str] = ""
    language: str = "en"

class Article(BaseModel):
    id: str
    ts: str
    source: str
    title: str
    body: str
    url: Optional[str] = ""

class Event(BaseModel):
    date: str
    type: str # results, guidance, split, macro etc
    details: Optional[str] = ""

class NewsInput(BaseModel):
    instrument: Dict[str, str]
    as_of: str
    time_horizon: str
    news: Dict[str, List[Dict[str, Any]]] # headlines, articles, social
    events_calendar: Dict[str, Any] # earnings, corporate_actions, macro
    context: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, Any]] = None

# =============================================================================
# STATE
# =============================================================================

class NewsAgentState(TypedDict):
    inputs: Dict[str, Any]
    validated: bool
    missing_data: List[str]
    
    # Internal processing
    deduped_headlines: List[Dict[str, Any]]
    
    # Output
    output: Dict[str, Any]

# =============================================================================
# NODES
# =============================================================================

def validate_inputs(state: NewsAgentState) -> Dict[str, Any]:
    """
    Node 1: Validate inputs.
    Require: instrument.symbol, as_of, and at least one news/event data point.
    """
    inputs = state.get("inputs", {})
    missing = []
    
    if "instrument" not in inputs or "symbol" not in inputs["instrument"]:
        missing.append("instrument.symbol")
    if "as_of" not in inputs:
        missing.append("as_of")
        
    news = inputs.get("news", {})
    events = inputs.get("events_calendar", {})
    
    has_news = any(len(news.get(k, [])) > 0 for k in ["headlines", "articles", "social"])
    has_events = any(len(events.get(k, [])) > 0 for k in ["earnings", "corporate_actions", "macro"])
    
    if not has_news and not has_events:
        missing.append("news_or_events")
        
    if missing:
        return {
            "validated": False,
            "missing_data": missing,
            "output": {
                "overall_risk_label": "insufficient_data",
                "data_gaps": missing,
                "recommended_next_data_to_fetch": ["Please provide instrument, timestamp, and news/events."]
            }
        }
        
    return {"validated": True, "missing_data": []}

def dedupe_and_rank(state: NewsAgentState) -> Dict[str, Any]:
    """
    Node 2: Deduplicate and sort headlines.
    Simple string similarity check.
    """
    inputs = state["inputs"]
    headlines = inputs.get("news", {}).get("headlines", [])
    
    # 1. Sort by recency (assuming ISO format)
    headlines.sort(key=lambda x: x.get("ts", ""), reverse=True)
    
    # 2. Dedupe
    unique_headlines = []
    seen_titles = []
    
    for h in headlines:
        title = h.get("title", "")
        # Simple fuzzy check against already active titles
        is_duplicate = False
        for seen in seen_titles:
            ratio = SequenceMatcher(None, title, seen).ratio()
            if ratio > 0.8: # 80% similarity
                is_duplicate = True
                break
                
        if not is_duplicate:
            unique_headlines.append(h)
            seen_titles.append(title)
            
    # Limit logic (e.g. top 50) could go here
    return {"deduped_headlines": unique_headlines[:50]}

def generate_risk_brief(state: NewsAgentState) -> Dict[str, Any]:
    """
    Node 3: LLM Risk Analysis.
    """
    inputs = state["inputs"]
    deduped_headlines = state.get("deduped_headlines", [])
    events = inputs.get("events_calendar", {})
    
    # Context for LLM
    llm_context = {
        "instrument": inputs.get("instrument"),
        "time_horizon": inputs.get("time_horizon"),
        "as_of": inputs.get("as_of"),
        "headlines": deduped_headlines,
        "events": events,
        "context": inputs.get("context")
    }
    
    api_key = os.environ.get("GEMINI_API_KEY")
    from llm_client import run_llm
    
    system_prompt = """You are NewsEventRiskAgent, a senior buy-side event-risk analyst.
    Assess the provided news items for MATERIAL MARKET IMPACT.
    Classification: confirmed_fact vs rumor vs opinion.
    Score impacts from -10 (Crash) to +10 (Moon) to 0 (Noise).
    
    OUTPUT JSON:
    {
        "overall_risk_label": "LOW|MED|HIGH|EXTREME",
        "market_moving_events": [{"headline": "...", "impact_score": 8, "reason": "..."}],
        "summary": "..."
    }
    """
    
    user_prompt = f"Analyze these news items: {json.dumps(llm_context, default=str)}"

    try:
        response_text = run_llm(system_prompt, user_prompt)
        parsed = json.loads(response_text.replace("```json", "").replace("```", "").strip())
        return {"output": parsed}
    except Exception as e:
        logger.error(f"News Analysis Failed: {e}")
        return {
            "output": {
                "overall_risk_label": "UNKNOWN",
                "summary": f"Error: {e}"
            }
        }

# =============================================================================
# GRAPH
# =============================================================================

def build_news_graph():
    graph = StateGraph(NewsAgentState)
    
    graph.add_node("validate_inputs", validate_inputs)
    graph.add_node("dedupe_and_rank", dedupe_and_rank)
    graph.add_node("generate_risk_brief", generate_risk_brief)
    
    graph.add_edge(START, "validate_inputs")
    
    def validation_router(state):
        if state.get("validated"):
            return "dedupe_and_rank"
        return END
        
    graph.add_conditional_edges("validate_inputs", validation_router, {
        "dedupe_and_rank": "dedupe_and_rank",
        END: END
    })
    
    graph.add_edge("dedupe_and_rank", "generate_risk_brief")
    graph.add_edge("generate_risk_brief", END)
    
    return graph.compile()

news_agent_app = build_news_graph()

def news_agent_node(state: dict):
    """
    Wrapper for LangGraph integration.
    """
    inputs = state.get("news_input", {})
    
    result = news_agent_app.invoke({
        "inputs": inputs,
        "validated": False,
        "missing_data": [],
        "deduped_headlines": [],
        "output": {}
    })
    
    return {"news_output": result["output"]}
