
import json
import logging
import os
from typing import TypedDict, Dict, Any, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

logger = logging.getLogger("global_analyst")

class CommitteeReport(TypedDict):
    symbol: str
    timestamp: str
    agent_outputs: Dict[str, Any] # Keyed by agent name (e.g., "fundamental_agent": {...})

class GlobalAnalystState(TypedDict):
    report: CommitteeReport
    final_decision: Dict[str, Any]

def global_analyst_node(state: Dict[str, Any]):
    """
    The Chairman Agent: Synthesizes reports from all 10 sub-agents.
    """
    report = state.get("committee_report", {})
    if not report:
        return {"final_output": {"error": "No report provided"}}

    agent_outputs = report.get("agent_outputs", {})
    
    # Prepare context for LLM
    # We strip down the outputs to avoid context window overflow if necessary, 
    # but initially we'll try to pass the full JSONs of the 10 agents.
    
    context_str = json.dumps(agent_outputs, indent=2, default=str)
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"final_output": {"decision": "HOLD", "reason": "API Key Missing"}}

    from llm_client import run_llm
    
    system_prompt = """You are the GlobalAnalyst (Chairman) of a hedge fund investment committee.
    
    You have received reports from 10 specialized agents:
    1. Fundamentals (Value)
    2. Technicals (Trends)
    3. News/Risk (Events)
    4. Sentiment (Crowd)
    5. Institutional (Flow)
    6. Options (Greeks)
    7. VolSurface (Arb)
    8. Liquidity (Execution)
    9. Correlation (Portfolio Risk)
    10. ChartPattern (Discretionary)

    YOUR TASK:
    - Synthesize these divergent views into a single firm MARKET REGIME DECISION for the symbol.
    - Resolve conflicts (e.g. valid Fundamentals but Bearish Technicals -> "Wait for pullback").
    - Prioritize Risk/Liquidity warnings (BLOCK trade if Liquidity/Correlation/News agents flag high risk).
    
    OUTPUT SCHEMA (JSON ONLY):
    {
        "symbol": "...",
        "timestamp": "...",
        "final_decision": "BUY|SELL|HOLD|EXIT_ALL",
        "conviction_score": 0-100,
        "primary_thesis": "...",
        "key_risks": ["..."],
        "agent_consensus": {
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0
        },
        "action_plan": {
            "entry_zone": "...",
            "stop_loss": "...",
            "target": "..."
        }
    }
    """
    
    user_prompt = f"""
    Review the Committee Report for {report.get('symbol')}.
    
    AGENT OUTPUTS:
    {context_str}
    """
    
    try:
        response_text = run_llm(system_prompt, user_prompt)
        
        content = response_text.replace("```json", "").replace("```", "").strip()
        decision_json = json.loads(content)
        
        return {"final_output": decision_json}
        
    except Exception as e:
        logger.error(f"Global Analyst Failed: {e}")
        return {"final_output": {"decision": "HOLD", "reason": f"Analysis Error: {str(e)}"}}
