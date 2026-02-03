"""
Fundamentals Agent - Fundamental Analysis & Research
Responsibility:
- Web search for financial news, earnings, and macro events
- Analyze business models and risks
- Generate investment theses
"""
import os
import logging
from typing import Dict, Any, List
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from graph_architecture import TradingGraphState, store_large_output

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize tools
search_tool = DuckDuckGoSearchRun()

# Initialize LLM
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
from llm_client import run_llm

SYSTEM_PROMPT = """You are a Senior Fundamental Analyst for a top-tier hedge fund.
Your goal is to provide a comprehensive investment thesis based on the provided search results.

Focus on:
1. Business Model & Competitive Advantage (Moat)
2. Recent Financial Performance (Revenue, Earnings, Growth)
3. Key Risks (Regulatory, Market, Operational)
4. Macro Environment Impact
5. Catalyst for future price movement

Format your output as a professional research memo. 
Be concise, data-driven, and objective. 
Conclude with a clear 'Bullish', 'Bearish', or 'Neutral' outlook.
"""

def fundamentals_agent_node(state: TradingGraphState) -> Dict[str, Any]:
    """
    LangGraph node for Fundamental Analysis.
    Performs web search and generates an investment thesis.
    """
    logger.info("Setting up Fundamentals Agent...")
    
    # Extract intent or messages
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    query = last_message.content if last_message else "Analyze the current market sentiment"
    
    # Check if a specific symbol is mentioned in the query
    # Simple heuristic to extract potential symbol if not explicitly provided
    # ideally parsing logic should be robust or passed in state
    
    logger.info(f"Fundamentals Agent processing query: {query}")
    
    try:
        # 1. Perform Research
        # We search for a few key aspects to get a broad view
        search_queries = [
            f"{query} latest financial news",
            f"{query} business model analysis",
            f"{query} key investment risks",
            f"{query} analyst ratings and target price"
        ]
        
        research_results = []
        for q in search_queries:
            try:
                logger.info(f"Searching for: {q}")
                result = search_tool.invoke(q)
                research_results.append(f"Query: {q}\nResult: {result}\n---\n")
            except Exception as e:
                logger.error(f"Search failed for '{q}': {e}")
                research_results.append(f"Query: {q}\nResult: Search failed.\n---\n")
        
        full_research_text = "\n".join(research_results)
        
        # 2. Generate Thesis
        logger.info("Synthesizing investment thesis...")
        user_prompt = f"""
        Research Data:
        {full_research_text}
        
        User Query:
        {query}
        
        Based ONLY on the research data above, write a fundamental analysis report.
        """
        
        # Replaced llm.invoke with run_llm as per instruction
        response_content = run_llm(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
        
        thesis = response_content
        
        # 3. Store result
        # Store large research data as artifact
        research_artifact_key = f"fundamentals_research_{state.get('request_id', 'unknown')}"
        store_large_output(research_artifact_key, {
            "query": query,
            "raw_research": research_results
        })
        
        return {
            "current_node": "fundamentals_worker",
            "messages": [
                HumanMessage(content=f"Fundamental Analysis for: {query}"),
                response 
            ],
            "result": {
                "type": "fundamental_analysis",
                "thesis": thesis,
                "research_artifact": research_artifact_key
            },
            # Always return to orchestrator
            "return_to": "orchestrator", 
            "path_history": state.get("path_history", []) + ["fundamentals_worker"]
        }
        
    except Exception as e:
        logger.error(f"Error in fundamentals agent: {e}")
        return {
            "current_node": "fundamentals_worker",
            "error": str(e),
            "messages": [HumanMessage(content=f"Error performing fundamental analysis: {e}")],
             "return_to": "orchestrator"
        }
