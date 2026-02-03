
import json
import logging
import sys
import os
from dotenv import load_dotenv

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from news_agent import news_agent_node
from graph_architecture import TradingGraphState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_news")

def test_news_agent():
    print("Testing News Event Risk Agent...")
    
    # Mock Input Data
    mock_input = {
        "instrument": {"symbol": "INFY", "exchange": "NSE"},
        "as_of": "2024-01-01T10:00:00Z",
        "time_horizon": "swing",
        "news": {
            "headlines": [
                {"id": "1", "ts": "2024-01-01T09:00:00Z", "title": "Infosys Q3 Results Preview", "source": "Reuters"},
                {"id": "2", "ts": "2024-01-01T09:05:00Z", "title": "Infosys Q3 Results Preview - Expect margins to hold", "source": "Bloomberg"}, # Duplicate check
                {"id": "3", "ts": "2024-01-01T08:00:00Z", "title": "Tech sector down on weak global cues", "source": "Moneycontrol"}
            ],
            "articles": [],
            "social": []
        },
        "events_calendar": {
            "earnings": {"date": "2024-01-10", "type": "results"}
        }
    }
    
    # Mock State
    state = {
        "news_input": mock_input,
        "messages": []
    }
    
    # MOCKING LLM INVOKE to test logic flow without API
    from unittest.mock import MagicMock, patch
    
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "agent": "news_event_risk",
        "overall_risk_label": "moderate",
        "overall_bias_from_news": "neutral",
        "key_developments": [
            {
                "headline_id": "1",
                "what_happened": "Q3 Results approaching",
                "time_sensitivity": "scheduled",
                "source_quality": "high"
            }
        ],
        "deduplication_stats": "Mocked deduplication happened"
    })
    
    print("\n[Test 1] Running full agent flow with mock data (MOCKED LLM)...")
    with patch("langchain_google_genai.ChatGoogleGenerativeAI.invoke", return_value=mock_response):
        try:
            result = news_agent_node(state)
            
            output = result.get("news_output", {})
            print("\nAgent Output:")
            print(json.dumps(output, indent=2))
            
            if "overall_risk_label" in output and output["overall_risk_label"] != "insufficient_data":
                print("\nSUCCESS: Generated risk brief (mocked).")
            else:
                print(f"\nFAILURE: Risk label is {output.get('overall_risk_label')}")
                
        except Exception as e:
            print(f"\nERROR: Agent execution failed: {e}")

if __name__ == "__main__":
    test_news_agent()
