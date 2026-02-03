
import json
import logging
import sys
import os
from dotenv import load_dotenv
from unittest.mock import MagicMock, patch

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from sentiment_agent import sentiment_agent_node

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_sentiment")

def test_sentiment_agent():
    print("Testing Sentiment Agent...")
    
    # Mock Input
    mock_input = {
        "instrument": {"symbol": "TATASTEEL", "exchange": "NSE"},
        "as_of": "2024-01-01T12:00:00Z",
        "time_horizon": "swing",
        "lookback": {"hours": 24},
        "sources": {
            "social": [
                {"text": "Steel prices going up! Buy TATASTEEL", "platform": "X"},
                {"text": "Steel prices up! Buy TATASTEEL", "platform": "Twitter"}, # Dupe
                {"text": "Selling my holding", "platform": "Reddit"}
            ],
            "news": [],
            "analyst": []
        }
    }
    
    state = {"sentiment_input": mock_input, "messages": []}
    
    # Mock LLM Response
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "agent": "sentiment",
        "overall_sentiment": "bullish",
        "confidence_0_to_100": 75,
        "sentiment_breakdown": {
            "retail_social": {"direction": "bullish", "intensity_0_to_100": 80, "notes": ["High engagement on X"]},
            "news_tone": {"direction": "neutral", "intensity_0_to_100": 0, "notes": ["No news"]},
            "analyst_tone": {"direction": "neutral", "intensity_0_to_100": 0, "notes": []},
            "positioning": {"direction": "unknown", "crowding_risk": "low"}
        },
        "narratives": [
            {"theme": "Global Commodity Cycle", "direction": "bullish", "share_of_voice_estimate": "high"}
        ],
        "crowding_and_contrarian_flags": [],
        "momentum_of_sentiment": {"shift": "improving", "reason": "Better macro data"}
    })
    
    print("\n[Test 1] Running full agent flow with mock data (MOCKED LLM)...")
    
    with patch("langchain_google_genai.ChatGoogleGenerativeAI.invoke", return_value=mock_response):
        try:
            result = sentiment_agent_node(state)
            
            output = result.get("sentiment_output", {})
            print("\nAgent Output:")
            print(json.dumps(output, indent=2))
            
            if "overall_sentiment" in output and output["overall_sentiment"] != "insufficient_data":
                print("\nSUCCESS: Generated sentiment brief (mocked).")
            else:
                print(f"\nFAILURE: Sentiment is {output.get('overall_sentiment')}")
                
        except Exception as e:
            print(f"\nERROR: Agent execution failed: {e}")

if __name__ == "__main__":
    test_sentiment_agent()
