
import json
import logging
import sys
import os
from dotenv import load_dotenv

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from technicals_agent import technicals_agent_node
from graph_architecture import TradingGraphState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_technicals")

def test_technicals_agent():
    print("Testing Technicals Agent...")
    
    # Mock Input Data
    mock_input = {
        "instrument": {
            "symbol": "AAPL",
            "exchange": "NASDAQ",
            "type": "equity"
        },
        "time_horizon": "swing",
        "timeframes": ["1D", "1h"],
        "market_data": {
            "ohlcv": {
                "1D": [{"ts": "2024-01-01", "close": 150.0}] * 100, # Mock 100 candles
                "1h": [{"ts": "2024-01-01T10:00", "close": 150.5}] * 50
            },
            "session_info": {"timezone": "US/Eastern", "market": "NASDAQ"}
        }
    }
    
    # Mock State
    state = {
        "technicals_input": mock_input,
        "messages": []
    }
    
    # MOCKING LLM INVOKE to test logic flow without API
    from unittest.mock import MagicMock, patch
    
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "agent": "technicals",
        "stance": "bullish",
        "conviction_0_to_100": 85,
        "primary_timeframe": "1D",
        "market_regime": {"regime": "trend_up", "evidence": ["Higher highs"]},
        "setups": [{"setup_name": "pullback", "trigger": "test trigger"}]
    })
    
    print("\n[Test 1] Running full agent flow with mock data (MOCKED LLM)...")
    with patch("langchain_google_genai.ChatGoogleGenerativeAI.invoke", return_value=mock_response):
        try:
            result = technicals_agent_node(state)
            
            output = result.get("technicals_output", {})
            print("\nAgent Output:")
            print(json.dumps(output, indent=2))
            
            if "stance" in output and output["stance"] != "insufficient_data":
                print("\nSUCCESS: Generated analysis (mocked).")
            else:
                print(f"\nFAILURE: Stance is {output.get('stance')}")
                
        except Exception as e:
            print(f"\nERROR: Agent execution failed: {e}")

if __name__ == "__main__":
    test_technicals_agent()
