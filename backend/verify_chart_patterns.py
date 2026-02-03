
import json
import logging
import sys
import os
from dotenv import load_dotenv
from unittest.mock import MagicMock, patch

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from chart_pattern_agent import chart_pattern_agent_node

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_chart_patterns")

def test_chart_pattern_agent():
    print("Testing Chart Pattern Agent...")
    
    # Generate dummy OHLCV data (sine wave + trend)
    ohlcv = []
    base_price = 100
    for i in range(100):
        trend = i * 0.1
        wave = 5 * (i % 20) / 20 # Sawtooth
        o = base_price + trend + wave
        c = o + 1
        h = max(o, c) + 2
        l = min(o, c) - 2
        ohlcv.append({
            "ts": f"2024-01-01T{i:02d}:00:00",
            "open": o, "high": h, "low": l, "close": c,
            "volume": 1000 + i*10
        })
    
    # Mock Input Data
    mock_input = {
        "instrument": {"symbol": "BTC", "timeframe": "1h"},
        "ohlcv": ohlcv,
        "context": {"higher_timeframe_trend": "bullish"}
    }
    
    state = {"chart_pattern_input": mock_input, "messages": []}
    
    # Mock LLM Response
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "agent": "chart_patterns",
        "primary_pattern": "bull_flag",
        "pattern_strength_0_to_100": 85,
        "pattern_details": {
            "name": "Bullish Flag",
            "measured_move_target": 150
        },
        "entry_setup": {
            "ideal_entry": 115,
            "stop_loss": 105
        }
    })
    
    print("\n[Test 1] Running full agent flow with mock data (MOCKED LLM)...")
    
    with patch("langchain_google_genai.ChatGoogleGenerativeAI.invoke", return_value=mock_response):
        try:
            result = chart_pattern_agent_node(state)
            
            output = result.get("chart_pattern_output", {})
            print("\nAgent Output:")
            print(json.dumps(output, indent=2))
            
            if "primary_pattern" in output and output["primary_pattern"] != "insufficient_data":
                print("\nSUCCESS: Generated pattern verdict (mocked).")
            else:
                print(f"\nFAILURE: Pattern is {output.get('primary_pattern')}")
                
        except Exception as e:
            print(f"\nERROR: Agent execution failed: {e}")

if __name__ == "__main__":
    test_chart_pattern_agent()
