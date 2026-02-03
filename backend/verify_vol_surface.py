
import json
import logging
import sys
import os
from dotenv import load_dotenv
from unittest.mock import MagicMock, patch

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from vol_surface_agent import vol_surface_agent_node

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_vol_surface")

def test_vol_surface_agent():
    print("Testing Vol Surface Agent...")
    
    # Mock Input Data
    mock_input = {
        "instrument": {"symbol": "NIFTY", "spot": 21500},
        "time_horizon": "swing",
        "surface_date": "2024-01-01",
        "vol_surface": {
            "strikes": [21000, 21500, 22000, 22500, 23000],
            "expiries": ["7d", "30d", "60d"],
            "iv_matrix": {
                "7d": {"21500": 12.0, "22000": 11.0},
                "30d": {"21500": 13.5, "22000": 12.5}
            }
        },
        "market_data": {"risk_free_rate": 0.07}
    }
    
    state = {"vol_surface_input": mock_input, "messages": []}
    
    # Mock LLM Response
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "agent": "vol_surface",
        "instrument": {"symbol": "NIFTY"},
        "overall_opportunity": "term_structure",
        "conviction_0_to_100": 85,
        "edge_estimate_pct": 1.5,
        "surface_diagnostics": {
            "term_structure": "contango",
            "skew_slope": {"normal": True},
            "rv_vs_iv": "iv_rich"
        },
        "top_relative_value": [
            {
                "type": "back_month_cheap",
                "expiry": "60d",
                "edge_pct": 2.0,
                "structure": "calendar_spread"
            }
        ],
        "recommended_trade": {
            "name": "calendar_spread",
            "direction": "long_vol",
            "legs": [
                {"action": "sell", "expiry": "7d", "strike_pct": 100},
                {"action": "buy", "expiry": "30d", "strike_pct": 100}
            ],
            "rationale": "Steep contango allows cheap entry for long veg"
        }
    })
    
    print("\n[Test 1] Running full agent flow with mock data (MOCKED LLM)...")
    
    with patch("langchain_google_genai.ChatGoogleGenerativeAI.invoke", return_value=mock_response):
        try:
            result = vol_surface_agent_node(state)
            
            output = result.get("vol_surface_output", {})
            print("\nAgent Output:")
            print(json.dumps(output, indent=2))
            
            if "overall_opportunity" in output and output["overall_opportunity"] != "insufficient_data":
                print("\nSUCCESS: Generated vol surface brief (mocked).")
            else:
                print(f"\nFAILURE: Opportunity is {output.get('overall_opportunity')}")
                
        except Exception as e:
            print(f"\nERROR: Agent execution failed: {e}")

if __name__ == "__main__":
    test_vol_surface_agent()
