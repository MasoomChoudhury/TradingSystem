
import json
import logging
import sys
import os
from dotenv import load_dotenv
from unittest.mock import MagicMock, patch

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from liquidity_agent import liquidity_agent_node

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_liquidity")

def test_liquidity_agent():
    print("Testing Liquidity Constraints Agent...")
    
    # Mock Input Data
    mock_input = {
        "instrument": {"symbol": "INFY", "exchange": "NSE"},
        "proposed_trade": {
            "direction": "long", 
            "size": 500000, # Large size to test impact
            "proposed_price": 1500
        },
        "market_snapshot": {
            "ltp": 1500,
            "bid": 1499, "ask": 1501,
            "avg_volume_20d": 10000000, # 5% of ADV
            "free_float": 500000000
        }
    }
    
    state = {"liquidity_input": mock_input, "messages": []}
    
    # Mock LLM Response
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "agent": "liquidity_constraints",
        "tradeability_verdict": "caution",
        "confidence_0_to_100": 90,
        "liquidity_assessment": {
            "size_vs_avg_volume_pct": 5.0,
            "expected_impact_pct": 0.35,
            "notes": ["Size is 5% of ADV, expect slicing"]
        },
        "spread_cost": {
            "half_spread_cost_pct": 0.03
        },
        "execution_estimate": {
            "slippage_pct": 0.4,
            "recommended_order_type": "iceberg",
            "slicing_needed": True
        }
    })
    
    print("\n[Test 1] Running full agent flow with mock data (MOCKED LLM)...")
    
    with patch("langchain_google_genai.ChatGoogleGenerativeAI.invoke", return_value=mock_response):
        try:
            result = liquidity_agent_node(state)
            
            output = result.get("liquidity_output", {})
            print("\nAgent Output:")
            print(json.dumps(output, indent=2))
            
            if "tradeability_verdict" in output and output["tradeability_verdict"] != "insufficient_data":
                print("\nSUCCESS: Generated liquidity verdict (mocked).")
            else:
                print(f"\nFAILURE: Verdict is {output.get('tradeability_verdict')}")
                
        except Exception as e:
            print(f"\nERROR: Agent execution failed: {e}")

if __name__ == "__main__":
    test_liquidity_agent()
