
import json
import logging
import sys
import os
from dotenv import load_dotenv
from unittest.mock import MagicMock, patch

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from institutional_agent import institutional_agent_node
from graph_architecture import TradingGraphState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_institutional")

def test_institutional_agent():
    print("Testing Institutional Flow Agent...")
    
    # Mock Input Data
    mock_input = {
        "instrument": {"symbol": "HDFCBANK", "exchange": "NSE"},
        "as_of": "2024-01-01T15:30:00Z",
        "time_horizon": "swing",
        "lookback": {"days": 5},
        "institutional_flows": {
            "fii_net": [{"date": "2024-01-01", "net_qty": 500000}], # Buying
            "dii_net": [],
            "trend": {"last_5d_net": 1000}
        },
        "delivery_data": {
            "daily_delivery": [{"date": "2024-01-01", "delivery_pct": 0.65}], # High delivery
            "delivery_trend": {"avg_5d_pct": 0.40} # Spike vs avg
        },
        "block_bulk_deals": [
            {"qty": 100000, "deal_type": "block", "buyer": "FII Fund"}
        ],
        "price_volume_context": {"float_shares": 10000000}
    }
    
    state = {"institutional_input": mock_input, "messages": []}
    
    # Mock LLM Response
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "agent": "institutional_flow",
        "smart_money_stance": "accumulate",
        "conviction_0_to_100": 85,
        "ride_with_them_bias": "bullish",
        "flow_alignment": {
            "fii_view": {"direction": "accumulating", "intensity_0_to_100": 80},
            "delivery_view": {"direction": "holding", "delivery_pct_trend": "rising"},
            "block_bulk_view": {"net_direction": "buy"}
        },
        "key_flow_signals": [
            {"signal_type": "delivery_spike", "impact": "high", "smart_money_read": "bullish"}
        ],
        "flow_momentum": {"short_term": "accelerating"},
        "tradeability_from_flow": {"can_ride_flow": True, "preferred_direction": "long"}
    })
    
    print("\n[Test 1] Running full agent flow with mock data (MOCKED LLM)...")
    
    with patch("langchain_google_genai.ChatGoogleGenerativeAI.invoke", return_value=mock_response):
        try:
            result = institutional_agent_node(state)
            
            output = result.get("institutional_output", {})
            print("\nAgent Output:")
            print(json.dumps(output, indent=2))
            
            if "smart_money_stance" in output and output["smart_money_stance"] != "insufficient_data":
                print("\nSUCCESS: Generated flow brief (mocked).")
            else:
                print(f"\nFAILURE: Stance is {output.get('smart_money_stance')}")
                
        except Exception as e:
            print(f"\nERROR: Agent execution failed: {e}")

if __name__ == "__main__":
    test_institutional_agent()
