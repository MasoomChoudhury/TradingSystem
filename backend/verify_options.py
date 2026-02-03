
import json
import logging
import sys
import os
from dotenv import load_dotenv
from unittest.mock import MagicMock, patch

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from options_agent import options_agent_node
from graph_architecture import TradingGraphState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_options")

def test_options_agent():
    print("Testing Options Analytics Agent...")
    
    # Mock Input Data (Universal Schema)
    mock_input = {
        "instrument": {
            "symbol": "SPX", 
            "exchange": "CBOE", 
            "spot": 4500,
            "lot_size": 100
        },
        "time_horizon": "swing",
        "view": {"direction": "bullish", "conviction": 85},
        "options_chain": {
            "calls": [
                {"strike": 4600, "bid": 20, "ask": 22, "ltp": 21, "iv_ltp": 0.15}
            ],
            "puts": []
        },
        "market_data": {"atm_iv": 0.14, "risk_free_rate": 0.05}
    }
    
    state = {"options_input": mock_input, "messages": []}
    
    # Mock LLM Response
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "agent": "options_analytics",
        "instrument": {"symbol": "SPX", "spot": 4500},
        "overall_recommendation": "bull_call_spread",
        "conviction_0_to_100": 80,
        "risk_reward": {"est_edge": 0.25, "max_loss_pct": 100},
        "greek_decomposition": {
            "primary_exposure": "delta",
            "delta_exposure": {"net_delta": 45, "directional_bias": "long"},
            "gamma_exposure": {"convexity": "positive", "notes": "Accelerates profit if SPX rallies fast"},
            "vega_exposure": {"net_vega": 10, "vol_regime_view": "long_vol"},
            "theta_exposure": {"daily_decay_pct": 2, "horizon_impact": "Manageable for swing"}
        },
        "recommended_structure": {
            "name": "bull_call_spread",
            "legs": [
                {"action": "buy", "type": "call", "strike": 4500, "premium_paid_received": 50},
                {"action": "sell", "type": "call", "strike": 4600, "premium_paid_received": 20}
            ],
            "net_debit_credit": 30,
            "net_delta": 25
        },
        "stress_test": {
            "spot_up_2pct": {"pnl_pct": 50},
            "spot_down_2pct": {"pnl_pct": -80}
        },
        "liquidity_assessment": {"warnings": ["None"]}
    })
    
    print("\n[Test 1] Running full agent flow with mock data (MOCKED LLM)...")
    
    with patch("langchain_google_genai.ChatGoogleGenerativeAI.invoke", return_value=mock_response):
        try:
            result = options_agent_node(state)
            
            output = result.get("options_output", {})
            print("\nAgent Output:")
            print(json.dumps(output, indent=2))
            
            if "overall_recommendation" in output and output["overall_recommendation"] != "insufficient_data":
                print("\nSUCCESS: Generated options brief (mocked).")
            else:
                print(f"\nFAILURE: Recommendation is {output.get('overall_recommendation')}")
                
        except Exception as e:
            print(f"\nERROR: Agent execution failed: {e}")

if __name__ == "__main__":
    test_options_agent()
