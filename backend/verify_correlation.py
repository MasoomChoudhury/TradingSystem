
import json
import logging
import sys
import os
from dotenv import load_dotenv
from unittest.mock import MagicMock, patch

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from correlation_agent import correlation_agent_node

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_correlation")

def test_correlation_agent():
    print("Testing Correlation Exposure Agent...")
    
    # Mock Input Data
    mock_input = {
        "as_of": "2024-01-01",
        "portfolio": {
            "total_aum": 1000000,
            "holdings": [
                {"symbol": "NIFTY", "weight_pct": 20},
                {"symbol": "BANKNIFTY", "weight_pct": 15}
            ]
        },
        "proposed_trade": {
            "symbol": "HDFCBANK",
            "size": 50000,
            "expected_weight_pct": 5.0
        },
        "correlation_matrix": {
            "symbols": ["NIFTY", "BANKNIFTY", "HDFCBANK"],
            "corr_30d": [
                [1.0, 0.8, 0.75],
                [0.8, 1.0, 0.85],
                [0.75, 0.85, 1.0]
            ]
        },
        "concentration_limits": {"max_single_name_pct": 10}
    }
    
    state = {"correlation_input": mock_input, "messages": []}
    
    # Mock LLM Response
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "agent": "correlation_exposure",
        "portfolio_impact_verdict": "approved",
        "risk_score_0_to_100": 30,
        "post_trade_exposures": {
            "concentration_breaches": []
        },
        "correlation_analysis": {
            "highest_corr_pair": {"symbol": "BANKNIFTY", "corr_30d": 0.85}
        },
        "risk_recommendations": []
    })
    
    print("\n[Test 1] Running full agent flow with mock data (MOCKED LLM)...")
    
    with patch("langchain_google_genai.ChatGoogleGenerativeAI.invoke", return_value=mock_response):
        try:
            result = correlation_agent_node(state)
            
            output = result.get("correlation_output", {})
            print("\nAgent Output:")
            print(json.dumps(output, indent=2))
            
            if "portfolio_impact_verdict" in output and output["portfolio_impact_verdict"] != "insufficient_data":
                print("\nSUCCESS: Generated correlation verdict (mocked).")
            else:
                print(f"\nFAILURE: Verdict is {output.get('portfolio_impact_verdict')}")
                
        except Exception as e:
            print(f"\nERROR: Agent execution failed: {e}")

if __name__ == "__main__":
    test_correlation_agent()
