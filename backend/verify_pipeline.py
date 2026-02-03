
import asyncio
import logging
import json
import sys
import os
from dotenv import load_dotenv

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from multi_agent_pipeline import MultiAgentPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_pipeline")

async def test_pipeline():
    print("Testing Full Multi-Agent Pipeline (REAL EXECUTION)...")
    
    pipeline = MultiAgentPipeline()
    symbol = "SBIN" # Use a real symbol if possible
    
    print(f"\n[Test 1] Running pipeline for {symbol}...")
    
    try:
        result = await pipeline.run_pipeline(symbol)
        
        print("\nPipeline Result:")
        print(json.dumps(result, indent=2))
        
        final_report = result.get("final_report", {})
        if final_report and "final_decision" in final_report:
            print(f"\nSUCCESS: Pipeline generated decision: {final_report['final_decision']}")
        else:
            print("\nFAILURE: No final decision produced.")
            
    except Exception as e:
        print(f"\nERROR: Pipeline execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_pipeline())
