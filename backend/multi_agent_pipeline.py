
import asyncio
import logging
import datetime
from typing import Dict, Any, List

# Import all agent nodes
from fundamentals_agent import fundamentals_agent_node
from technicals_agent import technicals_agent_node
from news_agent import news_agent_node
from sentiment_agent import sentiment_agent_node
from institutional_agent import institutional_agent_node
from options_agent import options_agent_node
from vol_surface_agent import vol_surface_agent_node
from liquidity_agent import liquidity_agent_node
from correlation_agent import correlation_agent_node
from chart_pattern_agent import chart_pattern_agent_node
from global_analyst_agent import global_analyst_node

from data_collector import DataCollector

logger = logging.getLogger("multi_agent_pipeline")

class MultiAgentPipeline:
    """
    Orchestrates the parallel execution of 10 agents and 1 synthesizer.
    """
    
    def __init__(self):
        self.collector = DataCollector()
        
    async def run_pipeline(self, symbol: str) -> Dict[str, Any]:
        """
        1. Collect Data (Factory)
        2. Run 10 Agents (Parallel)
        3. Run Global Analyst (Sequential)
        """
        logger.info(f"Starting Multi-Agent Pipeline for {symbol}...")
        
        # 1. Data Collection
        # In a real async DB scenario, this might be async too.
        raw_inputs = self.collector.collect_all(symbol)
        
        # 2. Parallel Execution Setup
        # Create coroutines for each agent
        # Note: Nodes are typically synchronous logic calling async LLMs or sync functions. 
        # If they are sync, we run them in thread pool. If async, await directly.
        # Assuming nodes are currently synchronous wrappers calling sync/async methods. 
        # For true parallelism with sync nodes, use loop.run_in_executor.
        
        loop = asyncio.get_running_loop()
        
        # Helper to run node in thread
        def run_node(node_func, input_key, output_key):
            # Capture the exact input state for this agent
            agent_input = raw_inputs.get(input_key, {})
            state = {input_key: agent_input}
            
            trace_entry = {
                "agent_name": node_func.__name__.replace("_node", "").replace("_agent", ""),
                "inputs": agent_input,
                "output": None,
                "error": None
            }

            try:
                res = node_func(state)
                
                # Capture full raw output before filtering
                trace_entry["output"] = res.get(output_key, res)
                
                # Extract specific output key if exists, else return whole
                if output_key in res:
                    return {output_key: res[output_key], "_trace": trace_entry}
                return {**res, "_trace": trace_entry}
                
            except Exception as e:
                logger.error(f"Node {node_func.__name__} failed: {e}")
                trace_entry["error"] = str(e)
                return {output_key: {"error": str(e)}, "_trace": trace_entry}

        # Launch all 10 agents
        logger.info("Launching Committee of 10...")
        
        tasks = [
            loop.run_in_executor(None, run_node, fundamentals_agent_node, "fundamentals_input", "fundamentals_output"),
            loop.run_in_executor(None, run_node, technicals_agent_node, "technicals_input", "technicals_output"),
            loop.run_in_executor(None, run_node, news_agent_node, "news_input", "news_output"),
            loop.run_in_executor(None, run_node, sentiment_agent_node, "sentiment_input", "sentiment_output"),
            loop.run_in_executor(None, run_node, institutional_agent_node, "institutional_input", "institutional_output"),
            loop.run_in_executor(None, run_node, options_agent_node, "options_input", "options_output"),
            loop.run_in_executor(None, run_node, vol_surface_agent_node, "vol_surface_input", "vol_surface_output"),
            loop.run_in_executor(None, run_node, liquidity_agent_node, "liquidity_input", "liquidity_output"),
            loop.run_in_executor(None, run_node, correlation_agent_node, "correlation_input", "correlation_output"),
            loop.run_in_executor(None, run_node, chart_pattern_agent_node, "chart_pattern_input", "chart_pattern_output"),
        ]
        
        results = await asyncio.gather(*tasks)
        
        # 3. Stitch Results & Collect Traces
        stitched_outputs = {}
        agent_traces = {}
        
        for res in results:
            # Extract trace if present
            if "_trace" in res:
                trace = res.pop("_trace")
                agent_name = trace["agent_name"]
                agent_traces[agent_name] = trace
                
            stitched_outputs.update(res)
            
        committee_report = {
            "symbol": symbol,
            "timestamp": datetime.datetime.now().isoformat(),
            "agent_outputs": stitched_outputs
        }
        
        logger.info(f"Agents finished. Report size: {len(str(stitched_outputs))} chars.")
        
        # 4. Global Analyst
        logger.info("Running Global Analyst...")
        final_decision = global_analyst_node({"committee_report": committee_report})
        
        # Add Global Analyst to trace
        agent_traces["global_analyst"] = {
            "agent_name": "global_analyst",
            "inputs": committee_report,
            "output": final_decision.get("final_output")
        }
        
        return {
            "pipeline_status": "success",
            "symbol": symbol,
            "final_report": final_decision.get("final_output"),
            "trace": agent_traces
        }

if __name__ == "__main__":
    # Test run
    logging.basicConfig(level=logging.INFO)
    pipeline = MultiAgentPipeline()
    res = asyncio.run(pipeline.run_pipeline("NIFTY"))
    print(json.dumps(res, indent=2))
