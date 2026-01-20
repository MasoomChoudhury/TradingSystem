"""
Dedicated Worker Agents - Specialized Tools Only

Each worker has ONLY the tools it needs:
- MarketData: quotes, history, search, expiry
- Indicators: calculate, validate, list
- Options: greeks, symbols, strategies
- Accounts: funds, positions, holdings, margin

Workers CANNOT:
- Place orders
- Modify orders
- Cancel orders
- Execute any broker side effects

All workers return results to Supervisor for routing.
"""
import os
import json
import logging
from typing import Annotated, TypedDict, Sequence, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBW90vlRHW6uJgWbjzoNOYYKndHPp33ctk")


# =============================================================================
# WORKER STATE (Shared by all workers)
# =============================================================================

class WorkerState(TypedDict):
    """State for worker agents."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    task: Optional[str]
    result: Optional[dict]


# =============================================================================
# BASE WORKER CLASS
# =============================================================================

class BaseWorker:
    """Base class for all worker agents."""
    
    def __init__(self, name: str, tools: list, system_prompt: str):
        from rate_limiter import RateLimitedLLM
        
        self.name = name
        self.tools = tools
        self.system_prompt = system_prompt
        
        base_llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            google_api_key=GEMINI_API_KEY,
            temperature=0.2,
        )
        
        self.llm = RateLimitedLLM(base_llm, estimated_tokens_per_call=2000)
        
        self.llm_with_tools = self.llm.bind_tools(tools)
        self.tool_node = ToolNode(tools)
        self.memory = InMemorySaver()
        self.graph = self._build_graph()
        self.app = self.graph.compile(checkpointer=self.memory)
        
        logger.info(f"{name} Worker initialized with {len(tools)} tools")
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(WorkerState)
        
        workflow.add_node("worker", self._work)
        workflow.add_node("tools", self.tool_node)
        
        workflow.set_entry_point("worker")
        
        workflow.add_conditional_edges(
            "worker",
            self._should_continue,
            {"continue": "tools", "end": END}
        )
        
        workflow.add_edge("tools", "worker")
        
        return workflow
    
    def _work(self, state: WorkerState) -> dict:
        import time
        time.sleep(1)
        
        messages = list(state["messages"])
        full_messages = [SystemMessage(content=self.system_prompt)] + messages
        response = self.llm_with_tools.invoke(full_messages)
        
        return {"messages": [response]}
    
    def _should_continue(self, state: WorkerState) -> str:
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        return "end"
    
    async def process(self, task: str, thread_id: str = None) -> dict:
        """Process a task and return results."""
        thread_id = thread_id or f"{self.name}_default"
        config = {"configurable": {"thread_id": thread_id}}
        
        input_state = {
            "messages": [HumanMessage(content=task)],
            "task": task,
            "result": None,
        }
        
        tool_calls_made = []
        final_response = ""
        
        try:
            async for event in self.app.astream(input_state, config):
                for node_name, node_output in event.items():
                    if node_name == "tools":
                        for msg in node_output.get("messages", []):
                            if hasattr(msg, "name"):
                                tool_calls_made.append({
                                    "tool": msg.name,
                                    "result": msg.content[:500] if len(msg.content) > 500 else msg.content
                                })
                    elif node_name == "worker":
                        messages = node_output.get("messages", [])
                        if messages:
                            last_msg = messages[-1]
                            if isinstance(last_msg, AIMessage) and last_msg.content:
                                content = last_msg.content
                                if isinstance(content, str):
                                    final_response = content
                                elif isinstance(content, list):
                                    texts = [b.get("text", "") if isinstance(b, dict) else str(b) for b in content]
                                    final_response = "\n".join(texts)
            
            return {
                "worker": self.name,
                "response": final_response,
                "tool_calls": tool_calls_made,
            }
            
        except Exception as e:
            logger.error(f"{self.name} Worker error: {e}")
            return {"worker": self.name, "response": f"Error: {str(e)}", "tool_calls": []}


# =============================================================================
# MARKET DATA WORKER
# =============================================================================

def get_market_data_tools():
    """Get tools scoped to Market Data operations only."""
    
    @tool
    def get_quotes(
        symbol: Annotated[str, "Trading symbol"],
        exchange: Annotated[str, "Exchange code"]
    ) -> str:
        """Get real-time quotes for a symbol."""
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            result = client.quotes(symbol=symbol, exchange=exchange.upper())
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    @tool
    def get_market_depth(
        symbol: Annotated[str, "Trading symbol"],
        exchange: Annotated[str, "Exchange code"]
    ) -> str:
        """Get market depth (order book) data."""
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            result = client.depth(symbol=symbol, exchange=exchange.upper())
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    @tool
    def get_history(
        symbol: Annotated[str, "Trading symbol"],
        exchange: Annotated[str, "Exchange code"],
        interval: Annotated[str, "Interval: 1m, 5m, 15m, 30m, 60m, D"],
        start_date: Annotated[str, "Start date YYYY-MM-DD"],
        end_date: Annotated[str, "End date YYYY-MM-DD"]
    ) -> str:
        """Get historical OHLC data."""
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            df = client.history(
                symbol=symbol, exchange=exchange.upper(),
                interval=interval, start_date=start_date, end_date=end_date
            )
            if hasattr(df, 'tail'):
                data = df.tail(30).reset_index().to_dict(orient='records')
                return json.dumps({"status": "success", "count": len(data), "data": data}, default=str, indent=2)
            return json.dumps(df, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    @tool
    def search_symbols(
        query: Annotated[str, "Search query"],
        exchange: Annotated[str, "Exchange filter (optional)"] = ""
    ) -> str:
        """Search for symbols across exchanges."""
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            if exchange:
                result = client.search(query=query, exchange=exchange.upper())
            else:
                result = client.search(query=query)
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    @tool
    def get_symbol_info(
        symbol: Annotated[str, "Symbol to lookup"],
        exchange: Annotated[str, "Exchange code"]
    ) -> str:
        """Get symbol details (lot size, token, expiry)."""
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            result = client.symbol(symbol=symbol, exchange=exchange.upper())
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    @tool
    def get_expiry_dates(
        symbol: Annotated[str, "Underlying symbol"],
        exchange: Annotated[str, "Exchange"],
        instrument_type: Annotated[str, "futures or options"]
    ) -> str:
        """Get expiry dates for F&O."""
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            result = client.expiry(
                symbol=symbol, exchange=exchange.upper(),
                instrumenttype=instrument_type.lower()
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    @tool
    def get_intervals() -> str:
        """Get supported time intervals."""
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            result = client.intervals()
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    return [get_quotes, get_market_depth, get_history, search_symbols, 
            get_symbol_info, get_expiry_dates, get_intervals]


class MarketDataWorker(BaseWorker):
    """Market Data Worker - quotes, history, search, expiry ONLY."""
    
    def __init__(self):
        tools = get_market_data_tools()
        system_prompt = """You are the MARKET DATA WORKER - specialized in fetching market data.

## YOUR TOOLS
- get_quotes: Real-time quotes
- get_market_depth: Order book data
- get_history: Historical OHLC
- search_symbols: Symbol search
- get_symbol_info: Symbol details (lot size, token)
- get_expiry_dates: F&O expiry dates
- get_intervals: Supported timeframes

## YOUR CONSTRAINTS
- You can ONLY fetch data
- You CANNOT place orders
- You CANNOT modify anything
- You CANNOT access account data

## OUTPUT FORMAT
Return structured JSON with the requested data."""

        super().__init__("MarketData", tools, system_prompt)


# =============================================================================
# INDICATORS WORKER
# =============================================================================

def get_indicators_tools():
    """Get tools scoped to Indicator calculations only."""
    
    @tool
    def list_indicators(
        category: Annotated[str, "Category: trend, momentum, volatility, volume, all"] = "all"
    ) -> str:
        """List available indicators."""
        from tools.openalgo_indicators import INDICATORS_BY_CATEGORY, ALL_INDICATORS
        
        if category.lower() == "all":
            return json.dumps({"indicators": ALL_INDICATORS, "count": len(ALL_INDICATORS)})
        
        cat = category.lower()
        if cat in INDICATORS_BY_CATEGORY:
            return json.dumps({"category": cat, "indicators": INDICATORS_BY_CATEGORY[cat]})
        
        return json.dumps({"error": f"Unknown category: {category}"})
    
    @tool
    def validate_indicator(
        indicator: Annotated[str, "Indicator name to validate"]
    ) -> str:
        """Check if indicator is in the approved list."""
        from tools.openalgo_indicators import is_valid_indicator, get_indicator_category
        
        is_valid = is_valid_indicator(indicator)
        category = get_indicator_category(indicator) if is_valid else None
        
        return json.dumps({
            "indicator": indicator,
            "is_valid": is_valid,
            "category": category
        })
    
    @tool
    def calculate_indicator(
        indicator: Annotated[str, "Indicator name (RSI, EMA, MACD, etc.)"],
        ohlcv_json: Annotated[str, "JSON with OHLCV data"],
        period: Annotated[int, "Period/length"] = 14,
        extra_params: Annotated[str, "Additional params as JSON"] = "{}"
    ) -> str:
        """Calculate a technical indicator on OHLCV data."""
        from tools.openalgo_indicators import is_valid_indicator
        
        # MUST validate first
        if not is_valid_indicator(indicator):
            return json.dumps({
                "error": f"Invalid indicator: {indicator}",
                "hint": "Use validate_indicator first"
            })
        
        try:
            from openalgo import ta
            import pandas as pd
            
            ohlcv = json.loads(ohlcv_json)
            extra = json.loads(extra_params) if extra_params else {}
            
            df = pd.DataFrame({
                'open': ohlcv.get('open', []),
                'high': ohlcv.get('high', []),
                'low': ohlcv.get('low', []),
                'close': ohlcv.get('close', []),
                'volume': ohlcv.get('volume', [])
            })
            
            indicator_lower = indicator.lower()
            result = {}
            
            if indicator_lower in ["sma", "ema", "wma", "dema", "tema", "hma"]:
                func = getattr(ta, indicator_lower, None)
                if func:
                    values = func(df['close'], period)
                    result = {"values": values.tail(20).tolist()}
            elif indicator_lower == "rsi":
                values = ta.rsi(df['close'], period)
                result = {"values": values.tail(20).tolist()}
            elif indicator_lower == "macd":
                fast = extra.get('fast', 12)
                slow = extra.get('slow', 26)
                signal = extra.get('signal', 9)
                macd_line, signal_line, histogram = ta.macd(df['close'], fast, slow, signal)
                result = {
                    "macd": macd_line.tail(20).tolist(),
                    "signal": signal_line.tail(20).tolist(),
                    "histogram": histogram.tail(20).tolist()
                }
            elif indicator_lower == "atr":
                values = ta.atr(df['high'], df['low'], df['close'], period)
                result = {"values": values.tail(20).tolist()}
            elif indicator_lower == "bollingerbands":
                std_dev = extra.get('std_dev', 2)
                upper, middle, lower = ta.bollinger(df['close'], period, std_dev)
                result = {"upper": upper.tail(20).tolist(), "middle": middle.tail(20).tolist(), "lower": lower.tail(20).tolist()}
            else:
                func = getattr(ta, indicator_lower, None)
                if func:
                    try:
                        values = func(df['close'], period)
                        result = {"values": values.tail(20).tolist() if hasattr(values, 'tolist') else list(values)[-20:]}
                    except:
                        result = {"error": f"Could not calculate {indicator}"}
            
            result["indicator"] = indicator
            result["period"] = period
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    return [list_indicators, validate_indicator, calculate_indicator]


class IndicatorsWorker(BaseWorker):
    """Indicators Worker - technical analysis calculations ONLY."""
    
    def __init__(self):
        tools = get_indicators_tools()
        system_prompt = """You are the INDICATORS WORKER - specialized in technical analysis.

## YOUR TOOLS
- list_indicators: List available indicators by category
- validate_indicator: Check if indicator is valid (MUST call before calculate)
- calculate_indicator: Calculate indicator on OHLCV data

## WORKFLOW REQUIREMENT
You MUST call validate_indicator BEFORE calculate_indicator.

## YOUR CONSTRAINTS
- You can ONLY calculate indicators
- You CANNOT fetch market data (request from MarketData worker)
- You CANNOT place orders
- You CANNOT access accounts

## OUTPUT FORMAT
Return calculated indicator values as JSON."""

        super().__init__("Indicators", tools, system_prompt)


# =============================================================================
# OPTIONS WORKER
# =============================================================================

def get_options_tools():
    """Get tools scoped to Options operations only."""
    
    @tool
    def get_option_greeks(
        symbol: Annotated[str, "Option symbol"],
        exchange: Annotated[str, "Exchange (NFO, BFO, MCX)"],
        interest_rate: Annotated[float, "Interest rate"] = 6.5
    ) -> str:
        """Calculate Option Greeks (Delta, Gamma, Theta, Vega, Rho, IV)."""
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            result = client.optiongreeks(
                symbol=symbol, exchange=exchange.upper(),
                interest_rate=interest_rate
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    @tool
    def get_option_symbol(
        underlying: Annotated[str, "Underlying symbol (NIFTY, BANKNIFTY)"],
        exchange: Annotated[str, "Exchange"],
        expiry_date: Annotated[str, "Expiry date (28NOV24)"],
        strike_int: Annotated[int, "Strike interval"],
        offset: Annotated[str, "ATM, ITM1-50, OTM1-50"],
        option_type: Annotated[str, "CE or PE"]
    ) -> str:
        """Resolve option symbol based on offset."""
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            result = client.optionsymbol(
                underlying=underlying, exchange=exchange.upper(),
                expiry_date=expiry_date, strike_int=strike_int,
                offset=offset.upper(), option_type=option_type.upper()
            )
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    @tool
    def analyze_option_chain(
        underlying: Annotated[str, "Underlying symbol"],
        exchange: Annotated[str, "Exchange"],
        expiry_date: Annotated[str, "Expiry date"]
    ) -> str:
        """Analyze option chain for strategy opportunities."""
        # This would typically fetch and analyze multiple strikes
        return json.dumps({
            "status": "analysis",
            "underlying": underlying,
            "exchange": exchange,
            "expiry": expiry_date,
            "note": "Use get_option_symbol with different offsets to build chain"
        })
    
    return [get_option_greeks, get_option_symbol, analyze_option_chain]


class OptionsWorker(BaseWorker):
    """Options Worker - Greeks, symbol resolution ONLY."""
    
    def __init__(self):
        tools = get_options_tools()
        system_prompt = """You are the OPTIONS WORKER - specialized in options analysis.

## YOUR TOOLS
- get_option_greeks: Calculate Greeks (Delta, Gamma, Theta, Vega, IV)
- get_option_symbol: Resolve ATM/ITM/OTM symbols
- analyze_option_chain: Analyze chain for opportunities

## YOUR CONSTRAINTS
- You can ONLY analyze options
- You CANNOT place option orders (that goes to Executor)
- You CANNOT access account data

## OUTPUT FORMAT
Return Greeks and symbol info as JSON."""

        super().__init__("Options", tools, system_prompt)


# =============================================================================
# ACCOUNTS WORKER
# =============================================================================

def get_accounts_tools():
    """Get tools scoped to Account information only."""
    
    @tool
    def get_funds() -> str:
        """Get available funds and margin."""
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            result = client.funds()
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    @tool
    def get_positions() -> str:
        """Get current open positions."""
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            result = client.positionbook()
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    @tool
    def get_holdings() -> str:
        """Get stock holdings."""
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            result = client.holdings()
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    @tool
    def get_orderbook() -> str:
        """Get all orders for the day."""
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            result = client.orderbook()
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    @tool
    def get_tradebook() -> str:
        """Get executed trades."""
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            result = client.tradebook()
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    @tool
    def calculate_margin(
        positions_json: Annotated[str, "JSON array of positions to calculate margin for"]
    ) -> str:
        """Calculate margin requirement for positions."""
        try:
            positions = json.loads(positions_json)
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            result = client.margin(positions=positions)
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    @tool
    def get_analyzer_status() -> str:
        """Check if in live or analyze (simulated) mode."""
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            result = client.analyzerstatus()
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    return [get_funds, get_positions, get_holdings, get_orderbook, 
            get_tradebook, calculate_margin, get_analyzer_status]


class AccountsWorker(BaseWorker):
    """Accounts Worker - funds, positions, holdings ONLY."""
    
    def __init__(self):
        tools = get_accounts_tools()
        system_prompt = """You are the ACCOUNTS WORKER - specialized in account information.

## YOUR TOOLS
- get_funds: Available cash and margins
- get_positions: Open positions
- get_holdings: Stock holdings
- get_orderbook: Today's orders
- get_tradebook: Executed trades
- calculate_margin: Margin requirement calculator
- get_analyzer_status: Check live/simulate mode

## YOUR CONSTRAINTS
- You can ONLY read account data
- You CANNOT place orders
- You CANNOT modify positions
- You CANNOT toggle analyzer mode (that's Supervisor)

## OUTPUT FORMAT
Return account data as JSON."""

        super().__init__("Accounts", tools, system_prompt)


# =============================================================================
# WORKER REGISTRY
# =============================================================================

class WorkerRegistry:
    """Registry to manage all worker agents."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._workers = {}
            cls._instance._initialized = False
        return cls._instance
    
    def initialize(self):
        """Initialize all workers."""
        if self._initialized:
            return
        
        self._workers = {
            "market_data": MarketDataWorker(),
            "indicators": IndicatorsWorker(),
            "options": OptionsWorker(),
            "accounts": AccountsWorker(),
        }
        self._initialized = True
        logger.info(f"Worker Registry initialized with {len(self._workers)} workers")
    
    def get_worker(self, name: str) -> Optional[BaseWorker]:
        """Get a worker by name."""
        if not self._initialized:
            self.initialize()
        return self._workers.get(name)
    
    def list_workers(self) -> dict:
        """List all available workers and their tools."""
        if not self._initialized:
            self.initialize()
        
        return {
            name: {
                "tools": [t.name for t in worker.tools],
                "tool_count": len(worker.tools)
            }
            for name, worker in self._workers.items()
        }


def get_worker_registry() -> WorkerRegistry:
    """Get the global worker registry."""
    registry = WorkerRegistry()
    registry.initialize()
    return registry
