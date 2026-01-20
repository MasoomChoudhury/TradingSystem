"""
Efficiency Optimizations - Latency, Token, and Cost Reduction

Implements:
1. Parallel execution of independent reads
2. Caching for symbol info, expiry, intervals, history
3. Batch operations to reduce tool chatter
4. Dynamic tool selection to reduce prompt size

Goal: Fewer LLM calls, fewer tool calls, lower time-to-trade.
"""
import os
import json
import asyncio
import logging
import hashlib
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass
from functools import wraps
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_DB = os.path.join(os.path.dirname(__file__), "cache.db")


# =============================================================================
# CACHING SYSTEM
# =============================================================================

@dataclass
class CacheConfig:
    """Cache configuration for different data types."""
    ttl_seconds: int
    max_entries: int = 1000


# Default cache TTLs
CACHE_CONFIGS = {
    "symbol_info": CacheConfig(ttl_seconds=86400),      # 24 hours
    "expiry_dates": CacheConfig(ttl_seconds=3600),       # 1 hour
    "intervals": CacheConfig(ttl_seconds=86400),         # 24 hours
    "history_daily": CacheConfig(ttl_seconds=3600),      # 1 hour
    "history_intraday": CacheConfig(ttl_seconds=60),     # 1 minute
    "quotes": CacheConfig(ttl_seconds=5),                # 5 seconds
    "funds": CacheConfig(ttl_seconds=30),                # 30 seconds
    "positions": CacheConfig(ttl_seconds=10),            # 10 seconds
}


class DataCache:
    """
    High-performance cache for market data.
    Reduces API calls and improves latency.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_db()
            cls._instance._memory_cache = {}  # Fast in-memory layer
        return cls._instance
    
    def _init_db(self):
        conn = sqlite3.connect(CACHE_DB)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                cache_key TEXT PRIMARY KEY,
                cache_type TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                hit_count INTEGER DEFAULT 0
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_expires ON cache_entries(expires_at)")
        conn.commit()
        conn.close()
    
    def _generate_key(self, cache_type: str, params: dict) -> str:
        """Generate unique cache key."""
        param_str = json.dumps(params, sort_keys=True)
        return f"{cache_type}:{hashlib.md5(param_str.encode()).hexdigest()[:16]}"
    
    def get(self, cache_type: str, params: dict) -> Optional[Any]:
        """Get cached value if valid."""
        key = self._generate_key(cache_type, params)
        
        # Check memory cache first (fastest)
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if datetime.now() < entry["expires_at"]:
                return entry["data"]
            else:
                del self._memory_cache[key]
        
        # Check DB cache
        conn = sqlite3.connect(CACHE_DB)
        row = conn.execute(
            "SELECT data, expires_at FROM cache_entries WHERE cache_key = ?", (key,)
        ).fetchone()
        
        if row:
            expires = datetime.fromisoformat(row[1])
            if datetime.now() < expires:
                # Update hit count
                conn.execute(
                    "UPDATE cache_entries SET hit_count = hit_count + 1 WHERE cache_key = ?", (key,)
                )
                conn.commit()
                conn.close()
                
                data = json.loads(row[0])
                # Populate memory cache
                self._memory_cache[key] = {"data": data, "expires_at": expires}
                return data
        
        conn.close()
        return None
    
    def set(self, cache_type: str, params: dict, data: Any) -> None:
        """Set cached value."""
        config = CACHE_CONFIGS.get(cache_type, CacheConfig(ttl_seconds=60))
        key = self._generate_key(cache_type, params)
        expires = datetime.now() + timedelta(seconds=config.ttl_seconds)
        
        # Set in memory cache
        self._memory_cache[key] = {"data": data, "expires_at": expires}
        
        # Persist to DB
        conn = sqlite3.connect(CACHE_DB)
        conn.execute("""
            INSERT OR REPLACE INTO cache_entries (cache_key, cache_type, data, expires_at)
            VALUES (?, ?, ?, ?)
        """, (key, cache_type, json.dumps(data), expires.isoformat()))
        conn.commit()
        conn.close()
    
    def invalidate(self, cache_type: str = None, params: dict = None) -> int:
        """Invalidate cache entries."""
        conn = sqlite3.connect(CACHE_DB)
        
        if cache_type and params:
            key = self._generate_key(cache_type, params)
            cursor = conn.execute("DELETE FROM cache_entries WHERE cache_key = ?", (key,))
            if key in self._memory_cache:
                del self._memory_cache[key]
        elif cache_type:
            cursor = conn.execute("DELETE FROM cache_entries WHERE cache_type = ?", (cache_type,))
            self._memory_cache = {k: v for k, v in self._memory_cache.items() if not k.startswith(cache_type)}
        else:
            cursor = conn.execute("DELETE FROM cache_entries")
            self._memory_cache.clear()
        
        count = cursor.rowcount
        conn.commit()
        conn.close()
        return count
    
    def cleanup_expired(self) -> int:
        """Remove expired entries."""
        conn = sqlite3.connect(CACHE_DB)
        cursor = conn.execute(
            "DELETE FROM cache_entries WHERE expires_at < ?", (datetime.now().isoformat(),)
        )
        count = cursor.rowcount
        conn.commit()
        conn.close()
        
        # Cleanup memory cache
        now = datetime.now()
        self._memory_cache = {
            k: v for k, v in self._memory_cache.items() 
            if v["expires_at"] > now
        }
        
        return count
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        conn = sqlite3.connect(CACHE_DB)
        total = conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]
        by_type = conn.execute("""
            SELECT cache_type, COUNT(*), SUM(hit_count) 
            FROM cache_entries GROUP BY cache_type
        """).fetchall()
        conn.close()
        
        return {
            "total_entries": total,
            "memory_entries": len(self._memory_cache),
            "by_type": {row[0]: {"count": row[1], "hits": row[2]} for row in by_type}
        }


def get_cache() -> DataCache:
    return DataCache()


def cached(cache_type: str):
    """Decorator to cache function results."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            params = {"args": args, "kwargs": kwargs}
            
            # Check cache
            cached_result = cache.get(cache_type, params)
            if cached_result is not None:
                logger.debug(f"Cache HIT: {cache_type}")
                return cached_result
            
            # Execute and cache
            result = func(*args, **kwargs)
            cache.set(cache_type, params, result)
            logger.debug(f"Cache SET: {cache_type}")
            return result
        
        return wrapper
    return decorator


# =============================================================================
# PARALLEL EXECUTION
# =============================================================================

async def parallel_reads(functions: List[Callable], timeout: float = 10.0) -> List[Any]:
    """
    Execute multiple read functions in parallel.
    
    Example:
        results = await parallel_reads([
            get_funds,
            get_positions,
            lambda: get_quotes("RELIANCE", "NSE"),
            lambda: get_symbol_info("RELIANCE", "NSE")
        ])
    """
    async def run_sync(func):
        """Run sync function in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func)
    
    tasks = [asyncio.create_task(run_sync(func)) for func in functions]
    
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=timeout
        )
        return results
    except asyncio.TimeoutError:
        # Cancel remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        logger.warning(f"Parallel reads timed out after {timeout}s")
        return [None] * len(functions)


class ParallelDataFetcher:
    """
    Fetches multiple data points in parallel.
    Optimized for pre-trade data collection.
    """
    
    async def fetch_pre_trade_data(self, symbol: str, exchange: str) -> dict:
        """
        Fetch all data needed before trade execution in parallel.
        
        Returns funds, positions, quotes, symbol info concurrently.
        """
        cache = get_cache()
        
        # Check cache first
        cached_results = {}
        functions_to_run = []
        function_keys = []
        
        data_requests = [
            ("funds", "funds", {}),
            ("positions", "positions", {}),
            ("quotes", "quotes", {"symbol": symbol, "exchange": exchange}),
            ("symbol_info", "symbol_info", {"symbol": symbol, "exchange": exchange}),
        ]
        
        for key, cache_type, params in data_requests:
            cached = cache.get(cache_type, params)
            if cached:
                cached_results[key] = cached
            else:
                function_keys.append((key, cache_type, params))
                functions_to_run.append(self._get_fetch_function(key, symbol, exchange))
        
        # Fetch missing data in parallel
        if functions_to_run:
            start = time.time()
            results = await parallel_reads(functions_to_run)
            elapsed = time.time() - start
            logger.info(f"Parallel fetch of {len(functions_to_run)} items in {elapsed:.2f}s")
            
            # Cache results
            for i, (key, cache_type, params) in enumerate(function_keys):
                if results[i] and not isinstance(results[i], Exception):
                    cached_results[key] = results[i]
                    cache.set(cache_type, params, results[i])
        
        return cached_results
    
    def _get_fetch_function(self, key: str, symbol: str, exchange: str) -> Callable:
        """Get the fetch function for a data type."""
        def fetch_funds():
            try:
                from tools.openalgo_tools import get_openalgo_client
                return get_openalgo_client().funds()
            except:
                return None
        
        def fetch_positions():
            try:
                from tools.openalgo_tools import get_openalgo_client
                return get_openalgo_client().positionbook()
            except:
                return None
        
        def fetch_quotes():
            try:
                from tools.openalgo_tools import get_openalgo_client
                return get_openalgo_client().quotes(symbol=symbol, exchange=exchange)
            except:
                return None
        
        def fetch_symbol_info():
            try:
                from tools.openalgo_tools import get_openalgo_client
                return get_openalgo_client().symbol(symbol=symbol, exchange=exchange)
            except:
                return None
        
        functions = {
            "funds": fetch_funds,
            "positions": fetch_positions,
            "quotes": fetch_quotes,
            "symbol_info": fetch_symbol_info,
        }
        return functions.get(key, lambda: None)


def get_parallel_fetcher() -> ParallelDataFetcher:
    return ParallelDataFetcher()


# =============================================================================
# BATCH OPERATIONS
# =============================================================================

class BatchOperations:
    """
    Batch operations to reduce tool chatter.
    Combines multiple calls into single operations.
    """
    
    def batch_get_quotes(self, symbols: List[tuple]) -> Dict[str, dict]:
        """
        Get quotes for multiple symbols at once.
        
        Args:
            symbols: List of (symbol, exchange) tuples
        
        Returns:
            Dict mapping symbol to quote data
        """
        results = {}
        cache = get_cache()
        
        # Check cache first, collect uncached
        uncached = []
        for symbol, exchange in symbols:
            params = {"symbol": symbol, "exchange": exchange}
            cached = cache.get("quotes", params)
            if cached:
                results[symbol] = cached
            else:
                uncached.append((symbol, exchange))
        
        # Fetch uncached in parallel
        if uncached:
            async def fetch_all():
                def make_fetcher(sym, exch):
                    def fetch():
                        try:
                            from tools.openalgo_tools import get_openalgo_client
                            return get_openalgo_client().quotes(symbol=sym, exchange=exch)
                        except:
                            return None
                    return fetch
                
                functions = [make_fetcher(s, e) for s, e in uncached]
                return await parallel_reads(functions)
            
            fetched = asyncio.run(fetch_all())
            
            for i, (symbol, exchange) in enumerate(uncached):
                if fetched[i]:
                    results[symbol] = fetched[i]
                    cache.set("quotes", {"symbol": symbol, "exchange": exchange}, fetched[i])
        
        return results
    
    def batch_get_symbol_info(self, symbols: List[tuple]) -> Dict[str, dict]:
        """Get symbol info for multiple symbols."""
        results = {}
        cache = get_cache()
        
        for symbol, exchange in symbols:
            params = {"symbol": symbol, "exchange": exchange}
            cached = cache.get("symbol_info", params)
            if cached:
                results[symbol] = cached
            else:
                try:
                    from tools.openalgo_tools import get_openalgo_client
                    data = get_openalgo_client().symbol(symbol=symbol, exchange=exchange)
                    results[symbol] = data
                    cache.set("symbol_info", params, data)
                except:
                    pass
        
        return results
    
    def batch_calculate_indicators(
        self, 
        ohlcv_data: dict, 
        indicators: List[dict]
    ) -> Dict[str, Any]:
        """
        Calculate multiple indicators in one pass.
        
        Args:
            ohlcv_data: OHLCV data
            indicators: List of {"name": "RSI", "period": 14}
        
        Returns:
            Dict mapping indicator name to values
        """
        results = {}
        
        try:
            from openalgo import ta
            import pandas as pd
            
            df = pd.DataFrame(ohlcv_data)
            
            for ind in indicators:
                name = ind.get("name", "").lower()
                period = ind.get("period", 14)
                
                try:
                    if name in ["sma", "ema", "wma"]:
                        func = getattr(ta, name)
                        values = func(df['close'], period)
                        results[name] = values.tail(20).tolist()
                    elif name == "rsi":
                        values = ta.rsi(df['close'], period)
                        results["rsi"] = values.tail(20).tolist()
                    elif name == "macd":
                        macd, signal, hist = ta.macd(df['close'], 12, 26, 9)
                        results["macd"] = {
                            "macd": macd.tail(20).tolist(),
                            "signal": signal.tail(20).tolist(),
                            "histogram": hist.tail(20).tolist()
                        }
                    elif name == "atr":
                        values = ta.atr(df['high'], df['low'], df['close'], period)
                        results["atr"] = values.tail(20).tolist()
                except Exception as e:
                    results[name] = {"error": str(e)}
            
        except ImportError:
            results["error"] = "openalgo library not available"
        
        return results


def get_batch_ops() -> BatchOperations:
    return BatchOperations()


# =============================================================================
# DYNAMIC TOOL SELECTION
# =============================================================================

class ToolSelector:
    """
    Dynamically select relevant tools to reduce prompt size.
    Improves tool-choice accuracy by limiting options.
    """
    
    # Tool categories
    TOOL_CATEGORIES = {
        "market_data": [
            "get_quotes", "get_market_depth", "get_history",
            "search_symbols", "get_symbol_info", "get_expiry_dates", "get_intervals"
        ],
        "indicators": [
            "list_indicators", "validate_indicator", "calculate_indicator"
        ],
        "options": [
            "get_option_greeks", "get_option_symbol", "analyze_option_chain"
        ],
        "accounts": [
            "get_funds", "get_positions", "get_holdings",
            "get_orderbook", "get_tradebook", "calculate_margin", "get_analyzer_status"
        ],
        "execution": [
            "place_order", "place_smart_order", "place_basket_order", "place_split_order"
        ],
        "management": [
            "get_order_status", "modify_order", "cancel_order", "get_open_position"
        ],
        "emergency": [
            "emergency_cancel_all", "emergency_close_all",
            "trip_circuit_breaker", "reset_circuit_breaker"
        ],
        "validation": [
            "validate_policy", "validate_risk", "validate_data_integrity"
        ]
    }
    
    # Intent to category mapping
    INTENT_CATEGORIES = {
        "price": ["market_data"],
        "quote": ["market_data"],
        "history": ["market_data"],
        "indicator": ["indicators"],
        "rsi": ["indicators"],
        "macd": ["indicators"],
        "option": ["options"],
        "greeks": ["options"],
        "fund": ["accounts"],
        "position": ["accounts"],
        "holding": ["accounts"],
        "margin": ["accounts"],
        "buy": ["execution", "validation"],
        "sell": ["execution", "validation"],
        "order": ["execution", "management"],
        "modify": ["management"],
        "cancel": ["management"],
        "emergency": ["emergency"],
        "close all": ["emergency"],
    }
    
    def select_tools(self, query: str, max_tools: int = 10) -> List[str]:
        """
        Select relevant tools based on query intent.
        Returns subset of tools to include in prompt.
        """
        query_lower = query.lower()
        selected_categories = set()
        
        # Find matching categories
        for keyword, categories in self.INTENT_CATEGORIES.items():
            if keyword in query_lower:
                selected_categories.update(categories)
        
        # Default to market_data if no match
        if not selected_categories:
            selected_categories.add("market_data")
        
        # Collect tools from selected categories
        tools = []
        for category in selected_categories:
            tools.extend(self.TOOL_CATEGORIES.get(category, []))
        
        # Deduplicate and limit
        tools = list(dict.fromkeys(tools))[:max_tools]
        
        logger.debug(f"Selected {len(tools)} tools for query: {query[:50]}...")
        return tools
    
    def get_category_tools(self, category: str) -> List[str]:
        """Get all tools for a category."""
        return self.TOOL_CATEGORIES.get(category, [])
    
    def estimate_token_savings(self, full_tools: int, selected_tools: int) -> dict:
        """Estimate token savings from tool selection."""
        # Approximate: each tool description ~100 tokens
        tokens_per_tool = 100
        saved = (full_tools - selected_tools) * tokens_per_tool
        
        return {
            "full_tools": full_tools,
            "selected_tools": selected_tools,
            "tokens_saved": saved,
            "reduction_pct": ((full_tools - selected_tools) / full_tools * 100) if full_tools > 0 else 0
        }


def get_tool_selector() -> ToolSelector:
    return ToolSelector()


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

class PerformanceTracker:
    """Track efficiency metrics."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._metrics = {
                "cache_hits": 0,
                "cache_misses": 0,
                "parallel_fetches": 0,
                "total_latency_ms": 0,
                "tool_calls": 0,
                "llm_calls": 0,
            }
        return cls._instance
    
    def record_cache_hit(self):
        self._metrics["cache_hits"] += 1
    
    def record_cache_miss(self):
        self._metrics["cache_misses"] += 1
    
    def record_latency(self, ms: float):
        self._metrics["total_latency_ms"] += ms
    
    def record_tool_call(self):
        self._metrics["tool_calls"] += 1
    
    def record_llm_call(self):
        self._metrics["llm_calls"] += 1
    
    def get_metrics(self) -> dict:
        m = self._metrics
        total_cache = m["cache_hits"] + m["cache_misses"]
        hit_rate = (m["cache_hits"] / total_cache * 100) if total_cache > 0 else 0
        
        return {
            **m,
            "cache_hit_rate": f"{hit_rate:.1f}%",
            "avg_latency_ms": m["total_latency_ms"] / max(m["tool_calls"], 1)
        }
    
    def reset(self):
        for key in self._metrics:
            self._metrics[key] = 0


def get_perf_tracker() -> PerformanceTracker:
    return PerformanceTracker()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "CacheConfig",
    "CACHE_CONFIGS",
    "DataCache",
    "get_cache",
    "cached",
    "parallel_reads",
    "ParallelDataFetcher",
    "get_parallel_fetcher",
    "BatchOperations",
    "get_batch_ops",
    "ToolSelector",
    "get_tool_selector",
    "PerformanceTracker",
    "get_perf_tracker",
]
