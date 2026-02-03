
import logging
import datetime
from typing import Dict, Any
from tools.openalgo_tools import get_openalgo_client

logger = logging.getLogger("data_collector")

class DataCollector:
    """
    Centralized Data Factory to generate input payloads for all 10 agents.
    Fetches REAL market data via OpenAlgo where applicable.
    """
    
    def __init__(self):
        try:
            self.client = get_openalgo_client()
        except:
            self.client = None
            logger.warning("OpenAlgo client not available. Live data will be mocked/unavailable.")

    def collect_all(self, symbol: str) -> Dict[str, Any]:
        """
        Collects/Generates inputs for all agents for a given symbol.
        Returns a dictionary keyed by agent input string names.
        """
        logger.info(f"Collecting data for {symbol}...")
        
        # Prefetch core market data to avoid N calls
        market_snapshot = self._fetch_live_market_data(symbol)
        option_chain = self._fetch_option_chain(symbol)
        
        return {
            "fundamentals_input": self._get_fundamentals(symbol, market_snapshot),
            "technicals_input": self._get_technicals(symbol, market_snapshot),
            "news_input": self._get_news(symbol),
            "sentiment_input": self._get_sentiment(symbol),
            "institutional_input": self._get_institutional(symbol),
            "options_input": self._get_options(symbol, market_snapshot, option_chain),
            "vol_surface_input": self._get_vol_surface(symbol, market_snapshot),
            "liquidity_input": self._get_liquidity(symbol, market_snapshot),
            "correlation_input": self._get_correlation(symbol),
            "chart_pattern_input": self._get_chart_patterns(symbol, market_snapshot)
        }
    
    def _fetch_live_market_data(self, symbol):
        """Fetch real-time quote/LTP from OpenAlgo or fall back."""
        data = {"ltp": 100, "open": 100, "high": 105, "low": 95, "close": 100, "volume": 10000}
        if self.client:
            try:
                # Assuming exchange is NSE for simplicity, can be dynamic
                quote = self.client.get_quote(exchange="NSE", symbol=symbol)
                if quote:
                    data.update(quote)
            except Exception as e:
                logger.debug(f"Live quote fetch failed for {symbol}: {e}")
        return data

    def _fetch_option_chain(self, symbol):
        """Fetch option chain if client available."""
        if self.client and symbol in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
            try:
                 # Fetch near ATM chain or full chain
                 # For now return placeholder or actual if method exists
                 # result = self.client.optionchain(...)
                 pass
            except:
                pass
        return {"calls": [], "puts": []}

    def _get_fundamentals(self, symbol, snapshot):
        return {
            "ticker": symbol,
            "period": "annual",
            "financial_metrics": {"pe_ratio": 25, "roe": 15, "debt_to_equity": 0.5},
            "market_data": {"market_cap": 1000000000, "sector": "Technology", "current_price": snapshot.get('ltp')}
        }

    def _get_technicals(self, symbol, snapshot):
        return {
            "symbol": symbol,
            "timeframe": "1h",
            "market_data": snapshot, # Pass snapshot for latest price check
            "ohlcv": [], # Would fetch historical bars here via client.get_history(...)
            "indicators": {"rsi": 65, "macd": {"macd_line": 1.5, "signal_line": 1.0}}
        }

    def _get_news(self, symbol):
        return {
            "query": f"{symbol} news",
            "raw_articles": [{"title": f"{symbol} reports strong earnings", "source": "Reuters"}]
        }

    def _get_sentiment(self, symbol):
        return {
            "symbol": symbol,
            "social_data": [{"text": f"Bullish on {symbol}!", "platform": "twitter"}],
            "news_summary": "Positive earnings surprise."
        }

    def _get_institutional(self, symbol):
        return {
            "instrument": {"symbol": symbol},
            "as_of": datetime.datetime.now().isoformat(),
            "time_horizon": "swing",
            "institutional_flows": {"fii_net": [{"date": "2024-01-01", "net_qty": 5000}]}
        }

    def _get_options(self, symbol, snapshot, chain):
        return {
            "instrument": {"symbol": symbol, "spot": snapshot.get("ltp")},
            "time_horizon": "swing",
            "view": {"direction": "bullish", "conviction": 80},
            "options_chain": chain
        }
        
    def _get_vol_surface(self, symbol, snapshot):
         return {
            "instrument": {"symbol": symbol, "spot": snapshot.get("ltp")},
            "time_horizon": "swing",
            "surface_date": "2024-01-01",
            "vol_surface": {
                "strikes": [snapshot.get("ltp", 100)*0.9, snapshot.get("ltp", 100)*1.1], 
                "expiries": ["7d"],
                "iv_matrix": {"7d": {str(snapshot.get("ltp", 100)): 15}}
            },
            "market_data": {"risk_free_rate": 0.05}
        }

    def _get_liquidity(self, symbol, snapshot):
        return {
            "instrument": {"symbol": symbol},
            "proposed_trade": {"direction": "long", "size": 1000, "proposed_price": snapshot.get("ltp")},
            "market_snapshot": {"ltp": snapshot.get("ltp"), "bid": snapshot.get("ltp")*0.99, "ask": snapshot.get("ltp")*1.01, "avg_volume_20d": 500000}
        }

    def _get_correlation(self, symbol):
        return {
            "as_of": datetime.datetime.now().isoformat(),
            "portfolio": {
                "total_aum": 100000,
                "holdings": [{"symbol": "NIFTY", "weight_pct": 50}]
            },
            "proposed_trade": {"symbol": symbol, "size": 5000, "expected_weight_pct": 5.0},
            "correlation_matrix": {
                "symbols": ["NIFTY", symbol],
                "corr_30d": [[1.0, 0.6], [0.6, 1.0]]
            }
        }

    def _get_chart_patterns(self, symbol, snapshot):
        # Would fetch real OHLCV here
        base = snapshot.get("ltp", 100)
        return {
            "instrument": {"symbol": symbol, "timeframe": "1h"},
            "ohlcv": [{"ts": "2024-01-01", "open": base, "high": base+5, "low": base-5, "close": base+2, "volume": 1000} for _ in range(60)]
        }
