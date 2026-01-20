"""
OpenAlgo Technical Indicators Tools for LangGraph Agents

Provides indicator calculation tools using the OpenAlgo ta library.
"""
import json
import logging
from typing import Annotated, List, Dict, Any, Optional
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# =============================================================================
# APPROVED INDICATORS REGISTRY
# =============================================================================

TREND_INDICATORS = [
    "SMA", "EMA", "WMA", "DEMA", "TEMA", "HMA", "VWMA", "ALMA", "KAMA",
    "ZLEMA", "T3", "FRAMA", "TRIMA", "McGinley", "VIDYA", "Alligator",
    "MovingAverageEnvelopes", "Supertrend", "Ichimoku", "ChandeKrollStop"
]

MOMENTUM_INDICATORS = [
    "RSI", "MACD", "Stochastic", "CCI", "WilliamsR", "BOP",
    "ElderRay", "Fisher", "CRSI"
]

VOLATILITY_INDICATORS = [
    "ATR", "BollingerBands", "Keltner", "Donchian", "Chaikin", "NATR",
    "RVI", "ULTOSC", "TRANGE", "MASS", "BBPercent", "BBWidth",
    "ChandelierExit", "HistoricalVolatility", "UlcerIndex", "STARC"
]

VOLUME_INDICATORS = [
    "OBV", "OBVSmoothed", "VWAP", "MFI", "ADL", "CMF", "EMV", "FI",
    "NVI", "PVI", "VOLOSC", "VROC", "KlingerVolumeOscillator",
    "PriceVolumeTrend", "RVOL"
]

OSCILLATORS = [
    "ROC", "CMO", "TRIX", "UO", "AO", "AC", "PPO", "PO", "DPO",
    "AROONOSC", "StochRSI", "RVI", "CHO", "CHOP", "KST", "TSI",
    "VI", "STC", "GatorOscillator", "Coppock"
]

STATISTICAL_INDICATORS = [
    "LINREG", "LRSLOPE", "CORREL", "BETA", "VAR", "TSF",
    "MEDIAN", "MedianBands", "MODE"
]

HYBRID_INDICATORS = [
    "ADX", "Aroon", "PivotPoints", "SAR", "DMI", "WilliamsFractals", "RWI"
]

UTILITY_FUNCTIONS = [
    "crossover", "crossunder", "highest", "lowest", "change", "roc",
    "stdev", "exrem", "flip", "valuewhen", "rising", "falling", "cross"
]

# All approved indicators
ALL_INDICATORS = (
    TREND_INDICATORS + MOMENTUM_INDICATORS + VOLATILITY_INDICATORS +
    VOLUME_INDICATORS + OSCILLATORS + STATISTICAL_INDICATORS +
    HYBRID_INDICATORS
)

# Categorized dictionary for easy lookup
INDICATORS_BY_CATEGORY = {
    "trend": TREND_INDICATORS,
    "momentum": MOMENTUM_INDICATORS,
    "volatility": VOLATILITY_INDICATORS,
    "volume": VOLUME_INDICATORS,
    "oscillators": OSCILLATORS,
    "statistical": STATISTICAL_INDICATORS,
    "hybrid": HYBRID_INDICATORS,
    "utility": UTILITY_FUNCTIONS
}

# Most commonly used indicators for quick reference
COMMON_INDICATORS = [
    "SMA", "EMA", "RSI", "MACD", "ATR", "BollingerBands",
    "Supertrend", "ADX", "VWAP", "OBV", "Stochastic"
]


def is_valid_indicator(indicator_name: str) -> bool:
    """Check if an indicator is in the approved list."""
    return indicator_name in ALL_INDICATORS or indicator_name in UTILITY_FUNCTIONS


def get_indicator_category(indicator_name: str) -> Optional[str]:
    """Get the category of an indicator."""
    for category, indicators in INDICATORS_BY_CATEGORY.items():
        if indicator_name in indicators:
            return category
    return None


def _safe_tail(values, n: int = 20) -> list:
    """Safely convert indicator values to list, handling various types."""
    try:
        if hasattr(values, 'tail'):
            # pandas Series
            return values.tail(n).tolist()
        elif hasattr(values, 'tolist'):
            # numpy array
            return values[-n:].tolist()
        elif isinstance(values, list):
            return values[-n:]
        else:
            return list(values)[-n:]
    except Exception:
        return []


# =============================================================================
# INDICATOR TOOLS
# =============================================================================

@tool
def openalgo_list_indicators(
    category: Annotated[str, "Category: trend, momentum, volatility, volume, oscillators, statistical, hybrid, utility, or 'all'"] = "all"
) -> str:
    """
    List available technical indicators by category.
    Use this to discover what indicators are available for analysis.
    
    Categories:
    - trend: Moving averages and trend-following indicators
    - momentum: RSI, MACD, and momentum-based indicators
    - volatility: ATR, Bollinger Bands, volatility measures
    - volume: OBV, VWAP, and volume-based indicators
    - oscillators: ROC, CMO, and oscillator indicators
    - statistical: Regression, correlation, statistical measures
    - hybrid: ADX, SAR, and multi-purpose indicators
    - utility: Helper functions (crossover, highest, lowest, etc.)
    - all: List all indicators
    
    Returns:
        JSON list of indicator names.
    """
    category = category.lower()
    
    if category == "all":
        return json.dumps({
            "total": len(ALL_INDICATORS),
            "indicators": ALL_INDICATORS,
            "utilities": UTILITY_FUNCTIONS
        }, indent=2)
    elif category in INDICATORS_BY_CATEGORY:
        indicators = INDICATORS_BY_CATEGORY[category]
        return json.dumps({
            "category": category,
            "count": len(indicators),
            "indicators": indicators
        }, indent=2)
    else:
        return json.dumps({
            "error": f"Unknown category: {category}",
            "valid_categories": list(INDICATORS_BY_CATEGORY.keys()) + ["all"]
        })


@tool
def openalgo_get_common_indicators() -> str:
    """
    Get the most commonly used technical indicators.
    These are recommended for typical trading analysis.
    
    Returns:
        JSON list of common indicator names with descriptions.
    """
    common_info = {
        "SMA": "Simple Moving Average - Basic trend indicator",
        "EMA": "Exponential Moving Average - Faster trend response",
        "RSI": "Relative Strength Index - Overbought/oversold levels",
        "MACD": "Moving Average Convergence Divergence - Trend and momentum",
        "ATR": "Average True Range - Volatility measurement",
        "BollingerBands": "Volatility bands around moving average",
        "Supertrend": "Trend-following indicator with stop levels",
        "ADX": "Average Directional Index - Trend strength",
        "VWAP": "Volume Weighted Average Price - Fair value indicator",
        "OBV": "On Balance Volume - Volume trend indicator",
        "Stochastic": "Stochastic Oscillator - Momentum oscillator"
    }
    return json.dumps(common_info, indent=2)


@tool
def openalgo_calculate_indicator(
    indicator: Annotated[str, "Indicator name (e.g., 'RSI', 'EMA', 'MACD')"],
    ohlcv_json: Annotated[str, "JSON string with OHLCV data: {open: [], high: [], low: [], close: [], volume: []}"],
    period: Annotated[int, "Period/length for the indicator calculation"] = 14,
    extra_params: Annotated[str, "JSON string with additional parameters"] = "{}"
) -> str:
    """
    Calculate a technical indicator on OHLCV data.
    
    Args:
        indicator: Name of the indicator (must be from approved list)
        ohlcv_json: OHLCV data as JSON with arrays for open, high, low, close, volume
        period: Primary period parameter (default 14)
        extra_params: Additional parameters as JSON (indicator-specific)
    
    Returns:
        JSON with calculated indicator values.
    """
    # Validate indicator
    if not is_valid_indicator(indicator):
        return json.dumps({
            "error": f"Unknown indicator: {indicator}",
            "hint": "Use openalgo_list_indicators to see available indicators"
        })
    
    try:
        from openalgo import ta
        import pandas as pd
        import numpy as np
        
        # Parse OHLCV data
        ohlcv = json.loads(ohlcv_json)
        extra = json.loads(extra_params) if extra_params else {}
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': ohlcv.get('open', []),
            'high': ohlcv.get('high', []),
            'low': ohlcv.get('low', []),
            'close': ohlcv.get('close', []),
            'volume': ohlcv.get('volume', [])
        })
        
        # Calculate based on indicator type
        result = {}
        indicator_lower = indicator.lower()
        
        # Map indicator names to lowercase functions
        # Trend Indicators (single output)
        if indicator in ["SMA", "sma"]:
            values = ta.sma(df['close'], period)
            result = {"values": _safe_tail(values, 20)}
            
        elif indicator in ["EMA", "ema"]:
            values = ta.ema(df['close'], period)
            result = {"values": _safe_tail(values, 20)}
            
        elif indicator in ["WMA", "wma"]:
            values = ta.wma(df['close'], period)
            result = {"values": _safe_tail(values, 20)}
            
        elif indicator in ["DEMA", "dema"]:
            values = ta.dema(df['close'], period)
            result = {"values": _safe_tail(values, 20)}
            
        elif indicator in ["TEMA", "tema"]:
            values = ta.tema(df['close'], period)
            result = {"values": _safe_tail(values, 20)}
            
        elif indicator in ["HMA", "hma"]:
            values = ta.hma(df['close'], period)
            result = {"values": _safe_tail(values, 20)}
            
        elif indicator in ["ZLEMA", "zlema"]:
            values = ta.zlema(df['close'], period)
            result = {"values": _safe_tail(values, 20)}
            
        elif indicator in ["VWMA", "vwma"]:
            values = ta.vwma(df['close'], df['volume'], period)
            result = {"values": _safe_tail(values, 20)}
            
        elif indicator in ["KAMA", "kama"]:
            fast = extra.get('fast_length', 2)
            slow = extra.get('slow_length', 30)
            values = ta.kama(df['close'], period, fast, slow)
            result = {"values": _safe_tail(values, 20)}
            
        elif indicator in ["ALMA", "alma"]:
            offset = extra.get('offset', 0.85)
            sigma = extra.get('sigma', 6.0)
            values = ta.alma(df['close'], period, offset, sigma)
            result = {"values": _safe_tail(values, 20)}
            
        # Supertrend
        elif indicator in ["Supertrend", "supertrend"]:
            multiplier = extra.get('multiplier', 3.0)
            supertrend, direction = ta.supertrend(df['high'], df['low'], df['close'], period, multiplier)
            result = {
                "supertrend": _safe_tail(supertrend, 20),
                "direction": _safe_tail(direction, 20),
                "note": "direction: -1=uptrend(green), 1=downtrend(red)"
            }
            
        # Ichimoku Cloud
        elif indicator in ["Ichimoku", "ichimoku"]:
            conversion_periods = extra.get('conversion_periods', 9)
            base_periods = extra.get('base_periods', 26)
            lagging_span2_periods = extra.get('lagging_span2_periods', 52)
            displacement = extra.get('displacement', 26)
            conversion, base, span_a, span_b, lagging = ta.ichimoku(
                df['high'], df['low'], df['close'],
                conversion_periods, base_periods, lagging_span2_periods, displacement
            )
            result = {
                "conversion_line": _safe_tail(conversion, 20),
                "base_line": _safe_tail(base, 20),
                "span_a": _safe_tail(span_a, 20),
                "span_b": _safe_tail(span_b, 20),
                "lagging_span": _safe_tail(lagging, 20)
            }
            
        # Momentum Indicators
        elif indicator in ["RSI", "rsi"]:
            values = ta.rsi(df['close'], period)
            result = {"values": _safe_tail(values, 20)}
            
        elif indicator in ["MACD", "macd"]:
            fast = extra.get('fast', 12)
            slow = extra.get('slow', 26)
            signal = extra.get('signal', 9)
            macd_line, signal_line, histogram = ta.macd(df['close'], fast, slow, signal)
            result = {
                "macd": _safe_tail(macd_line, 20),
                "signal": _safe_tail(signal_line, 20),
                "histogram": _safe_tail(histogram, 20)
            }
            
        elif indicator in ["Stochastic", "stochastic"]:
            k_period = extra.get('k', 14)
            d_period = extra.get('d', 3)
            smooth_k = extra.get('smooth_k', 3)
            k, d = ta.stochastic(df['high'], df['low'], df['close'], k_period, d_period, smooth_k)
            result = {
                "k": _safe_tail(k, 20),
                "d": _safe_tail(d, 20)
            }
            
        elif indicator in ["CCI", "cci"]:
            values = ta.cci(df['high'], df['low'], df['close'], period)
            result = {"values": _safe_tail(values, 20)}
            
        # Volatility Indicators
        elif indicator in ["ATR", "atr"]:
            values = ta.atr(df['high'], df['low'], df['close'], period)
            result = {"values": _safe_tail(values, 20)}
            
        elif indicator in ["BollingerBands", "bollingerbands"]:
            std_dev = extra.get('std_dev', 2)
            upper, middle, lower = ta.bollingerbands(df['close'], period, std_dev)
            result = {
                "upper": _safe_tail(upper, 20),
                "middle": _safe_tail(middle, 20),
                "lower": _safe_tail(lower, 20)
            }
            
        # Hybrid Indicators
        elif indicator in ["ADX", "adx"]:
            values = ta.adx(df['high'], df['low'], df['close'], period)
            result = {"values": _safe_tail(values, 20)}
            
        # Volume Indicators
        elif indicator in ["OBV", "obv"]:
            values = ta.obv(df['close'], df['volume'])
            result = {"values": _safe_tail(values, 20)}
            
        elif indicator in ["VWAP", "vwap"]:
            values = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
            result = {"values": _safe_tail(values, 20)}
            
        elif indicator in ["MFI", "mfi"]:
            values = ta.mfi(df['high'], df['low'], df['close'], df['volume'], period)
            result = {"values": _safe_tail(values, 20)}
            
        else:
            # Generic attempt - try lowercase function name
            try:
                indicator_func = getattr(ta, indicator_lower, None)
                if indicator_func is None:
                    indicator_func = getattr(ta, indicator, None)
                if indicator_func is None:
                    return json.dumps({"error": f"Indicator function '{indicator}' not found in ta library"})
                    
                values = indicator_func(df['close'], period)
                result = {"values": _safe_tail(values, 20)}
            except Exception as e:
                result = {"error": f"Could not calculate {indicator}: {str(e)}"}
        
        result["indicator"] = indicator
        result["period"] = period
        return json.dumps(result, indent=2)
        
    except ImportError:
        return json.dumps({"error": "OpenAlgo library not installed. Run: pip install openalgo"})
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"Calculation error: {str(e)}"})


@tool
def openalgo_validate_indicator(
    indicator: Annotated[str, "Indicator name to validate"]
) -> str:
    """
    Check if an indicator is in the approved OpenAlgo indicators list.
    
    Args:
        indicator: Name of the indicator to check
        
    Returns:
        JSON with validation result and category if valid.
    """
    is_valid = is_valid_indicator(indicator)
    category = get_indicator_category(indicator)
    
    if is_valid:
        return json.dumps({
            "valid": True,
            "indicator": indicator,
            "category": category,
            "message": f"{indicator} is an approved indicator in the {category} category"
        })
    else:
        # Find similar indicators
        similar = [ind for ind in ALL_INDICATORS if indicator.lower() in ind.lower()][:5]
        return json.dumps({
            "valid": False,
            "indicator": indicator,
            "message": f"{indicator} is NOT an approved indicator",
            "similar": similar if similar else "No similar indicators found"
        })


# =============================================================================
# Export all indicator tools
# =============================================================================

OPENALGO_INDICATOR_TOOLS = [
    openalgo_list_indicators,
    openalgo_get_common_indicators,
    openalgo_calculate_indicator,
    openalgo_validate_indicator,
]

# Export the registries for use by other modules
__all__ = [
    "OPENALGO_INDICATOR_TOOLS",
    "ALL_INDICATORS",
    "COMMON_INDICATORS",
    "INDICATORS_BY_CATEGORY",
    "TREND_INDICATORS",
    "MOMENTUM_INDICATORS",
    "VOLATILITY_INDICATORS",
    "VOLUME_INDICATORS",
    "OSCILLATORS",
    "STATISTICAL_INDICATORS",
    "HYBRID_INDICATORS",
    "UTILITY_FUNCTIONS",
    "is_valid_indicator",
    "get_indicator_category",
]
