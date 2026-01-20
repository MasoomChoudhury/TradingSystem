"""
Options Symbol Generator

Handles:
- Strike price selection based on current spot price
- Expiry date selection (weekly/monthly)
- CE/PE symbol generation for NSE/NFO
"""
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Weekly expiry is every Thursday
WEEKLY_EXPIRY_DAY = 3  # Thursday


@dataclass
class OptionContract:
    """Represents an option contract."""
    symbol: str  # e.g., "BANKNIFTY24JAN52000CE"
    underlying: str  # e.g., "BANKNIFTY"
    strike: int
    option_type: str  # "CE" or "PE"
    expiry: str  # "2026-01-23"
    expiry_display: str  # "23JAN26"
    lot_size: int
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "underlying": self.underlying,
            "strike": self.strike,
            "option_type": self.option_type,
            "expiry": self.expiry,
            "expiry_display": self.expiry_display,
            "lot_size": self.lot_size
        }


class OptionsSymbolGenerator:
    """Generates option symbols for trading."""
    
    # Lot sizes
    LOT_SIZES = {
        "BANKNIFTY": 15,
        "NIFTY": 25,
        "FINNIFTY": 25,
        "MIDCPNIFTY": 50
    }
    
    # Strike intervals
    STRIKE_INTERVALS = {
        "BANKNIFTY": 100,
        "NIFTY": 50,
        "FINNIFTY": 50,
        "MIDCPNIFTY": 25
    }
    
    def __init__(self):
        logger.info("📊 Options Symbol Generator initialized")
    
    def get_next_expiry(self, weekly: bool = True) -> datetime:
        """Get the next expiry date (Thursday for weekly)."""
        today = datetime.now()
        
        # Find next Thursday
        days_until_thursday = (WEEKLY_EXPIRY_DAY - today.weekday()) % 7
        if days_until_thursday == 0 and today.hour >= 15:  # After market close on expiry day
            days_until_thursday = 7
        
        next_expiry = today + timedelta(days=days_until_thursday)
        return next_expiry.replace(hour=15, minute=30, second=0, microsecond=0)
    
    def format_expiry_for_symbol(self, expiry: datetime) -> str:
        """Format expiry for option symbol (e.g., '23JAN26')."""
        return expiry.strftime("%d%b%y").upper()
    
    def get_atm_strike(self, spot_price: float, underlying: str) -> int:
        """Get At-The-Money strike price."""
        interval = self.STRIKE_INTERVALS.get(underlying, 100)
        return round(spot_price / interval) * interval
    
    def select_strike(
        self, 
        spot_price: float, 
        underlying: str, 
        option_type: str,
        offset: int = 0
    ) -> int:
        """
        Select strike price based on strategy.
        
        Args:
            spot_price: Current spot price
            underlying: BANKNIFTY, NIFTY, etc.
            option_type: CE or PE
            offset: Number of strikes away from ATM (positive = OTM, negative = ITM)
        
        Returns:
            Selected strike price
        """
        atm = self.get_atm_strike(spot_price, underlying)
        interval = self.STRIKE_INTERVALS.get(underlying, 100)
        
        if option_type == "CE":
            # For CE: higher strike = OTM
            return atm + (offset * interval)
        else:  # PE
            # For PE: lower strike = OTM
            return atm - (offset * interval)
    
    def generate_symbol(
        self,
        underlying: str,
        strike: int,
        option_type: str,
        expiry: datetime
    ) -> str:
        """Generate the trading symbol for an option."""
        expiry_str = self.format_expiry_for_symbol(expiry)
        # Format: BANKNIFTY23JAN2652000CE
        return f"{underlying}{expiry_str}{strike}{option_type}"
    
    def get_option_contract(
        self,
        underlying: str,
        spot_price: float,
        bias: str,  # "LONG" or "SHORT" or "BULLISH" or "BEARISH"
        strike_offset: int = 0
    ) -> OptionContract:
        """
        Get the appropriate option contract based on market bias.
        
        For BULLISH/LONG bias: Buy CE (Call)
        For BEARISH/SHORT bias: Buy PE (Put)
        """
        # Determine option type based on bias
        bias_upper = bias.upper()
        if bias_upper in ["LONG", "BULLISH"]:
            option_type = "CE"
        else:  # SHORT, BEARISH
            option_type = "PE"
        
        # Get next weekly expiry
        expiry = self.get_next_expiry(weekly=True)
        
        # Select strike (ATM by default, or with offset)
        strike = self.select_strike(spot_price, underlying, option_type, strike_offset)
        
        # Generate symbol
        symbol = self.generate_symbol(underlying, strike, option_type, expiry)
        
        lot_size = self.LOT_SIZES.get(underlying, 15)
        
        contract = OptionContract(
            symbol=symbol,
            underlying=underlying,
            strike=strike,
            option_type=option_type,
            expiry=expiry.strftime("%Y-%m-%d"),
            expiry_display=self.format_expiry_for_symbol(expiry),
            lot_size=lot_size
        )
        
        logger.info(f"📊 Generated option: {contract.symbol}")
        return contract
    
    def rank_strikes(
        self,
        underlying: str,
        spot_price: float,
        option_type: str,
        num_strikes: int = 5
    ) -> List[dict]:
        """
        Rank strikes by probability/premium trade-off.
        Returns list of strike options with reasoning.
        """
        atm = self.get_atm_strike(spot_price, underlying)
        interval = self.STRIKE_INTERVALS.get(underlying, 100)
        expiry = self.get_next_expiry()
        
        strikes = []
        for offset in range(-2, num_strikes - 2):  # -2 to +2 from ATM
            strike = atm + (offset * interval)
            
            if option_type == "CE":
                moneyness = "ITM" if strike < spot_price else ("ATM" if strike == atm else "OTM")
                otm_distance = strike - spot_price
            else:
                moneyness = "ITM" if strike > spot_price else ("ATM" if strike == atm else "OTM")
                otm_distance = spot_price - strike
            
            # Simple ranking: ATM gets highest score, decreases as you go OTM
            rank_score = max(0, 100 - abs(offset) * 20)
            
            symbol = self.generate_symbol(underlying, strike, option_type, expiry)
            
            strikes.append({
                "symbol": symbol,
                "strike": strike,
                "moneyness": moneyness,
                "distance_from_spot": otm_distance,
                "rank_score": rank_score,
                "reasoning": f"{moneyness} strike, {abs(offset)} strikes from ATM"
            })
        
        # Sort by rank score
        strikes.sort(key=lambda x: x["rank_score"], reverse=True)
        return strikes


# Singleton
_options_generator = None

def get_options_generator() -> OptionsSymbolGenerator:
    """Get the singleton options generator."""
    global _options_generator
    if _options_generator is None:
        _options_generator = OptionsSymbolGenerator()
    return _options_generator


async def get_spot_price(underlying: str = "BANKNIFTY") -> Optional[float]:
    """Get current spot price from OpenAlgo."""
    try:
        from tools.openalgo_tools import get_openalgo_client
        client = get_openalgo_client()
        
        # Try to get quotes
        result = client.quotes(symbol=underlying, exchange="NSE")
        if result.get("status") == "success":
            return float(result.get("ltp", 0))
        
        # Fallback: use a reasonable default
        defaults = {
            "BANKNIFTY": 52000,
            "NIFTY": 25700,
            "FINNIFTY": 24500
        }
        return defaults.get(underlying, 50000)
    except Exception as e:
        logger.warning(f"Could not get spot price: {e}")
        # Return approximate values
        if underlying == "BANKNIFTY":
            return 52000
        elif underlying == "NIFTY":
            return 25700
        return 50000
