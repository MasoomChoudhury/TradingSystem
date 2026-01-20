"""
Trading Session Manager

Manages the daily trading session lifecycle:
- Parse market reports into standardized strategies
- Track active positions, P&L, risk limits
- Coordinate strategy switching based on Analyst recommendations
"""
import os
import json
import sqlite3
import logging
from datetime import datetime, date
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SESSION_DB = "trading_session.db"


class SessionState(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    PAUSED = "paused"
    CLOSED = "closed"


class MarketBias(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class RiskLimits:
    """Risk limits for the trading session."""
    max_lots: int = 1
    max_daily_loss: float = 1500.0  # INR
    max_position_value: float = 500000.0  # INR
    
    
@dataclass
class StandardizedStrategy:
    """Standardized strategy format that Supervisor can accept."""
    name: str  # e.g., "ORB_LONG", "MEAN_REVERT_SHORT"
    bias: str  # "LONG" or "SHORT"
    entry_condition: str  # Human-readable entry logic
    exit_condition: str  # Human-readable exit logic
    stop_loss: float  # Percentage or points
    target: float  # Percentage or points
    symbol: str = "BANKNIFTY"
    exchange: str = "NFO"
    product: str = "MIS"  # Intraday
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TradingSession:
    """Represents a daily trading session."""
    session_id: str
    date: str
    state: SessionState
    market_report: str  # Original user input
    market_bias: MarketBias
    strategies: List[StandardizedStrategy]
    active_strategy_index: int
    risk_limits: RiskLimits
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    positions: List[dict] = field(default_factory=list)
    trade_count: int = 0
    created_at: str = ""
    updated_at: str = ""
    
    def get_active_strategy(self) -> Optional[StandardizedStrategy]:
        if 0 <= self.active_strategy_index < len(self.strategies):
            return self.strategies[self.active_strategy_index]
        return None
    
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "date": self.date,
            "state": self.state.value,
            "market_bias": self.market_bias.value,
            "strategies": [s.to_dict() for s in self.strategies],
            "active_strategy_index": self.active_strategy_index,
            "active_strategy": self.get_active_strategy().to_dict() if self.get_active_strategy() else None,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "positions": self.positions,
            "trade_count": self.trade_count,
            "risk_limits": asdict(self.risk_limits),
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


class TradingSessionManager:
    """Manages trading sessions with SQLite persistence."""
    
    def __init__(self, db_path: str = SESSION_DB):
        self.db_path = db_path
        self.current_session: Optional[TradingSession] = None
        self._init_db()
        logger.info("📊 Trading Session Manager initialized")
    
    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                date TEXT NOT NULL,
                state TEXT NOT NULL,
                market_report TEXT,
                market_bias TEXT,
                strategies TEXT,
                active_strategy_index INTEGER,
                realized_pnl REAL,
                unrealized_pnl REAL,
                positions TEXT,
                trade_count INTEGER,
                risk_limits TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    def parse_market_report(self, report: str) -> Dict[str, Any]:
        """
        Parse a free-form market report into structured data.
        Uses LLM to extract key trading parameters.
        """
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage, SystemMessage
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.environ.get("GEMINI_API_KEY"),
            temperature=0.1,
        )
        
        system_prompt = """You are a trading strategy parser. Extract key information from market reports.

OUTPUT STRICT JSON:
{
    "market_bias": "BULLISH" | "BEARISH" | "NEUTRAL",
    "primary_strategy": {
        "name": "strategy name (e.g., ORB_LONG, MEAN_REVERT, TREND_FOLLOW)",
        "bias": "LONG" | "SHORT",
        "entry_condition": "clear entry condition",
        "exit_condition": "clear exit condition",
        "stop_loss_points": 50,
        "target_points": 100,
        "symbol": "BANKNIFTY"
    },
    "alternate_strategy": {
        "name": "opposite strategy if market reverses",
        "bias": "SHORT" | "LONG",
        "entry_condition": "reversal entry condition",
        "exit_condition": "reversal exit condition",
        "stop_loss_points": 50,
        "target_points": 100,
        "symbol": "BANKNIFTY"
    },
    "key_levels": {
        "resistance": [25850, 25900, 26000],
        "support": [25500, 25550, 25600]
    },
    "summary": "one line trading plan summary"
}"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Parse this market report:\n\n{report[:8000]}")  # Truncate if too long
        ]
        
        try:
            response = llm.invoke(messages)
            content = response.content
            
            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            return json.loads(content.strip())
        except Exception as e:
            logger.error(f"Failed to parse market report: {e}")
            # Return default structure if parsing fails
            return {
                "market_bias": "NEUTRAL",
                "primary_strategy": {
                    "name": "SCALP",
                    "bias": "LONG",
                    "entry_condition": "Buy on support",
                    "exit_condition": "Sell on resistance",
                    "stop_loss_points": 50,
                    "target_points": 50,
                    "symbol": "BANKNIFTY"
                },
                "alternate_strategy": {
                    "name": "SCALP_SHORT",
                    "bias": "SHORT",
                    "entry_condition": "Sell on resistance breakdown",
                    "exit_condition": "Cover on support",
                    "stop_loss_points": 50,
                    "target_points": 50,
                    "symbol": "BANKNIFTY"
                },
                "summary": "Range-bound trading day"
            }
    
    def start_session(self, market_report: str) -> TradingSession:
        """
        Start a new trading session from a market report.
        """
        import uuid
        
        # Parse the report
        parsed = self.parse_market_report(market_report)
        
        # Create strategies
        primary = StandardizedStrategy(
            name=parsed["primary_strategy"]["name"],
            bias=parsed["primary_strategy"]["bias"],
            entry_condition=parsed["primary_strategy"]["entry_condition"],
            exit_condition=parsed["primary_strategy"]["exit_condition"],
            stop_loss=parsed["primary_strategy"].get("stop_loss_points", 50),
            target=parsed["primary_strategy"].get("target_points", 100),
            symbol=parsed["primary_strategy"].get("symbol", "BANKNIFTY")
        )
        
        alternate = StandardizedStrategy(
            name=parsed["alternate_strategy"]["name"],
            bias=parsed["alternate_strategy"]["bias"],
            entry_condition=parsed["alternate_strategy"]["entry_condition"],
            exit_condition=parsed["alternate_strategy"]["exit_condition"],
            stop_loss=parsed["alternate_strategy"].get("stop_loss_points", 50),
            target=parsed["alternate_strategy"].get("target_points", 100),
            symbol=parsed["alternate_strategy"].get("symbol", "BANKNIFTY")
        )
        
        # Create session
        session = TradingSession(
            session_id=str(uuid.uuid4())[:8],
            date=date.today().isoformat(),
            state=SessionState.ACTIVE,
            market_report=market_report[:5000],  # Store truncated
            market_bias=MarketBias(parsed["market_bias"].lower()),
            strategies=[primary, alternate],
            active_strategy_index=0,
            risk_limits=RiskLimits(max_lots=1, max_daily_loss=1500.0),
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        self.current_session = session
        self._save_session(session)
        
        # Log to agent comms
        from agent_comms import send_agent_message
        send_agent_message(
            from_agent="orchestrator",
            to_agent="supervisor",
            message_type="info",
            content=f"Session started: {parsed.get('summary', 'New trading session')}",
            metadata={"session_id": session.session_id, "bias": session.market_bias.value}
        )
        
        logger.info(f"📊 Trading session started: {session.session_id}")
        
        # Generate and display detailed trading plan (NO IMMEDIATE EXECUTION)
        strategy = session.get_active_strategy()
        if strategy:
            # Create detailed plan based on strategy
            plan_details = {
                "strategy_name": strategy.name,
                "bias": strategy.bias,
                "option_type": "PE" if strategy.bias.upper() in ["SHORT", "BEARISH"] else "CE",
                "entry_condition": strategy.entry_condition,
                "exit_condition": strategy.exit_condition,
                "stop_loss_points": strategy.stop_loss,
                "target_points": strategy.target,
                "underlying": strategy.symbol,
                "action": "Waiting for Analyst recommendation",
                "note": "Trade will execute when Market Analyst provides KEEP/SWITCH/STOP signal"
            }
            
            # Log the detailed plan
            send_agent_message(
                from_agent="orchestrator",
                to_agent="analyst",
                message_type="info",
                content=f"📋 TRADING PLAN: {strategy.bias} bias → Will buy {plan_details['option_type']} on {strategy.symbol}",
                metadata=plan_details
            )
            
            # Send detailed strategy breakdown
            send_agent_message(
                from_agent="orchestrator",
                to_agent="supervisor",
                message_type="info",
                content=f"""📊 Strategy Details:
• Option: {plan_details['option_type']} (Put for Short, Call for Long)
• Entry: {strategy.entry_condition}
• Exit: {strategy.exit_condition}
• SL: {strategy.stop_loss} pts | Target: {strategy.target} pts
⏳ Awaiting Analyst signal to execute...""",
                metadata=plan_details
            )
        
        return session
    
    def switch_strategy(self, reason: str) -> Optional[StandardizedStrategy]:
        """
        Switch to the alternate strategy.
        Called when Analyst recommends SWITCH.
        """
        if not self.current_session:
            return None
        
        # Toggle between 0 and 1
        new_index = 1 - self.current_session.active_strategy_index
        self.current_session.active_strategy_index = new_index
        self.current_session.updated_at = datetime.now().isoformat()
        
        new_strategy = self.current_session.get_active_strategy()
        
        # Log the switch
        from agent_comms import send_agent_message
        send_agent_message(
            from_agent="orchestrator",
            to_agent="supervisor",
            message_type="info",
            content=f"Strategy switched to {new_strategy.name} ({new_strategy.bias}). Reason: {reason}",
            metadata={"new_strategy": new_strategy.to_dict()}
        )
        
        self._save_session(self.current_session)
        logger.info(f"📊 Strategy switched to: {new_strategy.name}")
        return new_strategy
    
    def pause_session(self, reason: str):
        """Pause trading - called when Analyst recommends STOP."""
        if self.current_session:
            self.current_session.state = SessionState.PAUSED
            self.current_session.updated_at = datetime.now().isoformat()
            self._save_session(self.current_session)
            
            from agent_comms import send_agent_message
            send_agent_message(
                from_agent="orchestrator",
                to_agent="executor",
                message_type="info",
                content=f"Trading PAUSED. Reason: {reason}",
                metadata={"session_id": self.current_session.session_id}
            )
            logger.info(f"📊 Session paused: {reason}")
    
    def resume_session(self):
        """Resume trading."""
        if self.current_session and self.current_session.state == SessionState.PAUSED:
            self.current_session.state = SessionState.ACTIVE
            self.current_session.updated_at = datetime.now().isoformat()
            self._save_session(self.current_session)
            logger.info("📊 Session resumed")
    
    def update_pnl(self, realized: float = 0, unrealized: float = 0):
        """Update session P&L."""
        if self.current_session:
            self.current_session.realized_pnl += realized
            self.current_session.unrealized_pnl = unrealized
            self.current_session.updated_at = datetime.now().isoformat()
            
            # Check daily loss limit
            total_loss = -self.current_session.realized_pnl if self.current_session.realized_pnl < 0 else 0
            if total_loss >= self.current_session.risk_limits.max_daily_loss:
                self.pause_session(f"Daily loss limit hit: ₹{total_loss}")
            
            self._save_session(self.current_session)
    
    def end_session(self) -> dict:
        """End the trading session and generate summary."""
        if not self.current_session:
            return {"error": "No active session"}
        
        self.current_session.state = SessionState.CLOSED
        self.current_session.updated_at = datetime.now().isoformat()
        self._save_session(self.current_session)
        
        summary = {
            "session_id": self.current_session.session_id,
            "date": self.current_session.date,
            "realized_pnl": self.current_session.realized_pnl,
            "trade_count": self.current_session.trade_count,
            "strategies_used": [s.name for s in self.current_session.strategies]
        }
        
        logger.info(f"📊 Session ended. P&L: ₹{self.current_session.realized_pnl}")
        self.current_session = None
        return summary
    
    def get_session_status(self) -> dict:
        """Get current session status."""
        if not self.current_session:
            return {"state": "idle", "message": "No active session"}
        return self.current_session.to_dict()
    
    def check_risk_limits(self, proposed_trade: dict) -> tuple[bool, str]:
        """Check if a proposed trade is within risk limits."""
        if not self.current_session:
            return False, "No active session"
        
        limits = self.current_session.risk_limits
        
        # Check lot limit
        qty = proposed_trade.get("quantity", 0)
        if qty > limits.max_lots * 25:  # Assuming 25 is BankNifty lot size
            return False, f"Exceeds max lots ({limits.max_lots})"
        
        # Check if already at daily loss limit
        if self.current_session.realized_pnl <= -limits.max_daily_loss:
            return False, "Daily loss limit already hit"
        
        return True, "Within risk limits"
    
    def _save_session(self, session: TradingSession):
        """Persist session to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO sessions 
            (session_id, date, state, market_report, market_bias, strategies, 
             active_strategy_index, realized_pnl, unrealized_pnl, positions, 
             trade_count, risk_limits, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session.session_id,
            session.date,
            session.state.value,
            session.market_report,
            session.market_bias.value,
            json.dumps([s.to_dict() for s in session.strategies]),
            session.active_strategy_index,
            session.realized_pnl,
            session.unrealized_pnl,
            json.dumps(session.positions),
            session.trade_count,
            json.dumps(asdict(session.risk_limits)),
            session.created_at,
            session.updated_at
        ))
        conn.commit()
        conn.close()


# Singleton instance
_session_manager = None

def get_session_manager() -> TradingSessionManager:
    """Get the singleton session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = TradingSessionManager()
    return _session_manager
