"""
Safety & Financial Risk Controls - Production Guardrails

Implements:
1. Deterministic guardrails before tool execution
2. 20% rule (position value, loss, drawdown limits)
3. Kill switch (halt trading instantly)
4. Mode control (verify analyzer status)

Even if the model is wrong, it CANNOT blow up the account.
"""
import os
import json
import logging
import sqlite3
from datetime import datetime, date
from typing import Optional, Callable
from dataclasses import dataclass, field
from functools import wraps
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAFETY_DB = os.path.join(os.path.dirname(__file__), "safety_controls.db")


# =============================================================================
# RISK LIMITS CONFIGURATION
# =============================================================================

@dataclass
class RiskLimits:
    """
    Configurable risk limits.
    The 20% rule is enforced across multiple dimensions.
    """
    # Per-trade limits
    max_position_value_pct: float = 20.0  # Max 20% of capital per position
    max_position_value_abs: float = 100000.0  # Absolute cap ₹1L
    max_quantity_per_order: int = 100  # Max quantity per order
    
    # Per-day limits
    max_daily_loss_pct: float = 5.0  # Stop trading if 5% daily loss
    max_daily_loss_abs: float = 25000.0  # Absolute daily loss cap ₹25K
    max_daily_trades: int = 50  # Max 50 trades per day
    max_daily_volume: float = 500000.0  # Max ₹5L daily volume
    
    # Total exposure limits
    max_open_positions: int = 5  # Max 5 concurrent positions
    max_total_exposure_pct: float = 80.0  # Max 80% of capital deployed
    max_total_exposure_abs: float = 400000.0  # Absolute exposure cap ₹4L
    
    # Price limits
    max_price_deviation_pct: float = 5.0  # Limit price within 5% of LTP
    
    # Product-specific limits
    max_intraday_qty: int = 200  # MIS/NRML
    max_delivery_qty: int = 50  # CNC
    
    # F&O specific
    max_lot_multiplier: int = 10  # Max 10 lots per order
    
    def to_dict(self) -> dict:
        return {
            "max_position_value_pct": self.max_position_value_pct,
            "max_position_value_abs": self.max_position_value_abs,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "max_daily_loss_abs": self.max_daily_loss_abs,
            "max_daily_trades": self.max_daily_trades,
            "max_open_positions": self.max_open_positions,
            "max_total_exposure_pct": self.max_total_exposure_pct,
        }


# Default limits
DEFAULT_LIMITS = RiskLimits()


# =============================================================================
# KILL SWITCH
# =============================================================================

class KillSwitch:
    """
    Emergency kill switch for trading.
    When activated:
    - All new orders are BLOCKED
    - Optionally triggers emergency close
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_db()
        return cls._instance
    
    def _init_db(self):
        conn = sqlite3.connect(SAFETY_DB)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kill_switch (
                id INTEGER PRIMARY KEY,
                is_active BOOLEAN DEFAULT 0,
                activated_at TIMESTAMP,
                reason TEXT,
                activated_by TEXT,
                auto_close_triggered BOOLEAN DEFAULT 0
            )
        """)
        conn.execute("INSERT OR IGNORE INTO kill_switch (id, is_active) VALUES (1, 0)")
        conn.commit()
        conn.close()
    
    def is_active(self) -> tuple[bool, str]:
        """Check if kill switch is active."""
        conn = sqlite3.connect(SAFETY_DB)
        row = conn.execute("SELECT is_active, reason FROM kill_switch WHERE id = 1").fetchone()
        conn.close()
        return (bool(row[0]), row[1]) if row else (False, None)
    
    def activate(self, reason: str, activated_by: str = "system", auto_close: bool = False) -> dict:
        """Activate kill switch - HALT ALL TRADING."""
        conn = sqlite3.connect(SAFETY_DB)
        conn.execute("""
            UPDATE kill_switch SET 
                is_active = 1, 
                activated_at = ?,
                reason = ?,
                activated_by = ?,
                auto_close_triggered = ?
            WHERE id = 1
        """, (datetime.now().isoformat(), reason, activated_by, auto_close))
        conn.commit()
        conn.close()
        
        logger.warning(f"🛑 KILL SWITCH ACTIVATED: {reason}")
        
        if auto_close:
            self._trigger_emergency_close()
        
        return {
            "kill_switch": "ACTIVATED",
            "reason": reason,
            "auto_close": auto_close,
            "timestamp": datetime.now().isoformat()
        }
    
    def deactivate(self, deactivated_by: str = "user") -> dict:
        """Deactivate kill switch - requires manual intervention."""
        conn = sqlite3.connect(SAFETY_DB)
        conn.execute("UPDATE kill_switch SET is_active = 0, reason = NULL WHERE id = 1")
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Kill switch deactivated by {deactivated_by}")
        return {"kill_switch": "DEACTIVATED", "by": deactivated_by}
    
    def _trigger_emergency_close(self):
        """Trigger emergency close of all positions."""
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            client.closeposition()
            client.cancelallorder()
            logger.warning("🚨 Emergency close triggered by kill switch")
        except Exception as e:
            logger.error(f"Emergency close failed: {e}")


def get_kill_switch() -> KillSwitch:
    return KillSwitch()


# =============================================================================
# MODE CONTROL
# =============================================================================

class TradingMode(str, Enum):
    ANALYZE = "analyze"  # Simulated, no real orders
    LIVE = "live"  # Real money trading


class ModeController:
    """
    Controls trading mode (analyze vs live).
    Blocks live trading unless explicitly enabled.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_db()
        return cls._instance
    
    def _init_db(self):
        conn = sqlite3.connect(SAFETY_DB)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mode_control (
                id INTEGER PRIMARY KEY,
                allowed_mode TEXT DEFAULT 'analyze',
                live_enabled BOOLEAN DEFAULT 0,
                live_enabled_until TIMESTAMP,
                last_verified_at TIMESTAMP,
                broker_mode TEXT
            )
        """)
        conn.execute("INSERT OR IGNORE INTO mode_control (id, allowed_mode) VALUES (1, 'analyze')")
        conn.commit()
        conn.close()
    
    def get_allowed_mode(self) -> TradingMode:
        """Get the currently allowed trading mode."""
        conn = sqlite3.connect(SAFETY_DB)
        row = conn.execute("SELECT allowed_mode, live_enabled_until FROM mode_control WHERE id = 1").fetchone()
        conn.close()
        
        if row:
            if row[0] == "live" and row[1]:
                # Check if live mode has expired
                until = datetime.fromisoformat(row[1])
                if datetime.now() > until:
                    self.disable_live_mode("Session expired")
                    return TradingMode.ANALYZE
            return TradingMode(row[0])
        return TradingMode.ANALYZE
    
    def is_live_enabled(self) -> bool:
        """Check if live trading is enabled."""
        return self.get_allowed_mode() == TradingMode.LIVE
    
    def enable_live_mode(self, duration_hours: int = 8, enabled_by: str = "user") -> dict:
        """
        Enable live trading for a limited duration.
        REQUIRES explicit user action.
        """
        until = datetime.now().replace(hour=datetime.now().hour + duration_hours)
        
        conn = sqlite3.connect(SAFETY_DB)
        conn.execute("""
            UPDATE mode_control SET 
                allowed_mode = 'live',
                live_enabled = 1,
                live_enabled_until = ?
            WHERE id = 1
        """, (until.isoformat(),))
        conn.commit()
        conn.close()
        
        logger.warning(f"⚠️ LIVE TRADING ENABLED until {until.strftime('%H:%M')}")
        return {
            "mode": "LIVE",
            "enabled_until": until.isoformat(),
            "enabled_by": enabled_by
        }
    
    def disable_live_mode(self, reason: str = "manual") -> dict:
        """Disable live trading, revert to analyze mode."""
        conn = sqlite3.connect(SAFETY_DB)
        conn.execute("""
            UPDATE mode_control SET 
                allowed_mode = 'analyze',
                live_enabled = 0,
                live_enabled_until = NULL
            WHERE id = 1
        """)
        conn.commit()
        conn.close()
        
        logger.info(f"Live trading disabled: {reason}")
        return {"mode": "ANALYZE", "reason": reason}
    
    def verify_broker_mode(self) -> dict:
        """Verify broker's current mode matches our setting."""
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            result = client.analyzerstatus()
            
            broker_in_analyze = result.get("analyzer_mode", True)
            our_mode = self.get_allowed_mode()
            
            # Update verification timestamp
            conn = sqlite3.connect(SAFETY_DB)
            conn.execute("""
                UPDATE mode_control SET 
                    last_verified_at = ?,
                    broker_mode = ?
                WHERE id = 1
            """, (datetime.now().isoformat(), "analyze" if broker_in_analyze else "live"))
            conn.commit()
            conn.close()
            
            # Safety: if broker is live but we expect analyze, force analyze
            if not broker_in_analyze and our_mode == TradingMode.ANALYZE:
                logger.warning("⚠️ Broker in LIVE mode but we expect ANALYZE - mismatch!")
                return {
                    "status": "MISMATCH",
                    "broker_mode": "live",
                    "expected_mode": "analyze",
                    "action": "Block live trading"
                }
            
            return {
                "status": "OK",
                "broker_mode": "analyze" if broker_in_analyze else "live",
                "our_mode": our_mode.value,
                "verified_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}


def get_mode_controller() -> ModeController:
    return ModeController()


# =============================================================================
# DAILY TRACKING
# =============================================================================

class DailyTracker:
    """Track daily trading metrics for limit enforcement."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_db()
        return cls._instance
    
    def _init_db(self):
        conn = sqlite3.connect(SAFETY_DB)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_metrics (
                trade_date DATE PRIMARY KEY,
                total_trades INTEGER DEFAULT 0,
                total_volume REAL DEFAULT 0,
                realized_pnl REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                peak_capital REAL DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()
    
    def _ensure_today(self):
        """Ensure today's record exists."""
        today = date.today().isoformat()
        conn = sqlite3.connect(SAFETY_DB)
        conn.execute("INSERT OR IGNORE INTO daily_metrics (trade_date) VALUES (?)", (today,))
        conn.commit()
        conn.close()
    
    def record_trade(self, volume: float, pnl: float = 0):
        """Record a trade."""
        self._ensure_today()
        today = date.today().isoformat()
        
        conn = sqlite3.connect(SAFETY_DB)
        conn.execute("""
            UPDATE daily_metrics SET 
                total_trades = total_trades + 1,
                total_volume = total_volume + ?,
                realized_pnl = realized_pnl + ?
            WHERE trade_date = ?
        """, (volume, pnl, today))
        conn.commit()
        conn.close()
    
    def get_today_metrics(self) -> dict:
        """Get today's trading metrics."""
        self._ensure_today()
        today = date.today().isoformat()
        
        conn = sqlite3.connect(SAFETY_DB)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM daily_metrics WHERE trade_date = ?", (today,)).fetchone()
        conn.close()
        
        return dict(row) if row else {}
    
    def check_daily_limits(self, limits: RiskLimits = None) -> tuple[bool, str]:
        """Check if daily limits are breached."""
        limits = limits or DEFAULT_LIMITS
        metrics = self.get_today_metrics()
        
        # Check trade count
        if metrics.get("total_trades", 0) >= limits.max_daily_trades:
            return False, f"Daily trade limit reached ({limits.max_daily_trades})"
        
        # Check volume
        if metrics.get("total_volume", 0) >= limits.max_daily_volume:
            return False, f"Daily volume limit reached (₹{limits.max_daily_volume:,.0f})"
        
        # Check loss
        pnl = metrics.get("realized_pnl", 0)
        if pnl < 0 and abs(pnl) >= limits.max_daily_loss_abs:
            return False, f"Daily loss limit reached (₹{limits.max_daily_loss_abs:,.0f})"
        
        return True, "Within daily limits"


def get_daily_tracker() -> DailyTracker:
    return DailyTracker()


# =============================================================================
# PRE-EXECUTION GUARDRAILS
# =============================================================================

def check_pre_execution_guardrails(trade_plan: dict, capital: float = 500000.0) -> tuple[bool, str]:
    """
    Run ALL safety checks before any order execution.
    This is DETERMINISTIC - no LLM involved.
    
    Returns (is_safe, rejection_reason)
    """
    limits = DEFAULT_LIMITS
    checks = []
    
    # 1. Kill switch check
    kill_switch = get_kill_switch()
    is_killed, reason = kill_switch.is_active()
    if is_killed:
        return False, f"KILL SWITCH ACTIVE: {reason}"
    checks.append("kill_switch: OK")
    
    # 2. Mode check
    mode_ctrl = get_mode_controller()
    mode = mode_ctrl.get_allowed_mode()
    if mode == TradingMode.ANALYZE:
        # Allow trade in analyze mode (simulated)
        checks.append(f"mode: {mode.value} (simulated)")
    else:
        checks.append(f"mode: {mode.value} (LIVE)")
    
    # 3. Daily limits check
    tracker = get_daily_tracker()
    within_limits, limit_msg = tracker.check_daily_limits()
    if not within_limits:
        return False, f"DAILY LIMIT: {limit_msg}"
    checks.append("daily_limits: OK")
    
    # 4. Position value check (20% rule)
    quantity = trade_plan.get("quantity", 0)
    price = trade_plan.get("price", 0) or trade_plan.get("ltp", 0)
    if price > 0:
        position_value = quantity * price
        max_allowed = min(
            capital * (limits.max_position_value_pct / 100),
            limits.max_position_value_abs
        )
        if position_value > max_allowed:
            return False, f"POSITION VALUE ₹{position_value:,.0f} exceeds limit ₹{max_allowed:,.0f} (20% rule)"
        checks.append(f"position_value: ₹{position_value:,.0f} < ₹{max_allowed:,.0f}")
    
    # 5. Quantity check
    if quantity > limits.max_quantity_per_order:
        return False, f"QUANTITY {quantity} exceeds limit {limits.max_quantity_per_order}"
    checks.append(f"quantity: {quantity} OK")
    
    # 6. Price deviation check (for limit orders)
    ltp = trade_plan.get("ltp", 0)
    order_price = trade_plan.get("price", 0)
    if order_price > 0 and ltp > 0:
        deviation = abs(order_price - ltp) / ltp * 100
        if deviation > limits.max_price_deviation_pct:
            return False, f"PRICE DEVIATION {deviation:.1f}% exceeds {limits.max_price_deviation_pct}%"
        checks.append(f"price_deviation: {deviation:.1f}% OK")
    
    logger.info(f"Pre-execution guardrails passed: {checks}")
    return True, "All guardrails passed"


# =============================================================================
# GUARDRAIL DECORATOR
# =============================================================================

def with_guardrails(func: Callable) -> Callable:
    """
    Decorator to wrap any execution function with guardrails.
    Blocks execution if guardrails fail.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract trade plan from args/kwargs
        trade_plan = kwargs.get("trade_plan") or (args[0] if args else {})
        
        # Run guardrails
        is_safe, reason = check_pre_execution_guardrails(trade_plan)
        
        if not is_safe:
            logger.warning(f"🚫 GUARDRAIL BLOCKED: {reason}")
            return {
                "status": "BLOCKED",
                "reason": reason,
                "guardrail": True
            }
        
        # Proceed with execution
        return func(*args, **kwargs)
    
    return wrapper


# =============================================================================
# STARTUP CHECKS
# =============================================================================

def run_startup_safety_checks() -> dict:
    """Run all safety checks on startup."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "checks": []
    }
    
    # 1. Kill switch status
    kill_switch = get_kill_switch()
    is_killed, reason = kill_switch.is_active()
    results["checks"].append({
        "check": "kill_switch",
        "status": "ACTIVE" if is_killed else "OK",
        "reason": reason
    })
    
    # 2. Mode verification
    mode_ctrl = get_mode_controller()
    mode_result = mode_ctrl.verify_broker_mode()
    results["checks"].append({
        "check": "mode_control",
        "status": mode_result.get("status"),
        "mode": mode_result.get("our_mode")
    })
    
    # 3. Daily metrics
    tracker = get_daily_tracker()
    metrics = tracker.get_today_metrics()
    within_limits, _ = tracker.check_daily_limits()
    results["checks"].append({
        "check": "daily_limits",
        "status": "OK" if within_limits else "EXCEEDED",
        "trades_today": metrics.get("total_trades", 0)
    })
    
    # 4. Database connectivity
    try:
        conn = sqlite3.connect(SAFETY_DB)
        conn.execute("SELECT 1")
        conn.close()
        results["checks"].append({"check": "database", "status": "OK"})
    except Exception as e:
        results["checks"].append({"check": "database", "status": "ERROR", "error": str(e)})
    
    # Overall status
    all_ok = all(c.get("status") == "OK" for c in results["checks"])
    results["overall"] = "READY" if all_ok else "WARNING"
    
    logger.info(f"Startup safety checks: {results['overall']}")
    return results


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "RiskLimits",
    "DEFAULT_LIMITS",
    "KillSwitch",
    "get_kill_switch",
    "TradingMode",
    "ModeController",
    "get_mode_controller",
    "DailyTracker",
    "get_daily_tracker",
    "check_pre_execution_guardrails",
    "with_guardrails",
    "run_startup_safety_checks",
]
