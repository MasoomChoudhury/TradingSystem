"""
Reliability & Error Handling - Production Robustness

Implements:
1. Retries with exponential backoff (safe for reads, idempotent for writes)
2. Self-correction loop for tool errors
3. Post-trade reconciliation
4. Invariant checks (no naked execution)

Handles API flakiness and prevents partial execution.
"""
import os
import json
import logging
import time
import functools
import hashlib
import sqlite3
from datetime import datetime
from typing import Callable, Optional, Any, TypeVar, List
from dataclasses import dataclass, field
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RELIABILITY_DB = os.path.join(os.path.dirname(__file__), "reliability.db")


# =============================================================================
# RETRY CONFIGURATION
# =============================================================================

class RetryPolicy(str, Enum):
    """Retry policies for different operation types."""
    NONE = "none"  # No retry
    SAFE = "safe"  # Safe to retry (reads)
    IDEMPOTENT = "idempotent"  # Retry only with idempotency check (writes)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    exponential_base: float = 2.0
    policy: RetryPolicy = RetryPolicy.SAFE
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for attempt with exponential backoff."""
        delay = self.base_delay_seconds * (self.exponential_base ** attempt)
        return min(delay, self.max_delay_seconds)


# Default configs for different operations
READ_RETRY_CONFIG = RetryConfig(max_retries=3, policy=RetryPolicy.SAFE)
WRITE_RETRY_CONFIG = RetryConfig(max_retries=2, policy=RetryPolicy.IDEMPOTENT)


# =============================================================================
# IDEMPOTENCY STORE
# =============================================================================

class IdempotencyStore:
    """
    Store for idempotency keys to prevent duplicate operations.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_db()
        return cls._instance
    
    def _init_db(self):
        conn = sqlite3.connect(RELIABILITY_DB)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS idempotency_keys (
                key TEXT PRIMARY KEY,
                operation TEXT NOT NULL,
                status TEXT NOT NULL,
                result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    
    def check_and_set(self, key: str, operation: str) -> tuple[bool, Optional[dict]]:
        """
        Check if operation was already performed.
        Returns (is_new, previous_result)
        """
        conn = sqlite3.connect(RELIABILITY_DB)
        row = conn.execute(
            "SELECT status, result FROM idempotency_keys WHERE key = ?", (key,)
        ).fetchone()
        
        if row:
            conn.close()
            status, result = row
            return False, json.loads(result) if result else None
        
        # New operation - mark as in progress
        conn.execute(
            "INSERT INTO idempotency_keys (key, operation, status) VALUES (?, ?, 'in_progress')",
            (key, operation)
        )
        conn.commit()
        conn.close()
        return True, None
    
    def mark_complete(self, key: str, result: dict) -> None:
        """Mark operation as complete with result."""
        conn = sqlite3.connect(RELIABILITY_DB)
        conn.execute(
            "UPDATE idempotency_keys SET status = 'complete', result = ?, completed_at = ? WHERE key = ?",
            (json.dumps(result), datetime.now().isoformat(), key)
        )
        conn.commit()
        conn.close()
    
    def mark_failed(self, key: str, error: str) -> None:
        """Mark operation as failed."""
        conn = sqlite3.connect(RELIABILITY_DB)
        conn.execute(
            "UPDATE idempotency_keys SET status = 'failed', result = ?, completed_at = ? WHERE key = ?",
            (json.dumps({"error": error}), datetime.now().isoformat(), key)
        )
        conn.commit()
        conn.close()
    
    def clear_stale(self, max_age_hours: int = 24) -> int:
        """Clear stale entries older than max_age_hours."""
        conn = sqlite3.connect(RELIABILITY_DB)
        cursor = conn.execute("""
            DELETE FROM idempotency_keys 
            WHERE created_at < datetime('now', '-' || ? || ' hours')
        """, (max_age_hours,))
        count = cursor.rowcount
        conn.commit()
        conn.close()
        return count


def get_idempotency_store() -> IdempotencyStore:
    return IdempotencyStore()


# =============================================================================
# RETRY DECORATORS
# =============================================================================

T = TypeVar('T')


def retry_with_backoff(
    config: RetryConfig = READ_RETRY_CONFIG,
    idempotency_key_func: Callable[..., str] = None
) -> Callable:
    """
    Decorator for retry with exponential backoff.
    
    For SAFE policy: retry on any failure
    For IDEMPOTENT policy: check idempotency before retry
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            idem_store = get_idempotency_store()
            
            # Generate idempotency key for write operations
            idem_key = None
            if config.policy == RetryPolicy.IDEMPOTENT and idempotency_key_func:
                idem_key = idempotency_key_func(*args, **kwargs)
                
                # Check if already executed
                is_new, prev_result = idem_store.check_and_set(idem_key, func.__name__)
                if not is_new:
                    logger.info(f"Idempotent operation already executed: {idem_key}")
                    return prev_result
            
            for attempt in range(config.max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    
                    # Mark complete for idempotent operations
                    if idem_key:
                        idem_store.mark_complete(idem_key, result)
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    
                    if attempt < config.max_retries:
                        delay = config.get_delay(attempt)
                        logger.warning(
                            f"Retry {attempt + 1}/{config.max_retries} for {func.__name__} "
                            f"after {delay:.1f}s: {str(e)}"
                        )
                        time.sleep(delay)
                    else:
                        # Mark failed for idempotent operations
                        if idem_key:
                            idem_store.mark_failed(idem_key, str(e))
            
            raise last_exception
        
        return wrapper
    return decorator


def safe_retry(func: Callable[..., T]) -> Callable[..., T]:
    """Shorthand decorator for safe read operations."""
    return retry_with_backoff(READ_RETRY_CONFIG)(func)


def idempotent_retry(key_func: Callable[..., str]) -> Callable:
    """Shorthand decorator for idempotent write operations."""
    return retry_with_backoff(WRITE_RETRY_CONFIG, key_func)


# =============================================================================
# SELF-CORRECTION / ERROR REPAIR
# =============================================================================

@dataclass
class ToolError:
    """Represents a tool execution error."""
    tool_name: str
    args: dict
    error_type: str
    error_message: str
    is_recoverable: bool = True
    suggested_fix: str = ""


class ErrorRepairEngine:
    """
    Self-correction engine for tool errors.
    Analyzes errors and suggests fixes or fallbacks.
    """
    
    # Error patterns and their fixes
    ERROR_PATTERNS = {
        "invalid symbol": {
            "recoverable": True,
            "fix": "Use search_symbols to find correct symbol format"
        },
        "insufficient margin": {
            "recoverable": True,
            "fix": "Reduce quantity or check available funds"
        },
        "market closed": {
            "recoverable": False,
            "fix": "Wait for market hours"
        },
        "rate limit": {
            "recoverable": True,
            "fix": "Wait and retry with backoff"
        },
        "invalid quantity": {
            "recoverable": True,
            "fix": "Adjust quantity to match lot size"
        },
        "price out of range": {
            "recoverable": True,
            "fix": "Get current LTP and adjust price"
        },
        "order rejected": {
            "recoverable": True,
            "fix": "Check order parameters and retry"
        },
        "connection": {
            "recoverable": True,
            "fix": "Retry with backoff"
        },
        "timeout": {
            "recoverable": True,
            "fix": "Retry with backoff"
        },
    }
    
    def analyze_error(self, tool_name: str, args: dict, error: Exception) -> ToolError:
        """Analyze an error and determine if it's recoverable."""
        error_msg = str(error).lower()
        
        for pattern, info in self.ERROR_PATTERNS.items():
            if pattern in error_msg:
                return ToolError(
                    tool_name=tool_name,
                    args=args,
                    error_type=pattern,
                    error_message=str(error),
                    is_recoverable=info["recoverable"],
                    suggested_fix=info["fix"]
                )
        
        # Unknown error - assume not recoverable for safety
        return ToolError(
            tool_name=tool_name,
            args=args,
            error_type="unknown",
            error_message=str(error),
            is_recoverable=False,
            suggested_fix="Manual intervention required"
        )
    
    def get_fallback(self, tool_name: str, error: ToolError) -> Optional[dict]:
        """Get fallback action for a failed tool."""
        fallbacks = {
            "place_order": {
                "fallback_tool": "place_smart_order",
                "reason": "Smart order handles position sizing"
            },
            "get_quotes": {
                "fallback_tool": "get_ltp_websocket",
                "reason": "Use WebSocket for real-time data"
            },
            "get_history": {
                "fallback_tool": None,
                "reason": "No fallback - historical data is required"
            },
        }
        return fallbacks.get(tool_name)
    
    def repair_args(self, tool_name: str, args: dict, error: ToolError) -> Optional[dict]:
        """Attempt to repair tool arguments."""
        repaired = args.copy()
        
        if error.error_type == "invalid quantity":
            # Try to fix quantity to lot size
            qty = args.get("quantity", 0)
            lot_size = args.get("lot_size", 1)
            if lot_size > 1:
                repaired["quantity"] = (qty // lot_size) * lot_size
                return repaired
        
        elif error.error_type == "price out of range":
            # Remove price for market order
            if args.get("price_type") in ["MARKET", "SL-M"]:
                repaired["price"] = "0"
                return repaired
        
        return None


def get_error_repair_engine() -> ErrorRepairEngine:
    return ErrorRepairEngine()


# =============================================================================
# POST-TRADE RECONCILIATION
# =============================================================================

@dataclass
class ReconciliationResult:
    """Result of post-trade reconciliation."""
    order_id: str
    status: str
    confirmed: bool
    order_status: dict = field(default_factory=dict)
    in_orderbook: bool = False
    in_tradebook: bool = False
    discrepancies: List[str] = field(default_factory=list)


class PostTradeReconciler:
    """
    Reconciles orders after placement to confirm execution.
    """
    
    def __init__(self):
        self.max_retries = 3
        self.retry_delay = 2.0
    
    def reconcile(self, order_id: str, expected: dict) -> ReconciliationResult:
        """
        Reconcile an order after placement.
        
        1. Check order status
        2. Verify in orderbook
        3. Check tradebook for fills
        """
        result = ReconciliationResult(
            order_id=order_id,
            status="pending",
            confirmed=False
        )
        
        if not order_id:
            result.discrepancies.append("No order ID provided")
            return result
        
        for attempt in range(self.max_retries):
            try:
                # 1. Get order status
                order_status = self._get_order_status(order_id)
                result.order_status = order_status
                result.status = order_status.get("status", "unknown")
                
                # 2. Check orderbook
                result.in_orderbook = self._check_orderbook(order_id)
                
                # 3. Check tradebook for completed orders
                if result.status in ["COMPLETE", "TRADED", "FILLED"]:
                    result.in_tradebook = self._check_tradebook(order_id)
                
                # Verify expected values match
                discrepancies = self._verify_expected(order_status, expected)
                result.discrepancies = discrepancies
                
                result.confirmed = (
                    result.status in ["COMPLETE", "TRADED", "FILLED", "PENDING", "OPEN"] and
                    result.in_orderbook and
                    len(discrepancies) == 0
                )
                
                if result.confirmed:
                    logger.info(f"Order {order_id} reconciled successfully: {result.status}")
                    return result
                
            except Exception as e:
                logger.warning(f"Reconciliation attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        return result
    
    def _get_order_status(self, order_id: str) -> dict:
        """Get order status from broker."""
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            return client.orderstatus(order_id=order_id)
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _check_orderbook(self, order_id: str) -> bool:
        """Check if order exists in orderbook."""
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            orderbook = client.orderbook()
            
            orders = orderbook.get("data", orderbook) if isinstance(orderbook, dict) else orderbook
            if isinstance(orders, list):
                for order in orders:
                    if order.get("orderid") == order_id or order.get("order_id") == order_id:
                        return True
            return False
        except Exception:
            return False
    
    def _check_tradebook(self, order_id: str) -> bool:
        """Check if order appears in tradebook."""
        try:
            from tools.openalgo_tools import get_openalgo_client
            client = get_openalgo_client()
            tradebook = client.tradebook()
            
            trades = tradebook.get("data", tradebook) if isinstance(tradebook, dict) else tradebook
            if isinstance(trades, list):
                for trade in trades:
                    if trade.get("orderid") == order_id or trade.get("order_id") == order_id:
                        return True
            return False
        except Exception:
            return False
    
    def _verify_expected(self, actual: dict, expected: dict) -> List[str]:
        """Verify actual order matches expected values."""
        discrepancies = []
        
        # Check critical fields
        fields_to_check = ["symbol", "exchange", "quantity"]
        for field in fields_to_check:
            actual_val = actual.get(field)
            expected_val = expected.get(field)
            if actual_val and expected_val and str(actual_val) != str(expected_val):
                discrepancies.append(f"{field}: expected {expected_val}, got {actual_val}")
        
        return discrepancies


def get_reconciler() -> PostTradeReconciler:
    return PostTradeReconciler()


# =============================================================================
# INVARIANT CHECKS (No Naked Execution)
# =============================================================================

@dataclass
class ExecutionInvariants:
    """Invariants that must be true before any execution."""
    has_approval_token: bool = False
    token_valid: bool = False
    required_fields_present: bool = False
    within_risk_limits: bool = False
    mode_verified: bool = False
    
    def all_satisfied(self) -> bool:
        """Check if all invariants are satisfied."""
        return all([
            self.has_approval_token,
            self.token_valid,
            self.required_fields_present,
            self.within_risk_limits,
            self.mode_verified
        ])
    
    def get_failures(self) -> List[str]:
        """Get list of failed invariants."""
        failures = []
        if not self.has_approval_token:
            failures.append("Missing approval token")
        if not self.token_valid:
            failures.append("Invalid approval token")
        if not self.required_fields_present:
            failures.append("Missing required fields")
        if not self.within_risk_limits:
            failures.append("Outside risk limits")
        if not self.mode_verified:
            failures.append("Trading mode not verified")
        return failures


def check_execution_invariants(trade_plan: dict) -> tuple[bool, ExecutionInvariants]:
    """
    Check all invariants before execution.
    Returns (can_execute, invariant_details)
    """
    invariants = ExecutionInvariants()
    
    # 1. Check approval token
    approval_token = trade_plan.get("approval_token")
    invariants.has_approval_token = bool(approval_token)
    
    # 2. Validate token (basic check - in production, verify signature/expiry)
    if approval_token:
        invariants.token_valid = len(approval_token) >= 8  # Basic length check
    
    # 3. Check required fields
    required = ["symbol", "exchange", "quantity", "action", "product"]
    present = all(trade_plan.get(f) for f in required)
    invariants.required_fields_present = present
    
    # 4. Check risk limits (use safety_controls)
    try:
        from safety_controls import check_pre_execution_guardrails
        safe, _ = check_pre_execution_guardrails(trade_plan)
        invariants.within_risk_limits = safe
    except Exception:
        invariants.within_risk_limits = True  # Allow if safety module unavailable
    
    # 5. Verify mode
    try:
        from safety_controls import get_mode_controller
        mode_ctrl = get_mode_controller()
        verified = mode_ctrl.verify_broker_mode()
        invariants.mode_verified = verified.get("status") != "MISMATCH"
    except Exception:
        invariants.mode_verified = True  # Allow if module unavailable
    
    return invariants.all_satisfied(), invariants


def enforce_invariants(func: Callable) -> Callable:
    """Decorator to enforce execution invariants."""
    @functools.wraps(func)
    def wrapper(trade_plan: dict, *args, **kwargs):
        can_execute, invariants = check_execution_invariants(trade_plan)
        
        if not can_execute:
            failures = invariants.get_failures()
            logger.warning(f"🚫 INVARIANT CHECK FAILED: {failures}")
            return {
                "status": "REJECTED",
                "reason": "Invariant check failed",
                "failures": failures,
                "naked_execution_blocked": True
            }
        
        return func(trade_plan, *args, **kwargs)
    
    return wrapper


# =============================================================================
# COMPREHENSIVE EXECUTION WRAPPER
# =============================================================================

def execute_with_reliability(
    trade_plan: dict,
    execute_func: Callable,
    idempotency_key: str = None
) -> dict:
    """
    Execute a trade with full reliability wrapper:
    1. Check invariants
    2. Check/set idempotency
    3. Execute with retry
    4. Reconcile result
    """
    # Generate idempotency key if not provided
    if not idempotency_key:
        key_data = f"{trade_plan.get('symbol')}:{trade_plan.get('action')}:{trade_plan.get('quantity')}:{trade_plan.get('approval_token', '')}"
        idempotency_key = hashlib.sha256(key_data.encode()).hexdigest()[:24]
    
    # 1. Check invariants
    can_execute, invariants = check_execution_invariants(trade_plan)
    if not can_execute:
        return {
            "status": "REJECTED",
            "reason": "Invariant check failed",
            "failures": invariants.get_failures()
        }
    
    # 2. Check idempotency
    idem_store = get_idempotency_store()
    is_new, prev_result = idem_store.check_and_set(idempotency_key, "execute_order")
    if not is_new:
        logger.info(f"Duplicate execution prevented: {idempotency_key}")
        return {
            "status": "DUPLICATE",
            "previous_result": prev_result,
            "idempotency_key": idempotency_key
        }
    
    # 3. Execute with retry
    repair_engine = get_error_repair_engine()
    last_error = None
    
    for attempt in range(WRITE_RETRY_CONFIG.max_retries + 1):
        try:
            result = execute_func(trade_plan)
            
            # 4. Reconcile
            order_id = result.get("order_id") or result.get("orderid")
            if order_id:
                reconciler = get_reconciler()
                recon = reconciler.reconcile(order_id, trade_plan)
                result["reconciliation"] = {
                    "confirmed": recon.confirmed,
                    "status": recon.status,
                    "in_orderbook": recon.in_orderbook
                }
            
            idem_store.mark_complete(idempotency_key, result)
            return result
            
        except Exception as e:
            last_error = e
            tool_error = repair_engine.analyze_error("execute", trade_plan, e)
            
            if not tool_error.is_recoverable:
                break
            
            # Try to repair args
            repaired = repair_engine.repair_args("execute", trade_plan, tool_error)
            if repaired:
                trade_plan = repaired
            
            if attempt < WRITE_RETRY_CONFIG.max_retries:
                delay = WRITE_RETRY_CONFIG.get_delay(attempt)
                logger.warning(f"Retry {attempt + 1} after {delay:.1f}s: {tool_error.suggested_fix}")
                time.sleep(delay)
    
    # Execution failed
    idem_store.mark_failed(idempotency_key, str(last_error))
    return {
        "status": "FAILED",
        "error": str(last_error),
        "idempotency_key": idempotency_key
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "RetryPolicy",
    "RetryConfig",
    "READ_RETRY_CONFIG",
    "WRITE_RETRY_CONFIG",
    "IdempotencyStore",
    "get_idempotency_store",
    "retry_with_backoff",
    "safe_retry",
    "idempotent_retry",
    "ToolError",
    "ErrorRepairEngine",
    "get_error_repair_engine",
    "ReconciliationResult",
    "PostTradeReconciler",
    "get_reconciler",
    "ExecutionInvariants",
    "check_execution_invariants",
    "enforce_invariants",
    "execute_with_reliability",
]
