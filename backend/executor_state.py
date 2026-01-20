"""
Executor State Manager - Persistent state storage using SQLite

Manages:
- Current execution state (IDLE, WAITING_ENTRY, IN_POSITION, etc.)
- Active strategy information
- Pending orders and their status
- Position tracking
"""
import sqlite3
import json
import os
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database path
DB_PATH = os.path.join(os.path.dirname(__file__), "executor_state.db")


@dataclass
class ExecutorPosition:
    """Current position information."""
    symbol: str = ""
    side: str = ""  # LONG, SHORT, FLAT
    qty: int = 0
    entry_price: float = 0.0
    entry_time: str = ""
    

@dataclass
class ExecutorState:
    """Complete executor state."""
    # Execution state
    status: str = "IDLE"  # IDLE, WAITING_ENTRY, IN_POSITION, WAITING_EXIT, WAITING_FOR_BUILD, STOPPED
    
    # Active strategy
    strategy_name: str = ""
    strategy_priority: int = 0
    
    # Levels (from Supervisor)
    entry_level: float = 0.0
    target_level: float = 0.0
    stop_loss_level: float = 0.0
    invalidation_level: float = 0.0
    
    # Execution params
    qty: int = 0
    order_type: str = "LIMIT"
    product_type: str = "MIS"
    
    # Current position
    position_side: str = "FLAT"  # LONG, SHORT, FLAT
    position_qty: int = 0
    position_entry_price: float = 0.0
    position_entry_time: str = ""
    
    # Pending order tracking
    pending_order_id: str = ""
    pending_order_type: str = ""  # ENTRY, EXIT, STOP_LOSS
    pending_order_status: str = ""  # PENDING, FILLED, REJECTED, PARTIAL, TIMEOUT
    
    # Timestamps
    last_updated: str = ""
    last_command: str = ""
    last_command_time: str = ""
    last_orchestrator_request: str = ""  # Timestamp of last build request


class ExecutorStateManager:
    """Manages persistent state for the Executor Agent."""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()
        logger.info(f"ExecutorStateManager initialized with DB: {db_path}")
    
    def _init_db(self):
        """Initialize the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create state table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS executor_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                state_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # Create order history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS order_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT NOT NULL,
                order_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                qty INTEGER NOT NULL,
                price REAL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                broker_response TEXT
            )
        """)
        
        # Create command log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS command_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                command TEXT NOT NULL,
                payload TEXT,
                source TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        # Initialize state if not exists
        cursor.execute("SELECT COUNT(*) FROM executor_state")
        if cursor.fetchone()[0] == 0:
            default_state = ExecutorState()
            cursor.execute(
                "INSERT INTO executor_state (id, state_json, updated_at) VALUES (1, ?, ?)",
                (json.dumps(asdict(default_state)), datetime.now().isoformat())
            )
        
        conn.commit()
        conn.close()
    
    def get_state(self) -> ExecutorState:
        """Get current executor state."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT state_json FROM executor_state WHERE id = 1")
        row = cursor.fetchone()
        conn.close()
        
        if row:
            data = json.loads(row[0])
            return ExecutorState(**data)
        return ExecutorState()
    
    def save_state(self, state: ExecutorState):
        """Save executor state."""
        state.last_updated = datetime.now().isoformat()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE executor_state SET state_json = ?, updated_at = ? WHERE id = 1",
            (json.dumps(asdict(state)), state.last_updated)
        )
        conn.commit()
        conn.close()
        logger.info(f"State saved: status={state.status}, strategy={state.strategy_name}")
    
    def update_status(self, status: str):
        """Quick update of just the status field."""
        state = self.get_state()
        state.status = status
        self.save_state(state)
    
    def set_strategy(self, strategy_name: str, priority: int = 0):
        """Set the active strategy."""
        state = self.get_state()
        state.strategy_name = strategy_name
        state.strategy_priority = priority
        self.save_state(state)
    
    def set_levels(self, entry: float, target: float, stop_loss: float, invalidation: float):
        """Set trading levels."""
        state = self.get_state()
        state.entry_level = entry
        state.target_level = target
        state.stop_loss_level = stop_loss
        state.invalidation_level = invalidation
        self.save_state(state)
    
    def set_execution_params(self, qty: int, order_type: str = "LIMIT", product_type: str = "MIS"):
        """Set execution parameters."""
        state = self.get_state()
        state.qty = qty
        state.order_type = order_type
        state.product_type = product_type
        self.save_state(state)
    
    def enter_position(self, side: str, qty: int, price: float):
        """Record entering a position."""
        state = self.get_state()
        state.position_side = side
        state.position_qty = qty
        state.position_entry_price = price
        state.position_entry_time = datetime.now().isoformat()
        state.status = "IN_POSITION"
        self.save_state(state)
    
    def exit_position(self):
        """Record exiting a position."""
        state = self.get_state()
        state.position_side = "FLAT"
        state.position_qty = 0
        state.position_entry_price = 0.0
        state.position_entry_time = ""
        state.status = "IDLE"
        self.save_state(state)
    
    def set_pending_order(self, order_id: str, order_type: str, status: str = "PENDING"):
        """Track a pending order."""
        state = self.get_state()
        state.pending_order_id = order_id
        state.pending_order_type = order_type
        state.pending_order_status = status
        self.save_state(state)
    
    def clear_pending_order(self):
        """Clear pending order tracking."""
        state = self.get_state()
        state.pending_order_id = ""
        state.pending_order_type = ""
        state.pending_order_status = ""
        self.save_state(state)
    
    def log_command(self, command: str, payload: dict = None, source: str = "supervisor"):
        """Log a command received."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO command_log (command, payload, source, created_at) VALUES (?, ?, ?, ?)",
            (command, json.dumps(payload) if payload else None, source, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
        
        # Also update state
        state = self.get_state()
        state.last_command = command
        state.last_command_time = datetime.now().isoformat()
        self.save_state(state)
    
    def log_order(self, order_id: str, order_type: str, symbol: str, side: str, 
                  qty: int, price: float, status: str, broker_response: dict = None):
        """Log an order to history."""
        now = datetime.now().isoformat()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO order_history 
               (order_id, order_type, symbol, side, qty, price, status, created_at, updated_at, broker_response) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (order_id, order_type, symbol, side, qty, price, status, now, now, 
             json.dumps(broker_response) if broker_response else None)
        )
        conn.commit()
        conn.close()
    
    def update_order_status(self, order_id: str, status: str, broker_response: dict = None):
        """Update an order's status in history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE order_history SET status = ?, updated_at = ?, broker_response = ? WHERE order_id = ?",
            (status, datetime.now().isoformat(), json.dumps(broker_response) if broker_response else None, order_id)
        )
        conn.commit()
        conn.close()
    
    def get_recent_orders(self, limit: int = 10) -> list:
        """Get recent order history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM order_history ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        conn.close()
        return rows
    
    def reset(self):
        """Reset to default state (for testing/emergency)."""
        default_state = ExecutorState()
        self.save_state(default_state)
        logger.warning("Executor state reset to defaults")
