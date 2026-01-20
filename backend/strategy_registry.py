"""
Strategy Registry - Manages strategies and webhook IDs

Allows agents to create, store, and retrieve strategy configurations
with their associated webhook IDs from OpenAlgo.
"""
import os
import json
import sqlite3
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "strategy_registry.db")


class StrategyRegistry:
    """SQLite-based registry for strategy configurations and webhook IDs."""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                webhook_id TEXT,
                mode TEXT DEFAULT 'BOTH',
                exchange TEXT DEFAULT 'NSE',
                symbols TEXT,
                description TEXT,
                config TEXT,
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS strategy_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                quantity INTEGER,
                response TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (strategy_name) REFERENCES strategies(name)
            )
        """)
        conn.commit()
        conn.close()
        logger.info(f"Strategy registry initialized: {self.db_path}")
    
    def create_strategy(
        self,
        name: str,
        webhook_id: str = None,
        mode: str = "BOTH",
        exchange: str = "NSE",
        symbols: List[str] = None,
        description: str = "",
        config: Dict = None
    ) -> Dict:
        """Create or update a strategy."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO strategies (name, webhook_id, mode, exchange, symbols, description, config)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    webhook_id = COALESCE(excluded.webhook_id, webhook_id),
                    mode = excluded.mode,
                    exchange = excluded.exchange,
                    symbols = excluded.symbols,
                    description = excluded.description,
                    config = excluded.config,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                name,
                webhook_id,
                mode,
                exchange,
                json.dumps(symbols or []),
                description,
                json.dumps(config or {})
            ))
            conn.commit()
            logger.info(f"Strategy created/updated: {name}")
            return {"status": "success", "name": name, "webhook_id": webhook_id}
        except Exception as e:
            logger.error(f"Failed to create strategy: {e}")
            return {"status": "error", "error": str(e)}
        finally:
            conn.close()
    
    def set_webhook_id(self, name: str, webhook_id: str) -> Dict:
        """Set or update webhook ID for a strategy."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "UPDATE strategies SET webhook_id = ?, updated_at = CURRENT_TIMESTAMP WHERE name = ?",
                (webhook_id, name)
            )
            conn.commit()
            if cursor.rowcount > 0:
                return {"status": "success", "name": name, "webhook_id": webhook_id}
            else:
                return {"status": "error", "error": f"Strategy '{name}' not found"}
        finally:
            conn.close()
    
    def get_strategy(self, name: str) -> Optional[Dict]:
        """Get a strategy by name."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT * FROM strategies WHERE name = ?", (name,)
            ).fetchone()
            if row:
                return dict(row)
            return None
        finally:
            conn.close()
    
    def get_webhook_id(self, name: str) -> Optional[str]:
        """Get webhook ID for a strategy."""
        strategy = self.get_strategy(name)
        return strategy.get("webhook_id") if strategy else None
    
    def list_strategies(self, active_only: bool = True) -> List[Dict]:
        """List all strategies."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            query = "SELECT * FROM strategies"
            if active_only:
                query += " WHERE is_active = 1"
            query += " ORDER BY updated_at DESC"
            rows = conn.execute(query).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
    
    def log_order(self, strategy_name: str, symbol: str, action: str, 
                  quantity: int, response: Dict) -> None:
        """Log an order execution."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO strategy_orders (strategy_name, symbol, action, quantity, response)
                VALUES (?, ?, ?, ?, ?)
            """, (strategy_name, symbol, action, quantity, json.dumps(response)))
            conn.commit()
        finally:
            conn.close()
    
    def get_order_history(self, strategy_name: str = None, limit: int = 50) -> List[Dict]:
        """Get order history."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            if strategy_name:
                rows = conn.execute(
                    "SELECT * FROM strategy_orders WHERE strategy_name = ? ORDER BY created_at DESC LIMIT ?",
                    (strategy_name, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM strategy_orders ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
    
    def deactivate_strategy(self, name: str) -> Dict:
        """Deactivate a strategy."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "UPDATE strategies SET is_active = 0, updated_at = CURRENT_TIMESTAMP WHERE name = ?",
                (name,)
            )
            conn.commit()
            return {"status": "success", "name": name, "is_active": False}
        finally:
            conn.close()


# Global instance
_registry = None

def get_strategy_registry() -> StrategyRegistry:
    """Get the global strategy registry instance."""
    global _registry
    if _registry is None:
        _registry = StrategyRegistry()
    return _registry
