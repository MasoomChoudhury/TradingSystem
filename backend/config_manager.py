import json
import os
import logging
from typing import Dict, Any

logger = logging.getLogger("config_manager")

CONFIG_FILE = "trading_config.json"
DEFAULT_CONFIG = {
    "active_symbol": "BANKNIFTY",
    "market_start": "09:15",
    "market_end": "15:30",
    "run_interval_minutes": 15
}

class ConfigManager:
    @staticmethod
    def load_config() -> Dict[str, Any]:
        """Load config from disk or return defaults."""
        if not os.path.exists(CONFIG_FILE):
             ConfigManager.save_config(DEFAULT_CONFIG)
             return DEFAULT_CONFIG
        
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return DEFAULT_CONFIG

    @staticmethod
    def save_config(config: Dict[str, Any]):
        """Save config to disk."""
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    @staticmethod
    def get_active_symbol() -> str:
        """Helper to get just the symbol."""
        config = ConfigManager.load_config()
        return config.get("active_symbol", "BANKNIFTY")

    @staticmethod
    def set_active_symbol(symbol: str):
        """Helper to set just the symbol."""
        config = ConfigManager.load_config()
        config["active_symbol"] = symbol.upper()
        ConfigManager.save_config(config)
