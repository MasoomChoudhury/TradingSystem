"""
Inter-Agent Communication Layer

Provides a robust HTTP client for agents to communicate with each other.
Features:
- Async HTTP calls using httpx
- Automatic retries (3 attempts with backoff)
- Timeout protection (5s)
- Centralized logging
"""
import httpx
import logging
import asyncio
import json
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base URL for local API
BASE_URL = "http://127.0.0.1:8000/api"

class AgentComm:
    """Robust communication client for inter-agent messaging."""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=5.0)
    
    async def _post(self, endpoint: str, payload: Dict[str, Any], retries: int = 3) -> Optional[Dict]:
        """Send POST request with retries."""
        url = f"{BASE_URL}{endpoint}"
        attempt = 0
        
        while attempt < retries:
            try:
                response = await self.client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                attempt += 1
                logger.warning(f"Comm error ({url}): {e}. Retrying {attempt}/{retries}...")
                await asyncio.sleep(1 * attempt)  # Exponential backoff
            except Exception as e:
                logger.error(f"Unexpected comm error: {e}")
                break
        
        logger.error(f"Failed to communicate with {endpoint} after {retries} attempts")
        return None

    async def send_to_executor(self, command: str, strategy_name: str = "", levels: dict = None, execution_params: dict = None) -> bool:
        """
        Send command to Executor Agent.
        
        Args:
            command: LOCK, SWITCH, STOP
            strategy_name: Name of strategy file
            levels: Entry/target levels
            execution_params: Order params
        """
        payload = {
            "command": command,
            "strategy_name": strategy_name,
            "levels": levels,
            "execution_params": execution_params,
            "thread_id": "comm_layer_auto"
        }
        
        logger.info(f"Sending to Executor: {command} {strategy_name}")
        result = await self._post("/executor/command", payload)
        
        if result and result.get("status") != "UNKNOWN":
            return True
        return False

    async def send_to_orchestrator(self, message: str) -> bool:
        """
        Send chat message to Orchestrator (e.g., to request strategy build).
        """
        payload = {
            "message": message,
            "thread_id": "comm_layer_auto"
        }
        
        logger.info(f"Sending to Orchestrator: {message[:50]}...")
        result = await self._post("/chat", payload)
        
        if result and result.get("role") == "ai":
            return True
        return False

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# Singleton instance
agent_comm = AgentComm()
