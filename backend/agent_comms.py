"""
Agent Communication Bus

Centralized inter-agent messaging system with SQLite persistence.
Allows agents to send messages to each other with full visibility.
"""
import os
import json
import sqlite3
import uuid
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COMMS_DB = "agent_comms.db"


@dataclass
class AgentMessage:
    """Represents a message between agents."""
    id: str
    timestamp: str
    from_agent: str  # "orchestrator", "analyst", "supervisor", "executor", "user"
    to_agent: str
    message_type: str  # "request", "response", "advisory", "approval", "denial", "info"
    content: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> dict:
        return asdict(self)


class AgentCommunicationBus:
    """Central message bus for inter-agent communication."""
    
    def __init__(self, db_path: str = COMMS_DB):
        self.db_path = db_path
        self._init_db()
        logger.info("🔗 Agent Communication Bus initialized")
    
    def _init_db(self):
        """Initialize SQLite database for message storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                from_agent TEXT NOT NULL,
                to_agent TEXT NOT NULL,
                message_type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON messages(timestamp DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agents ON messages(from_agent, to_agent)
        """)
        conn.commit()
        conn.close()
    
    def send_message(
        self,
        from_agent: str,
        to_agent: str,
        message_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """
        Send a message from one agent to another.
        
        Args:
            from_agent: Sending agent name
            to_agent: Receiving agent name
            message_type: Type of message (request, response, advisory, approval, denial, info)
            content: Message content
            metadata: Optional additional data (e.g., trade_plan, approval_token)
            
        Returns:
            The created AgentMessage
        """
        message = AgentMessage(
            id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=message_type,
            content=content,
            metadata=metadata or {}
        )
        
        # Persist to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO messages (id, timestamp, from_agent, to_agent, message_type, content, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            message.id,
            message.timestamp,
            message.from_agent,
            message.to_agent,
            message.message_type,
            message.content,
            json.dumps(message.metadata)
        ))
        conn.commit()
        conn.close()
        
        logger.info(f"🔗 [{from_agent}] → [{to_agent}]: {message_type} - {content[:50]}...")
        return message
    
    def get_messages(
        self,
        limit: int = 50,
        filter_agent: Optional[str] = None,
        message_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent messages from the bus.
        
        Args:
            limit: Maximum number of messages to return
            filter_agent: Optional agent name to filter by (from or to)
            message_type: Optional message type filter
            
        Returns:
            List of message dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM messages"
        params = []
        conditions = []
        
        if filter_agent:
            conditions.append("(from_agent = ? OR to_agent = ?)")
            params.extend([filter_agent, filter_agent])
        
        if message_type:
            conditions.append("message_type = ?")
            params.append(message_type)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        messages = []
        for row in rows:
            messages.append({
                "id": row["id"],
                "timestamp": row["timestamp"],
                "from_agent": row["from_agent"],
                "to_agent": row["to_agent"],
                "message_type": row["message_type"],
                "content": row["content"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
            })
        
        return messages
    
    def get_conversation(
        self,
        agent1: str,
        agent2: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get conversation between two specific agents.
        
        Args:
            agent1: First agent name
            agent2: Second agent name
            limit: Maximum number of messages
            
        Returns:
            List of messages between the two agents
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM messages
            WHERE (from_agent = ? AND to_agent = ?)
               OR (from_agent = ? AND to_agent = ?)
            ORDER BY timestamp DESC
            LIMIT ?
        """, (agent1, agent2, agent2, agent1, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        messages = []
        for row in rows:
            messages.append({
                "id": row["id"],
                "timestamp": row["timestamp"],
                "from_agent": row["from_agent"],
                "to_agent": row["to_agent"],
                "message_type": row["message_type"],
                "content": row["content"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
            })
        
        return messages
    
    def clear_history(self):
        """Clear all message history (for testing)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages")
        conn.commit()
        conn.close()
        logger.info("🔗 Message history cleared")


# Singleton instance
_comms_bus = None

def get_comms_bus() -> AgentCommunicationBus:
    """Get the singleton communication bus instance."""
    global _comms_bus
    if _comms_bus is None:
        _comms_bus = AgentCommunicationBus()
    return _comms_bus


# Convenience functions for agents to use directly
def send_agent_message(
    from_agent: str,
    to_agent: str,
    message_type: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> AgentMessage:
    """Convenience function to send a message."""
    return get_comms_bus().send_message(from_agent, to_agent, message_type, content, metadata)


def get_agent_messages(limit: int = 50, filter_agent: Optional[str] = None) -> List[Dict[str, Any]]:
    """Convenience function to get messages."""
    return get_comms_bus().get_messages(limit, filter_agent)
