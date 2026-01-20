"""
Task Status Tracker - Real-time task status visibility

Tracks running agent tasks and their progress for UI display.
"""
import os
import json
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TASK_DB = os.path.join(os.path.dirname(__file__), "task_status.db")


class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskStep:
    """A single step in a task."""
    step_id: int
    name: str
    status: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[str] = None


@dataclass
class TaskStatus:
    """Status of a running task."""
    task_id: str
    name: str
    description: str
    state: TaskState
    progress: int  # 0-100
    current_step: str
    steps: List[TaskStep]
    started_at: str
    updated_at: str
    completed_at: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None


class TaskTracker:
    """
    Tracks running agent tasks for real-time UI updates.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_db()
            cls._instance._active_tasks: Dict[str, TaskStatus] = {}
        return cls._instance
    
    def _init_db(self):
        conn = sqlite3.connect(TASK_DB)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                state TEXT DEFAULT 'pending',
                progress INTEGER DEFAULT 0,
                current_step TEXT,
                started_at TIMESTAMP,
                updated_at TIMESTAMP,
                completed_at TIMESTAMP,
                result TEXT,
                error TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS task_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                step_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                result TEXT,
                FOREIGN KEY (task_id) REFERENCES tasks(task_id)
            )
        """)
        conn.commit()
        conn.close()
    
    def start_task(self, task_id: str, name: str, description: str = "", steps: List[str] = None) -> TaskStatus:
        """Start a new task."""
        now = datetime.now().isoformat()
        
        task = TaskStatus(
            task_id=task_id,
            name=name,
            description=description,
            state=TaskState.RUNNING,
            progress=0,
            current_step=steps[0] if steps else "Initializing...",
            steps=[TaskStep(i, s, "pending") for i, s in enumerate(steps or [])],
            started_at=now,
            updated_at=now
        )
        
        self._active_tasks[task_id] = task
        self._persist_task(task)
        
        logger.info(f"📋 Task started: {name} ({task_id})")
        return task
    
    def update_step(self, task_id: str, step_name: str, status: str = "running", result: str = None) -> Optional[TaskStatus]:
        """Update a task step."""
        task = self._active_tasks.get(task_id)
        if not task:
            return None
        
        now = datetime.now().isoformat()
        task.current_step = step_name
        task.updated_at = now
        
        # Update step in list
        for step in task.steps:
            if step.name == step_name:
                step.status = status
                if status == "running":
                    step.started_at = now
                elif status in ["completed", "failed"]:
                    step.completed_at = now
                    step.result = result
                break
        
        # Calculate progress
        completed = sum(1 for s in task.steps if s.status == "completed")
        task.progress = int((completed / len(task.steps)) * 100) if task.steps else 0
        
        self._persist_task(task)
        return task
    
    def complete_task(self, task_id: str, result: str = None) -> Optional[TaskStatus]:
        """Mark task as completed."""
        task = self._active_tasks.get(task_id)
        if not task:
            return None
        
        now = datetime.now().isoformat()
        task.state = TaskState.COMPLETED
        task.progress = 100
        task.current_step = "Completed"
        task.completed_at = now
        task.updated_at = now
        task.result = result
        
        self._persist_task(task)
        self._cleanup_task(task_id)
        
        logger.info(f"✅ Task completed: {task.name} ({task_id})")
        return task
    
    def fail_task(self, task_id: str, error: str) -> Optional[TaskStatus]:
        """Mark task as failed."""
        task = self._active_tasks.get(task_id)
        if not task:
            return None
        
        now = datetime.now().isoformat()
        task.state = TaskState.FAILED
        task.current_step = "Failed"
        task.updated_at = now
        task.completed_at = now
        task.error = error
        
        self._persist_task(task)
        self._cleanup_task(task_id)
        
        logger.error(f"❌ Task failed: {task.name} - {error}")
        return task
    
    def get_task(self, task_id: str) -> Optional[TaskStatus]:
        """Get task status."""
        return self._active_tasks.get(task_id)
    
    def get_active_tasks(self) -> List[Dict]:
        """Get all active tasks for UI display."""
        return [
            {
                "task_id": t.task_id,
                "name": t.name,
                "description": t.description,
                "state": t.state.value,
                "progress": t.progress,
                "current_step": t.current_step,
                "started_at": t.started_at,
                "updated_at": t.updated_at,
                "steps": [
                    {"name": s.name, "status": s.status, "result": s.result}
                    for s in t.steps
                ]
            }
            for t in self._active_tasks.values()
        ]
    
    def get_recent_tasks(self, limit: int = 10) -> List[Dict]:
        """Get recently completed tasks from DB."""
        conn = sqlite3.connect(TASK_DB)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT * FROM tasks 
            WHERE state IN ('completed', 'failed')
            ORDER BY completed_at DESC 
            LIMIT ?
        """, (limit,)).fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def _persist_task(self, task: TaskStatus):
        """Persist task to DB."""
        conn = sqlite3.connect(TASK_DB)
        conn.execute("""
            INSERT OR REPLACE INTO tasks 
            (task_id, name, description, state, progress, current_step, 
             started_at, updated_at, completed_at, result, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task.task_id, task.name, task.description, task.state.value,
            task.progress, task.current_step, task.started_at, task.updated_at,
            task.completed_at, task.result, task.error
        ))
        
        # Persist steps
        conn.execute("DELETE FROM task_steps WHERE task_id = ?", (task.task_id,))
        for step in task.steps:
            conn.execute("""
                INSERT INTO task_steps (task_id, step_id, name, status, started_at, completed_at, result)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (task.task_id, step.step_id, step.name, step.status, 
                  step.started_at, step.completed_at, step.result))
        
        conn.commit()
        conn.close()
    
    def _cleanup_task(self, task_id: str, delay_seconds: int = 30):
        """Remove task from active list after delay."""
        import threading
        def cleanup():
            import time
            time.sleep(delay_seconds)
            if task_id in self._active_tasks:
                del self._active_tasks[task_id]
        
        threading.Thread(target=cleanup, daemon=True).start()


def get_task_tracker() -> TaskTracker:
    return TaskTracker()


# Convenience functions for use in agents
def start_agent_task(name: str, steps: List[str] = None, description: str = "") -> str:
    """Start a task and return task_id."""
    import uuid
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    tracker = get_task_tracker()
    tracker.start_task(task_id, name, description, steps or ["Processing..."])
    return task_id


def update_agent_task(task_id: str, step: str, status: str = "running", result: str = None):
    """Update task step."""
    tracker = get_task_tracker()
    tracker.update_step(task_id, step, status, result)


def complete_agent_task(task_id: str, result: str = None):
    """Complete task."""
    tracker = get_task_tracker()
    tracker.complete_task(task_id, result)


def fail_agent_task(task_id: str, error: str):
    """Fail task."""
    tracker = get_task_tracker()
    tracker.fail_task(task_id, error)


__all__ = [
    "TaskState",
    "TaskStep",
    "TaskStatus",
    "TaskTracker",
    "get_task_tracker",
    "start_agent_task",
    "update_agent_task",
    "complete_agent_task",
    "fail_agent_task",
]
