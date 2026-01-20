"""
TradingSystem Backend - Agent-Only Architecture

A FastAPI server that provides:
- Orchestrator Agent: Strategy creation & chat
- Supervisor Agent: Market regime analysis
- Executor Agent: Trade command execution
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncio
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)

# ==================== FASTAPI SETUP ====================
app = FastAPI(title="TradingSystem Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "tauri://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== WEBSOCKETS ====================
active_websockets: list[WebSocket] = []

async def _broadcast_to_sockets(sockets_list: list[WebSocket], message: str):
    for ws in sockets_list[:]:
        try:
            await ws.send_text(message)
        except:
            if ws in sockets_list:
                sockets_list.remove(ws)

@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_websockets.remove(websocket)

# ==================== BASIC ROUTES ====================
@app.get("/")
def read_root():
    return {"status": "ok", "message": "TradingSystem Agent API"}

@app.get("/api/health")
def health_check():
    """Health check endpoint for Railway."""
    return {
        "status": "healthy",
        "service": "trading-agents",
        "version": "1.0.0"
    }

@app.get("/api/ping")
def ping():
    return {"message": "Pong from Python"}

# ==================== ORCHESTRATOR AGENT ====================
from orchestrator_agent import OrchestratorAgent

orchestrator = OrchestratorAgent()

class ChatMessage(BaseModel):
    message: str
    thread_id: str = "default"

@app.post("/api/chat")
async def chat_agent(body: ChatMessage):
    result = await orchestrator.send_message(body.message, body.thread_id)
    return {
        "role": "ai",
        "content": result["response"],
        "tool_calls": result.get("tool_calls", [])
    }

@app.get("/api/chat/history")
async def chat_history(thread_id: str = "default"):
    return orchestrator.get_history(thread_id)

# ==================== SUPERVISOR AGENT ====================
from supervisor_agent import SupervisorAgent

supervisor = SupervisorAgent()

class SupervisorRequest(BaseModel):
    message: str
    csv_data: str = ""
    image_path: str = ""
    strategy_json: Optional[dict] = None
    thread_id: str = "supervisor_default"

@app.post("/api/supervisor/analyze")
async def supervisor_analyze(body: SupervisorRequest):
    """Main Supervisor analysis endpoint."""
    
    # Fetch current Executor state
    executor_state = executor_agent.get_status()
    previous_state = {
        "position": executor_state.get("position", {}),
        "last_command": executor_state.get("last_command", "NONE"),
        "status": executor_state.get("status", "IDLE")
    }

    result = await supervisor.analyze(
        message=body.message,
        csv_data=body.csv_data,
        image_path=body.image_path,
        strategy_json=body.strategy_json,
        previous_state=previous_state,
        thread_id=body.thread_id
    )
    return {
        "role": "ai",
        "content": result["response"],
        "regime_status": result.get("regime_status", "UNKNOWN")
    }

@app.post("/api/supervisor/chat")
async def supervisor_chat(body: ChatMessage):
    """Simple chat with Supervisor (no market data)."""
    result = await supervisor.analyze(
        message=body.message,
        thread_id=body.thread_id
    )
    return {
        "role": "ai",
        "content": result["response"],
        "regime_status": result.get("regime_status", "UNKNOWN")
    }

@app.get("/api/supervisor/history")
async def supervisor_history(thread_id: str = "supervisor_default"):
    return supervisor.get_history(thread_id)

# ==================== EXECUTOR AGENT ====================
from executor_agent import ExecutorAgent

executor_agent = ExecutorAgent()

class ExecutorCommand(BaseModel):
    command: str  # LOCK, SWITCH, STOP
    strategy_name: str = ""
    levels: Optional[dict] = None
    execution_params: Optional[dict] = None
    thread_id: str = "executor_default"

@app.post("/api/executor/command")
async def executor_command(body: ExecutorCommand):
    """Receive command from Supervisor."""
    result = await executor_agent.execute_command(
        command=body.command,
        strategy_name=body.strategy_name,
        levels=body.levels,
        execution_params=body.execution_params,
        thread_id=body.thread_id
    )
    return {
        "role": "ai",
        "content": result["response"],
        "status": result.get("status", "UNKNOWN"),
        "strategy": result.get("strategy", ""),
        "position": result.get("position", "FLAT")
    }

@app.get("/api/executor/status")
async def executor_status():
    """Get current executor status."""
    return executor_agent.get_status()

@app.post("/api/executor/chat")
async def executor_chat(body: ChatMessage):
    """Chat with Executor agent."""
    result = await executor_agent.chat(body.message, body.thread_id)
    return {
        "role": "ai",
        "content": result["response"],
        "status": result.get("status", "UNKNOWN")
    }

@app.get("/api/executor/history")
async def executor_history(thread_id: str = "executor_default"):
    return executor_agent.get_history(thread_id)

# ==================== OPENALGO WEBSOCKET LOGS ====================
from tools import get_ws_logs, clear_ws_logs

@app.get("/api/openalgo/ws-logs")
async def openalgo_ws_logs(limit: int = 50):
    """Get recent OpenAlgo WebSocket logs."""
    logs = get_ws_logs(limit)
    return {"logs": logs}

@app.delete("/api/openalgo/ws-logs")
async def openalgo_clear_ws_logs():
    """Clear OpenAlgo WebSocket logs."""
    clear_ws_logs()
    return {"status": "cleared"}

@app.get("/api/openalgo/status")
async def openalgo_status():
    """Get OpenAlgo connection status."""
    try:
        from tools.openalgo_tools import get_openalgo_client
        import os
        
        host = os.environ.get("OPENALGO_HOST", "http://127.0.0.1:5000")
        ws_url = os.environ.get("OPENALGO_WS_URL", "ws://127.0.0.1:8765")
        
        # Try to initialize and test the client
        client = get_openalgo_client()
        
        # Test connection by calling a simple API
        try:
            # Try to get funds as a simple connectivity test
            result = client.funds()
            connected = result is not None and "error" not in str(result).lower()
        except Exception:
            connected = False
        
        return {
            "connected": connected,
            "authenticated": connected,
            "ws_url": ws_url,
            "host": host
        }
    except ImportError:
        return {"connected": False, "error": "OpenAlgo library not installed", "ws_url": "N/A"}
    except Exception as e:
        return {"connected": False, "error": str(e), "ws_url": "N/A"}

# ==================== OPENALGO STRATEGY LOGS ====================
from tools import get_strategy_logs

@app.get("/api/openalgo/strategy-logs")
async def openalgo_strategy_logs(limit: int = 20):
    """Get recent strategy order execution logs."""
    logs = get_strategy_logs(limit)
    return {"logs": logs, "count": len(logs)}

# ==================== OPENALGO ANALYZER CONTROL ====================
from tools import openalgo_analyzer_status, openalgo_analyzer_toggle

@app.get("/api/openalgo/analyzer-status")
async def get_analyzer_status():
    """Get current analyzer mode status."""
    return openalgo_analyzer_status()

# ==================== MARKET ANALYST (TRADING EYES) ====================
from market_analyst import market_analyst

# Keep-alive for Render free tier (prevents sleep after 15 min inactivity)
import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler

keep_alive_scheduler = AsyncIOScheduler()

async def keep_alive_ping():
    """Ping own health endpoint to prevent Render from sleeping."""
    try:
        port = os.environ.get("PORT", "8000")
        async with httpx.AsyncClient() as client:
            await client.get(f"http://127.0.0.1:{port}/api/health", timeout=5.0)
        logging.info("🏓 Keep-alive ping successful")
    except Exception as e:
        logging.warning(f"Keep-alive ping failed: {e}")

@app.on_event("startup")
async def start_scheduler():
    """Start the Market Analyst scheduler and keep-alive."""
    market_analyst.start()
    # Add keep-alive job (every 5 minutes)
    keep_alive_scheduler.add_job(keep_alive_ping, 'interval', minutes=5)
    keep_alive_scheduler.start()
    logging.info("🏓 Keep-alive scheduler started (every 5 min)")

@app.on_event("shutdown")
async def stop_scheduler():
    """Stop the Market Analyst scheduler and keep-alive."""
    market_analyst.stop()
    keep_alive_scheduler.shutdown()

@app.post("/api/analyst/run")
async def run_analyst():
    """Manually trigger the Market Analyst."""
    # Run in background to not block
    import asyncio
    asyncio.create_task(market_analyst.run_analysis_cycle())
    return {"status": "started", "message": "Market analysis triggered in background"}

@app.get("/api/analyst/latest")
async def get_latest_analysis():
    """Get the latest market analysis result and chart."""
    from market_analyst import market_analyst
    
    # Check if chart exists
    chart_path = "temp/latest_chart.png"
    has_chart = os.path.exists(chart_path)
    
    if hasattr(market_analyst, 'last_analysis_result') and market_analyst.last_analysis_result:
        # Return persisted result with inputs
        return {
            **market_analyst.last_analysis_result,
            "has_chart": has_chart,
            "chart_url": "/api/analyst/chart" if has_chart else None
        }
    
    # Placeholder response if no analysis yet
    return {
        "timestamp": datetime.now().isoformat(),
        "recommendation": "WAIT", 
        "confidence": 0.0,
        "observations": ["Waiting for first analysis..."],
        "has_chart": has_chart,
        "chart_url": "/api/analyst/chart" if has_chart else None
    }

@app.get("/api/analyst/chart")
async def get_analyst_chart():
    """Get the latest analyzed chart image."""
    from fastapi.responses import FileResponse
    chart_path = "temp/latest_chart.png"
    if os.path.exists(chart_path):
        return FileResponse(chart_path)
    return {"error": "No chart available"}
@app.post("/api/openalgo/analyzer-toggle")
async def toggle_analyzer_mode(mode: bool):
    """Toggle analyzer mode. mode=True for analyze (simulated), mode=False for live."""
    try:
        from tools.openalgo_tools import get_openalgo_client
        client = get_openalgo_client()
        result = client.analyzertoggle(mode=mode)
        return result
    except Exception as e:
        return {"status": "error", "error": str(e)}

# ==================== TASK STATUS API ====================
from task_tracker import get_task_tracker

@app.get("/api/tasks/active")
async def get_active_tasks():
    """Get all active/running tasks."""
    tracker = get_task_tracker()
    return {"tasks": tracker.get_active_tasks()}

@app.get("/api/tasks/recent")
async def get_recent_tasks(limit: int = 10):
    """Get recently completed tasks."""
    tracker = get_task_tracker()
    return {"tasks": tracker.get_recent_tasks(limit)}

@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a specific task."""
    tracker = get_task_tracker()
    task = tracker.get_task(task_id)
    if task:
        return {
            "task_id": task.task_id,
            "name": task.name,
            "state": task.state.value,
            "progress": task.progress,
            "current_step": task.current_step,
            "started_at": task.started_at,
            "updated_at": task.updated_at
        }
    return {"error": "Task not found"}

# ==================== AGENT COMMUNICATION API ====================
from agent_comms import get_comms_bus, send_agent_message

@app.get("/api/comms/history")
async def get_comms_history(limit: int = 50, agent: Optional[str] = None):
    """Get inter-agent communication history."""
    bus = get_comms_bus()
    messages = bus.get_messages(limit=limit, filter_agent=agent)
    return {"messages": messages}

@app.get("/api/comms/conversation")
async def get_conversation(agent1: str, agent2: str, limit: int = 20):
    """Get conversation between two specific agents."""
    bus = get_comms_bus()
    messages = bus.get_conversation(agent1, agent2, limit)
    return {"messages": messages}

@app.delete("/api/comms/history")
async def clear_comms_history():
    """Clear all inter-agent communication history."""
    bus = get_comms_bus()
    bus.clear_history()
    return {"status": "cleared"}

# ==================== TRADING SESSION API ====================
from trading_session import get_session_manager

class MarketReportRequest(BaseModel):
    report: str

@app.post("/api/session/start")
async def start_trading_session(request: MarketReportRequest):
    """Start a new trading session from a market report."""
    manager = get_session_manager()
    session = manager.start_session(request.report)
    return {"status": "started", "session": session.to_dict()}

@app.get("/api/session/status")
async def get_session_status():
    """Get current trading session status."""
    manager = get_session_manager()
    return manager.get_session_status()

@app.post("/api/session/switch")
async def switch_strategy(reason: str = "Manual switch"):
    """Switch to alternate strategy."""
    manager = get_session_manager()
    new_strategy = manager.switch_strategy(reason)
    if new_strategy:
        return {"status": "switched", "new_strategy": new_strategy.to_dict()}
    return {"status": "error", "message": "No active session"}

@app.post("/api/session/pause")
async def pause_session(reason: str = "Manual pause"):
    """Pause the trading session."""
    manager = get_session_manager()
    manager.pause_session(reason)
    return {"status": "paused"}

@app.post("/api/session/resume")
async def resume_session():
    """Resume a paused session."""
    manager = get_session_manager()
    manager.resume_session()
    return {"status": "resumed"}

@app.post("/api/session/end")
async def end_trading_session():
    """End the trading session and get summary."""
    manager = get_session_manager()
    # First close any open positions
    from auto_trade_executor import get_auto_executor
    executor = get_auto_executor()
    await executor.close_position("Session ended")
    summary = manager.end_session()
    return {"status": "ended", "summary": summary}

# ==================== POSITION & AUTO-TRADING API ====================
from auto_trade_executor import get_auto_executor

@app.get("/api/position/status")
async def get_position_status():
    """Get current position status."""
    executor = get_auto_executor()
    return executor.get_position_status()

@app.post("/api/position/close")
async def close_position(reason: str = "Manual close"):
    """Manually close current position."""
    executor = get_auto_executor()
    result = await executor.close_position(reason)
    return result

@app.post("/api/position/open")
async def open_position():
    """Open position based on current active strategy."""
    executor = get_auto_executor()
    manager = get_session_manager()
    
    if not manager.current_session:
        return {"status": "error", "message": "No active session"}
    
    strategy = manager.current_session.get_active_strategy()
    if not strategy:
        return {"status": "error", "message": "No active strategy"}
    
    result = await executor.open_position(strategy, "Manual entry")
    return result

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)



