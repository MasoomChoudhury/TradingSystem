# Trading System - Agents & Tools Reference

This document describes all agents in the system, their responsibilities, connected tools, and key functions.

---

## 1. 🎯 Orchestrator Agent

**File:** `backend/orchestrator_agent.py`  
**Layer:** 1 (Planning & Routing)  
**Model:** `gemini-3-flash-preview`

### Responsibilities
- Interprets user intent (info, analysis, trade, deploy)
- Creates structured task plans (`TaskPlan`)
- Routes requests to specialist workers (does **not** execute trades)

### Key Functions/Methods
| Function                | Description                                              |
|-------------------------|----------------------------------------------------------|
| `send_message()`        | Main entry point; processes user message and routes.    |
| `_orchestrate()`        | Core LLM orchestration logic.                            |
| `_should_continue()`    | Determines next action from state (tools or end).        |
| `get_history()`         | Retrieves conversation history for a thread.             |

### Connected Tools (5 Safe, Read-Only)
| Tool                       | Description                                                      |
|----------------------------|------------------------------------------------------------------|
| `create_task_plan`         | Creates a structured task plan with steps for execution.        |
| `route_to_worker`          | Routes a specific action to a specialist worker (e.g., `market_data`, `supervisor`). |
| `get_system_status`        | Returns available workers and their capabilities.               |
| `list_registered_strategies` | Lists all registered trading strategies (read-only).            |
| `get_strategy_config`      | Gets configuration of a specific strategy.                       |

---

## 2. 👁️ Market Analyst Agent ("Trading Eyes")

**File:** `backend/market_analyst.py`  
**Layer:** 1.5 (Visual Analysis & Advisory)  
**Model:** `gemini-1.5-pro` (Vision-enabled)

### Responsibilities
- Runs automatically every 15 minutes during market hours (**9:15 AM - 3:30 PM IST**)
- Fetches 15-minute candlestick data from OpenAlgo
- Generates chart images using `mplfinance`
- Sends chart + strategy context to Gemini for visual analysis
- Outputs recommendations: `KEEP`, `STOP`, or `SWITCH`

### Key Functions/Methods
| Function                  | Description                                                   |
|---------------------------|---------------------------------------------------------------|
| `start()`                 | Starts the APScheduler for periodic analysis.                 |
| `stop()`                  | Stops the scheduler.                                           |
| `fetch_data()`            | Fetches 7 days of 15m OHLCV data from OpenAlgo.               |
| `generate_chart()`        | Generates a candlestick chart image with EMA & volume.        |
| `analyze_with_vision()`   | Sends chart image + context to Gemini for visual analysis.    |
| `run_analysis_cycle()`    | Main scheduled task loop: fetch → chart → analyze → store.   |

### Connected Tools
*This agent uses direct OpenAlgo client calls (`client.history()`), not LangChain tools.*

---

## 2.5. 🤖 Auto Trade Executor

**File:** `backend/auto_trade_executor.py`  
**Layer:** 1.5 (Execution Bridge)

### Responsibilities
- Track current position (long/short/flat)
- Execute trades when Analyst signals STOP/SWITCH
- Close positions on session end
- Open initial positions on first KEEP signal

### Key Functions/Methods
| Function                | Description                                              |
|-------------------------|----------------------------------------------------------|
| `close_position()`      | Close current position and record P&L.                   |
| `open_position()`       | Open new position based on strategy bias.                |
| `switch_position()`     | Close current + open opposite position.                  |
| `get_position_status()` | Return current position state.                           |

### Auto-Execution Flow
| Analyst Signal | Action |
|----------------|--------|
| `KEEP` (no position) | Open initial position per strategy bias |
| `KEEP` (has position) | Hold current position |
| `STOP` | Close position, pause session |
| `SWITCH` | Close current, open opposite bias |

---

## 3. 🛡️ Supervisor Agent

**File:** `backend/supervisor_agent.py`  
**Layer:** 2 (Policy Gate & Risk Checks)  
**Model:** `gemini-3-flash-preview`

### Responsibilities
- Validates trade plans against policies
- Enforces risk limits (position size, margin, daily loss)
- Schema validation for `TradePlan` objects
- Issues approval tokens for the Executor

### Key Functions/Methods
| Function                    | Description                                                    |
|-----------------------------|----------------------------------------------------------------|
| `validate()`                | Main entry; validates a trade plan and returns approval/denial.|
| `chat()`                    | Conversational wrapper for supervisor queries.                 |
| `get_history()`             | Retrieves conversation history.                                |

### Connected Tools (7 Guard Tools)
| Tool                        | Description                                                    |
|-----------------------------|----------------------------------------------------------------|
| `check_account_status`      | Check account funds and positions (for exposure checks).       |
| `check_analyzer_mode`       | Check if system is in Analyze (simulated) or Live mode.        |
| `validate_symbol_info`      | Get symbol info for validation (lot size, tick size).          |
| `get_current_ltp`           | Get current LTP for price bound validation.                    |
| `calculate_margin_requirement` | Calculate margin required for a trade.                       |
| `validate_trade_request`    | Combined validation: schema, policy, risk, data integrity.     |
| `get_tool_scope`            | Returns allowed tools for a given worker type.                 |

### Validation Functions (Non-LLM)
| Function                   | Description                                                     |
|----------------------------|-----------------------------------------------------------------|
| `validate_trade_plan_schema` | Checks for required fields in the trade plan.                  |
| `validate_policy`          | Checks against policy rules (allowed exchanges, products, etc).|
| `validate_lot_size`        | Ensures quantity is a multiple of lot size.                     |
| `validate_price_bounds`    | Ensures limit price is within reasonable bounds of LTP.         |

---

## 4. ⚡ Executor Agent

**File:** `backend/executor_agent.py`  
**Layer:** 3 (Broker Execution - **HIGH RISK**)  
**Model:** `gemini-3-flash-preview`

### Responsibilities
- The **ONLY** agent that can place, modify, or cancel orders
- Requires an `ApprovedTradePlan` with a valid `approval_token` from the Supervisor
- Maintains an immutable `ExecutionLog` (SQLite)
- Implements a **circuit breaker** for emergency situations

### Key Functions/Methods
| Function                    | Description                                                     |
|-----------------------------|-----------------------------------------------------------------|
| `chat()`                    | Conversational wrapper for executor queries.                    |
| `execute()`                 | Main entry; executes an approved trade plan.                    |
| `get_history()`             | Retrieves conversation history.                                 |

### `ExecutionLog` Class
| Method                      | Description                                                     |
|-----------------------------|-----------------------------------------------------------------|
| `check_idempotency`         | Checks if a trade was already executed.                         |
| `log_execution`             | Logs an execution (immutable insert).                           |
| `is_circuit_breaker_tripped` | Checks if the circuit breaker is active.                        |
| `trip_circuit_breaker`      | Trips the circuit breaker.                                      |
| `reset_circuit_breaker`     | Resets the circuit breaker.                                     |
| `get_recent_executions`     | Gets recent executions from the log.                            |

### Connected Tools (9 Execution Tools)
| Tool                        | Description                                                     |
|-----------------------------|-----------------------------------------------------------------|
| `execute_approved_order`    | Execute an order with a valid approval token.                   |
| `check_order_status`        | Check status of an executed order.                              |
| `modify_order`              | Modify an existing order (requires approval token).             |
| `cancel_order`              | Cancel a specific order (requires approval token).              |
| `emergency_cancel_all`      | **EMERGENCY:** Cancel all pending orders.                       |
| `emergency_close_positions` | **EMERGENCY:** Close all open positions.                        |
| `emergency_trip_circuit`    | **EMERGENCY:** Trip the circuit breaker.                        |
| `emergency_reset_circuit`   | Reset the circuit breaker.                                      |
| `get_execution_history`     | Get recent executions from the log.                             |

---

## 5. Tool Categories (Available via `backend/tools/`)

These tools are organized by category and can be assigned to agents as needed.

### WebSocket Tools (`openalgo_tools.py`)
| Tool                        | Description                                       |
|-----------------------------|---------------------------------------------------|
| `openalgo_connect`          | Connect to WebSocket server.                      |
| `openalgo_disconnect`       | Disconnect from WebSocket server.                 |
| `openalgo_subscribe_ltp`    | Subscribe to LTP updates.                         |
| `openalgo_subscribe_quote`  | Subscribe to Quote (OHLC) updates.                |
| `openalgo_subscribe_depth`  | Subscribe to Market Depth updates.                |
| `openalgo_get_ltp`          | Get cached LTP data.                              |
| `openalgo_get_quotes`       | Get cached Quote data.                            |
| `openalgo_get_depth`        | Get cached Market Depth data.                     |
| `openalgo_unsubscribe_*`    | Unsubscribe from various feeds.                   |
| `openalgo_get_ws_logs`      | Get WebSocket activity logs.                      |

### Market Data Tools (`openalgo_marketdata.py`)
| Tool                        | Description                                       |
|-----------------------------|---------------------------------------------------|
| `openalgo_get_quotes`       | Get real-time quotes via REST API.                |
| `openalgo_get_market_depth` | Get market depth (order book).                    |
| `openalgo_get_history`      | Get historical OHLC data.                         |
| `openalgo_get_intervals`    | Get supported time intervals.                     |
| `openalgo_get_symbol_info`  | Get detailed symbol information.                  |
| `openalgo_search_symbols`   | Search for symbols across exchanges.              |
| `openalgo_get_expiry`       | Get expiry dates for F&O.                         |

### Indicator Tools (`openalgo_indicators.py`)
| Tool                          | Description                                     |
|-------------------------------|-------------------------------------------------|
| `openalgo_list_indicators`    | List all available indicators.                  |
| `openalgo_get_common_indicators` | Get commonly used indicators.                |
| `openalgo_calculate_indicator` | Calculate a specific indicator on data.        |
| `openalgo_validate_indicator` | Validate indicator parameters.                  |

### Account Tools (`openalgo_accounts.py`)
| Tool                        | Description                                       |
|-----------------------------|---------------------------------------------------|
| `openalgo_get_funds`        | Get account funds.                                |
| `openalgo_get_orderbook`    | Get order book.                                   |
| `openalgo_get_tradebook`    | Get trade book.                                   |
| `openalgo_get_positions`    | Get open positions.                               |
| `openalgo_get_holdings`     | Get holdings (for equity).                        |
| `openalgo_analyzer_status`  | Get analyzer mode status.                         |
| `openalgo_analyzer_toggle`  | Toggle analyzer mode (Live/Analyze).              |
| `openalgo_calculate_margin` | Calculate margin for a trade.                     |

### Order Tools (`openalgo_orders.py`)
| Tool                        | Description                                       |
|-----------------------------|---------------------------------------------------|
| `openalgo_place_order`      | Place a standard order.                           |
| `openalgo_place_smart_order`| Place an order with smart routing.                |
| `openalgo_basket_order`     | Place multiple orders at once.                    |
| `openalgo_split_order`      | Split a large order into smaller ones.            |
| `openalgo_order_status`     | Get status of an order.                           |
| `openalgo_open_position`    | Open a new position.                              |
| `openalgo_modify_order`     | Modify an existing order.                         |
| `openalgo_cancel_order`     | Cancel an order.                                  |
| `openalgo_cancel_all_orders`| Cancel all pending orders.                        |
| `openalgo_close_all_positions` | Close all open positions.                      |

### Options Tools (`openalgo_options.py`)
| Tool                        | Description                                       |
|-----------------------------|---------------------------------------------------|
| `openalgo_option_greeks`    | Calculate option greeks.                          |
| `openalgo_option_symbol`    | Get option symbol for a strike.                   |
| `openalgo_option_order`     | Place an option order.                            |
| `openalgo_build_iron_condor`| Build an Iron Condor strategy.                    |

### Strategy Tools (`openalgo_strategy.py`, `strategy_management.py`)
| Tool                        | Description                                       |
|-----------------------------|---------------------------------------------------|
| `openalgo_strategy_order`   | Place an order via a strategy webhook.            |
| `openalgo_close_position`   | Close a position via strategy.                    |
| `openalgo_get_strategy_logs`| Get strategy execution logs.                      |
| `create_strategy`           | Create a new strategy.                            |
| `set_strategy_webhook`      | Set a webhook for a strategy.                     |
| `list_strategies`           | List all strategies.                              |
| `get_strategy`              | Get details of a strategy.                        |
| `execute_strategy_order`    | Execute an order for a strategy.                  |
| `get_strategy_order_history`| Get order history for a strategy.                 |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER REQUEST                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │        🎯 ORCHESTRATOR (Layer 1)      │
        │  Intent → Plan → Route to Workers     │
        │   (5 Safe, Read-Only Tools)           │
        └───────────────────┬───────────────────┘
                            │
         ┌──────────────────┴──────────────────┐
         ▼                                     ▼
┌─────────────────────┐           ┌─────────────────────┐
│  👁️ MARKET ANALYST  │           │   🛡️ SUPERVISOR     │
│  (Visual Advisor)   │◄─────────►│   (Policy Gate)     │
│ 15m Chart Analysis  │  Advisory │   (7 Guard Tools)   │
└─────────────────────┘           └──────────┬──────────┘
                                            │ Approval Token
                                            ▼
                               ┌─────────────────────┐
                               │   ⚡ EXECUTOR        │
                               │  (Broker Execution) │
                               │  (9 Execution Tools)│
                               │  Circuit Breaker    │
                               └─────────────────────┘
```
