# OpenAlgo LangGraph Tools Reference

**Total: 56 Tools** for AI-powered algorithmic trading

---

## 📡 WebSocket Tools (12)

| Tool | Description |
|------|-------------|
| `openalgo_connect` | Connect to OpenAlgo WebSocket server |
| `openalgo_disconnect` | Disconnect from WebSocket |
| `openalgo_subscribe_ltp` | Subscribe to Last Traded Price updates |
| `openalgo_subscribe_quote` | Subscribe to OHLC quote updates |
| `openalgo_subscribe_depth` | Subscribe to market depth (order book) |
| `openalgo_get_ltp` | Get current LTP for subscribed instruments |
| `openalgo_get_quotes` | Get current quotes (OHLC) |
| `openalgo_get_depth` | Get current market depth (5 levels) |
| `openalgo_unsubscribe_ltp` | Unsubscribe from LTP feed |
| `openalgo_unsubscribe_quote` | Unsubscribe from quote feed |
| `openalgo_unsubscribe_depth` | Unsubscribe from depth feed |
| `openalgo_get_ws_logs` | Get WebSocket activity logs |

---

## 📊 Technical Indicators (4)

| Tool | Description |
|------|-------------|
| `openalgo_list_indicators` | List available indicators by category |
| `openalgo_get_common_indicators` | Get frequently used indicators |
| `openalgo_calculate_indicator` | Calculate indicator on OHLCV data |
| `openalgo_validate_indicator` | Check if indicator is in approved list |

**Supports 80+ indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, Supertrend, Ichimoku, etc.

---

## 🎯 Strategy Management (9)

| Tool | Description |
|------|-------------|
| `openalgo_strategy_order` | Execute webhook-based strategy order |
| `openalgo_close_position` | Close position via strategy webhook |
| `openalgo_get_strategy_logs` | Get strategy execution logs |
| `create_strategy` | Create strategy in local registry |
| `set_strategy_webhook` | Store webhook ID for a strategy |
| `list_strategies` | List all registered strategies |
| `get_strategy` | Get strategy configuration |
| `execute_strategy_order` | Execute order using stored webhook |
| `get_strategy_order_history` | View strategy order history |

---

## 💰 Account Management (8)

| Tool | Description |
|------|-------------|
| `openalgo_get_funds` | Get available cash and margin |
| `openalgo_get_orderbook` | Get all orders with statistics |
| `openalgo_get_tradebook` | Get executed trades |
| `openalgo_get_positions` | Get current open positions |
| `openalgo_get_holdings` | Get stock holdings with P&L |
| `openalgo_analyzer_status` | Check live/analyze mode |
| `openalgo_analyzer_toggle` | Switch between live and simulated |
| `openalgo_calculate_margin` | Calculate margin for positions |

---

## 📝 Order Management (11)

| Tool | Description |
|------|-------------|
| `openalgo_place_order` | Place regular order |
| `openalgo_place_smart_order` | Place order with position sizing |
| `openalgo_basket_order` | Place multiple orders at once |
| `openalgo_split_order` | Split large order into chunks |
| `openalgo_order_status` | Check specific order status |
| `openalgo_open_position` | Get position for a symbol |
| `openalgo_modify_order` | Modify pending order |
| `openalgo_cancel_order` | Cancel specific order |
| `openalgo_cancel_all_orders` | Cancel ALL pending orders |
| `openalgo_close_all_positions` | Close ALL open positions |
| `openalgo_get_order_logs` | Get order execution history |

---

## 📈 Market Data (7)

| Tool | Description |
|------|-------------|
| `openalgo_get_quotes` | Get real-time quotes (REST) |
| `openalgo_get_market_depth` | Get order book depth |
| `openalgo_get_history` | Get historical OHLC data |
| `openalgo_get_intervals` | Get supported timeframes |
| `openalgo_get_symbol_info` | Get symbol details (lotsize, token) |
| `openalgo_search_symbols` | Search symbols across exchanges |
| `openalgo_get_expiry` | Get F&O expiry dates |

---

## 🎲 Options Trading (4)

| Tool | Description |
|------|-------------|
| `openalgo_option_greeks` | Calculate Delta, Gamma, Theta, Vega, IV |
| `openalgo_option_symbol` | Resolve ATM/ITM/OTM option symbol |
| `openalgo_option_order` | Place option order with offset |
| `openalgo_build_iron_condor` | Build 4-leg Iron Condor strategy |

---

## 🛠️ Utility Tools (5)

| Tool | Description |
|------|-------------|
| `list_files` | List files in directory |
| `read_file` | Read file contents |
| `write_file` | Write to file |
| `execute_python_code` | Execute Python code |
| `deploy_strategy` | Deploy strategy to production |

---

## Quick Reference by Use Case

### Getting Started
```
openalgo_connect → openalgo_subscribe_ltp → openalgo_get_ltp
```

### Place a Trade
```
openalgo_get_funds → openalgo_place_order → openalgo_order_status
```

### Technical Analysis
```
openalgo_get_history → openalgo_calculate_indicator → analyze signals
```

### Options Strategy
```
openalgo_option_symbol → openalgo_option_greeks → openalgo_option_order
```

### Strategy Automation
```
create_strategy → set_strategy_webhook → execute_strategy_order
```

---

## Environment Variables

```bash
OPENALGO_API_KEY=your-api-key
OPENALGO_HOST=http://127.0.0.1:5000
OPENALGO_WS_URL=ws://127.0.0.1:8765
GEMINI_API_KEY=your-gemini-key
```

---

*All tools are LangGraph-compatible and available to the Orchestrator, Supervisor, and Executor agents.*
