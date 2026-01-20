# OpenAlgo Strategy API

Webhook-based strategy management for automated trading.

## Import
```python
from openalgo import Strategy
```

## Initialization
```python
client = Strategy(
    host_url="http://127.0.0.1:5000",  # OpenAlgo server URL
    webhook_id="your-webhook-id"        # From OpenAlgo strategy section
)
```

## Strategy Order
```python
# Parameters: symbol, action, position_size
response = client.strategyorder("RELIANCE", "BUY", 1)   # Long entry
response = client.strategyorder("ZOMATO", "SELL", 1)    # Short entry
response = client.strategyorder("RELIANCE", "SELL", 0)  # Close long
response = client.strategyorder("ZOMATO", "BUY", 0)     # Close short
```

## Strategy Modes
| Mode | Description |
|------|-------------|
| LONG_ONLY | Only processes BUY signals |
| SHORT_ONLY | Only processes SELL signals |
| BOTH | Processes both BUY and SELL with position sizing |

## Position Size Logic
- `position_size > 0`: Enter position with specified quantity
- `position_size = 0`: Close existing position

## Integration Points
- Custom trading systems
- Technical analysis platforms
- Alert systems
- Automated trading bots
- Any HTTP-capable system
