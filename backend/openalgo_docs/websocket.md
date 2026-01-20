# OpenAlgo WebSocket Documentation

## Verbose Control

The `verbose` parameter manages SDK-level logging for WebSocket feed operations (LTP, Quote, Depth).

### Verbose Levels

| Level | Value | Description |
|-------|-------|-------------|
| Silent | `False` or `0` | Errors only (default) |
| Basic | `True` or `1` | Connection, authentication, subscription logs |
| Debug | `2` | All market data updates (LTP/Quote/Depth) |

### Log Categories

| Tag | Meaning |
|-----|---------|
| `[WS]` | WebSocket connection events |
| `[AUTH]` | Authentication requests & responses |
| `[SUB]` | Subscription operations |
| `[UNSUB]` | Unsubscription logs |
| `[LTP]` | LTP updates (verbose=2) |
| `[QUOTE]` | Quote updates (verbose=2) |
| `[DEPTH]` | Market depth updates (verbose=2) |
| `[ERROR]` | Error messages (always shown) |

### Usage Example

```python
from openalgo import api

# Silent mode (default)
client = api(api_key="...", host="...", ws_url="...", verbose=False)

# Basic logging
client = api(api_key="...", host="...", ws_url="...", verbose=True)

# Full debug
client = api(api_key="...", host="...", ws_url="...", verbose=2)
```

### Subscription Methods

- `client.connect()` - Connect to WebSocket
- `client.disconnect()` - Disconnect from WebSocket
- `client.subscribe_ltp(instruments, on_data_received=callback)` - Subscribe to LTP
- `client.subscribe_quote(instruments, on_data_received=callback)` - Subscribe to quotes
- `client.subscribe_depth(instruments, on_data_received=callback)` - Subscribe to depth
- `client.unsubscribe_ltp(instruments)` - Unsubscribe from LTP
- `client.unsubscribe_quote(instruments)` - Unsubscribe from quotes
- `client.unsubscribe_depth(instruments)` - Unsubscribe from depth
- `client.get_ltp()` - Get cached LTP data
- `client.get_quotes()` - Get cached quote data
- `client.get_depth()` - Get cached depth data

### Instrument Format

```python
instruments = [
    {"exchange": "NSE_INDEX", "symbol": "NIFTY"},
    {"exchange": "NSE", "symbol": "INFY"}
]
```
