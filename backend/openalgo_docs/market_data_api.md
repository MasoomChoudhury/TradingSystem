# OpenAlgo Market Data API

## REST Data API

### quotes(symbol, exchange)
Get real-time quotes.
```python
result = client.quotes(symbol="RELIANCE", exchange="NSE")
```

### depth(symbol, exchange)
Get market depth (top 5 bids/asks).
```python
result = client.depth(symbol="RELIANCE", exchange="NSE")
```

### history(symbol, exchange, interval, start_date, end_date)
Get historical OHLC data as DataFrame.
```python
result = client.history(
    symbol="RELIANCE", exchange="NSE", interval="5m",
    start_date="2024-01-01", end_date="2024-01-31"
)
```

### intervals()
Get supported time intervals.
```python
result = client.intervals()
# Returns: seconds: [1s], minutes: [1m,2m,3m,5m,10m,15m,30m,60m], days: [D]
```

### symbol(symbol, exchange)
Get symbol details (lotsize, token, expiry, etc.).
```python
result = client.symbol(symbol="NIFTY24APR25FUT", exchange="NFO")
```

### search(query, exchange=None)
Search for symbols.
```python
result = client.search(query="NIFTY", exchange="NFO")
# Exchanges: NSE, NFO, BSE, BFO, MCX, CDS, BCD, NCDEX, NSE_INDEX, BSE_INDEX, MCX_INDEX
```

### expiry(symbol, exchange, instrumenttype)
Get expiry dates for F&O.
```python
result = client.expiry(symbol="NIFTY", exchange="NFO", instrumenttype="futures")
result = client.expiry(symbol="NIFTY", exchange="NFO", instrumenttype="options")
```

## WebSocket Feed API

### LTP Feed
```python
client.connect()
client.subscribe_ltp(instruments, on_data_received=callback)
client.get_ltp()  # Returns: {"ltp": {"MCX": {"GOLD...": {"ltp": 9529.0}}}}
client.unsubscribe_ltp(instruments)
```

### Quote Feed
```python
client.subscribe_quote(instruments)
client.get_quotes()  # Returns: open, high, low, close, ltp
client.unsubscribe_quote(instruments)
```

### Depth Feed
```python
client.subscribe_depth(instruments)
client.get_depth()  # Returns: buyBook, sellBook with 5 levels
client.unsubscribe_depth(instruments)
```
