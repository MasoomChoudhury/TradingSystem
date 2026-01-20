# OpenAlgo Accounts API

Account management with the `api` client.

## Methods

### funds()
Get available cash and margin details.
```python
result = client.funds()
# Returns: availablecash, collateral, m2mrealized, m2munrealized, utiliseddebits
```

### orderbook()
Get all orders with statistics.
```python
result = client.orderbook()
# Returns: order list, total buy/sell, completed/open/rejected counts
```

### tradebook()
Get executed trades.
```python
result = client.tradebook()
# Returns: symbol, action, quantity, avgprice, tradevalue, timestamp, orderid
```

### positionbook()
Get current positions.
```python
result = client.positionbook()
# Returns: symbol, exchange, product, quantity, avgprice
```

### holdings()
Get stock holdings with P&L.
```python
result = client.holdings()
# Returns: holdings list, totalholdingvalue, totalinvvalue, totalpnlpercentage
```

### analyzerstatus()
Get analyze/live mode status.
```python
result = client.analyzerstatus()
# Returns: analyze_mode (bool), mode (live/analyze), total_logs
```

### analyzertoggle(mode)
Switch between analyze (simulated) and live mode.
```python
client.analyzertoggle(mode=True)   # Analyze mode (simulated)
client.analyzertoggle(mode=False)  # Live mode (real orders)
```

### margin(positions)
Calculate margin for positions (max 50).
```python
result = client.margin(positions=[{
    "symbol": "SBIN",
    "exchange": "NSE",      # NSE, BSE, NFO, BFO, CDS, MCX
    "action": "BUY",
    "product": "MIS",       # CNC (delivery), MIS (intraday), NRML (F&O)
    "pricetype": "LIMIT",   # MARKET, LIMIT, SL, SL-M
    "quantity": "10",
    "price": "750.50"
}])
# Returns: total_margin_required, span_margin, exposure_margin
```

## Notes (Kotak Broker)
- Single position only, aggregated for multiple positions
