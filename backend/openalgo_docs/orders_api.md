# OpenAlgo Orders API

Core order management with the `api` client.

## Order Placement

### placeorder()
Regular order.
```python
result = client.placeorder(
    symbol="RELIANCE", exchange="NSE", action="BUY",
    quantity=1, price_type="MARKET", product="MIS"
)
```

### placesmartorder()
Order with position sizing.
```python
result = client.placesmartorder(
    symbol="RELIANCE", exchange="NSE", action="BUY",
    quantity=1, position_size=100, price_type="MARKET", product="MIS"
)
```

### basketorder(orders)
Multiple orders simultaneously.
```python
orders = [
    {"symbol": "RELIANCE", "exchange": "NSE", "action": "BUY",
     "quantity": 1, "pricetype": "MARKET", "product": "MIS"},
    {"symbol": "INFY", "exchange": "NSE", "action": "SELL",
     "quantity": 1, "pricetype": "MARKET", "product": "MIS"}
]
result = client.basketorder(orders=orders)
```

### splitorder()
Split large order into smaller ones.
```python
result = client.splitorder(
    symbol="YESBANK", exchange="NSE", action="SELL",
    quantity=105, splitsize=20, price_type="MARKET", product="MIS"
)
```

## Order Management

### orderstatus(order_id, strategy)
Check order status.
```python
result = client.orderstatus(order_id="24120900146469", strategy="Test Strategy")
```

### openposition(symbol, exchange, product)
Get open position for symbol.
```python
result = client.openposition(symbol="YESBANK", exchange="NSE", product="CNC")
```

### modifyorder()
Modify existing order.
```python
result = client.modifyorder(
    order_id="24120900146469", symbol="RELIANCE", action="BUY",
    exchange="NSE", quantity=2, price="2100", product="MIS", price_type="LIMIT"
)
```

### cancelorder(order_id)
Cancel specific order.
```python
result = client.cancelorder(order_id="24120900146469")
```

### cancelallorder()
Cancel all open orders.
```python
result = client.cancelallorder()
```

### closeposition()
Close all open positions.
```python
result = client.closeposition()
```

## Parameters Reference
- **exchange**: NSE, BSE, NFO, BFO, CDS, MCX
- **product**: CNC (delivery), MIS (intraday), NRML (F&O carry)
- **price_type**: MARKET, LIMIT, SL, SL-M
- **action**: BUY, SELL
