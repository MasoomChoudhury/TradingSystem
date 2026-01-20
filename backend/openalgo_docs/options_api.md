# OpenAlgo Options API

Advanced options trading with Greeks, auto-symbol resolution, and smart orders.

## optiongreeks(symbol, exchange, ...)
Calculate Delta, Gamma, Theta, Vega, Rho, and IV using Black-Scholes.
Requires: `pip install mibian`

```python
# Auto-detect spot price
greeks = client.optiongreeks(symbol="NIFTY28NOV2526000CE", exchange="NFO")

# With custom interest rate
greeks = client.optiongreeks(symbol="BANKNIFTY28NOV2550000CE", exchange="NFO", interest_rate=6.5)

# Using futures as underlying
greeks = client.optiongreeks(
    symbol="NIFTY28NOV2526000CE", exchange="NFO",
    underlying_symbol="NIFTY28NOV25FUT", underlying_exchange="NFO"
)

# Returns: delta, gamma, theta, vega, rho, implied_volatility, spot_price, days_to_expiry
```

## optionsymbol(underlying, exchange, expiry_date, strike_int, offset, option_type)
Get option symbol details without placing order.

```python
# ATM call
symbol_info = client.optionsymbol(
    underlying="NIFTY", exchange="NSE_INDEX", expiry_date="28NOV24",
    strike_int=50, offset="ATM", option_type="CE"
)

# OTM put
symbol_info = client.optionsymbol(
    underlying="BANKNIFTY", exchange="NSE_INDEX", expiry_date="28NOV24",
    strike_int=100, offset="OTM2", option_type="PE"
)
```

**Offset Options**: ATM, ITM1-ITM50, OTM1-OTM50

## optionsorder(...)
Place option orders with auto-symbol resolution.

```python
# Market order
result = client.optionsorder(
    strategy="test", underlying="NIFTY", exchange="NSE_INDEX",
    expiry_date="28NOV24", strike_int=50, offset="ATM", option_type="CE",
    action="BUY", quantity=75, price_type="MARKET", product="MIS"
)

# Limit order
result = client.optionsorder(
    strategy="scalping", underlying="NIFTY", exchange="NSE_INDEX",
    expiry_date="28NOV24", strike_int=50, offset="OTM1", option_type="PE",
    action="SELL", quantity=75, price_type="LIMIT", product="MIS", price="50.0"
)
```

## Strategy Example: Iron Condor
```python
common = {"underlying": "NIFTY", "exchange": "NSE_INDEX", ...}
client.optionsorder(offset="OTM1", option_type="CE", action="SELL", **common)
client.optionsorder(offset="OTM1", option_type="PE", action="SELL", **common)
client.optionsorder(offset="OTM3", option_type="CE", action="BUY", **common)
client.optionsorder(offset="OTM3", option_type="PE", action="BUY", **common)
```
