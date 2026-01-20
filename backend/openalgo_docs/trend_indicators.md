# OpenAlgo Trend Indicators - Detailed Usage

All trend indicators use the `ta` module: `from openalgo import ta`

## Data Setup Pattern
```python
from openalgo import api, ta
import pandas as pd

client = api(api_key='your_api_key', host='http://127.0.0.1:5000')
df = client.history(symbol="SBIN", exchange="NSE", interval="5m",
                   start_date="2025-04-01", end_date="2025-04-08")
```

## Indicator Reference

### SMA - Simple Moving Average
```python
df['SMA_20'] = ta.sma(df['close'], 20)
```
- **Params**: `data`, `period`
- **Returns**: pandas.Series

### EMA - Exponential Moving Average
```python
df['EMA_20'] = ta.ema(df['close'], 20)
```
- **Params**: `data`, `period`
- **Returns**: pandas.Series
- More responsive to recent prices than SMA

### WMA - Weighted Moving Average
```python
df['WMA_20'] = ta.wma(df['close'], 20)
```
- **Params**: `data`, `period`
- **Returns**: numpy.ndarray

### HMA - Hull Moving Average
```python
df['HMA_16'] = ta.hma(df['close'], 16)
```
- **Params**: `data`, `period`
- **Returns**: pandas.Series
- Minimizes lag while improving smoothing

### VWMA - Volume Weighted Moving Average
```python
df['VWMA_20'] = ta.vwma(df['close'], df['volume'], 20)
```
- **Params**: `data`, `volume`, `period`
- **Returns**: pandas.Series
- Weights by volume

### KAMA - Kaufman's Adaptive Moving Average
```python
df['KAMA_14'] = ta.kama(df['close'])  # defaults: 14, 2, 30
df['KAMA_custom'] = ta.kama(df['close'], length=14, fast_length=2, slow_length=30)
```
- **Params**: `data`, `length=14`, `fast_length=2`, `slow_length=30`
- **Returns**: pandas.Series
- Adapts to market volatility

### Supertrend
```python
df['Supertrend'], df['Direction'] = ta.supertrend(df['high'], df['low'], df['close'])
df['ST_Fast'], df['ST_Dir'] = ta.supertrend(df['high'], df['low'], df['close'], period=7, multiplier=2.0)
```
- **Params**: `high`, `low`, `close`, `period=10`, `multiplier=3.0`
- **Returns**: tuple(supertrend_values, direction) 
  - direction: -1 = uptrend (green), 1 = downtrend (red)

### Ichimoku Cloud
```python
conversion, base, span_a, span_b, lagging = ta.ichimoku(df['high'], df['low'], df['close'])
```
- **Params**: `high`, `low`, `close`, `conversion_periods=9`, `base_periods=26`, `lagging_span2_periods=52`, `displacement=26`
- **Returns**: tuple(conversion_line, base_line, leading_span_a, leading_span_b, lagging_span)

### ALMA - Arnaud Legoux Moving Average
```python
df['ALMA'] = ta.alma(df['close'])  # defaults: 21, 0.85, 6.0
df['ALMA_custom'] = ta.alma(df['close'], period=14, offset=0.9, sigma=4.0)
```
- **Params**: `data`, `period=21`, `offset=0.85`, `sigma=6.0`
- **Returns**: pandas.Series

### ZLEMA - Zero Lag EMA
```python
df['ZLEMA_20'] = ta.zlema(df['close'], 20)
```
- **Params**: `data`, `period`
- **Returns**: pandas.Series
- Eliminates lag using price momentum

### DEMA - Double Exponential Moving Average
```python
df['DEMA_20'] = ta.dema(df['close'], 20)
```
- **Params**: `data`, `period`
- **Returns**: pandas.Series

### TEMA - Triple Exponential Moving Average
```python
df['TEMA_20'] = ta.tema(df['close'], 20)
```
- **Params**: `data`, `period`
- **Returns**: pandas.Series

## Trading Signal Patterns

### MA Crossover
```python
df['MA_Bullish'] = (df['close'] > df['SMA_20']) & (df['EMA_20'] > df['SMA_20'])
```

### Supertrend Signal
```python
df['ST_Bullish'] = df['ST_Direction'] == -1
df['Trend_Change'] = df['ST_Direction'].diff() != 0
```

### Ichimoku Cloud Signal
```python
df['Cloud_Top'] = df[['SpanA', 'SpanB']].max(axis=1)
df['Above_Cloud'] = df['close'] > df['Cloud_Top']
df['TK_Bullish'] = (df['Conversion'] > df['Base']) & (df['Conversion'].shift(1) <= df['Base'].shift(1))
```

### Combined Signal Example
```python
df['Combined_Signal'] = (df['MA_Bullish'] & df['ST_Bullish'] & df['Ichimoku_Bullish']).astype(int)
```
