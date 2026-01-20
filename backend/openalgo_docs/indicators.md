# OpenAlgo Technical Indicators

High-performance Python library with 80+ technical indicators.

## Import Statement
```python
from openalgo import ta
```

## Indicator Categories

### 1. Trend Indicators
| Indicator | Full Name |
|-----------|-----------|
| SMA | Simple Moving Average |
| EMA | Exponential Moving Average |
| WMA | Weighted Moving Average |
| DEMA | Double Exponential Moving Average |
| TEMA | Triple Exponential Moving Average |
| HMA | Hull Moving Average |
| VWMA | Volume Weighted Moving Average |
| ALMA | Arnaud Legoux Moving Average |
| KAMA | Kaufman's Adaptive Moving Average |
| ZLEMA | Zero Lag Exponential Moving Average |
| T3 | T3 Moving Average |
| FRAMA | Fractal Adaptive Moving Average |
| TRIMA | Triangular Moving Average |
| McGinley | McGinley Dynamic |
| VIDYA | Variable Index Dynamic Average |
| Alligator | Bill Williams Alligator |
| MovingAverageEnvelopes | Moving Average Envelopes |
| Supertrend | Supertrend Indicator |
| Ichimoku | Ichimoku Cloud |
| ChandeKrollStop | Chande Kroll Stop |

### 2. Momentum Indicators
| Indicator | Full Name |
|-----------|-----------|
| RSI | Relative Strength Index |
| MACD | Moving Average Convergence Divergence |
| Stochastic | Stochastic Oscillator |
| CCI | Commodity Channel Index |
| WilliamsR | Williams %R |
| BOP | Balance of Power |
| ElderRay | Elder Ray Index (Bull/Bear Power) |
| Fisher | Fisher Transform |
| CRSI | Connors RSI |

### 3. Volatility Indicators
| Indicator | Full Name |
|-----------|-----------|
| ATR | Average True Range |
| BollingerBands | Bollinger Bands |
| Keltner | Keltner Channel |
| Donchian | Donchian Channel |
| Chaikin | Chaikin Volatility |
| NATR | Normalized Average True Range |
| RVI | Relative Volatility Index |
| ULTOSC | Ultimate Oscillator |
| TRANGE | True Range |
| MASS | Mass Index |
| BBPercent | Bollinger Bands %B |
| BBWidth | Bollinger Bandwidth |
| ChandelierExit | Chandelier Exit |
| HistoricalVolatility | Historical Volatility |
| UlcerIndex | Ulcer Index |
| STARC | STARC Bands |

### 4. Volume Indicators
| Indicator | Full Name |
|-----------|-----------|
| OBV | On Balance Volume |
| OBVSmoothed | On Balance Volume with Smoothing |
| VWAP | Volume Weighted Average Price |
| MFI | Money Flow Index |
| ADL | Accumulation/Distribution Line |
| CMF | Chaikin Money Flow |
| EMV | Ease of Movement |
| FI | Elder Force Index |
| NVI | Negative Volume Index |
| PVI | Positive Volume Index |
| VOLOSC | Volume Oscillator |
| VROC | Volume Rate of Change |
| KlingerVolumeOscillator | Klinger Volume Oscillator |
| PriceVolumeTrend | Price Volume Trend |
| RVOL | Relative Volume |

### 5. Oscillators
| Indicator | Full Name |
|-----------|-----------|
| ROC | Rate of Change |
| CMO | Chande Momentum Oscillator |
| TRIX | Triple Exponential Average |
| UO | Ultimate Oscillator |
| AO | Awesome Oscillator |
| AC | Accelerator Oscillator |
| PPO | Percentage Price Oscillator |
| PO | Price Oscillator |
| DPO | Detrended Price Oscillator |
| AROONOSC | Aroon Oscillator |
| StochRSI | Stochastic RSI |
| RVI | Relative Vigor Index |
| CHO | Chaikin Oscillator |
| CHOP | Choppiness Index |
| KST | Know Sure Thing |
| TSI | True Strength Index |
| VI | Vortex Indicator |
| STC | Schaff Trend Cycle |
| GatorOscillator | Gator Oscillator |
| Coppock | Coppock Curve |

### 6. Statistical Indicators
| Indicator | Full Name |
|-----------|-----------|
| LINREG | Linear Regression |
| LRSLOPE | Linear Regression Slope |
| CORREL | Pearson Correlation Coefficient |
| BETA | Beta Coefficient |
| VAR | Variance |
| TSF | Time Series Forecast |
| MEDIAN | Rolling Median |
| MedianBands | Median with Bands |
| MODE | Rolling Mode |

### 7. Hybrid Indicators
| Indicator | Full Name |
|-----------|-----------|
| ADX | Average Directional Index |
| Aroon | Aroon Indicator |
| PivotPoints | Pivot Points |
| SAR | Parabolic SAR |
| DMI | Directional Movement Index |
| WilliamsFractals | Williams Fractals |
| RWI | Random Walk Index |

### 8. Utility Functions
| Function | Description |
|----------|-------------|
| crossover | Series crossover detection |
| crossunder | Series crossunder detection |
| highest | Highest value over period |
| lowest | Lowest value over period |
| change | Change in value |
| roc | Rate of change |
| stdev | Standard deviation |
| exrem | Excess removal |
| flip | Flip function |
| valuewhen | Value when condition |
| rising | Rising detection |
| falling | Falling detection |
| cross | Cross detection (both directions) |
