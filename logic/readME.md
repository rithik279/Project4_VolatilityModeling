# QUANT PROJECT 4: VOLATILITY MODELING WITH ARCH, EWMA & GARCH

This project applies **ARCH, EWMA, and GARCH models** to estimate and forecast the daily volatility of **JPMorgan Chase (JPM)** stock returns.  
It demonstrates how econometric volatility models compare to realized volatility and evaluates their predictive performance.

---

## OBJECTIVE
- Compute **daily returns** from JPM close prices  
- Fit **ARCH(1)** and **GARCH(1,1)** models using the `arch` library  
- Implement **EWMA volatility** (λ = 0.94) to estimate time-varying variance  
- Forecast volatility (1-day and 5-day horizons) and compare against realized volatility  
- Backtest performance on 2024 daily data  

---

## SETUP INSTRUCTIONS
```bash
git clone https://github.com/rithik279/Project4_VolatilityModeling
cd Project4_VolatilityModeling
pip install -r requirements.txt
PROJECT PIPELINE
1. Data Collection
Source: Yahoo Finance (yfinance) and AlphaVantage (CSV API)

Ticker: JPM (JPMorgan Chase)

Range: 2022-01-01 to 2025-01-01

2. Preprocessing
Compute daily returns (%) from closing prices

Handle missing data with forward-fill

Ensure consistent DatetimeIndex

3. Models Implemented
ARCH(1): models variance as a function of lagged squared returns

GARCH(1,1): variance depends on lagged squared returns and lagged variance

EWMA: exponentially weighted moving variance with decay factor λ = 0.94

4. Forecasting
ARCH & GARCH: 5-day volatility forecasts

EWMA: rolling 1-step-ahead daily forecasts

5. Performance Evaluation
Metrics on 2024 backtest:

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

Forecast bias

Correlation with realized volatility

Overshoot frequency (%)

Out-of-sample test: Jan 2, 2025 daily volatility prediction

VISUALIZATIONS
Daily return series of JPM

EWMA volatility vs realized volatility (2024)

ARCH/GARCH forecasts vs realized volatility

Feature interpretation: ARCH (μ, ω, α), GARCH (α, β) parameters

KEY FINDINGS
ARCH(1): captures volatility clustering but ignores persistence

GARCH(1,1): models persistence well (high β) but requires tuning (p, q)

EWMA: simple, efficient, and competitive for short-horizon forecasts

Realized vs predicted volatility differences highlight model calibration importance

GARCH long-term variance stabilizes, while EWMA responds faster to shocks

REQUIREMENTS
bash
Copy
Edit
pip install -r requirements.txt
Dependencies:

yfinance

pandas, numpy

matplotlib

arch (ARCH/GARCH modeling)

FILES INCLUDED
notebooks/vol_modeling.ipynb — end-to-end ARCH, EWMA, GARCH workflow

logic/arch_model.py — ARCH(1) implementation

logic/ewma.py — EWMA volatility estimator

logic/garch_model.py — GARCH(1,1) implementation

requirements.txt — dependencies

README.md — project documentation

AUTHOR NOTES
This project builds intuition for time-varying volatility models and their role in risk management, derivatives pricing, and volatility trading.
It also prepares the ground for VIX-linked strategies and more advanced econometric models (EGARCH, GJR-GARCH).

TO-DO / NEXT STEPS
Tune GARCH(p, q) parameters for better fit

Test on other financials (BAC, C, MS)

Add realized volatility measures: Parkinson, Garman-Klass, Yang-Zhang

Compare against implied volatility (VIX)

Deploy as an automated vol forecasting API

LIMITATIONS
ARCH: underestimates persistence (ignores lagged variance)

GARCH: fit depends heavily on (p, q) choice

EWMA: assumes fixed λ; optimizing decay could improve accuracy

Realized volatility depends on return definition (abs vs std)

© 2025 Rithik Singh — Quant Finance Project Series