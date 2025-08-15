# Step 1: Download the data
import yfinance as yf
import numpy as np
import pandas as pd

df = yf.download("JPM", start="2022-01-01", end="2025-01-01")

# --- Make yfinance output robust ---
# Ensure DatetimeIndex
df.index = pd.to_datetime(df.index)

# Flatten MultiIndex columns if present (can happen with some yfinance versions)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Step 2: Calculate daily returns (PERCENT, your choice)
df["returns"] = df["Close"].pct_change() * 100
df = df.dropna()

# Step 3: Set lambda value for EWMA Model
lambda1 = 0.94

# Step 4: Initialize variance and calculate EWMA (your core logic)
ewma_variance = []
var_t = df["returns"].var()

for ret in df["returns"]:
    variance_tplus1 = lambda1 * var_t + (1 - lambda1) * (ret ** 2)
    ewma_variance.append(variance_tplus1)

# Volatility is the square root of variance (percent units)
df["ewma_vol"] = np.sqrt(ewma_variance)

# ---- Evaluation prep (create columns BEFORE slicing) ----
df["ewma_forecast"] = df["ewma_vol"].shift(1)   # t+1 forecast
df["realized_next"] = df["returns"].abs()       # |r_t|

# ---- Slice 2024 safely (no string indexing) ----
mask_2024 = (df.index >= pd.Timestamp("2024-01-01")) & (df.index < pd.Timestamp("2025-01-01"))
df_2024 = df.loc[mask_2024, ["ewma_forecast", "realized_next"]].dropna()

# ---- Metrics (still percent units) ----
mae  = (df_2024["ewma_forecast"] - df_2024["realized_next"]).abs().mean()
rmse = np.sqrt(((df_2024["ewma_forecast"] - df_2024["realized_next"])**2).mean())
bias = (df_2024["ewma_forecast"] - df_2024["realized_next"]).mean()
corr = df_2024[["ewma_forecast", "realized_next"]].corr().iloc[0, 1]
overshoot_pct = (df_2024["ewma_forecast"] > df_2024["realized_next"]).mean() * 100

print("=== EWMA Model Backtest — 2024 ===")
print(f"MAE (%):   {mae:.3f}")
print(f"RMSE (%):  {rmse:.3f}")
print(f"Bias (%):  {bias:.3f}")
print(f"Corr:      {corr:.3f}")
print(f"Overshoot: {overshoot_pct:.1f}% of days")

# ---- Jan 2, 2025 check (next trading day after 2024-12-31) ----
next_day = pd.Timestamp("2024-12-31") + pd.tseries.offsets.BDay(1)  # should be 2025-01-02
day_after = next_day + pd.tseries.offsets.BDay(1)

real_df = yf.download("JPM", start=str(next_day.date()), end=str(day_after.date()))
if isinstance(real_df.columns, pd.MultiIndex):
    real_df.columns = real_df.columns.get_level_values(0)

real_df["returns"] = real_df["Close"].pct_change() * 100
real_df = real_df.dropna()

predicted_tplus1 = float(df["ewma_vol"].iloc[-1])                 # %
realized_nextday = float(real_df["returns"].iloc[0]) if len(real_df) else float("nan")

print("\n=== Jan 2, 2025 Check ===")
print(f"Predicted daily vol (%): {predicted_tplus1:.3f}")
print(f"Realized |return|  (%):  {abs(realized_nextday):.3f}")
