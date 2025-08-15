# Step 1: Download the data
import yfinance as yf
import numpy as np
import pandas as pd

df = yf.download("JPM", start= "2022-01-01", end = "2025-01-01")

#Step 2: Calculate daily returns
df["returns"] = df["Close"].pct_change()*100
df = df.dropna()

#Step3: Set lambda value for EWMA Model

lambda1 = 0.94

#Step 4: Initialize variance and calculate EWMA

ewma_variance = []
var_t = df['returns'].var()

for ret in df['returns']:
    variance_tplus1 = lambda1*var_t + (1-lambda1)*(ret**2) 
    ewma_variance.append(variance_tplus1)

#print(ewma_variance)

#Volatility is the square root of variance

df["ewma_vol"] = np.sqrt(ewma_variance)
#print(df["ewma_vol"])

#We can use this model to predict the volatility of only one day

#Step 5: Predicted Volatility

latest_daily_volatility  = df["ewma_vol"].iloc[-1]#The predicted volatility
#print(latest_daily_volatility)

#Step 6: Realized volatility, only for one day prediction

start_date = pd.to_datetime("2024-12-31")
end_date = pd.to_datetime("2025-01-03") #buffer for weekends and holidays, #Jan 2,3,6,7,8 2025, accounting for holdiay and the saturday sunday


real_df = yf.download("JPM", start_date, end_date)
real_df["returns"] = real_df["Close"].pct_change()*100
real_df = real_df.dropna()

realized_vol = real_df['returns'].iloc[-1]#No standard deviation and no adjustment (only for one day)
#print(realized_vol)

print("EWMA Predicted Volatility: ", latest_daily_volatility)
print("EWMA Actual Volatility: ", realized_vol)