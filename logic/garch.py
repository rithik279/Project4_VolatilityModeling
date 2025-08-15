#Module 2 - GARCH Model implementation

import pandas as pd
import yfinance as yf
from arch import arch_model
import numpy as np

symbol = "JPM"

# 1) Download daily close prices
prices = yf.download(symbol, start="2022-01-01", end="2025-01-01", progress=False)["Close"]

print(prices)

# 2) Compute daily % returns & drop missing value
returns = prices.pct_change().dropna() * 100

print(returns)

# 3) Fit GARCH(2)
model = arch_model(returns, vol="GARCH", p=1, q = 1) # Call the model
results = model.fit(disp="off") # train the model

#4) Show Summary
print(results.summary())

#Analysis of GARCH model

#mu --> 0.1146: the model estimates theat the average daily return is 0.1146
#omega --> 0.0220: the long run base level of vairance
#alpha [1]--> 0.0165: how much yesterday's squared shock impacts today's variance
#beta[1]--> 0.9737: How much yesterday;s variance impacts today's variance


forecast = results.forecast(horizon = 5)
predicted_variance = forecast.variance

#Volatility = square root(Variance), Volatility is the standard deviation
predicted_volatility = predicted_variance**0.5
print(predicted_volatility)

#Calulate the average of the predicted volatility

predicted_volatility = [1.623483,  1.622252,  1.621032,  1.619824,  1.618626] #2022- 2024
predicted_avg_col = sum(predicted_volatility)/len(predicted_volatility)

start_date = pd.to_datetime("2024-12-31")
end_date = pd.to_datetime("2025-01-09") #buffer for weekends and holidays, #Jan 2,3,6,7,8 2025, accounting for holdiay and the saturday sunday

print(start_date, end_date)

real_df = yf.download("JPM", start_date, end_date)
real_df["returns"] = real_df["Close"].pct_change()*100
real_df = real_df.dropna()

print(real_df)

realized_vol = real_df['returns'].std()*np.sqrt(5)

print("GARCH Model Predicted Volatility: ", predicted_avg_col)
print("GARCH Model Actual Volatility: ", realized_vol)

#Values are far from each other, need to adjust the model, changing p and q will make the values insignificant