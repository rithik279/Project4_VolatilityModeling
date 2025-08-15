
"""
#Could use Yahoo Finance or Alpha Vantage
import pandas as pd
import requests
from io import StringIO

#Download Data From AlphaVantage

api_key = "XZA9L8BHKDJM2FQL"
symbol = "JPM"
#url to get stock data from AlphaVantage
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=JPM&apikey={api_key}&datatype=csv&outputsize=full'

#Fetch the Data

response = requests.get(url)

#Convert the csv text into a dataframe
df = pd.read_csv(StringIO(response.text()))
df.index = df["timestamp"]

df = df.loc[:"2015-01-01"]
df = df['close']

print(df)"""



#ARCH Model

#Step 1: Get the data from yahoo finance
#Step 2: Calculate the Daily Returns, don't deal with the stock price data, use the returns for the model

import pandas as pd
import yfinance as yf
from arch import arch_model
import numpy as np

symbol = "JPM"

# 1) Download daily close prices
prices = yf.download(symbol, start="2022-01-01", end="2025-01-01", progress=False)["Close"]

print(prices)

# 2) Compute daily % returns
returns = prices.pct_change().dropna() * 100

print(returns)

# 3) Fit ARCH(1)
model = arch_model(returns, vol="ARCH", p=1, mean="Constant", dist="normal")
results = model.fit(disp="off")

print(results.summary())


#Analysis of the ARCH Model

#mu --> 0.1260: The model estimates that the average daily return is 0.1260
#omega --> 1.6656: the long run base level of variance
#alpha [1] --> how much yesterday's shock impacts today's variance

#Forecasting volatility of the next 5 days

forecast = results.forecast(horizon = 5)
predicted_variance = forecast.variance

#Volatility = square root(Variance), Volatility is the standard deviation
predicted_volatility = predicted_variance**0.5
print(predicted_volatility)

predicted_volatility = [1.290793,  1.538573,  1.631696, 1.669358, 1.684966] #2015 - 2024

#predicted_volatility = [1.479269,  1.567023,  1.577429,  1.578698,  1.578854] #2022- 2024
predicted_avg_col = sum(predicted_volatility)/len(predicted_volatility)

#Step 7: get the data and calculate the realized volatility

start_date = pd.to_datetime("2024-12-31")
end_date = pd.to_datetime("2025-01-09") #buffer for weekends and holidays, #Jan 2,3,6,7,8 2025, accounting for holdiay and the saturday sunday

print(start_date, end_date)

real_df = yf.download("JPM", start_date, end_date)
real_df["returns"] = real_df["Close"].pct_change()*100
real_df = real_df.dropna()

print(real_df)

realized_vol = real_df['returns'].std()*np.sqrt(5)#Make the adjustment since it is the # of days for which the volatility is found, Predicting the volatility of past one year, we multiply by square root of 252

print("ARCH Model Predicted Volatility: ", predicted_avg_col)
print("ARCH Model Actual Volatility: ", realized_vol)

#We haven't yet finetuned the model, change to last 3 years

#ARCH Model does not consider past volatility that is why we are not getting a good result, covered in garch model
#Can change the ticker to bank of america
#How can we use this model to get the daily volatility, then we can make an automated VIX trading strategy




