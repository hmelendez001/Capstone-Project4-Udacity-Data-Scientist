import pandas as pd
import numpy as np
import yfinance as yf 
import datetime

df_future = pd.DataFrame()
start_date = '2021-01-01'
today = datetime.datetime.today()
tomorrow = today + datetime.timedelta(days=1)

symbol = 'TSLA'
df_ret3 = yf.download(symbol, start_date, tomorrow)
df_ret3 = df_ret3.reset_index()
df_ret3.insert(0, 'Symbol', symbol)
df_ret3['Symbol'] = symbol

df_future = df_future.append(df_ret3)

symbol = 'AAPL'
df_ret3 = yf.download(symbol, start_date, tomorrow)
df_ret3 = df_ret3.reset_index()
df_ret3.insert(0, 'Symbol', symbol)
df_ret3['Symbol'] = symbol

df_future = df_future.append(df_ret3)

symbol = 'MSFT'
df_ret3 = yf.download(symbol, start_date, tomorrow)
df_ret3 = df_ret3.reset_index()
df_ret3.insert(0, 'Symbol', symbol)
df_ret3['Symbol'] = symbol
df_ret3["Close"].plot()

df_future = df_future.append(df_ret3)

print(df_future[df_future['Symbol'] == 'TSLA']['Date'])

df_future.to_csv("./initial_stocks.csv", index=False)
