import yfinance as yf
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import cufflinks as cf
from cufflinks.ta import validate
from matplotlib.pyplot import title
from plotly.offline import iplot
import plotly.graph_objects as go

cf.go_offline()

sns.set_palette(sns.color_palette("pastel"))

start = datetime.datetime(2006, 1, 1)
end = datetime.datetime(2016, 1, 1)
BAC = yf.download("BAC", start, end)
C = yf.download("C", start, end)
GS = yf.download("GS", start, end)
JPM = yf.download("JPM", start, end)
MS = yf.download("MS", start, end)
WFC = yf.download("WFC", start, end)

tickers = ["BAC", "C", "GS", "JPM", "MS", "WFC"]
tickers.sort()

bank_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC], keys=tickers, axis=1)
print("bank_stocks:\n", bank_stocks.head())

# ****** What is the max Close price for each bank's stock throughout the time period? ****** #
# max_close = bank_stocks.xs("Close", 1, level="Price").max()
# print("\n\nmax close for each bank:\n", max_close)

# ***** We can use pandas pct_change() method on the Close column to create a column representing
# this return value. Create a for loop that goes and for each Bank Stock Ticker creates this
# returns column and set's it as a column in the returns DataFrame. ***** #
# returns = pd.DataFrame()
# for i in tickers:
#     returns[i+"Return"] = bank_stocks[i]["Close"].pct_change()
# print("\n\nreturns:\n", returns)

# ***** Create a pairplot using seaborn of the returns dataframe.
# What stock stands out to you? Can you figure out why? ***** #
# sns.pairplot(returns, plot_kws={'s': 7})
# plt.show()

# ***** Using this returns DataFrame, figure out on what dates each bank stock had the best
# and worst single day returns. You should notice that 4 of the banks share the same day for
# the worst drop, did anything significant happen that day? ***** #
# min_return = returns.idxmin()
# print("\n\nworst day for each bank:\n", min_return)
# max_return = returns.idxmax()
# print("\n\nbest day for each bank:\n", max_return)
# risky_returns = returns.loc["2015-01-01":"2015-12-31"].std()
# print("\n\nthe riskiest stock from 2015:\n", risky_returns.sort_values(ascending=False))

# ***** Create a distplot using seaborn of the 2015 returns for Morgan Stanley ***** #
# sns.displot(returns.loc["2015-01-01":"2015-12-31", "MSReturn"], bins=30, kde=True)
# plt.show()

# ***** Create a distplot using seaborn of the 2008 returns for CitiGroup ***** #
# sns.displot(returns.loc["2008-01-01":"2008-12-31", "CReturn"], bins=30, kde=True)
# plt.show()

# ***** Create a line plot showing Close price for each bank for the entire index of time. ***** #
# bank_stocks.xs("Close", 1, level="Price").plot(figsize=(12, 6))
# plt.show()

# ***** Plot the rolling 30 day avg against the Close Price for BAC's stock for the year 2008 ***** #
# fig, ax = plt.subplots(figsize=(8, 6))
# BAC_2008 = BAC["Close"].loc["2008-01-01":"2008-12-31"]
# BAC_2008.plot(ax=ax, label="BAC close price")
# BAC_2008.rolling(window=30).mean().plot(ax=ax, label="30-day avg")
# ax.set_xlabel("Date")
# ax.set_ylabel("Price")
# ax.legend()
# plt.show()

# ***** Create a heatmap of the correlation between the stocks Close Price. ***** #
# ***** Create a clustermap using seaborn to cluster the correlations between the stocks Close Price
# sns.heatmap(bank_stocks.xs("Close", 1, level="Price").corr(), cmap="PiYG")
# sns.clustermap(bank_stocks.xs("Close", 1, level="Price").corr(), cmap="PiYG")
# plt.show()

# ***** Use iplot(kind='candle') to create a candle plot of Bank of America's stock
# from Jan 1st 2015 to Jan 1st 2016. ***** #
# BAC_2015 = BAC.loc["2015-01-01":"2016-01-01"]
# BAC_2015.columns = BAC_2015.columns.get_level_values(0)
# print("\n\nBAC 2015 data:\n", BAC_2015.head())
# fig = go.Figure(data=[go.Candlestick(
#     x=BAC_2015.index,
#     open=BAC_2015["Open"],
#     high=BAC_2015["High"],
#     low=BAC_2015["Low"],
#     close=BAC_2015["Close"])])
# fig.update_layout(
#     title="Bank of America 2015 Stock Price", yaxis_title="Stock Price", xaxis_title="Date")
# fig.show()

# ***** DOESN'T WORK ***** #
# ***** Use .ta_plot(study='sma') to create a
# Simple Moving Averages plot of Morgan Stanley for the year 2015.
# MS_2015 = MS.loc["2015-01-01":"2016-01-01"]
# MS_2015.columns = MS_2015.columns.get_level_values(0)
# print("\n\nMS 2015 data:\n", MS_2015.head())
# MS_2015.ta_plot(study="sma", title="Morgan Stanley 2015 Simple Moving Averages", xTitle="Date", yTitle="Stock Price")
# fig = go.Figure(data=go.add_sma(MS_2015["Close"]))
# qf = cf.QuantFig(MS_2015, title="Morgan Stanley SMA 2015", name="MS")
# qf.add_sma(periods=[20, 50])
# qf.iplot()

# fig, ax = plt.subplots(figsize=(8, 6))
# MS_2015.plot(ax=ax, label="MS 2015 price")
# Add the 20-day SMA
# MS_a30 = MS_2015["Close"].rolling(window=30).mean()
# fig.add_trace(go.Scatter(x=MS_2015.index, y=MS_a30, name="30-day average"))
# MS_2015.rolling(window=30).mean().plot(ax=ax, label="30-day avg")
# plt.show()
