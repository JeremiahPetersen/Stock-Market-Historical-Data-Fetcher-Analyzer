import pandas as pd
import numpy as np
import logging
import os
import plotly.graph_objects as go
from scipy.stats import zscore
import seaborn as sns
from ta.trend import ADXIndicator
from ta.momentum import StochasticOscillator
from ta.volatility import BollingerBands
from ta.utils import dropna
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(filename="app.log", level=logging.INFO, format='%(asctime)s - %(message)s')

# Load the data
def load_data(filename):
    if not os.path.exists(filename):
        logging.error(f"File {filename} does not exist.")
        raise FileNotFoundError(f"File {filename} does not exist.")
    
    all_data = pd.read_csv(filename, header=[0, 1], index_col=0, parse_dates=True)

    return all_data

# Validate the data
def validate_data(data):
    if data.isnull().values.any():
        logging.warning("Data contains null values. Filling null values with the method ffill.")
        data.fillna(method='ffill', inplace=True)
    
    return data

# Filter data by date range
def filter_data_by_date_range(data, start_date, end_date):
    return data.loc[start_date:end_date]

# Define function to compute RSI
def compute_rsi(data, window=14):
    diff = data.diff()
    up = diff.where(diff > 0, 0.0)
    down = -diff.where(diff < 0, 0.0)
    ema_up = up.ewm(alpha=1/window).mean()
    ema_down = down.ewm(alpha=1/window).mean()
    rs = ema_up/ema_down
    return 100 - (100 / (1 + rs))

# Compute ADX
def compute_adx(data):
    high = data['High']
    low = data['Low']
    close = data['Close']
    adx = ADXIndicator(high, low, close).adx()
    return adx

# Compute Stochastic Oscillator
def compute_stochastic_oscillator(data):
    high = data['High']
    low = data['Low']
    close = data['Close']
    so = StochasticOscillator(high, low, close).stoch()
    return so

# Compute Simple Moving Average (SMA)
def compute_sma(data, window=14):
    return data.rolling(window).mean()

# Compute Exponential Moving Average (EMA)
def compute_ema(data, window=14):
    return data.ewm(span=window, adjust=False).mean()

# Compute Bollinger Bands
def compute_bollinger_bands(data, window=20):
    indicator_bb = BollingerBands(close=data["Close"], window=20, window_dev=2)
    data['bb_bbm'] = indicator_bb.bollinger_mavg()
    data['bb_bbh'] = indicator_bb.bollinger_hband()
    data['bb_bbl'] = indicator_bb.bollinger_lband()
    return data

# Compute Volatility (standard deviation)
def compute_volatility(data, window=14):
    return data['Close'].rolling(window).std()

# Compute Sharpe Ratio
def compute_sharpe_ratio(data, risk_free_rate=0.01):
    returns = data['Close'].pct_change().dropna()
    sharpe_ratio = (returns.mean() - risk_free_rate) / returns.std()
    return sharpe_ratio

# Functions for all plots
def plot_ohlc(data, ticker):
    ohlc_data = data[['Open', 'High', 'Low', 'Close']]
    fig = go.Figure(data=[go.Candlestick(x=ohlc_data.index, open=ohlc_data['Open'], high=ohlc_data['High'], low=ohlc_data['Low'], close=ohlc_data['Close'])])
    fig.update_layout(title=f'{ticker} OHLC Prices Over Time', yaxis_title='Price')
    fig.show()

def plot_histogram(data, ticker):
    plt.hist(data['Close'], bins=50, color='blue')
    plt.title(f'{ticker} Distribution of Closing Prices')
    plt.show()

def plot_rsi(rsi, ticker):
    plt.figure(figsize=(14, 6))
    plt.title(f'{ticker} RSI Over Time')
    plt.plot(rsi, label='RSI')
    plt.fill_between(rsi.index, y1=30, y2=70, color='#adccff', alpha=0.3)
    plt.legend()
    plt.show()

def plot_adx(adx, ticker):
    plt.figure(figsize=(14, 6))
    plt.title(f'{ticker} ADX Over Time')
    plt.plot(adx, label='ADX')
    plt.legend()
    plt.show()

def plot_stochastic_oscillator(so, ticker):
    plt.figure(figsize=(14, 6))
    plt.title(f'{ticker} Stochastic Oscillator Over Time')
    plt.plot(so, label='Stochastic Oscillator')
    plt.legend()
    plt.show()
    
# Add new plotting functions for the new indicators
def plot_sma_ema(data, ticker, sma, ema):
    plt.figure(figsize=(14, 6))
    plt.title(f'{ticker} SMA and EMA Over Time')
    plt.plot(data['Close'], label='Close')
    plt.plot(sma, label='SMA')
    plt.plot(ema, label='EMA')
    plt.legend()
    plt.show()

def plot_bollinger_bands(data, ticker):
    plt.figure(figsize=(14, 6))
    plt.title(f'{ticker} Bollinger Bands Over Time')
    plt.plot(data['Close'], label='Close')
    plt.plot(data['bb_bbm'], label='Middle Band')
    plt.plot(data['bb_bbh'], label='Upper Band')
    plt.plot(data['bb_bbl'], label='Lower Band')
    plt.legend()
    plt.show()

def plot_volatility(volatility, ticker):
    plt.figure(figsize=(14, 6))
    plt.title(f'{ticker} Volatility Over Time')
    plt.plot(volatility, label='Volatility')
    plt.legend()
    plt.show()
    
# Load and validate the data
filename = 'market_all_historical_data.csv'
all_data = load_data(filename)
all_data = validate_data(all_data)

# Set date range
start_date = '2022-01-01'
end_date = '2023-06-13'
all_data = filter_data_by_date_range(all_data, start_date, end_date)

# Get tickers
tickers = all_data.columns.get_level_values(0).unique()

# Main loop for all tickers
for ticker in tickers:
    ticker_data = all_data[ticker]
    plot_ohlc(ticker_data, ticker)

    rsi = compute_rsi(ticker_data['Close'])
    plot_rsi(rsi, ticker)

    adx = compute_adx(ticker_data)
    plot_adx(adx, ticker)

    so = compute_stochastic_oscillator(ticker_data)
    plot_stochastic_oscillator(so, ticker)

    plot_histogram(ticker_data, ticker)
    
    # Compute and plot new indicators
    sma = compute_sma(ticker_data['Close'])
    ema = compute_ema(ticker_data['Close'])
    plot_sma_ema(ticker_data, ticker, sma, ema)

    ticker_data = compute_bollinger_bands(ticker_data)
    plot_bollinger_bands(ticker_data, ticker)

    volatility = compute_volatility(ticker_data)
    plot_volatility(volatility, ticker)
    
    sharpe_ratio = compute_sharpe_ratio(ticker_data)
    logging.info(f'Sharpe Ratio for {ticker}: {sharpe_ratio}')

# Correlation Matrix Heatmap for all tickers
closing_prices = pd.DataFrame({ticker: zscore(all_data[ticker]['Close']) for ticker in tickers})
correlation = closing_prices.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()
