import pandas as pd
import numpy as np


def moving_average(df, days):
    """
    Calculates moving average of closing prices
    
    :param 1 df
    :param 2 days: days of window to calculate moving average
    :return: pandas series MA values
    """
    ma = df['Close'].rolling(window=days, min_periods=1).mean()
    ma = ma.round(3)
    return ma


def macd_hist(df):
    """
    Moving Average Convergence Divergence with respect to closing price - Based on difference between 12 & 26 day exponential moving average
    """
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    
    return hist


def bollinger(df):
    """
    Bollinger Bands Calcluation with respect to closing prices
    """
    price = (df['Close'] + df['High'] + df['Low']) / 3
    sma = moving_average(df, '20D')
    upper = sma + 2 * df['Close'].rolling('20D').std()
    lower = sma - 2 * df['Close'].rolling('20D').std()
    
    return upper, lower

def rsi(df, days):
    """
    RSI calculation with respect to closing prices
    """
    delta = df['Close'].diff()
    
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)

    ma_up = up.rolling(window = days, min_periods=1).mean()
    ma_down = down.rolling(window = days, min_periods=1).mean()

    res = ma_up / ma_down
    res = 100 - (100/(1 + res))

    return res
