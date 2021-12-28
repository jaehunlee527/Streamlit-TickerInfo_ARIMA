from indicators import *

import math
import pandas as pd
import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st
from bokeh.plotting import figure, show
from bokeh.models import Span, LinearAxis, Range1d

import statsmodels
import statsmodels.api as sm  
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA

import warnings
warnings.filterwarnings('ignore')

@st.cache
def df_caller(name):

    df = yf.download(name)

    if len(df) == 0:
        st.warning('Invalid Ticker')
        st.stop()
    else:
        df[['Open','High','Low','Close','Adj Close']]
        df_close = pd.DataFrame(df['Close'])

    return df, df_close

# Slice dataframe with particular indices, or dates
def df_slicer(df, start, end):
    if start < df.index[0]:
        start = df.index[0]

    if type(start) == str:
        # Convert start and end into datetime
        start = datetime.strptime(start, "%Y%m%d")
        end = datetime.strptime(end, "%Y%m%d")
        return df[start:end]
    elif type(start) == int:
        return df[start:end]
    else:
    	return df[start:end]

    raise ValueError

def plot_price(df, company, days, show_bollinger):
    
    p = figure(
            width=800,
            height=400,
            title="%s Historical Price" %company,
            x_axis_type='datetime'
            )

    p.title.align = "center"
    p.title.text_font_size = "24px"
    p.line(df.index, df['Close'], legend_label='Price')

    if days:
        rolmean = moving_average(df, days)
        p.line(df.index, rolmean, line_color='red', legend_label="Moving Average (%sD)" %days)

    if show_bollinger:
        upper, lower = bollinger(df)
        df['bollinger_upper'] = upper
        df['bollinger_lower'] = lower
        p.varea(df.index, lower, upper, color="brown", alpha=0.2)

    st.bokeh_chart(p)

def plot_pct(df):
   
    df_pct = (df['Close'] - df['Close'].shift(1))/df['Close'] * 100 
   
    p1 = figure(
            width=800,
            height=400,
            title='Daily Percentage Change',
            x_axis_type='datetime'
        )

    p1.title.align = "center"
    p1.title.text_font_size = "24px"
    p1.line(df_pct.index, df_pct, legend_label="% Change")

    hline = Span(location=0, dimension='width', line_color='black', line_width=2)
    p1.renderers.extend([hline])

    st.bokeh_chart(p1)

def plot_ind(df, show_macd, show_rsi):
    p2 = figure(
            width=800,
            height=400,
            #title=
            x_axis_type='datetime'
        )

    if show_macd and show_rsi:
        df_macd = macd_hist(df)
        p2.line(df_macd.index, df_macd, legend_label="MACD")
        p2.y_range = Range1d(math.floor(min(df_macd)), math.ceil(max(df_macd)))

        p2.extra_y_ranges = {"rsi": Range1d(start=0, end=100)}
        p2.add_layout(LinearAxis(y_range_name="rsi"), 'right')  # Add the second y-axis
        
        df_rsi = rsi(df, 14)
        p2.line(df_macd.index, df_rsi, line_color = 'red', legend_label="RSI", y_range_name="rsi")

    elif show_macd:
        df_macd = macd_hist(df)
        p2.line(df_macd.index, df_macd, legend_label="MACD")

    elif show_rsi:
        df_rsi = rsi(df, 14) # 14 day is the standard 
        p2.line(df_rsi.index, df_rsi, line_color = 'red', legend_label="RSI")

    st.bokeh_chart(p2)

def adf(df):
    dftest = adfuller(df, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    dfoutput['#Lags Used'] = int(dfoutput['#Lags Used'])
    
    for k,v in dftest[4].items():
        dfoutput["Critical Value (%s)" %k] = v
    print(dfoutput)

# differencing
def df_shift(df):
    df_diff = df - df.shift(1)
    df_diff.dropna(inplace=True)
    return df_diff

def acf_plot(df):
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(df, lags=20, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(df, lags=20, ax=ax2)

def pdq_calc(df):
    # Define the p, d and q parameters to take any value between 0 and 3
    p = q = range(1, 3)
    d = range(1,3)
    # Generate all different combinations of p, q and q
    pdq = list(itertools.product(p, d, q))

    best_aic = math.inf
    # Compute AIC for each (p,d,q)
    for param in pdq:
        model = sm.tsa.statespace.SARIMAX(df, order=param, enforce_stationarity=True)
        result = model.fit()
        if result.aic < best_aic:
            best_aic = result.aic
            best_param = param

    return best_param

def predict(df, best_param, day_range): 
    fig, ax = plt.subplots(figsize=(12,6))

    #ax = df['Close'].plot(ax=ax)
    #x_range_day = pd.date_range(df.index[0], df.index[-1] + timedelta(days=50))   
    
    model = ARIMA(df, order=best_param)
    model_fit = model.fit(disp=0)
    fig = model_fit.plot_predict(start=1, end=len(df)+day_range, ax=ax)
  
    #x_range = list(range(len(df) + 50))
    #plt.xticks(x_range, x_range_day)
    #plt.locator_params(nbins=8)

    #ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    #ax.xaxis.set_minor_formatter(mdates.DateFormatter("%Y-%m"))

    st.pyplot(fig)
