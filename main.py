# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 11:41:04 2021

@author: PSOMIADIS
"""


from datetime import datetime
from datetime import timedelta
from enum import Enum
from io import BytesIO, StringIO
from typing import Union
import streamlit as st
import pandas as pd

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation
from pypfopt.risk_models import CovarianceShrinkage

from stocks import stocks_list, download_from_yahoo
from portfolio_functions import momentum_score, capm_return, get_latest_prices
from portfolio_functions import select_columns, download_button
from portfolio_functions import get_portfolio,rebalance_portfolio, backtest_portfolio

from sklearn.metrics import mean_absolute_error
from fbprophet import Prophet
from matplotlib import pyplot

import logging

logging.getLogger('fbprophet').setLevel(logging.WARNING)

def load_data(tickers_sp500, start, end, interval='1d'):
    return download_from_yahoo(tickers_sp500, start, end, '1d')

pd.set_option('max_columns',200)
pd.set_option('max_rows',200)
# get current date as end_date
end_date = datetime.strftime(datetime.now().date(), '%Y-%m-%d')
# get as start date 1500 days ago
start_date = datetime.strftime(
    datetime.now() - timedelta(days=1500), '%Y-%m-%d')
# Load rows of data into a dataframe.
stocks = stocks_list()
stocks_data = load_data(stocks, start_date, end_date, '1d')
# create the closing prices dataframe
l_close = pd.DataFrame(columns=['stock', 'date', 'last_price', 'len_prices'])
close_data = stocks_data['Adj Close']
for ticker in stocks:
    last_close = stocks_data['Adj Close'].iloc[-1][ticker]
    last_date = datetime.strftime(stocks_data['Adj Close'].index[-1], '%Y-%m-%d')
    len_values = len(stocks_data)
    l_close = l_close.append({'stock': ticker, 'date': last_date, 'last_price': last_close,
                              'len_prices': len_values}, ignore_index=True)
l_close_min = l_close['len_prices'].min()
l_close=l_close.drop('len_prices',axis=1)

l_close['pct_change_5days']=[0]*len(l_close)
l_close['pct_change_10days']=[0]*len(l_close)
l_close['pct_change_15days']=[0]*len(l_close)
l_close['pct_change_20days']=[0]*len(l_close)
l_close['price_5days']=[0]*len(l_close)
l_close['price_10days']=[0]*len(l_close)
l_close['price_15days']=[0]*len(l_close)
l_close['price_20days']=[0]*len(l_close)

df = close_data
df=df.reset_index()
new_data={}
for stock in list(df.columns):
    df_temp=pd.DataFrame(columns=['ds','y'])
    df_temp['ds']=df.Date
    df_temp['y']=df[stock]
    # define the model
    model = Prophet()
    # fit the model
    model.fit(df_temp)
    # define the period for which we want a prediction
    future = model.make_future_dataframe(periods=20)
    # use the model to make a forecast
    forecast = model.predict(future)
    # summarize the forecast
    df_temp['trend']=forecast['trend'][:-20]
    df_temp['yhat']=forecast['yhat'][:-20]
    df_temp['trend_lower']=forecast['trend_lower'][:-20]
    df_temp['yhat_lower']=forecast['yhat_lower'][:-20]
    new_data[stock]=df_temp
    
    forecast_values=forecast[['ds', 'trend_lower', 'yhat_lower']].tail(20).reset_index(drop=True)
    l_close.loc[l_close['stock']==stock,'price_5days'] = forecast_values.loc[4,'trend_lower']
    l_close.loc[l_close['stock']==stock,'price_10days']= forecast_values.loc[9,'trend_lower']
    l_close.loc[l_close['stock']==stock,'price_15days']= forecast_values.loc[14,'trend_lower']
    l_close.loc[l_close['stock']==stock,'price_20days']= forecast_values.loc[19,'trend_lower']

l_close['last_price']=round(l_close['last_price'],2)
l_close['price_5days']=round(l_close['price_5days'],2)
l_close['price_10days']=round(l_close['price_10days'],2)
l_close['price_15days']=round(l_close['price_15days'],2)
l_close['price_20days']=round(l_close['price_20days'],2)
l_close['pct_change_5days'] = round(l_close['price_5days']/l_close['last_price']-1,3)
l_close['pct_change_10days']= round(l_close['price_10days']/l_close['last_price']-1,3)
l_close['pct_change_15days']= round(l_close['price_15days']/l_close['last_price']-1,3)
l_close['pct_change_20days']= round(l_close['price_20days']/l_close['last_price']-1,3)


portfolio_value=50000
portfolio_size=20
cutoff=0.05
universe_df=l_close.sort_values(by=['pct_change_5days','pct_change_10days','pct_change_15days','pct_change_20days'], ascending=False).head(100).tail(90).head(portfolio_size).reset_index(drop=True)
universe=universe_df['stock'].to_list()
(new_portfolio,non_trading_cash)=get_portfolio(universe, df, portfolio_value, cutoff)
total_invested=round(new_portfolio['value'].sum()-non_trading_cash,2)
new_portfolio['price_in_5days']=[0]*len(new_portfolio)
for stock in new_portfolio.index.to_list()[:-1]:
    new_portfolio.loc[new_portfolio.index==stock,'price_in_5days']=l_close.loc[l_close['stock']==stock,'price_5days'].values[0]
new_portfolio.loc[new_portfolio.index=='CASH','price_in_5days']=new_portfolio.loc[new_portfolio.index=='CASH','value']
new_portfolio['value_in_5days']=new_portfolio['price_in_5days']*new_portfolio['shares']
new_portfolio_value=round(new_portfolio['value_in_5days'].sum(),2)
