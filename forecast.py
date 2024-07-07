import yfinance as yf
import pandas as pd
import numpy as np
import itertools
from ta import add_all_ta_features
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
import streamlit as st
import plotly.graph_objects as go

# Function to fetch stock data from Yahoo Finance
def get_stock_data(ticker, start, end):
    stock_data = yf.download(ticker, start=start, end=end)
    if stock_data.empty:
        st.error("No data fetched. Please check the ticker and date range.")
        return None
    st.write("Initial Data:")
    st.write(stock_data.tail())  # Debugging: show the initial fetched data
    return stock_data

# Function to add technical indicators
def add_technical_indicators(data):
    data_with_indicators = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume")
    st.write("Data with Technical Indicators:")
    st.write(data_with_indicators.tail())  # Debugging: show the data after adding technical indicators
    return data_with_indicators

# Function to homogenize the data
def homogenize_data(data):
    data = data.dropna()
    data = data.select_dtypes(include=[np.number])
    st.write("Homogenized Data:")
    st.write(data.tail())  # Debugging: show the homogenized data
    return data

# Function to perform SARIMA hyperparameter tuning
def sarima_hyperparameter_tuning(data, seasonal_period):
    p = d = q = range(0, 3)
    pdq = [(x[0], x[1], x[2]) for x in list(itertools.product(p, d, q))]
    seasonal_pdq = [(x[0], x[1], x[2], seasonal_period) for x in list(itertools.product(p, d, q))]
    
    best_aic = float("inf")
    best_params = None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = SARIMAX(data,
                              order=param,
                              seasonal_order=param_seasonal,
                              enforce_stationarity=False,
                              enforce_invertibility=False)
                results = mod.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_params = (param, param_seasonal)
            except:
                continue
    return best_params

# Function to fit SARIMA model
def fit_sarima_model(data, order, seasonal_order):
    model = SARIMAX(data, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit()
    return results

# Streamlit app
st.title('Stock Price Forecasting using SARIMA')

# User inputs
ticker = st.text_input('Enter Stock Ticker', 'AAPL')
start_date = st.date_input('Start Date', value=pd.to_datetime('2015-01-01'))
end_date = st.date_input('End Date', value=pd.to_datetime('2023-01-01'))

if st.button('Fetch Data and Forecast'):
    stock_data = get_stock_data(ticker, start_date, end_date)
    
    if stock_data is not None:
        stock_data_with_indicators = add_technical_indicators(stock_data)
        
        st.write("Columns after adding technical indicators:")
        st.write(stock_data_with_indicators.columns)
        
        homogenized_data = homogenize_data(stock_data_with_indicators)
        
        st.write("Columns after homogenizing data:")
        st.write(homogenized_data.columns)
        
        if 'Close' in homogenized_data.columns:
            close_prices = homogenized_data['Close']
        
            st.write('Data fetched successfully!')
            st.write("Close Prices Data:")
            st.write(close_prices.tail())  # Debugging: show the close prices data

            if not close_prices.empty:
                # Hyperparameter tuning
                seasonal_period = 12  # Monthly seasonality for stock prices
                best_order, best_seasonal_order = sarima_hyperparameter_tuning(close_prices, seasonal_period)

                st.write(f'Best SARIMA order: {best_order}')
                st.write(f'Best Seasonal order: {best_seasonal_order}')

                # Fit the SARIMA model
                sarima_model = fit_sarima_model(close_prices, best_order, best_seasonal_order)
                
                # Forecast
                forecast_steps = 30  # Forecast for the next 30 days
                forecast = sarima_model.get_forecast(steps=forecast_steps)
                
                # Debugging information
                st.write("Forecast object:", forecast)
                st.write("Close prices index:", close_prices.index)
                
                forecast_index = pd.date_range(start=close_prices.index[-1], periods=forecast_steps+1, closed='right')
                forecast_df = pd.DataFrame(forecast.predicted_mean, index=forecast_index, columns=['Forecast'])
                forecast_ci = forecast.conf_int()

                # Plot the results
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=close_prices.index, y=close_prices, mode='lines', name='Observed'))
                fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines', name='Forecast'))
                fig.add_trace(go.Scatter(x=forecast_ci.index, y=forecast_ci.iloc[:, 0], fill=None, mode='lines', line_color='gray', name='Lower CI'))
                fig.add_trace(go.Scatter(x=forecast_ci.index, y=forecast_ci.iloc[:, 1], fill='tonexty', mode='lines', line_color='gray', name='Upper CI'))

                st.plotly_chart(fig)
                st.write('Forecasted Values:')
                st.write(forecast_df)
            else:
                st.error("Close prices data is empty. Cannot perform forecasting.")
        else:
            st.error("Close prices not found in the homogenized data.")
    else:
        st.error("Failed to fetch data.")


