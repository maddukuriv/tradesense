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
from pmdarima import auto_arima
from datetime import timedelta, datetime
from scipy.signal import cwt, ricker, hilbert

# Function to fetch stock data from Yahoo Finance
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data.reset_index(inplace=True)
    return stock_data

# Function to calculate technical indicators
def calculate_technical_indicators(df):
    # Moving Average Convergence Divergence (MACD)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
                
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
                
    # Bollinger Bands
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Std_Dev'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_20'] + (df['Std_Dev'] * 2)
    df['Lower_Band'] = df['SMA_20'] - (df['Std_Dev'] * 2)
    df['BBW'] = (df['Upper_Band'] - df['Lower_Band']) / df['SMA_20']  # Bollinger Band Width
                
    # Exponential Moving Average (EMA)
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
                
    # Average True Range (ATR)
    df['High-Low'] = df['High'] - df['Low']
    df['High-PrevClose'] = np.abs(df['High'] - df['Close'].shift(1))
    df['Low-PrevClose'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['True_Range'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    df['ATR'] = df['True_Range'].rolling(window=14).mean()
                
    # Rate of Change (ROC)
    df['ROC'] = df['Close'].pct_change(periods=12) * 100
                
    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
                
    return df

# Functions for Fourier, Wavelet, and Hilbert transforms
def calculate_fourier(df, n=5):
    close_fft = np.fft.fft(df['Close'].values)
    fft_df = pd.DataFrame({'fft': close_fft})
    fft_df['absolute'] = np.abs(fft_df['fft'])
    fft_df['angle'] = np.angle(fft_df['fft'])
    fft_df = fft_df.sort_values(by='absolute', ascending=False).head(n)
    fft_df = fft_df.reindex(range(len(df)), fill_value=0)  # Ensure it matches the length of the stock data
    return fft_df['absolute']

def calculate_wavelet(df):
    widths = np.arange(1, 31)
    cwt_matrix = cwt(df['Close'], ricker, widths)
    max_wavelet = np.max(cwt_matrix, axis=0)
    return pd.Series(max_wavelet).reindex(range(len(df)), fill_value=0)  # Ensure it matches the length of the stock data

def calculate_hilbert(df):
    analytic_signal = hilbert(df['Close'])
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    return pd.Series(amplitude_envelope), pd.Series(instantaneous_phase)

# Streamlit UI
st.title("Stock Price Prediction with SARIMA Model")

# Sidebar for user input
st.sidebar.subheader("Settings")
ticker_symbol = st.sidebar.text_input("Enter Stock Symbol", value="CUMMINSIND.NS")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2021-06-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))
forecast_period = st.sidebar.number_input("Forecast Period (days)", value=10, min_value=1, max_value=30)

# Fetch the data
stock_data = get_stock_data(ticker_symbol, start_date, end_date)

# Calculate technical indicators
stock_data = calculate_technical_indicators(stock_data)

# Calculate Fourier, Wavelet, and Hilbert transforms
stock_data['Fourier'] = calculate_fourier(stock_data)
stock_data['Wavelet'] = calculate_wavelet(stock_data)
stock_data['Hilbert_Amplitude'], stock_data['Hilbert_Phase'] = calculate_hilbert(stock_data)

# Drop rows with NaN values
stock_data.dropna(inplace=True)

# Extract the closing prices and technical indicators
close_prices = stock_data['Close']
technical_indicators = stock_data[['MACD', 'Signal_Line', 'MACD_Hist', 'RSI', 'SMA_20', 'Upper_Band', 'Lower_Band', 'BBW', 'EMA_50', 'ATR', 'ROC', 'OBV', 'Fourier', 'Wavelet', 'Hilbert_Amplitude', 'Hilbert_Phase']]

# Check correlation with close prices
correlations = technical_indicators.corrwith(close_prices).sort_values()

# Display correlation as a bar chart
fig_corr = go.Figure(go.Bar(
    x=correlations.values,
    y=correlations.index,
    orientation='h'
))
fig_corr.update_layout(
    title="Correlation with Close Prices",
    xaxis_title="Correlation",
    yaxis_title="Indicators",
    yaxis=dict(tickmode='linear')
)
st.plotly_chart(fig_corr)

# Train SARIMA model with exogenous variables (technical indicators)
sarima_model = auto_arima(
    close_prices,
    exogenous=technical_indicators,
    start_p=1,
    start_q=1,
    max_p=3,
    max_q=3,
    m=7,  # Weekly seasonality
    start_P=0,
    seasonal=True,
    d=1,
    D=1,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)

# Forecasting the next n business days (excluding weekends)
future_technical_indicators = technical_indicators.tail(forecast_period).values
forecast = sarima_model.predict(n_periods=forecast_period, exogenous=future_technical_indicators)

# Generate the forecasted dates excluding weekends
forecasted_dates = pd.bdate_range(start=stock_data['Date'].iloc[-1], periods=forecast_period + 1)[1:]

forecasted_df = pd.DataFrame({'Date': forecasted_dates, 'Forecasted_Close': forecast})

# Plotly interactive figure with time slider
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=stock_data['Date'],
    y=stock_data['Close'],
    mode='lines',
    name='Close Price'
))

# Add forecasted prices
fig.add_trace(go.Scatter(
    x=forecasted_df['Date'],
    y=forecasted_df['Forecasted_Close'],
    mode='lines',
    name='Forecasted Close Price',
    line=dict(color='orange')
))

fig.update_layout(
    title=f'Stock Price Forecast for {ticker_symbol}',
    xaxis_title='Date',
    yaxis_title='Close Price',
    legend=dict(x=0.01, y=0.99),
)

st.plotly_chart(fig)

# Display forecasted prices
st.write("Forecasted Prices for the next {} days:".format(forecast_period))
st.dataframe(forecasted_df)

# Display model summary
st.write("Model Summary:")
st.text(sarima_model.summary())
