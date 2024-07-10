import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from scipy.signal import find_peaks, cwt, ricker, hilbert
from pmdarima import auto_arima
import plotly.graph_objects as go
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

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
    freqs = np.fft.fftfreq(len(df))
    dominant_freqs = freqs[np.argsort(np.abs(close_fft))[-n:]]
    positive_freqs = np.abs(dominant_freqs)
    cycles_in_days = 1 / positive_freqs
    fft_df['cycles_in_days'] = cycles_in_days
    return fft_df

def calculate_wavelet(df):
    widths = np.arange(1, 31)
    cwt_matrix = cwt(df['Close'], ricker, widths)
    return cwt_matrix

def calculate_hilbert(df):
    analytic_signal = hilbert(df['Close'])
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    return amplitude_envelope, instantaneous_phase

def ensure_stationarity(df):
    df['Close_diff'] = df['Close'].diff().dropna()
    return df.dropna()

# Function to inverse the differencing
def invert_difference(history, forecast):
    history_len = len(history)
    forecast_len = len(forecast)
    if forecast_len == 0 or history_len == 0:
        return pd.Series([])  # Return an empty series if inputs are empty
    
    # Ensure forecast length does not exceed history length
    if forecast_len > history_len:
        forecast = forecast[:history_len]

    forecast_series = pd.Series(forecast, index=history.index[-forecast_len:])
    inverted = history['Close'].iloc[-forecast_len - 1] + forecast_series.cumsum()
    return inverted

# Streamlit UI
st.title("Stock Price Prediction with SARIMA Model")

# Sidebar for user input
st.sidebar.subheader("Settings")
ticker_symbol = st.sidebar.text_input("Enter Stock Symbol", value="CUMMINSIND.NS")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-06-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-06-21"))
forecast_period = st.sidebar.number_input("Forecast Period (days)", value=10, min_value=1, max_value=30)

# Fetch the data
stock_data = get_stock_data(ticker_symbol, start_date, end_date)

# Ensure stationarity
stock_data = ensure_stationarity(stock_data)

# Calculate technical indicators
stock_data = calculate_technical_indicators(stock_data)

# Calculate Fourier, Wavelet, and Hilbert transforms
fft_df = calculate_fourier(stock_data)
cwt_matrix = calculate_wavelet(stock_data)
amplitude_envelope, instantaneous_phase = calculate_hilbert(stock_data)

# Drop rows with NaN values
stock_data.dropna(inplace=True)

# Extract the closing prices and technical indicators
close_prices = stock_data['Close_diff']
technical_indicators = stock_data[['MACD', 'Signal_Line', 'MACD_Hist', 'RSI', 'SMA_20', 'Upper_Band', 'Lower_Band', 'BBW', 'EMA_50', 'ATR', 'ROC', 'OBV']]

# Normalize the data
scaler = StandardScaler()
technical_indicators = pd.DataFrame(scaler.fit_transform(technical_indicators), columns=technical_indicators.columns)

# Split the data into training and test sets
train_size = int(len(close_prices) * 0.8)
X_train, X_test = technical_indicators[:train_size], technical_indicators[train_size:]
y_train, y_test = close_prices[:train_size], close_prices[train_size:]

# Custom cross-validation and model training
tscv = TimeSeriesSplit(n_splits=5)
cv_mse = []
for train_index, val_index in tscv.split(X_train):
    X_train_cv, X_val_cv = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]
    
    sarima_model = auto_arima(
        y_train_cv,
        exogenous=X_train_cv,
        start_p=1,
        start_q=1,
        max_p=5,
        max_q=5,
        m=7,  # Weekly seasonality
        start_P=0,
        seasonal=True,
        d=1,
        D=1,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,
        n_jobs=-1
    )
    
    y_val_pred = sarima_model.predict(n_periods=len(X_val_cv), exogenous=X_val_cv)
    mse = mean_squared_error(y_val_cv, y_val_pred)
    cv_mse.append(mse)

st.write(f"Cross-Validation MSE: {np.mean(cv_mse)}")

# Train final model on entire training set
sarima_model = auto_arima(
    y_train,
    exogenous=X_train,
    start_p=1,
    start_q=1,
    max_p=5,
    max_q=5,
    m=7,  # Weekly seasonality
    start_P=0,
    seasonal=True,
    d=1,
    D=1,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True,
    n_jobs=-1
)

# Generate predictions on the test set
y_pred_diff = sarima_model.predict(n_periods=len(X_test), exogenous=X_test)
y_pred = invert_difference(stock_data.iloc[train_size:], y_pred_diff)

# Ensure there are no NaN values
y_pred = y_pred.dropna()
y_test = y_test.iloc[-len(y_pred):]

# Calculate evaluation metrics
mse = mean_squared_error(stock_data['Close'].iloc[train_size:train_size+len(y_pred)], y_pred)
mae = mean_absolute_error(stock_data['Close'].iloc[train_size:train_size+len(y_pred)], y_pred)
r2 = r2_score(stock_data['Close'].iloc[train_size:train_size+len(y_pred)], y_pred)

st.write(f"Test MSE: {mse}")
st.write(f"Test MAE: {mae}")
st.write(f"Test R^2: {r2}")

# Forecasting the next n business days (excluding weekends)
future_technical_indicators = technical_indicators.tail(forecast_period).values
forecast_diff = sarima_model.predict(n_periods=forecast_period, exogenous=future_technical_indicators)
forecast = invert_difference(stock_data, forecast_diff)

# Ensure the forecast length does not exceed the history length
forecast = forecast[:forecast_period]

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
