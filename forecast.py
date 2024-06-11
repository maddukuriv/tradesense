import streamlit as st
from datetime import date
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import ta
import pandas_ta as pta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import hilbert
import pywt
import holidays
import mplfinance as mpf

# Step 2: Search box for stock ticker
ticker = st.text_input('Enter Stock Ticker', 'RVNL.NS')

# Create two columns
col1, col2 = st.columns(2)

# Set up the start and end date inputs
with col1:
    START = st.date_input('Start Date', pd.to_datetime("2015-01-01"))
with col2:
    END = st.date_input('End Date', pd.to_datetime("today"))

# Fetch historical data from Yahoo Finance
@st.cache_data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start_date, end_date)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(ticker, START, END)
data_load_state.text('Loading data...done!')

# Display the raw data
st.subheader('Historical data')
st.write(data.tail())

# Ensure the index is of datetime type
data.index = pd.to_datetime(data.index)

# Add calendar features
data['day_of_week'] = data['Date'].dt.dayofweek
data['month'] = data['Date'].dt.month
data['quarter'] = data['Date'].dt.quarter

# Add holiday feature
us_holidays = holidays.US()
data['holiday'] = data['Date'].apply(lambda x: 1 if x in us_holidays else 0)

# Calculate technical indicators
data['SMA_5'] = ta.trend.SMAIndicator(close=data['Close'], window=5).sma_indicator()
data['SMA_10'] = ta.trend.SMAIndicator(close=data['Close'], window=10).sma_indicator()
data['SMA_20'] = ta.trend.SMAIndicator(close=data['Close'], window=20).sma_indicator()

data['EMA_10'] = ta.trend.EMAIndicator(close=data['Close'], window=10).ema_indicator()
data['EMA_20'] = ta.trend.EMAIndicator(close=data['Close'], window=20).ema_indicator()

macd = ta.trend.MACD(close=data['Close'])
data['MACD'] = macd.macd()
data['MACD_Signal'] = macd.macd_signal()

rsi = ta.momentum.RSIIndicator(close=data['Close'])
data['RSI'] = rsi.rsi()

bb = ta.volatility.BollingerBands(close=data['Close'])
data['BB_High'] = bb.bollinger_hband()
data['BB_Low'] = bb.bollinger_lband()

adx = ta.trend.ADXIndicator(high=data['High'], low=data['Low'], close=data['Close'])
data['ADX'] = adx.adx()

data['5_DAYS_STD_DEV'] = data['Close'].rolling(window=5).std()
data['ATR'] = ta.volatility.AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close'], window=14).average_true_range()

# Calculate Volume Moving Average
data['Volume_MA_20'] = data['Volume'].rolling(window=20).mean()
data['Volume_MA_10'] = data['Volume'].rolling(window=10).mean()

# On-Balance Volume (OBV)
data['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=data['Close'], volume=data['Volume']).on_balance_volume()

# Fourier Transform
data['FFT'] = np.abs(fft(data['Close'].values))

# Wavelet Transform
def wavelet_transform(data, wavelet='db1'):
    coeffs = pywt.wavedec(data, wavelet)
    return np.concatenate(coeffs)

data['Wavelet'] = data['Close'].apply(lambda x: wavelet_transform(data['Close'].values)).apply(lambda x: x[0])

# Hilbert Transform
data['Hilbert'] = np.abs(hilbert(data['Close'].values))

# Define custom candlestick pattern detection functions
def is_hammer(data):
    return (data['Close'] > data['Open']) & ((data['High'] - data['Close']) <= (data['Open'] - data['Low']) * 2)

def is_doji(data):
    return (abs(data['Close'] - data['Open']) <= (data['High'] - data['Low']) * 0.1)

def is_engulfing(data):
    return ((data['Open'] < data['Close'].shift(1)) & (data['Close'] > data['Open'].shift(1))) | ((data['Open'] > data['Close'].shift(1)) & (data['Close'] < data['Open'].shift(1)))

# Apply the custom candlestick pattern functions
data['Hammer'] = is_hammer(data)
data['Doji'] = is_doji(data)
data['Engulfing'] = is_engulfing(data)

# Drop rows with NaN values after calculating indicators
data.dropna(inplace=True)

# Define features and target variable
features = ['Open', 'High', 'Low', 'Volume', 'SMA_5', 'SMA_10', 'SMA_20', 'EMA_10', 'EMA_20',
            'MACD', 'MACD_Signal', 'RSI', 'BB_High', 'BB_Low', '5_DAYS_STD_DEV', 'ATR',
            'Volume_MA_20', 'Volume_MA_10', 'OBV', 'FFT', 'Wavelet', 'Hilbert', 'day_of_week',
            'month', 'quarter', 'holiday', 'Hammer', 'Doji', 'Engulfing']
X = data[features]
y = data['Close']

# Standardization
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
scaled_features = scaler_X.fit_transform(X)
scaled_target = scaler_y.fit_transform(y.values.reshape(-1, 1))
X = pd.DataFrame(scaled_features, columns=features)
y = pd.DataFrame(scaled_target, columns=['Close'])

# Split data into training and testing sets using time series split
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Evaluate model performance
def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae

# Train and evaluate SARIMA model
st.text("Training SARIMA model...")
sarima_model = SARIMAX(data['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit(disp=False)

# Forecasting next 5 days
st.text("Forecasting next 5 days...")
forecast_steps = 5
forecast = sarima_result.get_forecast(steps=forecast_steps)
forecasted_values = forecast.predicted_mean
forecasted_values_conf_int = forecast.conf_int()

# Print and plot the forecast
forecast_labels = ['Day1', 'Day2', 'Day3', 'Day4', 'Day5']
forecast_df = pd.DataFrame({
    'Day': forecast_labels,
    'Forecasted_Price': forecasted_values,
    'Lower Bound': forecasted_values_conf_int.iloc[:, 0],
    'Upper Bound': forecasted_values_conf_int.iloc[:, 1]
})

st.write("Next 5 days forecast:")
st.dataframe(forecast_df)

forecast_df.set_index('Day')[['Forecasted_Price', 'Lower Bound', 'Upper Bound']].plot()
plt.fill_between(forecast_df.index, forecast_df['Lower Bound'], forecast_df['Upper Bound'], color='gray', alpha=0.2)
plt.title('Next 5 Days Forecast')
plt.xlabel('Day')
plt.ylabel('Price')
plt.show()

# Plot close using plotly with a time bar
forecast_fig = go.Figure()

# Prices plot
forecast_fig.add_trace(go.Scatter(x=forecast_labels, y=forecasted_values, mode='lines', name='Next 5 Days Forecast'))
forecast_fig.add_trace(go.Scatter(x=forecast_labels, y=forecast_df['Lower Bound'], mode='lines', name='Lower Bound', line=dict(dash='dash')))
forecast_fig.add_trace(go.Scatter(x=forecast_labels, y=forecast_df['Upper Bound'], mode='lines', name='Upper Bound', line=dict(dash='dash')))
# Show the plot
st.plotly_chart(forecast_fig)

# Train LSTM model
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

X_train_lstm = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_lstm = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

lstm_model = create_lstm_model()
lstm_model.fit(X_train_lstm, y_train, batch_size=1, epochs=50)

# Make predictions with LSTM model
y_pred_lstm = lstm_model.predict(X_test_lstm)
y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm)
y_test_inv = scaler_y.inverse_transform(y_test)

# Evaluate LSTM model
r2_lstm, rmse_lstm, mae_lstm = evaluate_model(y_test_inv, y_pred_lstm)

st.write("LSTM Model Performance:")
st.write(f"R2 Score: {r2_lstm}")
st.write(f"RMSE: {rmse_lstm}")
st.write(f"MAE: {mae_lstm}")

# Visualize LSTM predictions
lstm_fig = go.Figure()
lstm_fig.add_trace(go.Scatter(x=data.index[-len(y_test):], y=y_test_inv.flatten(), mode='lines', name='Actual Price'))
lstm_fig.add_trace(go.Scatter(x=data.index[-len(y_test):], y=y_pred_lstm.flatten(), mode='lines', name='Predicted Price'))
lstm_fig.update_layout(title='LSTM Model Predictions', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(lstm_fig)

# Ensemble method: Combine predictions
ensemble_pred = (forecasted_values.values + y_pred_lstm.flatten()[:forecast_steps]) / 2

ensemble_fig = go.Figure()
ensemble_fig.add_trace(go.Scatter(x=forecast_labels, y=forecasted_values.values, mode='lines', name='SARIMA Forecast'))
ensemble_fig.add_trace(go.Scatter(x=forecast_labels, y=y_pred_lstm.flatten()[:forecast_steps], mode='lines', name='LSTM Forecast'))
ensemble_fig.add_trace(go.Scatter(x=forecast_labels, y=ensemble_pred, mode='lines', name='Ensemble Forecast'))
st.plotly_chart(ensemble_fig)
