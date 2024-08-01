import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LassoCV
import joblib

# Function to fetch stock data from Yahoo Finance
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data.reset_index(inplace=True)
    return stock_data

# Function to calculate technical indicators for swing trading
def calculate_technical_indicators(df):
    # Short-term Moving Averages (e.g., 10, 20 days)
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
                
    # Exponential Moving Averages (EMAs)
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
                
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
    
    # Lagged Returns
    df['Return_1'] = df['Close'].pct_change(1)
    df['Return_5'] = df['Close'].pct_change(5)
                
    return df

# Streamlit UI
st.title("Stock Price Prediction for Swing Trading")

# Sidebar for user input
st.sidebar.subheader("Settings")
ticker_symbol = st.sidebar.text_input("Enter Stock Symbol", value="CUMMINSIND.NS")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2021-06-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))
forecast_period = st.sidebar.number_input("Forecast Period (days)", value=5, min_value=1, max_value=30)
model_save_path = "lasso_stock_model.pkl"

# Fetch the data
stock_data = get_stock_data(ticker_symbol, start_date, end_date)

# Calculate technical indicators
stock_data = calculate_technical_indicators(stock_data)

# Drop rows with NaN values
stock_data.dropna(inplace=True)

# Extract the closing prices and technical indicators
close_prices = stock_data['Close']
technical_indicators = stock_data[['SMA_10', 'SMA_20', 'EMA_10', 'EMA_20', 'MACD', 'Signal_Line', 'MACD_Hist', 'RSI', 'ATR', 'Return_1', 'Return_5']]

# Normalize the technical indicators
scaler = StandardScaler()
technical_indicators_scaled = scaler.fit_transform(technical_indicators)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(technical_indicators_scaled, close_prices, test_size=0.2, shuffle=False)

# Use Lasso regression for regularization
lasso = LassoCV(cv=TimeSeriesSplit(n_splits=5)).fit(X_train, y_train)

# Save the model
joblib.dump(lasso, model_save_path)

# Load the model
lasso = joblib.load(model_save_path)

# Predictions using Lasso
lasso_forecast_train = lasso.predict(X_train)
lasso_forecast_test = lasso.predict(X_test)

# Calculate R-squared and RMSE for Lasso
lasso_r2_train = r2_score(y_train, lasso_forecast_train)
lasso_r2_test = r2_score(y_test, lasso_forecast_test)
lasso_rmse_train = np.sqrt(mean_squared_error(y_train, lasso_forecast_train))
lasso_rmse_test = np.sqrt(mean_squared_error(y_test, lasso_forecast_test))

# Display R-squared and RMSE for Lasso
st.write(f"Lasso Training R-squared: {lasso_r2_train:.4f}")
st.write(f"Lasso Test R-squared: {lasso_r2_test:.4f}")
st.write(f"Lasso Training RMSE: {lasso_rmse_train:.4f}")
st.write(f"Lasso Test RMSE: {lasso_rmse_test:.4f}")

# Generate the forecasted dates excluding weekends for test set
start_date_test = stock_data['Date'].iloc[len(y_train)]
end_date_test = stock_data['Date'].iloc[-1]
forecasted_dates_test = pd.bdate_range(start=start_date_test, end=end_date_test)

forecasted_df_test_lasso = pd.DataFrame({'Date': forecasted_dates_test[:len(y_test)], 'Forecasted_Close': lasso_forecast_test})

# Plotly interactive figure with time slider
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=stock_data['Date'],
    y=stock_data['Close'],
    mode='lines',
    name='Close Price'
))

# Add forecasted prices for Lasso
fig.add_trace(go.Scatter(
    x=forecasted_df_test_lasso['Date'],
    y=forecasted_df_test_lasso['Forecasted_Close'],
    mode='lines',
    name='Lasso Forecasted Close Price',
    line=dict(color='purple')
))

fig.update_layout(
    title=f'Stock Price Forecast for {ticker_symbol}',
    xaxis_title='Date',
    yaxis_title='Close Price',
    legend=dict(x=0.01, y=0.99),
)

st.plotly_chart(fig)

# Display forecasted prices
st.write("Forecasted Prices for the test period:")
st.dataframe(forecasted_df_test_lasso)

# Summary of results
st.subheader("Summary of Results")
st.write(f"Training R-squared: {lasso_r2_train:.4f}")
st.write(f"Test R-squared: {lasso_r2_test:.4f}")
st.write(f"Training RMSE: {lasso_rmse_train:.4f}")
st.write(f"Test RMSE: {lasso_rmse_test:.4f}")

st.write(f"The model has been saved to {model_save_path} and can be loaded for future predictions.")
