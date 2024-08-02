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
from datetime import datetime, timedelta

import pandas_ta as ta

# Function to fetch stock data from Yahoo Finance
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data.reset_index(inplace=True)
    return stock_data

# Define a function to calculate William Arbitrage
def calculate_williams_alligator(df):
    jaw_length = 13
    teeth_length = 8
    lips_length = 5

    df['Jaw'] = df['Close'].shift(jaw_length).rolling(window=jaw_length).mean()
    df['Teeth'] = df['Close'].shift(teeth_length).rolling(window=teeth_length).mean()
    df['Lips'] = df['Close'].shift(lips_length).rolling(window=lips_length).mean()

    return df

# Define a function to calculate technical indicators
def calculate_indicators(df):
    df['CMO'] = ta.cmo(df['Close'], length=14)
    
    keltner = ta.kc(df['High'], df['Low'], df['Close'])
    df['Keltner_High'] = keltner['KCUe_20_2']
    df['Keltner_Low'] = keltner['KCLe_20_2']
    
    df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)
    df['Ultimate_Oscillator'] = ta.uo(df['High'], df['Low'], df['Close'])
    
    kvo = ta.kvo(df['High'], df['Low'], df['Close'], df['Volume'])
    df['Klinger'] = kvo['KVO_34_55_13']
    
    donchian = ta.donchian(df['High'], df['Low'])
    df['Donchian_High'] = donchian['DCU_20_20']
    df['Donchian_Low'] = donchian['DCL_20_20']
    
    df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume']).astype(float)
    
    distance_moved = ((df['High'] + df['Low']) / 2) - ((df['High'].shift(1) + df['Low'].shift(1)) / 2)
    box_ratio = (df['Volume'] / 1e8) / (df['High'] - df['Low'])
    emv = distance_moved / box_ratio
    df['Ease_of_Movement'] = emv.rolling(window=14).mean()
    
    df['Chaikin_MF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'])
    
    df['Williams_R'] = ta.willr(df['High'], df['Low'], df['Close'])
    
    trix = ta.trix(df['Close'])
    df['Trix'] = trix['TRIX_30_9']
    df['Trix_Signal'] = trix['TRIXs_30_9']
    
    vortex = ta.vortex(df['High'], df['Low'], df['Close'])
    df['Vortex_Pos'] = vortex['VTXP_14']
    df['Vortex_Neg'] = vortex['VTXM_14']
    
    supertrend = ta.supertrend(df['High'], df['Low'], df['Close'], length=7, multiplier=3.0)
    df['SuperTrend'] = supertrend['SUPERT_7_3.0']
    
    df['RVI'] = ta.rvi(df['High'], df['Low'], df['Close'])
    df['RVI_Signal'] = ta.ema(df['RVI'], length=14)
    
    bull_power = df['High'] - ta.ema(df['Close'], length=13)
    bear_power = df['Low'] - ta.ema(df['Close'], length=13)
    df['Elder_Ray_Bull'] = bull_power
    df['Elder_Ray_Bear'] = bear_power
    
    wad = ta.ad(df['High'], df['Low'], df['Close'], df['Volume'])
    df['Williams_AD'] = wad
    
    # Darvas Box Theory
    df['Darvas_High'] = df['High'].rolling(window=20).max()
    df['Darvas_Low'] = df['Low'].rolling(window=20).min()
    
    # Volume Profile calculation
    df['Volume_Profile'] = df.groupby(pd.cut(df['Close'], bins=20))['Volume'].transform('sum')

    # Additional technical indicators
    df['5_day_EMA'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['10_day_EMA'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['20_day_EMA'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['12_day_EMA'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['26_day_EMA'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['12_day_EMA'] - df['26_day_EMA']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stochastic_%K'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
    df['Stochastic_%D'] = df['Stochastic_%K'].rolling(window=3).mean()
    
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    df['A/D_line'] = (clv * df['Volume']).fillna(0).cumsum()
    
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    df['5_day_Volume_MA'] = df['Volume'].rolling(window=5).mean()
    df['10_day_Volume_MA'] = df['Volume'].rolling(window=10).mean()
    df['20_day_Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['20_day_SMA'] = df['Close'].rolling(window=20).mean()
    df['Std_Dev'] = df['Close'].rolling(window=20).std()
    df['BB_High'] = df['20_day_SMA'] + (df['Std_Dev'] * 2)
    df['BB_Low'] = df['20_day_SMA'] - (df['Std_Dev'] * 2)
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = high_low.combine(high_close, np.maximum).combine(low_close, np.maximum)
    df['ATR'] = tr.rolling(window=14).mean()
    
    # Parabolic SAR calculation
    df['Parabolic_SAR'] = calculate_parabolic_sar(df)
    
    # ADX calculation
    df['ADX'] = calculate_adx(df)
    
    # Ichimoku Cloud calculation
    df['Ichimoku_conv'], df['Ichimoku_base'], df['Ichimoku_A'], df['Ichimoku_B'] = calculate_ichimoku(df)
    
    # Other indicators
    df['ROC'] = ((df['Close'] - df['Close'].shift(12)) / df['Close'].shift(12)) * 100
    df['DPO'] = df['Close'] - df['Close'].shift(21).rolling(window=21).mean()
    df['Williams_%R'] = (high_14 - df['Close']) / (high_14 - low_14) * -100
    df['McClellan_Oscillator'] = df['Close'].ewm(span=19, adjust=False).mean() - df['Close'].ewm(span=39, adjust=False).mean()
    
    advances = (df['Close'] > df['Open']).astype(int)
    declines = (df['Close'] < df['Open']).astype(int)
    df['TRIN'] = (advances.rolling(window=14).sum() / declines.rolling(window=14).sum()) / (df['Volume'].rolling(window=14).mean() / df['Volume'].rolling(window=14).mean())
    df['Price_to_Volume'] = df['Close'] / df['Volume']
    df['Trend_Line'] = df['Close'].rolling(window=30).mean()
    
    # Pivot Points calculation
    df['Pivot_Point'], df['Resistance_1'], df['Support_1'], df['Resistance_2'], df['Support_2'], df['Resistance_3'], df['Support_3'] = calculate_pivot_points(df)
    
    # Fibonacci Levels calculation
    df = calculate_fibonacci_levels(df)
    
    # Gann Levels calculation
    df = calculate_gann_levels(df)
    
    # Advance Decline Line calculation
    df['Advance_Decline_Line'] = advances.cumsum() - declines.cumsum()
    
    # William Arbitrage calculation
    df = calculate_williams_alligator(df)
    
    return df

def calculate_parabolic_sar(df):
    af = 0.02
    uptrend = True
    df['Parabolic_SAR'] = np.nan
    ep = df['Low'][0] if uptrend else df['High'][0]
    df['Parabolic_SAR'].iloc[0] = df['Close'][0]
    for i in range(1, len(df)):
        if uptrend:
            df['Parabolic_SAR'].iloc[i] = df['Parabolic_SAR'].iloc[i - 1] + af * (ep - df['Parabolic_SAR'].iloc[i - 1])
            if df['Low'].iloc[i] < df['Parabolic_SAR'].iloc[i]:
                uptrend = False
                df['Parabolic_SAR'].iloc[i] = ep
                af = 0.02
                ep = df['High'].iloc[i]
        else:
            df['Parabolic_SAR'].iloc[i] = df['Parabolic_SAR'].iloc[i - 1] + af * (ep - df['Parabolic_SAR'].iloc[i - 1])
            if df['High'].iloc[i] > df['Parabolic_SAR'].iloc[i]:
                uptrend = True
                df['Parabolic_SAR'].iloc[i] = ep
                af = 0.02
                ep = df['Low'].iloc[i]
    return df['Parabolic_SAR']

def calculate_adx(df):
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr = pd.concat([df['High'] - df['Low'], 
                    (df['High'] - df['Close'].shift()).abs(), 
                    (df['Low'] - df['Close'].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1/14).mean() / atr))
    adx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).ewm(alpha=1/14).mean()
    return adx

def calculate_ichimoku(df):
    high_9 = df['High'].rolling(window=9).max()
    low_9 = df['Low'].rolling(window=9).min()
    high_26 = df['High'].rolling(window=26).max()
    low_26 = df['Low'].rolling(window=26).min()
    high_52 = df['High'].rolling(window=52).max()
    low_52 = df['Low'].rolling(window=52).min()
    ichimoku_conv = (high_9 + low_9) / 2
    ichimoku_base = (high_26 + low_26) / 2
    ichimoku_a = ((ichimoku_conv + ichimoku_base) / 2).shift(26)
    ichimoku_b = ((high_52 + low_52) / 2).shift(26)
    return ichimoku_conv, ichimoku_base, ichimoku_a, ichimoku_b

def calculate_pivot_points(df):
    pivot = (df['High'] + df['Low'] + df['Close']) / 3
    resistance_1 = (2 * pivot) - df['Low']
    support_1 = (2 * pivot) - df['High']
    resistance_2 = pivot + (df['High'] - df['Low'])
    support_2 = pivot - (df['High'] - df['Low'])
    resistance_3 = df['High'] + 2 * (pivot - df['Low'])
    support_3 = df['Low'] - 2 * (df['High'] - pivot)
    return pivot, resistance_1, support_1, resistance_2, support_2, resistance_3, support_3

def calculate_fibonacci_levels(df):
    high = df['High'].max()
    low = df['Low'].min()
    diff = high - low
    df['Fib_0.0'] = high
    df['Fib_0.236'] = high - 0.236 * diff
    df['Fib_0.382'] = high - 0.382 * diff
    df['Fib_0.5'] = high - 0.5 * diff
    df['Fib_0.618'] = high - 0.618 * diff
    df['Fib_1.0'] = low
    return df

def calculate_gann_levels(df):
    high = df['High'].max()
    low = df['Low'].min()
    diff = high - low
    df['Gann_0.25'] = low + 0.25 * diff
    df['Gann_0.5'] = low + 0.5 * diff
    df['Gann_0.75'] = low + 0.75 * diff
    return df

# Streamlit UI
st.title("Stock Price Prediction with Lasso Regression Model")
ticker = st.sidebar.text_input("Enter Stock Symbol", value="CUMMINSIND.NS")
start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=1500))
end_date = st.sidebar.date_input("End Date", value=datetime.now() + timedelta(days=1))
forecast_period = st.sidebar.number_input("Forecast Period (days)", value=10, min_value=1, max_value=30)

# Fetch the data
stock_data = get_stock_data(ticker, start_date, end_date)

# Calculate technical indicators
stock_data = calculate_indicators(stock_data)

# Drop rows with NaN values
stock_data.dropna(inplace=True)

# Extract the closing prices and technical indicators
close_prices = stock_data['Close']
technical_indicators = stock_data.drop(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

# Normalize the technical indicators
scaler = StandardScaler()
technical_indicators_scaled = scaler.fit_transform(technical_indicators)

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

# Use Lasso regression for regularization
lasso = LassoCV(cv=TimeSeriesSplit(n_splits=5)).fit(technical_indicators_scaled, close_prices)

# Forecasting the next n business days (excluding weekends)
future_technical_indicators = technical_indicators_scaled[-forecast_period:]
forecast = lasso.predict(future_technical_indicators)

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
    title=f'Stock Price Forecast for {ticker}',
    xaxis_title='Date',
    yaxis_title='Close Price',
    legend=dict(x=0.01, y=0.99),
)

st.plotly_chart(fig)

# Display forecasted prices
st.write("Forecasted Prices for the next {} days:".format(forecast_period))
st.dataframe(forecasted_df)
