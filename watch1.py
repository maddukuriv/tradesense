import streamlit as st
import yfinance as yf
import ta
import pandas as pd

# Define ticker categories (example, these should be defined with actual tickers)
bse_largecap = ["TCS.NS", "RELIANCE.NS"]
bse_midcap = ["MIDCAP1.NS", "MIDCAP2.NS"]
bse_smallcap = ["SMALLCAP1.NS", "SMALLCAP2.NS"]

# Adding GLAXO.BO for testing
bse_largecap.append("GLAXO.BO")

st.sidebar.subheader("Strategies")

submenu = st.sidebar.selectbox("Select Strategy", ["MACD", "Moving Average", "Bollinger Bands"])

# Dropdown for selecting ticker category
ticker_category = st.sidebar.selectbox("Select Ticker Category", ["BSE-LargeCap", "BSE-MidCap", "BSE-SmallCap"])

# Set tickers based on selected category
if ticker_category == "BSE-LargeCap":
    tickers = bse_largecap
elif ticker_category == "BSE-MidCap":
    tickers = bse_midcap
else:
    tickers = bse_smallcap

# MACD Strategy logic
def calculate_macd(data, slow=26, fast=12, signal=9):
    data['EMA_fast'] = data['Close'].ewm(span=fast, min_periods=fast).mean()
    data['EMA_slow'] = data['Close'].ewm(span=slow, min_periods=slow).mean()
    data['MACD'] = data['EMA_fast'] - data['EMA_slow']
    data['MACD_signal'] = data['MACD'].ewm(span=signal, min_periods=signal).mean()
    data['MACD_histogram'] = data['MACD'] - data['MACD_signal']
    return data

def check_macd_signal(data):
    recent_data = data[-5:]
    for i in range(1, len(recent_data)):
        if (recent_data['MACD'].iloc[i] > recent_data['MACD_signal'].iloc[i] and
            recent_data['MACD'].iloc[i-1] < recent_data['MACD_signal'].iloc[i-1] and
            recent_data['MACD'].iloc[i] > 0):
            return True
    return False

@st.cache_data
def get_stock_data(ticker_symbols, period, interval):
    try:
        stock_data = {}
        progress_bar = st.progress(0)
        for idx, ticker_symbol in enumerate(ticker_symbols):
            df = yf.download(ticker_symbol, period=period, interval=interval)
            if not df.empty:
                df.interpolate(method='linear', inplace=True)
                df = calculate_indicators(df)
                df.dropna(inplace=True)
                stock_data[ticker_symbol] = df
            progress_bar.progress((idx + 1) / len(ticker_symbols))
        return stock_data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return {}

@st.cache_data
def calculate_indicators(df):
    df['20_MA'] = ta.trend.WMAIndicator(close=df['Close'], window=20).wma()
    df['50_MA'] = ta.trend.WMAIndicator(close=df['Close'], window=50).wma()
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Histogram'] = macd.macd_diff()
    return df

macd_signal_list = []
negative_histogram_tickers = []
moving_average_tickers = []  # Placeholder for moving average tickers
bollinger_low_cross_tickers = []  # Placeholder for Bollinger bands tickers

progress_bar = st.progress(0)
progress_step = 1 / len(tickers)

for i, ticker in enumerate(tickers):
    progress_bar.progress((i + 1) * progress_step)
    data = yf.download(ticker, period="1y", interval="1d")
    if data.empty:
        continue
    data = calculate_macd(data)
    if submenu == "MACD":
        if check_macd_signal(data):
            macd_signal_list.append(ticker)
        

def fetch_latest_data(tickers):
    technical_data = []
    for ticker in tickers:
        data = yf.download(ticker, period='1y')
        if data.empty:
            continue
        data['5_day_EMA'] = ta.trend.ema_indicator(data['Close'], window=5)
        data['15_day_EMA'] = ta.trend.ema_indicator(data['Close'], window=15)
        data['MACD'] = ta.trend.macd(data['Close'])
        data['MACD_Hist'] = ta.trend.macd_diff(data['Close'])
        data['RSI'] = ta.momentum.rsi(data['Close'])
        data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'])
        data['Bollinger_High'] = ta.volatility.bollinger_hband(data['Close'])
        data['Bollinger_Low'] = ta.volatility.bollinger_lband(data['Close'])
        data['20_day_vol_MA'] = data['Volume'].rolling(window=20).mean()

        latest_data = data.iloc[-1]
        technical_data.append({
            'Ticker': ticker,
            'Close': latest_data['Close'],
            '5_day_EMA': latest_data['5_day_EMA'],
            '15_day_EMA': latest_data['15_day_EMA'],
            'MACD': latest_data['MACD'],
            'MACD_Hist': latest_data['MACD_Hist'],
            'RSI': latest_data['RSI'],
            'ADX': latest_data['ADX'],
            'Bollinger_High': latest_data['Bollinger_High'],
            'Bollinger_Low': latest_data['Bollinger_Low'],
            'Volume': latest_data['Volume'],
            '20_day_vol_MA': latest_data['20_day_vol_MA']
        })
    return pd.DataFrame(technical_data)

# Generate dataframes for the selected strategies
df_macd_signal = fetch_latest_data(macd_signal_list)
df_negative_histogram = fetch_latest_data(negative_histogram_tickers)
df_moving_average_signal = fetch_latest_data(moving_average_tickers)
df_bollinger_low_cross_signal = fetch_latest_data(bollinger_low_cross_tickers)

st.title("Stock Analysis Based on Selected Strategy")

if submenu == "Moving Average":
    st.write("Stocks with 10-day EMA crossing above 20-day EMA in the last 5 days:")
    st.dataframe(df_moving_average_signal)

elif submenu == "MACD":
    st.write("Stocks with Negative MACD Histogram Increasing:")
    st.dataframe(df_negative_histogram)

elif submenu == "Bollinger Bands":
    st.write("Stocks with price crossing below Bollinger Low in the last 5 days:")
    st.dataframe(df_bollinger_low_cross_signal)
