import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from utils.constants import bse_largecap, bse_smallcap, bse_midcap

def stock_screener_app():
    st.sidebar.subheader("Stock Screener")

    # Dropdown for selecting ticker category
    ticker_category = st.sidebar.selectbox("Select Index", ["BSE-LargeCap", "BSE-MidCap", "BSE-SmallCap"])

    # Dropdown for Strategies
    submenu = st.sidebar.selectbox("Select Strategy", ["MACD", "Moving Average", "Bollinger Bands", "Volume"])

    # Date inputs
    start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", value=datetime.now() + timedelta(days=1))

    # Set tickers based on selected category
    if ticker_category == "BSE-LargeCap":
        tickers = bse_largecap
    elif ticker_category == "BSE-MidCap":
        tickers = bse_midcap
    else:
        tickers = bse_smallcap

    # Define functions for strategy logic
    def calculate_macd(data):
        macd = ta.macd(data['Close'])
        data['MACD'] = macd['MACD_12_26_9']
        data['MACD_signal'] = macd['MACDs_12_26_9']
        data['MACD_histogram'] = macd['MACDh_12_26_9']
        return data

    def check_macd_signal(data):
        recent_data = data[-5:]
        for i in range(1, len(recent_data)):
            if (recent_data['MACD'].iloc[i] > recent_data['MACD_signal'].iloc[i] and
                recent_data['MACD'].iloc[i-1] < recent_data['MACD_signal'].iloc[i-1] and
                recent_data['MACD'].iloc[i] > 0 and
                recent_data['MACD_histogram'].iloc[i] > 0 and
                recent_data['MACD_histogram'].iloc[i-1] < 0 and
                recent_data['MACD_histogram'].iloc[i] > recent_data['MACD_histogram'].iloc[i-1] > recent_data['MACD_histogram'].iloc[i-2]):
                return recent_data.index[i]
        return None

    def check_bollinger_low_cross(data):
        recent_data = data[-5:]
        for i in range(1, len(recent_data)):
            if (recent_data['Close'].iloc[i] < recent_data['BB_Low'].iloc[i] and
                recent_data['Close'].iloc[i-1] >= recent_data['BB_Low'].iloc[i-1]):
                return recent_data.index[i]
        return None

    def calculate_ema(data):
        data['Short_EMA'] = ta.ema(data['Close'], length=10)
        data['Long_EMA'] = ta.ema(data['Close'], length=20)
        return data

    def check_moving_average_crossover(data):
        recent_data = data[-5:]
        for i in range(1, len(recent_data)):
            if (recent_data['Short_EMA'].iloc[i] > recent_data['Long_EMA'].iloc[i] and
                recent_data['Short_EMA'].iloc[i-1] <= recent_data['Long_EMA'].iloc[i-1]):
                return recent_data.index[i]
        return None

    def calculate_volume(data):
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        return data

    def check_volume_increase(data):
        recent_data = data[-5:]
        for i in range(1, len(recent_data)):
            if (recent_data['Volume'].iloc[i] > recent_data['Volume_MA'].iloc[i] and
                recent_data['Volume'].iloc[i-1] <= recent_data['Volume_MA'].iloc[i-1]):
                return recent_data.index[i]
        return None

    @st.cache_data
    def get_stock_data(ticker_symbols, start_date, end_date):
        try:
            stock_data = {}
            progress_bar = st.progress(0)
            for idx, ticker_symbol in enumerate(ticker_symbols):
                df = yf.download(ticker_symbol, start=start_date, end=end_date)
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
        df['20_MA'] = ta.sma(df['Close'], length=20)
        df['50_MA'] = ta.sma(df['Close'], length=50)

        macd = ta.macd(df['Close'])
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_Signal'] = macd['MACDs_12_26_9']
        df['MACD_Histogram'] = macd['MACDh_12_26_9']

        df['RSI'] = ta.rsi(df['Close'])

        df['Std_Dev'] = df['Close'].rolling(window=20).std()
        df['BB_High'] = df['20_MA'] + (df['Std_Dev'] * 2)
        df['BB_Low'] = df['20_MA'] - (df['Std_Dev'] * 2)

        return df

    def fetch_latest_data(tickers_with_dates):
        technical_data = []
        for ticker, occurrence_date in tickers_with_dates:
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                continue
            data['5_day_EMA'] = ta.ema(data['Close'], length=5)
            data['10_day_EMA'] = ta.ema(data['Close'], length=10)
            data['20_day_EMA'] = ta.ema(data['Close'], length=20)
            macd = ta.macd(data['Close'])
            data['MACD'] = macd['MACD_12_26_9']
            data['MACD_Hist'] = macd['MACDh_12_26_9']
            data['RSI'] = ta.rsi(data['Close'])
            adx = ta.adx(data['High'], data['Low'], data['Close'])
            data['ADX_14'] = adx['ADX_14']
            data['ADX_NEG'] = adx['DMP_14']
            data['ADX_POS'] = adx['DMN_14']
            data['20_MA'] = ta.sma(data['Close'], length=20)
            data['Std_Dev'] = data['Close'].rolling(window=20).std()
            data['BB_High'] = data['20_MA'] + (data['Std_Dev'] * 2)
            data['BB_Low'] = data['20_MA'] - (data['Std_Dev'] * 2)
            data['20_day_vol_MA'] = data['Volume'].rolling(window=20).mean()

            latest_data = data.iloc[-1]
            technical_data.append({
                'Ticker': ticker,
                'Date of Occurrence': occurrence_date,
                'Close': latest_data['Close'],
                '5_day_EMA': latest_data['5_day_EMA'],
                '10_day_EMA': latest_data['10_day_EMA'],
                '20_day_EMA': latest_data['20_day_EMA'],
                'MACD': latest_data['MACD'],
                'MACD_Hist': latest_data['MACD_Hist'],
                'RSI': latest_data['RSI'],
                'ADX_14': latest_data['ADX_14'],
                'ADX_NEG': latest_data['ADX_NEG'],
                'ADX_POS': latest_data['ADX_POS'],
                'Bollinger_High': latest_data['BB_High'],
                'Bollinger_Low': latest_data['BB_Low'],
                'Volume': latest_data['Volume'],
                '20_day_vol_MA': latest_data['20_day_vol_MA']
            })
        return pd.DataFrame(technical_data)

    macd_signal_list = []
    moving_average_tickers = []
    bollinger_low_cross_tickers = []
    volume_increase_tickers = []

    progress_bar = st.progress(0)
    progress_step = 1 / len(tickers)

    for i, ticker in enumerate(tickers):
        progress_bar.progress((i + 1) * progress_step)
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            continue
        data = calculate_indicators(data)
        if submenu == "MACD":
            data = calculate_macd(data)
            occurrence_date = check_macd_signal(data)
            if occurrence_date:
                macd_signal_list.append((ticker, occurrence_date))
        elif submenu == "Moving Average":
            data = calculate_ema(data)
            occurrence_date = check_moving_average_crossover(data)
            if occurrence_date:
                moving_average_tickers.append((ticker, occurrence_date))
        elif submenu == "Bollinger Bands":
            occurrence_date = check_bollinger_low_cross(data)
            if occurrence_date:
                bollinger_low_cross_tickers.append((ticker, occurrence_date))
        elif submenu == "Volume":
            data = calculate_volume(data)
            occurrence_date = check_volume_increase(data)
            if occurrence_date:
                volume_increase_tickers.append((ticker, occurrence_date))

    df_macd_signal = fetch_latest_data(macd_signal_list)
    df_moving_average_signal = fetch_latest_data(moving_average_tickers)
    df_bollinger_low_cross_signal = fetch_latest_data(bollinger_low_cross_tickers)
    df_volume_increase_signal = fetch_latest_data(volume_increase_tickers)

    def add_links(df):
        df['Ticker'] = df['Ticker'].apply(lambda x: f'<a target="_blank" href="/stock_analysis?ticker={x}">{x}</a>')
        return df

    st.title("Stock's Based on Selected Strategy")

    if submenu == "MACD":
        st.write("Stocks with MACD > MACD Signal and MACD > 0 in the last 5 days:")
        df_macd_signal = add_links(df_macd_signal)
        st.markdown(df_macd_signal.to_html(escape=False), unsafe_allow_html=True)

    elif submenu == "Moving Average":
        st.write("Stocks with 10-day EMA crossing above 20-day EMA in the last 5 days:")
        df_moving_average_signal = add_links(df_moving_average_signal)
        st.markdown(df_moving_average_signal.to_html(escape=False), unsafe_allow_html=True)

    elif submenu == "Bollinger Bands":
        st.write("Stocks with price crossing below Bollinger Low in the last 5 days:")
        df_bollinger_low_cross_signal = add_links(df_bollinger_low_cross_signal)
        st.markdown(df_bollinger_low_cross_signal.to_html(escape=False), unsafe_allow_html=True)

    elif submenu == "Volume":
        st.write("Stocks with volume above 20-day moving average in the last 5 days:")
        df_volume_increase_signal = add_links(df_volume_increase_signal)
        st.markdown(df_volume_increase_signal.to_html(escape=False), unsafe_allow_html=True)

