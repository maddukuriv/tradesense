import streamlit as st
from components.my_portfolio import get_user_id
from utils.mongodb import watchlists_collection
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import ta

# Helper function to calculate indicators
def calculate_indicators(data):
    try:
        data['5_day_EMA'] = ta.trend.ema_indicator(data['Close'], window=5)
        data['15_day_EMA'] = ta.trend.ema_indicator(data['Close'], window=15)
        data['MACD'] = ta.trend.macd(data['Close'])
        data['MACD_Hist'] = ta.trend.macd_diff(data['Close'])
        data['RSI'] = ta.momentum.rsi(data['Close'])
        data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'])
        data['Bollinger_High'] = ta.volatility.bollinger_hband(data['Close'])
        data['Bollinger_Low'] = ta.volatility.bollinger_lband(data['Close'])
        data['20_day_vol_MA'] = data['Volume'].rolling(window=20).mean()
        return data
    except Exception as e:
        raise ValueError("Error calculating indicators") from e



# Helper function to fetch ticker data from yfinance
def fetch_ticker_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1y")
        if data.empty:
            raise ValueError("Ticker not found")
        return data
    except Exception as e:
        raise ValueError("Ticker not found") from e

# Watchlist feature
def display_watchlist():
    st.header(f"{st.session_state.username}'s Watchlist")
    user_id = get_user_id(st.session_state.email)
    watchlist = list(watchlists_collection.find({"user_id": user_id}))

    # Add new ticker to watchlist
    new_ticker = st.sidebar.text_input("Add a new ticker to your watchlist")
    if st.sidebar.button("Add Ticker"):
        try:
            fetch_ticker_data(new_ticker)
            if not watchlists_collection.find_one({"user_id": user_id, "ticker": new_ticker}):
                watchlists_collection.insert_one({"user_id": user_id, "ticker": new_ticker})
                st.success(f"{new_ticker} added to your watchlist!")
            else:
                st.warning(f"{new_ticker} is already in your watchlist.")
        except ValueError as ve:
            st.error(ve)

    # Display watchlist
    if watchlist:
        watchlist_data = {}
        for entry in watchlist:
            ticker = entry['ticker']
            try:
                data = fetch_ticker_data(ticker)
                data = calculate_indicators(data)
                latest_data = data.iloc[-1]
                watchlist_data[ticker] = {
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
                }
            except ValueError as ve:
                st.error(f"Error fetching data for {ticker}: {ve}")

        watchlist_df = pd.DataFrame.from_dict(watchlist_data, orient='index')
        st.write("Your Watchlist:")
        st.dataframe(watchlist_df)

        # Option to remove ticker from watchlist
        ticker_to_remove = st.sidebar.selectbox("Select a ticker to remove", [entry['ticker'] for entry in watchlist])
        if st.sidebar.button("Remove Ticker"):
            watchlists_collection.delete_one({"user_id": user_id, "ticker": ticker_to_remove})
            st.success(f"{ticker_to_remove} removed from your watchlist.")
            st.experimental_rerun()  # Refresh the app to reflect changes
    else:
        st.write("Your watchlist is empty.")
