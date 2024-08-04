import streamlit as st
from components.my_portfolio import get_user_id
from utils.mongodb import watchlists_collection
import yfinance as yf
import pandas as pd
import ta

# Helper function to calculate indicators
def calculate_indicators(data):
    try:
        data['5_day_EMA'] = ta.trend.ema_indicator(data['Close'], window=5)
        data['15_day_EMA'] = ta.trend.ema_indicator(data['Close'], window=15)
        macd = ta.trend.MACD(data['Close'])
        data['MACD'] = macd.macd()
        data['MACD_Hist'] = macd.macd_diff()
        data['RSI'] = ta.momentum.rsi(data['Close'])
        data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'])
        bollinger = ta.volatility.BollingerBands(data['Close'])
        data['Bollinger_High'] = bollinger.bollinger_hband()
        data['Bollinger_Low'] = bollinger.bollinger_lband()
        data['20_day_vol_MA'] = data['Volume'].rolling(window=20).mean()
        return data
    except Exception as e:
        raise ValueError(f"Error calculating indicators: {str(e)}")

# Helper function to fetch ticker data from yfinance
def fetch_ticker_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1y")
        if data.empty:
            raise ValueError(f"Ticker {ticker} not found")
        return data
    except Exception as e:
        raise ValueError(f"Error fetching data for ticker {ticker}: {str(e)}")

# Helper function to fetch company info from yfinance
def get_company_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info.get('longName', 'N/A'), info.get('sector', 'N/A'), info.get('industry', 'N/A')
    except Exception as e:
        return 'N/A', 'N/A', 'N/A'

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
                company_name, sector, industry = get_company_info(ticker)
                watchlist_data[ticker] = {
                    'Company Name': company_name,
                    'Sector': sector,
                    'Industry': industry,
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

        if watchlist_data:
            watchlist_df = pd.DataFrame.from_dict(watchlist_data, orient='index')
            watchlist_df.reset_index(inplace=True)
            watchlist_df.rename(columns={'index': 'Ticker'}, inplace=True)

            st.write("Your Watchlist:")
            st.dataframe(watchlist_df.style.set_properties(**{'text-align': 'center'}).set_table_styles(
                [{'selector': 'th', 'props': [('text-align', 'center')]}]
            ))
        else:
            st.write("No valid data found for the tickers in your watchlist.")

        # Option to remove ticker from watchlist
        ticker_to_remove = st.sidebar.selectbox("Select a ticker to remove", [entry['ticker'] for entry in watchlist])
        if st.sidebar.button("Remove Ticker"):
            watchlists_collection.delete_one({"user_id": user_id, "ticker": ticker_to_remove})
            st.success(f"{ticker_to_remove} removed from your watchlist.")
            st.experimental_rerun()  # Refresh the app to reflect changes
    else:
        st.write("Your watchlist is empty.")

# Call the function to display the watchlist
if 'username' not in st.session_state:
    st.session_state.username = 'Guest'  # or handle the case where username is not set
if 'email' not in st.session_state:
    st.session_state.email = 'guest@example.com'  # or handle the case where email is not set
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False  # or handle the case where logged_in is not set

if st.session_state.logged_in:
    display_watchlist()
else:
    st.write("Please log in to view your watchlist.")
