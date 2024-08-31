import streamlit as st
from components.my_portfolio import get_user_id
from utils.mongodb import watchlists_collection
import yfinance as yf
import pandas as pd
from utils.constants import bse_largecap, bse_smallcap, bse_midcap, sp500_tickers, ftse100_tickers

# Helper function to calculate RSI
def rsi(series, window=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Helper function to calculate ADX
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
    return adx, plus_di, minus_di

# Helper function to calculate Bollinger Bands
def calculate_bollinger_bands(df, window=20):
    df['BB_Middle'] = df['Close'].rolling(window=window).mean()
    df['BB_Std'] = df['Close'].rolling(window=window).std()
    df['Bollinger_High'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['Bollinger_Low'] = df['BB_Middle'] - (df['BB_Std'] * 2)
    return df['Bollinger_High'], df['Bollinger_Low']

# Helper function to calculate indicators
def calculate_indicators(data):
    try:
        data['5_day_EMA'] = data['Close'].ewm(span=5, adjust=False).mean()
        data['15_day_EMA'] = data['Close'].ewm(span=15, adjust=False).mean()
        
        # MACD calculations
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

        # Manual RSI calculation
        data['RSI'] = rsi(data['Close'])

        # Manual ADX calculation
        data['ADX'], data['+DI'], data['-DI'] = calculate_adx(data)

        # Manual Bollinger Bands calculation
        data['Bollinger_High'], data['Bollinger_Low'] = calculate_bollinger_bands(data)

        # Volume MA calculation
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

# Function to get company names from tickers
def get_company_name(ticker):
    try:
        company_info = yf.Ticker(ticker)
        return company_info.info['shortName']
    except:
        return ticker  # Return ticker if company name not found



# Generate the ticker to company mapping
ticker_to_company = {ticker: get_company_name(ticker) for ticker in bse_largecap + bse_midcap }

# Convert the ticker_to_company dictionary to a list of company names
company_names = list(ticker_to_company.values())


# Watchlist feature
def display_watchlist():
    st.header(f"{st.session_state.username}'s Watchlist")
    user_id = get_user_id(st.session_state.email)
    watchlist = list(watchlists_collection.find({"user_id": user_id}))

    # Replace text input with a selectbox for company name auto-suggestion
    selected_company = st.sidebar.selectbox('Select or Enter Company Name:', company_names)

    # Retrieve the corresponding ticker for the selected company
    ticker = [ticker for ticker, company in ticker_to_company.items() if company == selected_company][0]

    # Add new ticker to watchlist
    if st.sidebar.button("Add Ticker"):
        try:
            fetch_ticker_data(ticker)
            if not watchlists_collection.find_one({"user_id": user_id, "ticker": ticker}):
                watchlists_collection.insert_one({"user_id": user_id, "ticker": ticker})
                st.success(f"{ticker} ({selected_company}) added to your watchlist!")
            else:
                st.warning(f"{ticker} ({selected_company}) is already in your watchlist.")
        except ValueError as ve:
            st.error(ve)

    # Display watchlist
    if watchlist:
        watchlist_data = {}
        ticker_to_name_map = {}
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
                    'MACD_Signal': latest_data['MACD_Signal'],
                    'MACD_Hist': latest_data['MACD_Hist'],
                    'RSI': latest_data['RSI'],
                    'ADX': latest_data['ADX'],
                    'Bollinger_High': latest_data['Bollinger_High'],
                    'Bollinger_Low': latest_data['Bollinger_Low'],
                    'Volume': latest_data['Volume'],
                    '20_day_vol_MA': latest_data['20_day_vol_MA']
                }
                ticker_to_name_map[ticker] = company_name
            except ValueError as ve:
                st.error(f"Error fetching data for {ticker}: {ve}")

        if watchlist_data:
            watchlist_df = pd.DataFrame.from_dict(watchlist_data, orient='index')
            watchlist_df.reset_index(inplace=True)
            watchlist_df.rename(columns={'index': 'Ticker'}, inplace=True)

            # Use Styler to format the DataFrame
            styled_df = watchlist_df.style.format(precision=2)

            st.write("Your Watchlist:")
            st.dataframe(styled_df.set_properties(**{'text-align': 'center'}).set_table_styles(
                [{'selector': 'th', 'props': [('text-align', 'center')]}]
            ))

            # Option to remove ticker from watchlist
            company_names_in_watchlist = [ticker_to_name_map[entry['ticker']] for entry in watchlist]
            company_to_remove = st.sidebar.selectbox("Select a company to remove", company_names_in_watchlist)
            ticker_to_remove = [ticker for ticker, name in ticker_to_name_map.items() if name == company_to_remove][0]
            if st.sidebar.button("Remove Ticker"):
                watchlists_collection.delete_one({"user_id": user_id, "ticker": ticker_to_remove})
                st.success(f"{company_to_remove} removed from your watchlist.")
                st.experimental_rerun()  # Refresh the app to reflect changes
        else:
            st.write("No valid data found for the tickers in your watchlist.")
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
