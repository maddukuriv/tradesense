import streamlit as st
from datetime import date
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import ta
import pandas_ta as pta
from scipy.stats import linregress
from streamlit_option_menu import option_menu
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import hashlib
import streamlit_authenticator as stauth
import plotly.io as pio
from scipy.fftpack import fft, ifft
import pywt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from ta.trend import SMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set wide mode as default layout
st.set_page_config(layout="wide", page_title="e-Trade")

# In-memory user storage (for demo purposes)
if 'users' not in st.session_state:
    st.session_state.users = {}

# Initialize session state for login status
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'username' not in st.session_state:
    st.session_state.username = ""

# Function to handle user signup
def signup():
    st.subheader("Sign Up")
    username = st.text_input("Enter a new username", key='signup_username')
    password = st.text_input("Enter a new password", type="password", key='signup_password')
    if st.button("Sign Up"):
        if username in st.session_state.users:
            st.error("Username already exists. Try a different username.")
        else:
            st.session_state.users[username] = password
            st.success("User registered successfully!")

# Function to handle user login
def login():
    st.subheader("Login")
    username = st.text_input("Enter your username", key='login_username')
    password = st.text_input("Enter your password", type="password", key='login_password')
    if st.button("Login"):
        if st.session_state.users.get(username) == password:
            st.success("Login successful!")
            st.session_state.logged_in = True
            st.session_state.username = username
        else:
            st.error("Invalid username or password.")

# Function to handle user logout
def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""

# Main menu after login
def main_menu():
    st.subheader("Main Menu")
    menu_options = ["Markets", "Stock Screener", "Technical Analysis", "Stock Price Forecasting", "Stock Watch", "Strategy Backtesting", "Watchlist", "My Portfolio"]
    choice = st.selectbox("Select an option", menu_options)
    return choice

# Sidebar menu
with st.sidebar:
    st.title("e-Trade")
    if st.session_state.logged_in:
        st.write(f"Logged in as: {st.session_state.username}")
        if st.button("Logout"):
            logout()
            st.experimental_rerun()  # Refresh the app
        else:
            choice = main_menu()  # Display the main menu in the sidebar if logged in
    else:
        selected = st.selectbox("Choose an option", ["Login", "Sign Up"])
        if selected == "Login":
            login()
        elif selected == "Sign Up":
            signup()
        choice = None

# Main content area
if not st.session_state.logged_in:
    st.subheader("Please login or sign up to access the e-Trade platform.")

    # Function to download data and calculate moving averages
    def get_stock_data(ticker_symbol, start_date, end_date):
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
        data['MA_15'] = data['Close'].rolling(window=15).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        data.dropna(inplace=True)
        return data

    # Function to create Plotly figure
    def create_figure(data, indicators, title):
        fig = go.Figure()
        if 'Close' in indicators:
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))

        if 'MA_15' in indicators:
            fig.add_trace(go.Scatter(x=data.index, y=data['MA_15'], mode='lines', name='15-day MA'))

        if 'MA_50' in indicators:
            fig.add_trace(go.Scatter(x=data.index, y=data['MA_50'], mode='lines', name='50-day MA'))

        fig.update_layout(
            title=title, 
            xaxis_title='Date', 
            yaxis_title='Price',
            xaxis_rangeslider_visible=True,
            plot_bgcolor='dark grey',
            paper_bgcolor='white',
            font=dict(color='black'),
            hovermode='x',
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type='date'
            ),
            yaxis=dict(fixedrange=False),
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="Reset Zoom",
                            method="relayout",
                            args=[{"xaxis.range": [None, None], "yaxis.range": [None, None]}]
                        )
                    ]
                )
            ]
        )
        return fig

    # Create two columns
    col1, col2 = st.columns(2)

    # Set up the start and end date inputs
    with col1:
        START = st.date_input('Start Date', pd.to_datetime("2015-01-01"))
    with col2:
        END = st.date_input('End Date', pd.to_datetime("today"))

    data_bse = get_stock_data("^BSESN", START, END)
    indicators = st.multiselect("Select Indicators", ['Close', 'MA_15', 'MA_50'], default=['Close'])

    fig_bse = create_figure(data_bse, indicators, 'BSE Price')
    st.plotly_chart(fig_bse)

else:
    if choice:
        if choice == "Markets":
            st.subheader("Markets")
            with st.sidebar:
                submenu = st.radio("Select Option", ["Equities", "Commodities", "Currencies", "Cryptocurrencies"])

            # Function to download data and calculate moving averages
            def get_stock_data(ticker_symbol, start_date, end_date):
                data = yf.download(ticker_symbol, start=start_date, end=end_date)
                data['MA_15'] = data['Close'].rolling(window=15).mean()
                data['MA_50'] = data['Close'].rolling(window=50).mean()
                data.dropna(inplace=True)
                return data

            # Function to create Plotly figure
            def create_figure(data, indicators, title):
                fig = go.Figure()
                if 'Close' in indicators:
                    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))

                if 'MA_15' in indicators:
                    fig.add_trace(go.Scatter(x=data.index, y=data['MA_15'], mode='lines', name='15-day MA'))

                if 'MA_50' in indicators:
                    fig.add_trace(go.Scatter(x=data.index, y=data['MA_50'], mode='lines', name='50-day MA'))

                fig.update_layout(
                    title=title, 
                    xaxis_title='Date', 
                    yaxis_title='Price',
                    xaxis_rangeslider_visible=True,
                    plot_bgcolor='darkgrey',
                    paper_bgcolor='white',
                    font=dict(color='black'),
                    hovermode='x',
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="YTD", step="year", stepmode="todate"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(step="all")
                            ])
                        ),
                        rangeslider=dict(visible=True),
                        type='date'
                    ),
                    yaxis=dict(fixedrange=False),
                    updatemenus=[
                        dict(
                            type="buttons",
                            buttons=[
                                dict(
                                    label="Reset Zoom",
                                    method="relayout",
                                    args=[{"xaxis.range": [None, None], "yaxis.range": [None, None]}]
                                )
                            ]
                        )
                    ]
                )
                return fig

            # Create two columns
            col1, col2 = st.columns(2)

            # Set up the start and end date inputs
            with col1:
                START = st.date_input('Start Date', pd.to_datetime("2015-01-01"))
            with col2:
                END = st.date_input('End Date', pd.to_datetime("today"))

            if submenu == "Equities":
                st.header("Equity Markets")
                data_nyse = get_stock_data("^NYA", START, END)
                data_bse = get_stock_data("^BSESN", START, END)
                indicators = st.multiselect("Select Indicators", ['Close', 'MA_15', 'MA_50'], default=['Close'])
                fig_nyse = create_figure(data_nyse, indicators, 'NYSE Price')
                fig_bse = create_figure(data_bse, indicators, 'BSE Price')
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_nyse)
                    st.write("Insights (NYSE):")
                    if data_nyse['MA_15'].iloc[-1] < data_nyse['MA_50'].iloc[-1]:
                        st.markdown("* Market sentiment is **Bearish**")
                    elif data_nyse['MA_15'].iloc[-1] > data_nyse['MA_50'].iloc[-1]:
                        st.markdown("* Market sentiment is **Bullish**")
                with col2:
                    st.plotly_chart(fig_bse)
                    st.write("Insights (SENSEX):")
                    if data_bse['MA_15'].iloc[-1] < data_bse['MA_50'].iloc[-1]:
                        st.markdown("* Market sentiment is **Bearish**")
                    elif data_bse['MA_15'].iloc[-1] > data_bse['MA_50'].iloc[-1]:
                        st.markdown("* Market sentiment is **Bullish**")

            elif submenu == "Commodities":
                st.header("Commodities")
                tickers = ["GC=F", "CL=F", "NG=F", "SI=F", "HG=F"]
                selected_tickers = st.multiselect("Select stock tickers to visualize", tickers, default=["GC=F", "CL=F"])
                indicators = st.multiselect("Select Indicators", ['Close', 'MA_15', 'MA_50'], default=['Close'])
                if not selected_tickers:
                    st.warning("Please select at least one ticker.")
                else:
                    columns = st.columns(len(selected_tickers))
                    for ticker, col in zip(selected_tickers, columns):
                        data = get_stock_data(ticker, START, END)
                        fig = create_figure(data, indicators, f'{ticker} Price')
                        col.plotly_chart(fig)

            elif submenu == "Currencies":
                st.header("Currencies")
                tickers = ["EURUSD=X", "GBPUSD=X", "CNYUSD=X", "INRUSD=X"]
                selected_tickers = st.multiselect("Select currency pairs to visualize", tickers, default=["INRUSD=X", "CNYUSD=X"])
                indicators = st.multiselect("Select Indicators", ['Close', 'MA_15', 'MA_50'], default=['Close'])
                if not selected_tickers:
                    st.warning("Please select at least one currency pair.")
                else:
                    columns = st.columns(len(selected_tickers))
                    for ticker, col in zip(selected_tickers, columns):
                        data = get_stock_data(ticker, START, END)
                        fig = create_figure(data, indicators, f'{ticker} Price')
                        col.plotly_chart(fig)

            elif submenu == "Cryptocurrencies":
                st.header("Cryptocurrencies")
                tickers = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "DOGE-USD"]
                selected_tickers = st.multiselect("Select cryptocurrencies to visualize", tickers, default=["BTC-USD", "ETH-USD"])
                indicators = st.multiselect("Select Indicators", ['Close', 'MA_15', 'MA_50'], default=['Close'])
                if not selected_tickers:
                    st.warning("Please select at least one cryptocurrency.")
                else:
                    columns = st.columns(len(selected_tickers))
                    for ticker, col in zip(selected_tickers, columns):
                        data = get_stock_data(ticker, START, END)
                        fig = create_figure(data, indicators, f'{ticker} Price')
                        col.plotly_chart(fig)

        elif choice == "Stock Screener":
            st.sidebar.subheader("Screens")
            submenu = st.sidebar.radio("Select Option", ["LargeCap-1", "LargeCap-2", "LargeCap-3", "MidCap", "SmallCap"])

            # Define ticker symbols for different market caps
            largecap3_tickers = ["ITC.NS", "JBCHEPHARM.BO", "JWL.BO", "JYOTHYLAB.BO", "KPRMILL.NS", "KAJARIACER.NS", "KEI.BO", "LTTS.NS", "LTIM.NS", "MANKIND.NS", "MARICO.NS", "METROBRAND.BO", "MOTILALOFS.NS", "MPHASIS.NS", "MUTHOOTFIN.NS", "NH.NS", "NAVINFLUOR.NS", "NAM-INDIA.BO", "NMDC.NS", "OFSS.NS", "PGHH.NS", "PIIND.NS", "PAGEIND.NS", "PERSISTENT.NS", "PETRONET.NS", "PFIZER.NS", "PIDILITIND.NS", "POLYMED.NS", "POLYCAB.NS", "RRKABEL.NS", "RVNL.NS", "RATNAMANI.NS", "RITES.NS", "SANOFI.NS", "SCHAEFFLER.NS", "SKFINDIA.NS", "SOLARINDS.NS", "SONACOMS.NS", "SUMICHEM.NS", "SUNTV.NS", "SUNDRMFAST.NS", "SUPREMEIND.BO", "TATAELXSI.NS", "TATATECH.NS", "TCS.NS", "TECHM.NS", "TIMKEN.NS", "TITAN.NS", "TRITURBINE.NS", "TIINDIA.NS", "UNITDSPR.BO", "VGUARD.NS", "MANYAVAR.NS", "VINATIORGA.NS", "WIPRO.NS", "ZYDUSLIFE.NS"]
            largecap2_tickers = ["CUMMINSIND.NS", "CYIENT.NS", "DATAPATTNS.NS", "DEEPAKNTR.NS", "DIVISLAB.NS", "LALPATHLAB.NS", "RDY", "ELGIEQUIP.NS", "EMAMILTD.NS", "FIVESTAR.BO", "GRINFRA.NS", "GILLETTE.NS", "GLAXO.NS", "GODFRYPHLP.NS", "GRINDWELL.NS", "HAVELLS.NS", "HCLTECH.NS", "HAL.BO", "HONAUT.BO", "IRCTC.NS", "ISEC.BO", "INFY.NS", "IPCALAB.BO"]
            largecap1_tickers = ["ABBOTINDIA.NS", "ADANIPOWER.NS", "AFFLE.BO", "AIAENG.BO", "AJANTPHARM.BO", "APLLTD.BO", "ALKEM.BO", "ARE&M.NS", "ANANDRATHI.BO", "APARINDS.BO", "ASIANPAINT.NS", "ASTRAL.NS", "ASTRAZEN.NS", "BAJFINANCE.NS", "BASF.NS", "BAYERCROP.BO", "BERGEPAINT.BO", "BDL.NS", "BEL.NS", "BSOFT.BO", "CDSL.NS", "CAMS.NS", "CARBORUNIV.BO", "CASTROLIND.NS", "CHAMBLFERT.BO", "COALINDIA.NS", "COFORGE.BO", "COLPAL.NS", "CONCORDBIO.BO", "COROMANDEL.BO", "CREDITACC.BO"]
            smallcap_tickers = ["TAPARIA.BO", "LKPFIN.BO", "EQUITAS.NS"]
            midcap_tickers = ["PNCINFRA.NS", "INDIASHLTR.NS", "RAYMOND.NS", "KAMAHOLD.BO", "BENGALASM.BO", "CHOICEIN.NS", "GRAVITA.NS", "HGINFRA.NS", "JKPAPER.NS", "MTARTECH.NS", "HAPPSTMNDS.NS", "SARDAEN.NS", "WELENT.NS", "LTFOODS.NS", "GESHIP.NS", "SHRIPISTON.NS", "SHAREINDIA.NS", "CYIENTDLM.NS", "VTL.NS", "EASEMYTRIP.NS", "LLOYDSME.NS", "ROUTE.NS", "VAIBHAVGBL.NS", "GOKEX.NS", "USHAMART.NS", "EIDPARRY.NS", "KIRLOSBROS.NS", "MANINFRA.NS", "CMSINFO.NS", "RALLIS.NS", "GHCL.NS", "NEULANDLAB.NS", "SPLPETRO.NS", "MARKSANS.NS", "NAVINFLUOR.NS", "ELECON.NS", "TANLA.NS", "KFINTECH.NS", "TIPSINDLTD.NS", "ACI.NS", "SURYAROSNI.NS", "GPIL.NS", "GMDCLTD.NS", "MAHSEAMLES.NS", "TDPOWERSYS.NS", "TECHNOE.NS", "JLHL.NS"]

            # Function to fetch and process stock data
            def get_stock_data(ticker_symbols, start_date, end_date):
                try:
                    stock_data = {}
                    for ticker_symbol in ticker_symbols:
                        df = yf.download(ticker_symbol, start=start_date, end=end_date)
                        print(f"Downloaded data for {ticker_symbol}: Shape = {df.shape}")
                        df.interpolate(method='linear', inplace=True)
                        df = calculate_indicators(df)
                        df.dropna(inplace=True)
                        print(f"Processed data for {ticker_symbol}: Shape = {df.shape}")
                        stock_data[ticker_symbol] = df
                    combined_df = pd.concat(stock_data.values(), axis=1)
                    combined_df.columns = ['_'.join([ticker, col]).strip() for ticker, df in stock_data.items() for col in df.columns]
                    return combined_df
                except Exception as e:
                    print(f"Error fetching data: {e}")
                    return pd.DataFrame()

            # Function to calculate technical indicators
            def calculate_indicators(df):
                # Calculate Moving Averages
                df['5_MA'] = df['Close'].rolling(window=5).mean()
                df['20_MA'] = df['Close'].rolling(window=20).mean()
                df['50_MA'] = df['Close'].rolling(window=50).mean()

                # Calculate MACD
                df['MACD'] = ta.trend.MACD(df['Close']).macd()
                df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()
                df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

                # Calculate ADX
                df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()

                # Calculate Parabolic SAR
                psar = pta.psar(df['High'], df['Low'], df['Close'])
                df['Parabolic_SAR'] = psar['PSARl_0.02_0.2']

                # Calculate RSI
                df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()

                # Calculate Volume Moving Average (20 days)
                df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
                df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()

                # Calculate Bollinger Bands
                bollinger = ta.volatility.BollingerBands(df['Close'])
                df['Bollinger_High'] = bollinger.bollinger_hband()
                df['Bollinger_Low'] = bollinger.bollinger_lband()
                df['Bollinger_Middle'] = bollinger.bollinger_mavg()

                return df

            # Function to query the stocks
            def query_stocks(df, conditions, tickers):
                results = []
                for ticker in tickers:
                    condition_met = True
                    for condition in conditions:
                        col1, op, col2 = condition
                        col1 = f"{ticker}_{col1}"
                        col2 = f"{ticker}_{col2}"
                        if col1 not in df.columns or col2 not in df.columns:
                            condition_met = False
                            break
                        if op == '>':
                            if not (df[col1] > df[col2]).iloc[-1]:
                                condition_met = False
                                break
                        elif op == '<':
                            if not (df[col1] < df[col2]).iloc[-1]:
                                condition_met = False
                                break
                        elif op == '>=':
                            if not (df[col1] >= df[col2]).iloc[-1]:
                                condition_met = False
                                break
                        elif op == '<=':
                            if not (df[col1] <= df[col2]).iloc[-1]:
                                condition_met = False
                                break
                    if condition_met:
                        row = {
                            'Ticker': ticker,
                            'MACD': df[f"{ticker}_MACD"].iloc[-1],
                            'MACD_Signal': df[f"{ticker}_MACD_Signal"].iloc[-1],
                            'RSI': df[f"{ticker}_RSI"].iloc[-1],
                            'ADX': df[f"{ticker}_ADX"].iloc[-1],
                            'Close': df[f"{ticker}_Close"].iloc[-1],
                            '5_MA': df[f"{ticker}_5_MA"].iloc[-1],
                            '20_MA': df[f"{ticker}_20_MA"].iloc[-1],
                            'Bollinger_High': df[f"{ticker}_Bollinger_High"].iloc[-1],
                            'Bollinger_Low': df[f"{ticker}_Bollinger_Low"].iloc[-1],
                            'Bollinger_Middle': df[f"{ticker}_Bollinger_Middle"].iloc[-1],
                            'Parabolic_SAR': df[f"{ticker}_Parabolic_SAR"].iloc[-1],
                            'Volume': df[f"{ticker}_Volume"].iloc[-1],
                            'Volume_MA_20': df[f"{ticker}_Volume_MA_20"].iloc[-1]
                        }
                        results.append(row)
                return pd.DataFrame(results)

            # Create two columns
            col1, col2 = st.columns(2)

            # Set up the start and end date inputs
            with col1:
                START = st.date_input('Start Date', pd.to_datetime("2015-01-01"))

            with col2:
                END = st.date_input('End Date', pd.to_datetime("today"))

            if submenu == "LargeCap-1":
                st.header("LargeCap-1")
                tickers = largecap1_tickers

            if submenu == "LargeCap-2":
                st.header("LargeCap-2")
                tickers = largecap2_tickers

            if submenu == "LargeCap-3":
                st.header("LargeCap-3")
                tickers = largecap3_tickers

            if submenu == "MidCap":
                st.header("MidCap")
                tickers = midcap_tickers

            if submenu == "SmallCap":
                st.header("SmallCap")
                tickers = smallcap_tickers

            # Fetch data and calculate indicators for each stock
            stock_data = get_stock_data(tickers, START, END)

            # Define first set of conditions
            first_conditions = [
                ('Volume', '>', 'Volume_MA_20'),
                ('MACD', '>', 'MACD_Signal')
            ]

            # Query stocks based on the first set of conditions
            first_query_df = query_stocks(stock_data, first_conditions, tickers)

            # Display the final results
            second_query_df = first_query_df[(first_query_df['RSI'] < 75) & (first_query_df['RSI'] > 55) & (first_query_df['ADX'] > 20)]
            st.write("Stocks in an uptrend with high volume:")
            st.dataframe(second_query_df)

            # Dropdown for stock selection
            st.subheader("Signal:")
            selected_stock = st.selectbox("Select Stock", second_query_df['Ticker'].tolist())

            # If a stock is selected, plot its data with the selected indicators
            if selected_stock:
                def load_data(ticker, start, end):
                    data = yf.download(ticker, start=start, end=end)
                    dates = data.index
                    prices = data['Close'].values

                    # Fourier Transform
                    def apply_fft(prices):
                        N = len(prices)
                        T = 1.0  # Assuming daily data, T=1 day
                        yf = fft(prices)
                        xf = np.fft.fftfreq(N, T)[:N // 2]
                        return xf, 2.0 / N * np.abs(yf[0:N // 2])

                    # Apply FFT
                    frequencies, magnitudes = apply_fft(prices)

                    # Inverse FFT for noise reduction (optional)
                    def inverse_fft(yf, threshold=0.1):
                        yf_filtered = yf.copy()
                        yf_filtered[np.abs(yf_filtered) < threshold] = 0
                        return ifft(yf_filtered)

                    yf_transformed = fft(prices)
                    filtered_signal = inverse_fft(yf_transformed)

                    # Wavelet Transform
                    def apply_wavelet(prices, wavelet='db4', level=4):
                        coeffs = pywt.wavedec(prices, wavelet, level=level)
                        return coeffs

                    # Apply Wavelet Transform
                    coeffs = apply_wavelet(prices)
                    reconstructed_signal = pywt.waverec(coeffs, 'db4')

                    # Buy/Sell Signal Generation (Example Strategy)
                    def generate_signals(prices, reconstructed_signal):
                        buy_signals = []
                        sell_signals = []
                        for i in range(1, len(prices)):
                            if reconstructed_signal[i] > reconstructed_signal[i - 1] and reconstructed_signal[i - 1] < reconstructed_signal[i - 2]:
                                buy_signals.append((i, prices[i]))
                            elif reconstructed_signal[i] < reconstructed_signal[i - 1] and reconstructed_signal[i - 1] > reconstructed_signal[i - 2]:
                                sell_signals.append((i, prices[i]))
                        return buy_signals, sell_signals

                    # Generate Signals
                    buy_signals, sell_signals = generate_signals(prices, reconstructed_signal)

                    # Plot signals using plotly with a time bar
                    signals_fig = go.Figure()

                    # Prices plot
                    signals_fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines', name='Prices'))

                    # Buy signals
                    signals_fig.add_trace(go.Scatter(x=[dates[sig[0]] for sig in buy_signals], y=[sig[1] for sig in buy_signals], mode='markers', marker=dict(color='green', size=10), name='Buy Signal'))

                    # Sell signals
                    signals_fig.add_trace(go.Scatter(x=[dates[sig[0]] for sig in sell_signals], y=[sig[1] for sig in sell_signals], mode='markers', marker=dict(color='red', size=10), name='Sell Signal'))

                    # Update layout with a time bar
                    signals_fig.update_layout(
                        title='Buy/Sell Signals',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        xaxis=dict(
                            rangeselector=dict(
                                buttons=list([
                                    dict(count=1, label='1m', step='month', stepmode='backward'),
                                    dict(count=3, label='3m', step='month', stepmode='backward'),
                                    dict(count=6, label='6m', step='month', stepmode='backward'),
                                    dict(step='all')
                                ])
                            ),
                            rangeslider=dict(visible=True),
                            type='date'
                        )
                    )

                    # Show the plot
                    st.plotly_chart(signals_fig)

                # Load data and plot
                load_data(selected_stock, START, END)

        elif choice == "Technical Analysis":
            st.sidebar.subheader("Interactive Charts")
            submenu = st.sidebar.radio("Select Option", ["Trend Analysis", "Volume Analysis", "Support & Resistance Levels"])

            # Create three columns
            col1, col2, col3 = st.columns(3)

            # Set up the start and end date inputs
            with col1:
                # List of stock symbols
                stock_symbols = ["ABBOTINDIA.NS", "ADANIPOWER.NS", "AFFLE.BO", "AIAENG.BO", "AJANTPHARM.BO", "APLLTD.BO", "ALKEM.BO", "ARE&M.NS", "ANANDRATHI.BO", "APARINDS.BO", "ASIANPAINT.NS", "ASTRAL.NS", "ASTRAZEN.NS", "BAJFINANCE.NS", "BASF.NS", "BAYERCROP.BO", "BERGEPAINT.BO", "BDL.NS", "BEL.NS", "BSOFT.BO", "CDSL.NS", "CAMS.NS", "CARBORUNIV.BO", "CASTROLIND.NS", "CHAMBLFERT.BO", "COALINDIA.NS", "COFORGE.BO", "COLPAL.NS", "CONCORDBIO.BO", "COROMANDEL.BO", "CREDITACC.BO", "CUMMINSIND.NS", "CYIENT.NS", "DATAPATTNS.NS", "DEEPAKNTR.NS", "DIVISLAB.NS", "LALPATHLAB.NS", "RDY", "ELGIEQUIP.NS", "EMAMILTD.NS", "FIVESTAR.BO", "GRINFRA.NS", "GILLETTE.NS", "GLAXO.NS", "GODFRYPHLP.NS", "GRINDWELL.NS", "HAVELLS.NS", "HCLTECH.NS", "HAL.BO", "HONAUT.BO", "IRCTC.NS", "ISEC.BO", "INFY.NS", "IPCALAB.BO", "ITC.NS", "JBCHEPHARM.BO", "JWL.BO", "JYOTHYLAB.BO", "KPRMILL.NS", "KAJARIACER.NS", "KEI.BO", "LTTS.NS", "LTIM.NS", "MANKIND.NS", "MARICO.NS", "METROBRAND.BO", "MOTILALOFS.NS", "MPHASIS.NS", "MUTHOOTFIN.NS", "NH.NS", "NAVINFLUOR.NS", "NAM-INDIA.BO", "NMDC.NS", "OFSS.NS", "PGHH.NS", "PIIND.NS", "PAGEIND.NS", "PERSISTENT.NS", "PETRONET.NS", "PFIZER.NS", "PIDILITIND.NS", "POLYMED.NS", "POLYCAB.NS", "RRKABEL.NS", "RVNL.NS", "RATNAMANI.NS", "RITES.NS", "SANOFI.NS", "SCHAEFFLER.NS", "SKFINDIA.NS", "SOLARINDS.NS", "SONACOMS.NS", "SUMICHEM.NS", "SUNTV.NS", "SUNDRMFAST.NS", "SUPREMEIND.BO", "TATAELXSI.NS", "TATATECH.NS", "TCS.NS", "TECHM.NS", "TIMKEN.NS", "TITAN.NS", "TRITURBINE.NS", "TIINDIA.NS", "UNITDSPR.BO", "VGUARD.NS", "MANYAVAR.NS", "VINATIORGA.NS", "WIPRO.NS", "ZYDUSLIFE.NS"]
                ticker = st.selectbox("Enter Stock symbol", stock_symbols)
                st.write(f"You selected: {ticker}")
            with col2:
                START = st.date_input('Start Date', pd.to_datetime("2015-01-01"))
            with col3:
                END = st.date_input('End Date', pd.to_datetime("today"))

            # Load data
            def load_data(ticker):
                df = yf.download(ticker, START, END)
                df.reset_index(inplace=True)
                return df

            # Handle null values
            def interpolate_dataframe(df):
                if df.isnull().values.any():
                    df = df.interpolate()
                return df

            # Load data for the given ticker
            df = load_data(ticker)

            if df.empty:
                st.write("No data available for the provided ticker.")
            else:
                df = interpolate_dataframe(df)

                # Ensure enough data points for the calculations
                if len(df) > 200:  # 200 is the maximum window size used in calculations
                    # Calculate Moving Averages
                    df['15_MA'] = df['Close'].rolling(window=15).mean()
                    df['20_MA'] = df['Close'].rolling(window=20).mean()
                    df['50_MA'] = df['Close'].rolling(window=50).mean()
                    df['200_MA'] = df['Close'].rolling(window=200).mean()

                    # Calculate MACD
                    df['MACD'] = ta.trend.MACD(df['Close']).macd()
                    df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()
                    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

                    # Calculate ADX
                    df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()

                    # Calculate Parabolic SAR using 'pandas_ta' library
                    psar = pta.psar(df['High'], df['Low'], df['Close'])
                    df['Parabolic_SAR'] = psar['PSARl_0.02_0.2']

                    # Calculate RSI
                    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()

                    # Calculate Volume Moving Average (20 days)
                    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()

                    # Calculate On-Balance Volume (OBV)
                    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()

                    # Calculate Volume Oscillator (20-day EMA and 50-day EMA)
                    df['Volume_EMA_20'] = ta.trend.EMAIndicator(df['Volume'], window=20).ema_indicator()
                    df['Volume_EMA_50'] = ta.trend.EMAIndicator(df['Volume'], window=50).ema_indicator()
                    df['Volume_Oscillator'] = df['Volume_EMA_20'] - df['Volume_EMA_50']

                    # Calculate Chaikin Money Flow (20 days)
                    df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume']).chaikin_money_flow()

                    # Identify Horizontal Support and Resistance
                    def find_support_resistance(df, window=20):
                        df['Support'] = df['Low'].rolling(window, center=True).min()
                        df['Resistance'] = df['High'].rolling(window, center=True).max()
                        return df

                    df = find_support_resistance(df)

                    # Draw Trendlines
                    def calculate_trendline(df, kind='support'):
                        if kind == 'support':
                            prices = df['Low']
                        elif kind == 'resistance':
                            prices = df['High']
                        else:
                            raise ValueError("kind must be either 'support' or 'resistance'")

                        indices = np.arange(len(prices))
                        slope, intercept, _, _, _ = linregress(indices, prices)
                        trendline = slope * indices + intercept
                        return trendline

                    df['Support_Trendline'] = calculate_trendline(df, kind='support')
                    df['Resistance_Trendline'] = calculate_trendline(df, kind='resistance')

                    # Calculate Fibonacci Retracement Levels
                    def fibonacci_retracement_levels(high, low):
                        diff = high - low
                        levels = {
                            'Level_0': high,
                            'Level_0.236': high - 0.236 * diff,
                            'Level_0.382': high - 0.382 * diff,
                            'Level_0.5': high - 0.5 * diff,
                            'Level_0.618': high - 0.618 * diff,
                            'Level_1': low
                        }
                        return levels

                    recent_high = df['High'].max()
                    recent_low = df['Low'].min()
                    fib_levels = fibonacci_retracement_levels(recent_high, recent_low)

                    # Calculate Pivot Points
                    def pivot_points(df):
                        df['Pivot'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
                        df['R1'] = 2 * df['Pivot'] - df['Low'].shift(1)
                        df['S1'] = 2 * df['Pivot'] - df['High'].shift(1)
                        df['R2'] = df['Pivot'] + (df['High'].shift(1) - df['Low'].shift(1))
                        df['S2'] = df['Pivot'] - (df['High'].shift(1) - df['Low'].shift(1))
                        return df

                    df = pivot_points(df)

                    # Calculate Bollinger Bands
                    bollinger = ta.volatility.BollingerBands(df['Close'])
                    df['Bollinger_High'] = bollinger.bollinger_hband()
                    df['Bollinger_Low'] = bollinger.bollinger_lband()
                    df['Bollinger_Middle'] = bollinger.bollinger_mavg()  # Middle band is typically the SMA

                    # Generate buy/sell signals
                    df['Buy_Signal'] = (df['MACD'] > df['MACD_Signal']) & (df['ADX'] > 20)
                    df['Sell_Signal'] = (df['Close'] < df['15_MA'])

                    # Create a new column 'Signal' based on 'Buy_Signal' and 'Sell_Signal' conditions
                    def generate_signal(row):
                        if row['Buy_Signal']:
                            return 'Buy'
                        elif row['Sell_Signal']:
                            return 'Sell'
                        else:
                            return 'Hold'

                    df['Signal'] = df.apply(generate_signal, axis=1)

                    if submenu == "Trend Analysis":
                        st.header("Trend Analysis")

                        indicators = st.multiselect(
                            "Select Indicators",
                            ['Close', '20_MA', '50_MA', '200_MA', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'RSI', 'Signal', 'ADX', 'Parabolic_SAR', 'Bollinger_High', 'Bollinger_Low', 'Bollinger_Middle'],
                            default=['Close', 'Signal']
                        )
                        timeframe = st.radio(
                            "Select Timeframe",
                            ['15 days', '30 days', '90 days', '180 days', '1 year', 'All'],
                            index=4,
                            horizontal=True
                        )

                        if timeframe == '15 days':
                            df = df[-15:]
                        elif timeframe == '30 days':
                            df = df[-30:]
                        elif timeframe == '90 days':
                            df = df[-90:]
                        elif timeframe == '180 days':
                            df = df[-180:]
                        elif timeframe == '1 year':
                            df = df[-365:]

                        fig = go.Figure()
                        colors = {'Close': 'blue', '20_MA': 'orange', '50_MA': 'green', '200_MA': 'red', 'MACD': 'purple', 'MACD_Signal': 'brown', 'RSI': 'pink', 'Signal': 'black', 'ADX': 'magenta', 'Parabolic_SAR': 'yellow', 'Bollinger_High': 'black', 'Bollinger_Low': 'cyan', 'Bollinger_Middle': 'grey'}

                        for indicator in indicators:
                            if indicator == 'Signal':
                                # Plot buy and sell signals
                                buy_signals = df[df['Signal'] == 'Buy']
                                sell_signals = df[df['Signal'] == 'Sell']
                                fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Close'], mode='markers', name='Buy Signal', marker=dict(color='green', symbol='triangle-up')))
                                fig.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['Close'], mode='markers', name='Sell Signal', marker=dict(color='red', symbol='triangle-down')))
                            elif indicator == 'MACD_Histogram':
                                fig.add_trace(go.Bar(x=df['Date'], y=df[indicator], name=indicator, marker_color='gray'))
                            else:
                                fig.add_trace(go.Scatter(x=df['Date'], y=df[indicator], mode='lines', name=indicator, line=dict(color=colors.get(indicator, 'black'))))

                        st.plotly_chart(fig)

                        # Generate insights
                        st.subheader("Insights:")
                        if 'MACD' in indicators and 'MACD_Signal' in indicators:
                            last_macd = df['MACD'].iloc[-1]
                            last_macd_signal = df['MACD_Signal'].iloc[-1]
                            if last_macd > last_macd_signal:
                                st.markdown("MACD is above the MACD Signal line - Bullish Signal")
                            else:
                                st.markdown("MACD is below the MACD Signal line - Bearish Signal")

                        if 'RSI' in indicators:
                            last_rsi = df['RSI'].iloc[-1]
                            if last_rsi > 70:
                                st.markdown("RSI is above 70 - Overbought")
                            elif last_rsi < 30:
                                st.markdown("RSI is below 30 - Oversold")

                        if 'ADX' in indicators:
                            last_adx = df['ADX'].iloc[-1]
                            if last_adx > 20:
                                st.markdown("ADX is above 20 - Strong Trend")
                            else:
                                st.markdown("ADX is below 20 - Weak Trend")

                        if 'Parabolic_SAR' in indicators:
                            last_close = df['Close'].iloc[-1]
                            last_psar = df['Parabolic_SAR'].iloc[-1]
                            if last_close > last_psar:
                                st.markdown("Price is above Parabolic SAR - Bullish Signal")
                            else:
                                st.markdown("Price is below Parabolic SAR - Bearish Signal")

                        if 'Bollinger_High' in indicators and 'Bollinger_Low' in indicators:
                            last_close = df['Close'].iloc[-1]
                            last_boll_high = df['Bollinger_High'].iloc[-1]
                            last_boll_low = df['Bollinger_Low'].iloc[-1]
                            if last_close > last_boll_high:
                                st.markdown("Price is above the upper Bollinger Band - Potentially Overbought")
                            elif last_close < last_boll_low:
                                st.markdown("Price is below the lower Bollinger Band - Potentially Oversold")

                    elif submenu == "Volume Analysis":
                        st.header("Volume Analysis")
                        volume_indicators = st.multiselect(
                            "Select Volume Indicators",
                            ['Volume', 'Volume_MA_20', 'OBV', 'Volume_Oscillator', 'CMF'],
                            default=['Volume']
                        )
                        volume_timeframe = st.radio(
                            "Select Timeframe",
                            ['15 days', '30 days', '90 days', '180 days', '1 year', 'All'],
                            index=4,
                            horizontal=True
                        )

                        if volume_timeframe == '15 days':
                            df = df[-15:]
                        elif volume_timeframe == '30 days':
                            df = df[-30:]
                        elif volume_timeframe == '90 days':
                            df = df[-90:]
                        elif volume_timeframe == '180 days':
                            df = df[-180:]
                        elif volume_timeframe == '1 year':
                            df = df[-365:]

                        fig = go.Figure()
                        for indicator in volume_indicators:
                            fig.add_trace(go.Scatter(x=df['Date'], y=df[indicator], mode='lines', name=indicator))

                        st.plotly_chart(fig)

                        # Generate insights
                        st.subheader("Insights:")
                        if 'Volume' in volume_indicators and 'Volume_MA_20' in volume_indicators:
                            last_volume = df['Volume'].iloc[-1]
                            last_volume_ma_20 = df['Volume_MA_20'].iloc[-1]
                            if last_volume > last_volume_ma_20:
                                st.markdown("Current volume is above the 20-day average - Increased buying/selling interest")
                            else:
                                st.markdown("Current volume is below the 20-day average - Decreased buying/selling interest")

                        if 'OBV' in volume_indicators:
                            last_obv = df['OBV'].iloc[-1]
                            if last_obv > df['OBV'].iloc[-2]:
                                st.markdown("OBV is increasing - Accumulation phase (buying pressure)")
                            else:
                                st.markdown("OBV is decreasing - Distribution phase (selling pressure)")

                        if 'CMF' in volume_indicators:
                            last_cmf = df['CMF'].iloc[-1]
                            if last_cmf > 0:
                                st.markdown("CMF is positive - Buying pressure")
                            else:
                                st.markdown("CMF is negative - Selling pressure")

                    elif submenu == "Support & Resistance Levels":
                        st.header("Support & Resistance Levels")
                        sr_indicators = st.multiselect(
                            "Select Indicators",
                            ['Close', '20_MA', '50_MA', '200_MA', 'Support', 'Resistance', 'Support_Trendline', 'Resistance_Trendline', 'Pivot', 'R1', 'S1', 'R2', 'S2'],
                            default=['Close']
                        )
                        sr_timeframe = st.radio(
                            "Select Timeframe",
                            ['15 days', '30 days', '90 days', '180 days', '1 year', 'All'],
                            index=4,
                            horizontal=True
                        )

                        if sr_timeframe == '15 days':
                            df = df[-15:]
                        elif sr_timeframe == '30 days':
                            df = df[-30:]
                        elif sr_timeframe == '90 days':
                            df = df[-90:]
                        elif sr_timeframe == '180 days':
                            df = df[-180:]
                        elif sr_timeframe == '1 year':
                            df = df[-365:]

                        fig = go.Figure()
                        for indicator in sr_indicators:
                            fig.add_trace(go.Scatter(x=df['Date'], y=df[indicator], mode='lines', name=indicator))

                        st.plotly_chart(fig)

                        # Generate insights
                        st.subheader("Insights:")
                        if 'Support' in sr_indicators and 'Resistance' in sr_indicators:
                            last_close = df['Close'].iloc[-1]
                            last_support = df['Support'].iloc[-1]
                            last_resistance = df['Resistance'].iloc[-1]
                            if last_close > last_resistance:
                                st.markdown("Price is above the resistance level - Potential breakout")
                            elif last_close < last_support:
                                st.markdown("Price is below the support level - Potential breakdown")

                        if 'Pivot' in sr_indicators:
                            last_close = df['Close'].iloc[-1]
                            last_pivot = df['Pivot'].iloc[-1]
                            if last_close > last_pivot:
                                st.markdown("Price is above the pivot point - Bullish sentiment")
                            else:
                                st.markdown("Price is below the pivot point - Bearish sentiment")

                else:
                    st.write("Not enough data points for technical analysis.")

        elif choice == "Stock Price Forecasting":
            # Step 2: Search box for stock ticker
            ticker = st.text_input('Enter Stock Ticker', 'AAPL')

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
                data = yf.download(ticker, START, END)
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

            # Calculate technical indicators
            data['SMA_5'] = SMAIndicator(close=data['Close'], window=5).sma_indicator()
            data['SMA_10'] = SMAIndicator(close=data['Close'], window=10).sma_indicator()
            data['SMA_20'] = SMAIndicator(close=data['Close'], window=20).sma_indicator()

            macd = MACD(close=data['Close'])
            data['MACD'] = macd.macd()
            data['MACD_Signal'] = macd.macd_signal()

            rsi = RSIIndicator(close=data['Close'])
            data['RSI'] = rsi.rsi()

            bb = BollingerBands(close=data['Close'])
            data['BB_High'] = bb.bollinger_hband()
            data['BB_Low'] = bb.bollinger_lband()

            adx = ADXIndicator(high=data['High'], low=data['Low'], close=data['Close'])
            data['ADX'] = adx.adx()

            data['5_DAYS_STD_DEV'] = data['Close'].rolling(window=5).std()
            data['ATR'] = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close'], window=14).average_true_range()

            # Calculate Volume Moving Average
            data['Volume_MA_20'] = data['Volume'].rolling(window=20).mean()
            data['Volume_MA_10'] = data['Volume'].rolling(window=10).mean()

            # Drop rows with NaN values after calculating indicators
            data.dropna(inplace=True)

            # Define features and target variable
            features = ['Open', 'High', 'Low', 'Volume', 'SMA_5', 'SMA_10', 'SMA_20', 'MACD', 'MACD_Signal', 'RSI', 'BB_High', 'BB_Low', '5_DAYS_STD_DEV', 'ATR', 'Volume_MA_20', 'Volume_MA_10']
            X = data[features]
            y = data['Close']

            # Standardization
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            scaled_features = scaler_X.fit_transform(X)
            scaled_target = scaler_y.fit_transform(y.values.reshape(-1, 1))
            X = pd.DataFrame(scaled_features, columns=features)
            y = pd.DataFrame(scaled_target, columns=['Close'])

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Evaluate model performance
            def evaluate_model(y_true, y_pred):
                r2 = r2_score(y_true, y_pred)
                rmse = mean_squared_error(y_true, y_pred, squared=False)
                mae = mean_absolute_error(y_true, y_pred)
                return r2, rmse, mae

            # Train and evaluate ARIMA/SARIMA model
            print("Training ARIMA/SARIMA model...")
            sarima_model = SARIMAX(data['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            sarima_result = sarima_model.fit(disp=False)

            # Forecasting next 5 days
            print("Forecasting next 5 days...")
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
            plt.fill_between(forecast_df['Day'], forecast_df['Lower Bound'], forecast_df['Upper Bound'], color='gray', alpha=0.2)
            plt.title('Next 5 Days Forecast')
            plt.xlabel('Day')
            plt.ylabel('Price')
            plt.show()

            # Plot close using plotly with a time bar
            forecast_fig = go.Figure()

            # Prices plot
            forecast_fig.add_trace(go.Scatter(x=forecast_labels, y=forecasted_values, mode='lines', name='Next 5 Days Forecast'))
            # Show the plot
            st.plotly_chart(forecast_fig)

        elif choice == "Stock Watch":
            st.title('Momentum Stocks Selector')
            tickers = ["ABBOTINDIA.NS", "ADANIPOWER.NS", "AFFLE.BO", "AIAENG.BO", "AJANTPHARM.BO", "APLLTD.BO", "ALKEM.BO", "ARE&M.NS", "ANANDRATHI.BO", "APARINDS.BO", "ASIANPAINT.NS", "ASTRAL.NS", "ASTRAZEN.NS", "BAJFINANCE.NS", "BASF.NS", "BAYERCROP.BO", "BERGEPAINT.BO", "BDL.NS", "BEL.NS", "BSOFT.BO", "CDSL.NS", "CAMS.NS", "CARBORUNIV.BO", "CASTROLIND.NS", "CHAMBLFERT.BO", "COALINDIA.NS", "COFORGE.BO", "COLPAL.NS", "CONCORDBIO.BO", "COROMANDEL.BO", "CREDITACC.BO", "CUMMINSIND.NS", "CYIENT.NS", "DATAPATTNS.NS", "DEEPAKNTR.NS", "DIVISLAB.NS", "LALPATHLAB.NS", "RDY", "ELGIEQUIP.NS", "EMAMILTD.NS", "FIVESTAR.BO", "GRINFRA.NS", "GILLETTE.NS", "GLAXO.NS", "GODFRYPHLP.NS", "GRINDWELL.NS", "HAVELLS.NS", "HCLTECH.NS", "HAL.BO", "HONAUT.BO", "IRCTC.NS", "ISEC.BO", "INFY.NS", "IPCALAB.BO", "ITC.NS", "JBCHEPHARM.BO", "JWL.BO", "JYOTHYLAB.BO", "KPRMILL.NS", "KAJARIACER.NS", "KEI.BO", "LTTS.NS", "LTIM.NS", "MANKIND.NS", "MARICO.NS", "METROBRAND.BO", "MOTILALOFS.NS", "MPHASIS.NS", "MUTHOOTFIN.NS", "NH.NS", "NAVINFLUOR.NS", "NAM-INDIA.BO", "NMDC.NS", "OFSS.NS", "PGHH.NS", "PIIND.NS", "PAGEIND.NS", "PERSISTENT.NS", "PETRONET.NS", "PFIZER.NS", "PIDILITIND.NS", "POLYMED.NS", "POLYCAB.NS", "RRKABEL.NS", "RVNL.NS", "RATNAMANI.NS", "RITES.NS", "SANOFI.NS", "SCHAEFFLER.NS", "SKFINDIA.NS", "SOLARINDS.NS", "SONACOMS.NS", "SUMICHEM.NS", "SUNTV.NS", "SUNDRMFAST.NS", "SUPREMEIND.BO", "TATAELXSI.NS", "TATATECH.NS", "TCS.NS", "TECHM.NS", "TIMKEN.NS", "TITAN.NS", "TRITURBINE.NS", "TIINDIA.NS", "UNITDSPR.BO", "VGUARD.NS", "MANYAVAR.NS", "VINATIORGA.NS", "WIPRO.NS", "ZYDUSLIFE.NS"]

            # User inputs for period and top percentage
            period = st.selectbox("Select period for measuring momentum", ['1mo', '3mo', '6mo', '1y'])
            top_percent = st.slider("Select top percentage of performers", 1, 100, 20)

            # Fetch historical data for the defined period
            try:
                data = yf.download(tickers, period=period, interval='1d')

                if not data.empty:
                    # Calculate returns over the period
                    returns = (data['Adj Close'].iloc[-1] / data['Adj Close'].iloc[0]) - 1

                    # Rank stocks by returns
                    ranked_stocks = returns.sort_values(ascending=False)

                    # Select the top performers
                    top_performers = ranked_stocks.head(int(len(ranked_stocks) * top_percent / 100))

                    st.subheader('Top Performing Stocks:')
                    st.dataframe(top_performers)
                else:
                    st.error("Failed to fetch data. Please check the tickers and try again.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

        elif choice == "Strategy Backtesting":
            # List of tickers
            tickers = ["ABBOTINDIA.NS", "ADANIPOWER.NS", "AFFLE.BO", "AIAENG.BO", "AJANTPHARM.BO", "APLLTD.BO", "ALKEM.BO", "ARE&M.NS", "ANANDRATHI.BO", "APARINDS.BO", "ASIANPAINT.NS", "ASTRAL.NS", "ASTRAZEN.NS", "BAJFINANCE.NS", "BASF.NS", "BAYERCROP.BO", "BERGEPAINT.BO", "BDL.NS", "BEL.NS", "BSOFT.BO", "CDSL.NS", "CAMS.NS", "CARBORUNIV.BO", "CASTROLIND.NS", "CHAMBLFERT.BO", "COALINDIA.NS", "COFORGE.BO", "COLPAL.NS", "CONCORDBIO.BO", "COROMANDEL.BO", "CREDITACC.BO", "CUMMINSIND.NS", "CYIENT.NS", "DATAPATTNS.NS", "DEEPAKNTR.NS", "DIVISLAB.NS", "LALPATHLAB.NS", "RDY", "ELGIEQUIP.NS", "EMAMILTD.NS", "FIVESTAR.BO", "GRINFRA.NS", "GILLETTE.NS", "GLAXO.NS", "GODFRYPHLP.NS", "GRINDWELL.NS", "HAVELLS.NS", "HCLTECH.NS", "HAL.BO", "HONAUT.BO", "IRCTC.NS", "ISEC.BO", "INFY.NS", "IPCALAB.BO", "ITC.NS", "JBCHEPHARM.BO", "JWL.BO", "JYOTHYLAB.BO", "KPRMILL.NS", "KAJARIACER.NS", "KEI.BO", "LTTS.NS", "LTIM.NS", "MANKIND.NS", "MARICO.NS", "METROBRAND.BO", "MOTILALOFS.NS", "MPHASIS.NS", "MUTHOOTFIN.NS", "NH.NS", "NAVINFLUOR.NS", "NAM-INDIA.BO", "NMDC.NS", "OFSS.NS", "PGHH.NS", "PIIND.NS", "PAGEIND.NS", "PERSISTENT.NS", "PETRONET.NS", "PFIZER.NS", "PIDILITIND.NS", "POLYMED.NS", "POLYCAB.NS", "RRKABEL.NS", "RVNL.NS", "RATNAMANI.NS", "RITES.NS", "SANOFI.NS", "SCHAEFFLER.NS", "SKFINDIA.NS", "SOLARINDS.NS", "SONACOMS.NS", "SUMICHEM.NS", "SUNTV.NS", "SUNDRMFAST.NS", "SUPREMEIND.BO", "TATAELXSI.NS", "TATATECH.NS", "TCS.NS", "TECHM.NS", "TIMKEN.NS", "TITAN.NS", "TRITURBINE.NS", "TIINDIA.NS", "UNITDSPR.BO", "VGUARD.NS", "MANYAVAR.NS", "VINATIORGA.NS", "WIPRO.NS", "ZYDUSLIFE.NS"]

            # Streamlit App
            st.title('Stock Analysis with Technical Indicators')
            # Select ticker
            selected_ticker = st.selectbox("Select Ticker", tickers)

            # Select time period
            time_period = st.slider("Select Time Period (years)", 1, 5)

            # Calculate start date
            end_date = pd.to_datetime("today").strftime("%Y-%m-%d")
            start_date = (pd.to_datetime("today") - pd.DateOffset(years=time_period)).strftime("%Y-%m-%d")

            # Download historical data for the selected ticker
            data = yf.download(selected_ticker, start=start_date, end=end_date)

            # Function definitions for the indicators (same as before)

            # Function definitions for the indicators
            def calculate_rsi(series, period):
                delta = series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))

            def calculate_stochastic(data, period=14):
                low_min = data['Low'].rolling(window=period).min()
                high_max = data['High'].rolling(window=period).max()
                k = (data['Close'] - low_min) * 100 / (high_max - low_min)
                d = k.rolling(window=3).mean()
                return k, d

            def calculate_macd(series, fast=12, slow=26, signal=9):
                exp1 = series.ewm(span=fast, adjust=False).mean()
                exp2 = series.ewm(span=slow, adjust=False).mean()
                macd = exp1 - exp2
                signal_line = macd.ewm(span=signal, adjust=False).mean()
                return macd, signal_line

            def calculate_obv(data):
                obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
                return obv

            def calculate_vwap(data):
                vwap = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()
                return vwap

            def calculate_adl(data):
                clv = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
                adl = (clv * data['Volume']).cumsum()
                return adl

            def calculate_bollinger_bands(series, window=20, num_sd=2):
                mean = series.rolling(window=window).mean()
                sd = series.rolling(window=window).std()
                upper_band = mean + (sd * num_sd)
                lower_band = mean - (sd * num_sd)
                return lower_band, upper_band

            def calculate_atr(data, period=14):
                high_low = data['High'] - data['Low']
                high_close = np.abs(data['High'] - data['Close'].shift())
                low_close = np.abs(data['Low'] - data['Close'].shift())
                tr = high_low.combine(high_close, max).combine(low_close, max)
                atr = tr.rolling(window=period).mean()
                return atr

            def calculate_psar(data):
                af = 0.02
                ep = data['High'][0]
                psar = data['Low'][0]
                psar_list = [psar]
                for i in range(1, len(data)):
                    if data['Close'][i-1] > psar:
                        psar = min(psar + af * (ep - psar), data['Low'][i-1], data['Low'][i-2])
                    else:
                        psar = max(psar - af * (psar - ep), data['High'][i-1], data['High'][i-2])
                    ep = max(ep, data['High'][i]) if data['Close'][i] > psar else min(ep, data['Low'][i])
                    psar_list.append(psar)
                return pd.Series(psar_list, index=data.index)

            def calculate_ichimoku(data):
                nine_period_high = data['High'].rolling(window=9).max()
                nine_period_low = data['Low'].rolling(window=9).min()
                senkou_span_a = ((nine_period_high + nine_period_low) / 2).shift(26)
                senkou_span_b = ((data['High'].rolling(window=52).max() + data['Low'].rolling(window=52).min()) / 2).shift(26)
                return senkou_span_a, senkou_span_b

            def calculate_pivot_points(data):
                pivot = (data['High'] + data['Low'] + data['Close']) / 3
                return pivot

            def calculate_roc(series, period=12):
                return ((series.diff(period) / series.shift(period)) * 100).fillna(0)

            def calculate_dpo(series, period=20):
                return series - series.shift(period // 2 + 1).rolling(window=period).mean()

            def calculate_williams_r(data, period=14):
                high = data['High'].rolling(window=period).max()
                low = data['Low'].rolling(window=period).min()
                williams_r = (high - data['Close']) / (high - low) * -100
                return williams_r

            def calculate_ad_line(data):
                clv = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
                ad_line = (clv * data['Volume']).cumsum()
                return ad_line

            def calculate_aroon(data, period=25):
                aroon_up = data['High'].rolling(window=period + 1).apply(lambda x: x.argmax(), raw=True)
                aroon_down = data['Low'].rolling(window=period + 1).apply(lambda x: x.argmin(), raw=True)
                aroon_up = 100 * (period - aroon_up) / period
                aroon_down = 100 * (period - aroon_down) / period
                return aroon_up, aroon_down

            # Apply indicators to data
            data['SMA'] = data['Close'].rolling(window=50).mean()
            data['EMA'] = data['Close'].ewm(span=50, adjust=False).mean()
            data['WMA'] = data['Close'].rolling(window=50).apply(lambda x: np.dot(x, np.arange(1, 51)) / np.arange(1, 51).sum(), raw=True)
            data['RSI'] = calculate_rsi(data['Close'], 14)
            data['Stochastic_K'], data['Stochastic_D'] = calculate_stochastic(data)
            data['MACD'], data['MACD_Signal'] = calculate_macd(data['Close'])
            data['OBV'] = calculate_obv(data)
            data['VWAP'] = calculate_vwap(data)
            data['A/D Line'] = calculate_adl(data)
            data['Lower Band'], data['Upper Band'] = calculate_bollinger_bands(data['Close'])
            data['ATR'] = calculate_atr(data)
            data['SD'] = data['Close'].rolling(window=20).std()
            data['Mean'] = data['Close'].rolling(window=20).mean()
            data['PSAR'] = calculate_psar(data)
            data['Ichimoku_Cloud_Top'], data['Ichimoku_Cloud_Bottom'] = calculate_ichimoku(data)
            data['Pivot'] = calculate_pivot_points(data)
            data['ROC'] = calculate_roc(data['Close'])  # Corrected ROC calculation
            data['DPO'] = calculate_dpo(data['Close'])
            data['Williams %R'] = calculate_williams_r(data)
            data['A/D Line Breadth'] = calculate_ad_line(data)
            data['Aroon_Up'], data['Aroon_Down'] = calculate_aroon(data)

            # Define the buy and sell signals for each indicator
            data['Buy_Signal_SMA'] = (data['Close'] > data['SMA']).astype(int)
            data['Sell_Signal_SMA'] = (data['Close'] < data['SMA']).astype(int)

            data['Buy_Signal_EMA'] = (data['Close'] > data['EMA']).astype(int)
            data['Sell_Signal_EMA'] = (data['Close'] < data['EMA']).astype(int)

            data['Buy_Signal_WMA'] = (data['Close'] > data['WMA']).astype(int)
            data['Sell_Signal_WMA'] = (data['Close'] < data['WMA']).astype(int)

            data['Buy_Signal_RSI'] = (data['RSI'] < 30).astype(int)
            data['Sell_Signal_RSI'] = (data['RSI'] > 70).astype(int)

            data['Buy_Signal_Stochastic'] = ((data['Stochastic_K'] < 20) & (data['Stochastic_K'] > data['Stochastic_D'])).astype(int)
            data['Sell_Signal_Stochastic'] = ((data['Stochastic_K'] > 80) & (data['Stochastic_K'] < data['Stochastic_D'])).astype(int)

            data['Buy_Signal_MACD'] = (data['MACD'] > data['MACD_Signal']).astype(int)
            data['Sell_Signal_MACD'] = (data['MACD'] < data['MACD_Signal']).astype(int)

            data['Buy_Signal_OBV'] = ((data['OBV'] > data['OBV'].shift(1)) & (data['Close'] > data['Close'].shift(1))).astype(int)
            data['Sell_Signal_OBV'] = ((data['OBV'] < data['OBV'].shift(1)) & (data['Close'] < data['Close'].shift(1))).astype(int)

            data['Buy_Signal_VWAP'] = (data['Close'] > data['VWAP']).astype(int)
            data['Sell_Signal_VWAP'] = (data['Close'] < data['VWAP']).astype(int)

            data['Buy_Signal_ADL'] = (data['A/D Line'] > data['A/D Line'].shift(1)).astype(int)
            data['Sell_Signal_ADL'] = (data['A/D Line'] < data['A/D Line'].shift(1)).astype(int)

            data['Buy_Signal_Bollinger'] = (data['Close'] < data['Lower Band']).astype(int)
            data['Sell_Signal_Bollinger'] = (data['Close'] > data['Upper Band']).astype(int)

            data['Buy_Signal_PSAR'] = (data['Close'] > data['PSAR']).astype(int)
            data['Sell_Signal_PSAR'] = (data['Close'] < data['PSAR']).astype(int)

            data['Buy_Signal_Ichimoku'] = (data['Close'] > data['Ichimoku_Cloud_Top']).astype(int)
            data['Sell_Signal_Ichimoku'] = (data['Close'] < data['Ichimoku_Cloud_Bottom']).astype(int)

            data['Buy_Signal_Pivot'] = (data['Close'] > data['Pivot']).astype(int)
            data['Sell_Signal_Pivot'] = (data['Close'] < data['Pivot']).astype(int)

            data['Buy_Signal_ROC'] = (data['ROC'] > 0).astype(int)
            data['Sell_Signal_ROC'] = (data['ROC'] < 0).astype(int)

            data['Buy_Signal_DPO'] = (data['DPO'] > 0).astype(int)
            data['Sell_Signal_DPO'] = (data['DPO'] < 0).astype(int)

            data['Buy_Signal_WilliamsR'] = (data['Williams %R'] < -80).astype(int)
            data['Sell_Signal_WilliamsR'] = (data['Williams %R'] > -20).astype(int)

            data['Buy_Signal_Aroon'] = ((data['Aroon_Up'] > 70) & (data['Aroon_Down'] < 30)).astype(int)
            data['Sell_Signal_Aroon'] = ((data['Aroon_Up'] < 30) & (data['Aroon_Down'] > 70)).astype(int)

            # Define top 5 combined signals based on historical returns
            data['Buy_Signal_Combined_1'] = ((data['RSI'] < 30) & (data['MACD'] > data['MACD_Signal']) & (data['Close'] > data['SMA'])).astype(int)
            data['Sell_Signal_Combined_1'] = ((data['RSI'] > 70) & (data['MACD'] < data['MACD_Signal']) & (data['Close'] < data['SMA'])).astype(int)

            data['Buy_Signal_Combined_2'] = ((data['Close'] > data['EMA']) & (data['MACD'] > data['MACD_Signal']) & (data['Close'] > data['VWAP'])).astype(int)
            data['Sell_Signal_Combined_2'] = ((data['Close'] < data['EMA']) & (data['MACD'] < data['MACD_Signal']) & (data['Close'] < data['VWAP'])).astype(int)

            data['Buy_Signal_Combined_3'] = ((data['Close'] > data['WMA']) & (data['RSI'] < 30) & (data['Stochastic_K'] > data['Stochastic_D'])).astype(int)
            data['Sell_Signal_Combined_3'] = ((data['Close'] < data['WMA']) & (data['RSI'] > 70) & (data['Stochastic_K'] < data['Stochastic_D'])).astype(int)

            data['Buy_Signal_Combined_4'] = ((data['OBV'] > data['OBV'].shift(1)) & (data['Close'] > data['Ichimoku_Cloud_Top']) & (data['MACD'] > data['MACD_Signal'])).astype(int)
            data['Sell_Signal_Combined_4'] = ((data['OBV'] < data['OBV'].shift(1)) & (data['Close'] < data['Ichimoku_Cloud_Bottom']) & (data['MACD'] < data['MACD_Signal'])).astype(int)

            data['Buy_Signal_Combined_5'] = ((data['Close'] > data['Lower Band']) & (data['RSI'] < 30) & (data['ATR'] > data['ATR'].shift(1))).astype(int)
            data['Sell_Signal_Combined_5'] = ((data['Close'] < data['Upper Band']) & (data['RSI'] > 70) & (data['ATR'] < data['ATR'].shift(1))).astype(int)

            # Balance-based backtesting framework
            def backtest_strategy_balance_based(data, buy_signal, sell_signal, initial_amount=100000):
                balance = initial_amount
                positions = 0
                for i in range(len(data)):
                    if data[buy_signal].iloc[i] == 1 and balance > data['Close'].iloc[i]:
                        positions += balance // data['Close'].iloc[i]
                        balance -= positions * data['Close'].iloc[i]
                    elif data[sell_signal].iloc[i] == 1 and positions > 0:
                        balance += positions * data['Close'].iloc[i]
                        positions = 0
                final_balance = balance + (positions * data['Close'].iloc[-1])
                return final_balance

            # Cumulative return backtesting framework
            def backtest_strategy_cumulative(data, buy_signal, sell_signal, initial_amount=100000):
                data['Position'] = 0
                data.loc[data[buy_signal] == 1, 'Position'] = 1
                data.loc[data[sell_signal] == 1, 'Position'] = -1
                data['Position'] = data['Position'].shift(1).fillna(0)

                data['Daily_Return'] = data['Close'].pct_change()
                data['Strategy_Return'] = data['Position'] * data['Daily_Return']

                data['Cumulative_Return'] = (1 + data['Strategy_Return']).cumprod() - 1
                final_balance = initial_amount * (1 + data['Cumulative_Return'].iloc[-1])
                return data['Cumulative_Return'], final_balance

            strategies = ['Combined_1', 'Combined_2', 'Combined_3', 'Combined_4', 'Combined_5']
            results = []

            for strategy in strategies:
                final_balance_balance_based = backtest_strategy_balance_based(data, f'Buy_Signal_{strategy}', f'Sell_Signal_{strategy}')
                return_percentage_balance_based = ((final_balance_balance_based - 100000) / 100000) * 100

                cumulative_return, final_balance_cumulative = backtest_strategy_cumulative(data, f'Buy_Signal_{strategy}', f'Sell_Signal_{strategy}')
                return_percentage_cumulative = ((final_balance_cumulative - 100000) / 100000) * 100

                results.append({
                    'Strategy': strategy,
                    'Initial Amount (USD)': 100000,
                    'Final Amount (Balance-Based) (USD)': final_balance_balance_based,
                    'Return (Balance-Based) (%)': return_percentage_balance_based,
                    'Final Amount (Cumulative) (USD)': final_balance_cumulative,
                    'Return (Cumulative) (%)': return_percentage_cumulative
                })

            results_df = pd.DataFrame(results)

            st.subheader('Stock Data')
            st.dataframe(data)

            st.subheader('Technical Indicators')
            indicator = st.selectbox('Select Indicator', ['SMA', 'EMA', 'WMA', 'RSI', 'MACD', 'OBV', 'VWAP', 'Bollinger Bands', 'Aroon'])

            if indicator == 'Bollinger Bands':
                st.line_chart(data[['Close', 'Lower Band', 'Upper Band']])
            elif indicator == 'Aroon':
                st.line_chart(data[['Aroon_Up', 'Aroon_Down']])
            else:
                st.line_chart(data[['Close', indicator]])

            st.subheader('Strategy Performance')
            st.table(results_df)

        elif choice == "Watchlist":
            # Initialize session state
            if 'watchlists' not in st.session_state:
                st.session_state['watchlists'] = {f"Watchlist {i}": [] for i in range(1, 11)}

            # Function to fetch stock data
            def get_stock_data(ticker):
                try:
                    df = yf.download(ticker, period='1y')
                    if df.empty:
                        st.warning(f"No data found for {ticker}.")
                        return pd.DataFrame()  # Return an empty DataFrame
                    df['2_MA'] = df['Close'].rolling(window=2).mean()
                    df['15_MA'] = df['Close'].rolling(window=15).mean()
                    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
                    df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
                    return df[['Close', '2_MA', '15_MA', 'RSI', 'ADX']].iloc[-1]
                except Exception as e:
                    st.error(f"Error fetching data for {ticker}: {e}")
                    return pd.Series(dtype='float64')  # Return an empty Series

            # Function to update watchlist
            def update_watchlist(watchlist_name, ticker):
                if ticker not in st.session_state['watchlists'][watchlist_name]:
                    if len(st.session_state['watchlists'][watchlist_name]) < 10:
                        st.session_state['watchlists'][watchlist_name].append(ticker)
                    else:
                        st.warning(f"{watchlist_name} already has 10 stocks. Remove a stock before adding a new one.")
                else:
                    st.warning(f"{ticker} is already in {watchlist_name}.")

            # Function to remove a ticker from watchlist
            def remove_from_watchlist(watchlist_name, ticker):
                if ticker in st.session_state['watchlists'][watchlist_name]:
                    st.session_state['watchlists'][watchlist_name].remove(ticker)
                    st.success(f"Ticker {ticker} removed from {watchlist_name}!")

            # Watch List Page
            st.sidebar.header("Watchlist Manager")
            selected_watchlist = st.sidebar.radio("Select Watchlist", list(st.session_state['watchlists'].keys()))

            # Sidebar - Add ticker to selected watchlist
            st.sidebar.subheader("Add Ticker")
            ticker_input = st.sidebar.text_input("Ticker Symbol (e.g., AAPL)")

            if st.sidebar.button("Add Ticker"):
                update_watchlist(selected_watchlist, ticker_input.upper())
                st.sidebar.success(f"Ticker {ticker_input.upper()} added to {selected_watchlist}!")

            # Main section - Display watchlist and stock data
            st.header(f"{selected_watchlist}")
            watchlist_tickers = st.session_state['watchlists'][selected_watchlist]

            if watchlist_tickers:
                # Fetch data for all tickers in the watchlist
                watchlist_data = {ticker: get_stock_data(ticker) for ticker in watchlist_tickers}

                # Convert the dictionary of series to a DataFrame
                watchlist_df = pd.DataFrame(watchlist_data).T  # Transpose to have tickers as rows
                st.write("Watchlist Data:")
                st.dataframe(watchlist_df)

                # Provide option to remove tickers
                st.subheader("Remove Ticker")
                ticker_to_remove = st.selectbox("Select Ticker to Remove", watchlist_tickers)
                if st.button("Remove Ticker"):
                    remove_from_watchlist(selected_watchlist, ticker_to_remove)
                    st.experimental_rerun()  # Refresh the app to reflect changes
            else:
                st.write("No tickers in this watchlist.")

            # Footer - Show all watchlists and their tickers
            st.sidebar.subheader("All Watchlists")
            for watchlist, tickers in st.session_state['watchlists'].items():
                st.sidebar.write(f"{watchlist}: {', '.join(tickers) if tickers else 'No tickers'}")

        elif choice == "My Portfolio":

            # Function to get stock data
            def get_stock_data(ticker):
                stock = yf.Ticker(ticker)
                data = stock.history(period="1d")
                return data['Close'][0]

            # Function to calculate portfolio value
            def calculate_portfolio_value(portfolio):
                total_value = 0
                for stock, shares in portfolio.items():
                    price = get_stock_data(stock)
                    total_value += price * shares
                return total_value

            # Main function for the app
            def main():
                st.title("My Trading Portfolio")

                # Portfolio dictionary
                if 'portfolio' not in st.session_state:
                    st.session_state.portfolio = {}

                portfolio = st.session_state.portfolio

                # Input fields to add a stock
                st.header("Add a Stock to Your Portfolio")
                new_stock = st.text_input("Stock Ticker", value="")
                new_shares = st.number_input("Number of Shares", min_value=0, value=0, step=1)
                
                if st.button("Add Stock"):
                    if new_stock and new_shares > 0:
                        portfolio[new_stock] = portfolio.get(new_stock, 0) + new_shares
                        st.session_state.portfolio = portfolio
                        st.success(f"Added {new_shares} shares of {new_stock} to your portfolio.")

                # Input fields to remove a stock
                st.header("Remove a Stock from Your Portfolio")
                remove_stock = st.text_input("Stock Ticker to Remove", value="")
                remove_shares = st.number_input("Number of Shares to Remove", min_value=0, value=0, step=1)
                
                if st.button("Remove Stock"):
                    if remove_stock in portfolio and remove_shares > 0:
                        if portfolio[remove_stock] > remove_shares:
                            portfolio[remove_stock] -= remove_shares
                            st.success(f"Removed {remove_shares} shares of {remove_stock} from your portfolio.")
                        elif portfolio[remove_stock] == remove_shares:
                            del portfolio[remove_stock]
                            st.success(f"Removed all shares of {remove_stock} from your portfolio.")
                        else:
                            st.error(f"You don't have enough shares of {remove_stock} to remove.")
                        st.session_state.portfolio = portfolio

                # Display the portfolio
                st.header("Your Portfolio")
                if portfolio:
                    portfolio_df = pd.DataFrame(list(portfolio.items()), columns=['Stock', 'Shares'])
                    st.table(portfolio_df)

                    # Calculate and display the total portfolio value
                    total_value = calculate_portfolio_value(portfolio)
                    st.subheader(f"Total Portfolio Value: ${total_value:,.2f}")
                else:
                    st.write("Your portfolio is empty.")

            if __name__ == "__main__":
                main()
