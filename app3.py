import os
import streamlit as st
import hashlib
import random
import string
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from password_validator import PasswordValidator
from dotenv import load_dotenv
import plotly.express as px
from functools import lru_cache

from scipy.stats import linregress
from scipy.fftpack import fft, ifft

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import ta
import pandas_ta as pta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit

from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import hilbert
import pywt
import holidays
import mplfinance as mpf
from datetime import datetime, timedelta


# Set wide mode as default layout
st.set_page_config(layout="wide", page_title="TradeSense",page_icon=":chart_with_upwards_trend:")

# Load environment variables from .env file
load_dotenv()

# Database setup
Base = declarative_base()


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)


class Watchlist(Base):
    __tablename__ = 'watchlists'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    ticker = Column(String, nullable=False)
    date_added = Column(Date, default=datetime.utcnow)


class Portfolio(Base):
    __tablename__ = 'portfolios'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    ticker = Column(String, nullable=False)
    shares = Column(Float, nullable=False)
    bought_price = Column(Float, nullable=False)
    date_added = Column(Date, default=datetime.utcnow)


# Create the database session
DATABASE_URL = "sqlite:///etrade.db"
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

# Initialize session state for login status and reset code
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'email' not in st.session_state:
    st.session_state.email = ""
if 'reset_code' not in st.session_state:
    st.session_state.reset_code = ""

# Password validation schema
password_schema = PasswordValidator()
password_schema \
    .min(8) \
    .max(100) \
    .has().uppercase() \
    .has().lowercase() \
    .has().digits() \
    .has().no().spaces()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def send_email(to_email, subject, body):
    from_email = os.getenv('EMAIL_ADDRESS')
    password = os.getenv('EMAIL_PASSWORD')

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False


def signup():
    st.subheader("Sign Up")
    name = st.text_input("Enter your name", key='signup_name')
    email = st.text_input("Enter your email", key='signup_email')
    password = st.text_input("Enter a new password", type="password", key='signup_password')
    confirm_password = st.text_input("Confirm your password", type="password", key='signup_confirm_password')

    if st.button("Sign Up"):
        existing_user = session.query(User).filter_by(email=email).first()
        if existing_user:
            st.error("Email already exists. Try a different email.")
        elif password != confirm_password:
            st.error("Passwords do not match.")
        elif not password_schema.validate(password):
            st.error("Password does not meet the requirements.")
        else:
            new_user = User(name=name, email=email, password=hash_password(password))
            session.add(new_user)
            session.commit()
            st.success("User registered successfully!")


def login():
    st.subheader("Login")
    email = st.text_input("Enter your email", key='login_email')
    password = st.text_input("Enter your password", type="password", key='login_password')

    if st.button("Login"):
        user = session.query(User).filter_by(email=email).first()
        if user and user.password == hash_password(password):
            st.success("Login successful!")
            st.session_state.logged_in = True
            st.session_state.username = user.name
            st.session_state.email = user.email
        else:
            st.error("Invalid email or password.")


def forgot_password():
    st.subheader("Forgot Password")
    email = st.text_input("Enter your email", key='forgot_email')

    if st.button("Send Reset Code"):
        user = session.query(User).filter_by(email=email).first()
        if user:
            reset_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
            st.session_state.reset_code = reset_code
            st.session_state.email = email  # Save email in session state to use during password reset
            send_email(email, "Password Reset Code", f"Your password reset code is {reset_code}")
            st.success("Reset code sent to your email.")
        else:
            st.error("Email not found.")

    reset_code = st.text_input("Enter the reset code sent to your email", key='reset_code_input')
    new_password = st.text_input("Enter a new password", type="password", key='new_password')
    confirm_new_password = st.text_input("Confirm your new password", type="password", key='confirm_new_password')

    if st.button("Reset Password"):
        if reset_code == st.session_state.reset_code:
            if new_password != confirm_new_password:
                st.error("Passwords do not match.")
            elif not password_schema.validate(new_password):
                st.error("Password does not meet the requirements.")
            else:
                user = session.query(User).filter_by(email=st.session_state.email).first()
                user.password = hash_password(new_password)
                session.commit()
                st.success("Password reset successfully.")
                st.session_state.reset_code = ""
        else:
            st.error("Invalid reset code.")


def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.email = ""


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


def main_menu():
    st.subheader("Main Menu")
    menu_options = ["Markets", "Stock Screener", "Technical Analysis", "Stock Price Forecasting", "Stock Watch","Stock Comparison","Market Stats",
                     f"{st.session_state.username}'s Watchlist",
                    f"{st.session_state.username}'s Portfolio"]
    choice = st.selectbox("Select an option", menu_options)
    return choice


# Sidebar menu
with st.sidebar:
    st.title("TradeSense") 
    if st.session_state.logged_in:
        st.write(f"Logged in as: {st.session_state.username}")
        if st.button("Logout"):
            logout()
            st.experimental_rerun()  # Refresh the app
        else:
            choice = main_menu()  # Display the main menu in the sidebar if logged in
    else:
        selected = st.selectbox("Choose an option", ["Login", "Sign Up", "Forgot Password"])
        if selected == "Login":
            login()
        elif selected == "Sign Up":
            signup()
        elif selected == "Forgot Password":
            forgot_password()
        choice = None
     

# Main content area
if not st.session_state.logged_in:
    # dashboard code---------------------------------------------
    tickers = ["ABBOTINDIA.NS", "ADANIPOWER.NS", "AFFLE.BO", "AIAENG.BO", "AJANTPHARM.BO", "APLLTD.BO", "ALKEM.BO",
                "ARE&M.NS", "ANANDRATHI.BO", "APARINDS.BO", "ASIANPAINT.NS", "ASTRAL.NS", "ASTRAZEN.NS", "BAJFINANCE.NS",
                "BASF.NS", "BAYERCROP.BO", "BERGEPAINT.BO", "BDL.NS", "BEL.NS", "BSOFT.BO", "CDSL.NS", "CAMS.NS",
                "CARBORUNIV.BO", "CASTROLIND.NS", "CHAMBLFERT.BO", "COALINDIA.NS", "COFORGE.BO", "COLPAL.NS",
                "CONCORDBIO.BO", "COROMANDEL.BO", "CREDITACC.BO", "CUMMINSIND.NS", "CYIENT.NS", "DATAPATTNS.NS",
                "DEEPAKNTR.NS", "DIVISLAB.NS", "LALPATHLAB.NS", "RDY", "ELGIEQUIP.NS", "EMAMILTD.NS", "FIVESTAR.BO",
                "GRINFRA.NS", "GILLETTE.NS", "GLAXO.NS", "GODFRYPHLP.NS", "GRINDWELL.NS", "HAVELLS.NS", "HCLTECH.NS",
                "HAL.BO", "HONAUT.BO", "IRCTC.NS", "ISEC.BO", "INFY.NS", "IPCALAB.BO", "ITC.NS", "JBCHEPHARM.BO",
                "JWL.BO", "JYOTHYLAB.BO", "KPRMILL.NS", "KAJARIACER.NS", "KEI.BO", "LTTS.NS", "LTIM.NS", "MANKIND.NS",
                "MARICO.NS", "METROBRAND.BO", "MOTILALOFS.NS", "MPHASIS.NS", "MUTHOOTFIN.NS", "NH.NS", "NAVINFLUOR.NS",
                "NAM-INDIA.BO", "NMDC.NS", "OFSS.NS", "PGHH.NS", "PIIND.NS", "PAGEIND.NS", "PERSISTENT.NS", "PETRONET.NS",
                "PFIZER.NS", "PIDILITIND.NS", "POLYMED.NS", "POLYCAB.NS", "RRKABEL.NS", "RVNL.NS", "RATNAMANI.NS",
                "RITES.NS", "SANOFI.NS", "SCHAEFFLER.NS", "SKFINDIA.NS", "SOLARINDS.NS", "SONACOMS.NS", "SUMICHEM.NS",
                "SUNTV.NS", "SUNDRMFAST.NS", "SUPREMEIND.BO", "TATAELXSI.NS", "TATATECH.NS", "TCS.NS", "TECHM.NS",
                "TIMKEN.NS", "TITAN.NS", "TRITURBINE.NS", "TIINDIA.NS", "UNITDSPR.BO", "VGUARD.NS", "MANYAVAR.NS",
                "VINATIORGA.NS", "WIPRO.NS", "ZYDUSLIFE.NS"]

    # Function to get stock data and calculate moving averages
    @st.cache_data
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
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Reset Zoom",
                            method="relayout",
                            args=[{"xaxis.range": [None, None],
                                    "yaxis.range": [None, None]}])]
            )]
        )
        return fig

    # Function to fetch data
    @st.cache_data
    def fetch_data(tickers, period='1d', interval='1m'):
        data = yf.download(tickers, period=period, interval=interval)
        return data['Close']

    # Function to reshape data for heatmap
    def reshape_for_heatmap(df, num_columns=10):
        num_rows = int(np.ceil(len(df) / num_columns))
        reshaped_data = np.zeros((num_rows, num_columns))
        reshaped_tickers = np.empty((num_rows, num_columns), dtype=object)
        reshaped_data[:] = np.nan
        reshaped_tickers[:] = ''
        index = 0
        for y in range(num_rows):
            for x in range(num_columns):
                if index < len(df):
                    reshaped_data[y, x] = df['% Change'].values[index]
                    reshaped_tickers[y, x] = df['Ticker'].values[index]
                    index += 1
        return reshaped_data, reshaped_tickers

    # Create annotated heatmaps using Plotly
    def create_horizontal_annotated_heatmap(df, title, num_columns=10):
        reshaped_data, tickers = reshape_for_heatmap(df, num_columns)
        annotations = []
        for y in range(reshaped_data.shape[0]):
            for x in range(reshaped_data.shape[1]):
                text = f'<b>{tickers[y, x]}</b><br>{reshaped_data[y, x]}%'
                annotations.append(
                    go.layout.Annotation(
                        text=text,
                        x=x,
                        y=y,
                        xref='x',
                        yref='y',
                        showarrow=False,
                        font=dict(size=10, color="black", family="Arial, sans-serif"),
                        align="left"
                    )
                )
        fig = go.Figure(data=go.Heatmap(
            z=reshaped_data,
            x=list(range(reshaped_data.shape[1])),
            y=list(range(reshaped_data.shape[0])),
            hoverinfo='text',
            colorscale='Blues',
            showscale=False,
        ))
        fig.update_layout(
            title=title,
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            annotations=annotations,
            autosize=False,
            width=1800,
            height=200 + 50 * len(reshaped_data),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        return fig

    # Function to fetch stock data and volume
    @st.cache_data
    def get_volume_data(ticker, start_date, end_date):
        data = yf.download(ticker, start=start_date, end=end_date)
        return data['Volume'].sum()

    # Function to fetch sector data
    @st.cache_data
    def get_sector_data(ticker_symbol, start_date, end_date):
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
        return data

    # Function to calculate sector performance
    def calculate_performance(data):
        if not data.empty:
            performance = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
            return performance
        return None

    # Function to fetch market data
    @st.cache_data
    def get_market_data(ticker_symbol, start_date, end_date):
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
        return data

    st.title("TradeSense")
    st.write("An ultimate platform for smart trading insights. Please log in or sign up to get started.")

    # Create tiles for different sections
    tile_selection = st.selectbox("Select a section", 
                                ["Major Indices", "Top Gainers and Losers", "Volume Chart", 
                                "Sector Performance Chart", "Market Performance"])

    # Major Indices
    if tile_selection == "Major Indices":
        st.subheader("Major Indices")
        col1, col2, col3 = st.columns(3)
        with col1:
            stock_symbols = ["^BSESN", "BSE-500.BO", "^BSEMD", "^BSESMLCAP", "^NSEI", "^NSMIDCP", "^NSEMDCP", "^NSESCP"]
            ticker = st.selectbox("Enter Stock symbol", stock_symbols)
            st.write(f"You selected: {ticker}")
        with col2:
            START = st.date_input('Start Date', pd.to_datetime("2023-06-06"))
        with col3:
            END = st.date_input('End Date', pd.to_datetime("today"))
        if ticker and START and END:
            data = get_stock_data(ticker, START, END)
            fig = create_figure(data, ['Close', 'MA_15', 'MA_50'], f"{ticker} Stock Prices")
            st.plotly_chart(fig)

    # Top Gainers and Losers
    elif tile_selection == "Top Gainers and Losers":
        st.subheader("Top Gainers and Losers")
        
        
        # Fetch data for different periods
        data_daily = fetch_data(tickers, period='1d', interval='1m')
        data_weekly = fetch_data(tickers, period='5d', interval='1d')
        data_monthly = fetch_data(tickers, period='1mo', interval='1d')

        # Clean and prepare data
        data_daily.dropna(axis=1, how='all', inplace=True)
        data_weekly.dropna(axis=1, how='all', inplace=True)
        data_monthly.dropna(axis=1, how='all', inplace=True)
        data_daily.fillna(method='ffill', inplace=True)
        data_weekly.fillna(method='ffill', inplace=True)
        data_monthly.fillna(method='ffill', inplace=True)
        data_daily.fillna(method='bfill', inplace=True)
        data_weekly.fillna(method='bfill', inplace=True)
        data_monthly.fillna(method='bfill', inplace=True)

        # Calculate changes
        daily_change = data_daily.iloc[-1] - data_daily.iloc[0]
        percent_change_daily = (daily_change / data_daily.iloc[0]) * 100
        weekly_change = data_weekly.iloc[-1] - data_weekly.iloc[0]
        percent_change_weekly = (weekly_change / data_weekly.iloc[0]) * 100
        monthly_change = data_monthly.iloc[-1] - data_monthly.iloc[0]
        percent_change_monthly = (monthly_change / data_monthly.iloc[0]) * 100

        # Create DataFrames
        df_daily = pd.DataFrame({'Ticker': data_daily.columns, 'Last Traded Price': data_daily.iloc[-1].values, '% Change': percent_change_daily.values})
        df_weekly = pd.DataFrame({'Ticker': data_weekly.columns, 'Last Traded Price': data_weekly.iloc[-1].values, '% Change': percent_change_weekly.values})
        df_monthly = pd.DataFrame({'Ticker': data_monthly.columns, 'Last Traded Price': data_monthly.iloc[-1].values, '% Change': percent_change_monthly.values})

        # Round off the % Change values and sort
        df_daily['% Change'] = df_daily['% Change'].round(2)
        df_weekly['% Change'] = df_weekly['% Change'].round(2)
        df_monthly['% Change'] = df_monthly['% Change'].round(2)
        df_daily_sorted = df_daily.sort_values(by='% Change', ascending=True)
        df_weekly_sorted = df_weekly.sort_values(by='% Change', ascending=True)
        df_monthly_sorted = df_monthly.sort_values(by='% Change', ascending=True)

        # Dropdown menu to select the period
        heatmap_option = st.selectbox('Select to view:', ['Daily Gainers/Losers', 'Weekly Gainers/Losers', 'Monthly Gainers/Losers'])

        # Display the selected heatmap
        if heatmap_option == 'Daily Gainers/Losers':
            fig = create_horizontal_annotated_heatmap(df_daily_sorted, 'Daily Gainers/Losers')
            st.plotly_chart(fig)
        elif heatmap_option == 'Weekly Gainers/Losers':
            fig = create_horizontal_annotated_heatmap(df_weekly_sorted, 'Weekly Gainers/Losers')
            st.plotly_chart(fig)
        elif heatmap_option == 'Monthly Gainers/Losers':
            fig = create_horizontal_annotated_heatmap(df_monthly_sorted, 'Monthly Gainers/Losers')
            st.plotly_chart(fig)

    # Volume Chart
    elif tile_selection == "Volume Chart":
        st.subheader("Volume Chart")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input('Start Date', datetime(2022, 1, 1), key='start_date')
        with col2:
            end_date = st.date_input('End Date', datetime.today(), key='end_date')
        volume_data = {ticker: get_volume_data(ticker, start_date, end_date) for ticker in tickers}
        volume_df = pd.DataFrame(list(volume_data.items()), columns=['Ticker', 'Volume'])
        fig = px.bar(volume_df, x='Ticker', y='Volume', title='Trading Volume of Stocks',
                    labels={'Volume': 'Total Volume'}, color='Volume',
                    color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig)

    # Sector Performance Chart
    elif tile_selection == "Sector Performance Chart":
        st.subheader("Sector Performance Chart")
        sector_indices = {
            'NIFTY_BANK': '^NSEBANK',
            'NIFTY_IT': '^CNXIT',
            'NIFTY_AUTO': '^CNXAUTO',
            'NIFTY_FMCG': '^CNXFMCG',
            'NIFTY_PHARMA': '^CNXPHARMA',
            'NIFTY_REALTY': '^CNXREALTY',
            'NIFTY_METAL': '^CNXMETAL',
            'NIFTY_MEDIA': '^CNXMEDIA',
            'NIFTY_PSU_BANK': '^CNXPSUBANK',
            'NIFTY_ENERGY': '^CNXENERGY',
            'NIFTY_COMMODITIES': '^CNXCOMMOD',
            'NIFTY_INFRASTRUCTURE': '^CNXINFRA',
            'NIFTY_SERVICES_SECTOR': '^CNXSERVICE',
            'NIFTY_FINANCIAL_SERVICES': '^CNXFINANCE',
            'NIFTY_MNC': '^CNXMNC',
            'NIFTY_PSE': '^CNXPSE',
            'NIFTY_CPSE': '^CNXCPSE',
            'NIFTY_100': '^CNX100',
            'NIFTY_200': '^CNX200',
            'NIFTY_500': '^CNX500',
            'NIFTY_MIDCAP_50': '^CNXMID50',
            'NIFTY_MIDCAP_100': '^CNXMIDCAP',
            'NIFTY_SMALLCAP_100': '^CNXSMCAP',
            'NIFTY_NEXT_50': '^CNXNIFTY'
        }
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input('Start Date', datetime(2022, 1, 1), key='start_date')
        with col2:
            end_date = st.date_input('End Date', datetime.today(), key='end_date')
        sector_performance = {sector: calculate_performance(get_sector_data(ticker, start_date, end_date)) for sector, ticker in sector_indices.items() if calculate_performance(get_sector_data(ticker, start_date, end_date)) is not None}
        performance_df = pd.DataFrame(list(sector_performance.items()), columns=['Sector', 'Performance'])
        fig = px.bar(performance_df, x='Sector', y='Performance', title='Sector Performance',
                    labels={'Performance': 'Performance (%)'}, color='Performance',
                    color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig)

    # Market Performance
    elif tile_selection == "Market Performance":
        st.subheader("Market Performance")
        market_indices = {
            'S&P 500': '^GSPC',
            'Dow Jones': '^DJI',
            'NASDAQ': '^IXIC',
            'Gold': 'GC=F',
            'Silver': 'SI=F',
            'Oil': 'CL=F',
            'EUR/USD': 'EURUSD=X',
            'GBP/USD': 'GBPUSD=X',
            'Bitcoin': 'BTC-USD',
            'Ethereum': 'ETH-USD'
        }
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input('Start Date', datetime(2022, 1, 1), key='start_date')
        with col2:
            end_date = st.date_input('End Date', datetime.today(), key='end_date')
        market_performance = {market: calculate_performance(get_market_data(ticker, start_date, end_date)) for market, ticker in market_indices.items() if calculate_performance(get_market_data(ticker, start_date, end_date)) is not None}
        performance_df = pd.DataFrame(list(market_performance.items()), columns=['Market', 'Performance'])
        fig = px.bar(performance_df, x='Market', y='Performance', title='Market Performance',
                    labels={'Performance': 'Performance (%)'}, color='Performance',
                    color_continuous_scale=px.colors.diverging.RdYlGn)
        st.plotly_chart(fig)

    st.markdown("-----------------------------------------------------------------------------------------------------------------------")
    st.subheader("Unlock your trading potential. Join TradeSense today!")
    

else:
    if choice:
        if choice == "Markets":
            #'Markets' code-----------------------------------------------------
            # Function to download data and calculate moving averages with caching
            @lru_cache(maxsize=32)
            def get_stock_data(ticker_symbol, start_date, end_date):
                data = yf.download(ticker_symbol, start=start_date, end=end_date)
                data['MA_15'] = data['Close'].rolling(window=15).mean()
                data['MA_50'] = data['Close'].rolling(window=50).mean()
                data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
                data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
                data['Upper_Band'] = data['Close'].rolling(20).mean() + (data['Close'].rolling(20).std() * 2)
                data['Lower_Band'] = data['Close'].rolling(20).mean() - (data['Close'].rolling(20).std() * 2)
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
                if 'MACD' in indicators:
                    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD'))
                    fig.add_trace(go.Scatter(x=data.index, y=data['Signal_Line'], mode='lines', name='Signal Line'))
                if 'Bollinger Bands' in indicators:
                    fig.add_trace(go.Scatter(x=data.index, y=data['Upper_Band'], mode='lines', name='Upper Band'))
                    fig.add_trace(go.Scatter(x=data.index, y=data['Lower_Band'], mode='lines', name='Lower Band'))

                fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price',
                                xaxis_rangeslider_visible=True,
                                plot_bgcolor='dark grey',
                                paper_bgcolor='white',
                                font=dict(color='black'),
                                hovermode='x',
                                xaxis=dict(rangeselector=dict(buttons=list([
                                    dict(count=1, label="1m", step="month", stepmode="backward"),
                                    dict(count=6, label="6m", step="month", stepmode="backward"),
                                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                                    dict(count=1, label="1y", step="year", stepmode="backward"),
                                    dict(step="all")
                                ])),
                                    rangeslider=dict(visible=True),
                                    type='date'),
                                yaxis=dict(fixedrange=False),
                                updatemenus=[dict(type="buttons",
                                                    buttons=[dict(label="Reset Zoom",
                                                                method="relayout",
                                                                args=[{"xaxis.range": [None, None],
                                                                        "yaxis.range": [None, None]}])])])
                return fig

            # Function to calculate correlation
            def calculate_correlation(data1, data2):
                return data1['Close'].corr(data2['Close'])

            # Function to plot correlation matrix
            def plot_correlation_matrix(correlation_matrix):
                fig = go.Figure(data=go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.index,
                    colorscale='Viridis'))
                fig.update_layout(title="Correlation Matrix", xaxis_title='Assets', yaxis_title='Assets')
                return fig

            # Function to calculate Sharpe Ratio
            def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
                return (returns.mean() - risk_free_rate) / returns.std()

            # Function to calculate Beta
            def calculate_beta(asset_returns, market_returns):
                # Align the series to have the same index
                aligned_returns = pd.concat([asset_returns, market_returns], axis=1).dropna()
                covariance_matrix = np.cov(aligned_returns.iloc[:, 0], aligned_returns.iloc[:, 1])
                beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
                return beta

            # Function to calculate Value at Risk (VaR)
            def calculate_var(returns, confidence_level=0.05):
                return np.percentile(returns, confidence_level * 100)

            # Main application
            st.title("Market Insights")

            # Date inputs
            col1, col2 = st.columns(2)
            with col1:
                START = st.date_input('Start Date', pd.to_datetime("2023-06-06"))
            with col2:
                END = st.date_input('End Date', pd.to_datetime("today"))

            # Markets submenu
            submenu = st.sidebar.radio("Select Option", ["Equities", "Commodities", "Currencies", "Cryptocurrencies", "Analysis"])

            if submenu == "Equities":
                st.subheader("Equity Markets")
                data_nyse = get_stock_data("^NYA", START, END)
                data_bse = get_stock_data("^BSESN", START, END)
                indicators = st.multiselect("Select Indicators", ['Close', 'MA_15', 'MA_50', 'MACD', 'Bollinger Bands'], default=['Close'])
                fig_nyse = create_figure(data_nyse, indicators, 'NYSE Price')
                fig_bse = create_figure(data_bse, indicators, 'BSE Price')
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_nyse)

                with col2:
                    st.plotly_chart(fig_bse)


            elif submenu == "Commodities":
                st.subheader("Commodities")
                tickers = ["GC=F", "CL=F", "NG=F", "SI=F", "HG=F"]
                selected_tickers = st.multiselect("Select stock tickers to visualize", tickers, default=["GC=F", "CL=F"])
                indicators = st.multiselect("Select Indicators", ['Close', 'MA_15', 'MA_50', 'MACD', 'Bollinger Bands'], default=['Close'])
                if not selected_tickers:
                    st.warning("Please select at least one ticker.")
                else:
                    columns = st.columns(len(selected_tickers))
                    for ticker, col in zip(selected_tickers, columns):
                        data = get_stock_data(ticker, START, END)
                        fig = create_figure(data, indicators, f'{ticker} Price')
                        col.plotly_chart(fig)

            elif submenu == "Currencies":
                st.subheader("Currencies")
                tickers = ["EURUSD=X", "GBPUSD=X", "CNYUSD=X", "INRUSD=X"]
                selected_tickers = st.multiselect("Select currency pairs to visualize", tickers, default=["INRUSD=X", "CNYUSD=X"])
                indicators = st.multiselect("Select Indicators", ['Close', 'MA_15', 'MA_50', 'MACD', 'Bollinger Bands'], default=['Close'])
                if not selected_tickers:
                    st.warning("Please select at least one currency pair.")
                else:
                    columns = st.columns(len(selected_tickers))
                    for ticker, col in zip(selected_tickers, columns):
                        data = get_stock_data(ticker, START, END)
                        fig = create_figure(data, indicators, f'{ticker} Price')
                        col.plotly_chart(fig)

            elif submenu == "Cryptocurrencies":
                st.subheader("Cryptocurrencies")
                tickers = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "DOGE-USD"]
                selected_tickers = st.multiselect("Select cryptocurrencies to visualize", tickers, default=["BTC-USD", "ETH-USD"])
                indicators = st.multiselect("Select Indicators", ['Close', 'MA_15', 'MA_50', 'MACD', 'Bollinger Bands'], default=['Close'])
                if not selected_tickers:
                    st.warning("Please select at least one cryptocurrency.")
                else:
                    columns = st.columns(len(selected_tickers))
                    for ticker, col in zip(selected_tickers, columns):
                        data = get_stock_data(ticker, START, END)
                        fig = create_figure(data, indicators, f'{ticker} Price')
                        col.plotly_chart(fig)

            elif submenu == "Analysis":
                st.subheader("Detailed Market Analysis")
                st.write("This section provides an in-depth analysis of the markets, commodities, forex, and cryptos.")

                # Get data for all categories
                data_nyse = get_stock_data("^NYA", START, END)
                data_bse = get_stock_data("^BSESN", START, END)
                data_gold = get_stock_data("GC=F", START, END)
                data_oil = get_stock_data("CL=F", START, END)
                data_eurusd = get_stock_data("EURUSD=X", START, END)
                data_gbpusd = get_stock_data("GBPUSD=X", START, END)
                data_btc = get_stock_data("BTC-USD", START, END)
                data_eth = get_stock_data("ETH-USD", START, END)

                # Calculate correlations
                correlation_data = {
                    'NYSE': data_nyse['Close'],
                    'BSE': data_bse['Close'],
                    'Gold': data_gold['Close'],
                    'Oil': data_oil['Close'],
                    'EURUSD': data_eurusd['Close'],
                    'GBPUSD': data_gbpusd['Close'],
                    'BTC': data_btc['Close'],
                    'ETH': data_eth['Close']
                }
                df_correlation = pd.DataFrame(correlation_data)
                correlation_matrix = df_correlation.corr()

                # Plot correlation matrix
                fig_corr_matrix = plot_correlation_matrix(correlation_matrix)
                st.plotly_chart(fig_corr_matrix)

                # Calculate market returns for beta calculation (assuming S&P 500 as market index)
                data_sp500 = get_stock_data("^GSPC", START, END)
                market_returns = data_sp500['Close'].pct_change().dropna()

                # Trend and Additional Insights Analysis
                st.write("**Trend Analysis and Insights:**")
                analysis_data = {
                    "Assets": ['NYSE', 'BSE', 'Gold', 'Oil', 'EURUSD', 'GBPUSD', 'BTC', 'ETH'],
                    "Trend": [
                        "Bullish" if data_nyse['MA_15'].iloc[-1] > data_nyse['MA_50'].iloc[-1] else "Bearish",
                        "Bullish" if data_bse['MA_15'].iloc[-1] > data_bse['MA_50'].iloc[-1] else "Bearish",
                        "Bullish" if data_gold['MA_15'].iloc[-1] > data_gold['MA_50'].iloc[-1] else "Bearish",
                        "Bullish" if data_oil['MA_15'].iloc[-1] > data_oil['MA_50'].iloc[-1] else "Bearish",
                        "Bullish" if data_eurusd['MA_15'].iloc[-1] > data_eurusd['MA_50'].iloc[-1] else "Bearish",
                        "Bullish" if data_gbpusd['MA_15'].iloc[-1] > data_gbpusd['MA_50'].iloc[-1] else "Bearish",
                        "Bullish" if data_btc['MA_15'].iloc[-1] > data_btc['MA_50'].iloc[-1] else "Bearish",
                        "Bullish" if data_eth['MA_15'].iloc[-1] > data_eth['MA_50'].iloc[-1] else "Bearish"
                    ],
                    "Volatility (Daily)": [
                        np.std(data_nyse['Close']),
                        np.std(data_bse['Close']),
                        np.std(data_gold['Close']),
                        np.std(data_oil['Close']),
                        np.std(data_eurusd['Close']),
                        np.std(data_gbpusd['Close']),
                        np.std(data_btc['Close']),
                        np.std(data_eth['Close'])
                    ],
                    "Average Return (%) (Daily)": [
                        np.mean(data_nyse['Close'].pct_change()) * 100,
                        np.mean(data_bse['Close'].pct_change()) * 100,
                        np.mean(data_gold['Close'].pct_change()) * 100,
                        np.mean(data_oil['Close'].pct_change()) * 100,
                        np.mean(data_eurusd['Close'].pct_change()) * 100,
                        np.mean(data_gbpusd['Close'].pct_change()) * 100,
                        np.mean(data_btc['Close'].pct_change()) * 100,
                        np.mean(data_eth['Close'].pct_change()) * 100
                    ],
                    "Sharpe Ratio (Daily)": [
                        calculate_sharpe_ratio(data_nyse['Close'].pct_change()),
                        calculate_sharpe_ratio(data_bse['Close'].pct_change()),
                        calculate_sharpe_ratio(data_gold['Close'].pct_change()),
                        calculate_sharpe_ratio(data_oil['Close'].pct_change()),
                        calculate_sharpe_ratio(data_eurusd['Close'].pct_change()),
                        calculate_sharpe_ratio(data_gbpusd['Close'].pct_change()),
                        calculate_sharpe_ratio(data_btc['Close'].pct_change()),
                        calculate_sharpe_ratio(data_eth['Close'].pct_change())
                    ],
                    "Max Drawdown (%)": [
                        (data_nyse['Close'].max() - data_nyse['Close'].min()) / data_nyse['Close'].max() * 100,
                        (data_bse['Close'].max() - data_bse['Close'].min()) / data_bse['Close'].max() * 100,
                        (data_gold['Close'].max() - data_gold['Close'].min()) / data_gold['Close'].max() * 100,
                        (data_oil['Close'].max() - data_oil['Close'].min()) / data_oil['Close'].max() * 100,
                        (data_eurusd['Close'].max() - data_eurusd['Close'].min()) / data_eurusd['Close'].max() * 100,
                        (data_gbpusd['Close'].max() - data_gbpusd['Close'].min()) / data_gbpusd['Close'].max() * 100,
                        (data_btc['Close'].max() - data_btc['Close'].min()) / data_btc['Close'].max() * 100,
                        (data_eth['Close'].max() - data_eth['Close'].min()) / data_eth['Close'].max() * 100
                    ],
                    "Beta": [
                        calculate_beta(data_nyse['Close'].pct_change().dropna(), market_returns),
                        calculate_beta(data_bse['Close'].pct_change().dropna(), market_returns),
                        calculate_beta(data_gold['Close'].pct_change().dropna(), market_returns),
                        calculate_beta(data_oil['Close'].pct_change().dropna(), market_returns),
                        calculate_beta(data_eurusd['Close'].pct_change().dropna(), market_returns),
                        calculate_beta(data_gbpusd['Close'].pct_change().dropna(), market_returns),
                        calculate_beta(data_btc['Close'].pct_change().dropna(), market_returns),
                        calculate_beta(data_eth['Close'].pct_change().dropna(), market_returns)
                    ],
                    "Value at Risk (VaR) 5%": [
                        calculate_var(data_nyse['Close'].pct_change().dropna()),
                        calculate_var(data_bse['Close'].pct_change().dropna()),
                        calculate_var(data_gold['Close'].pct_change().dropna()),
                        calculate_var(data_oil['Close'].pct_change().dropna()),
                        calculate_var(data_eurusd['Close'].pct_change().dropna()),
                        calculate_var(data_gbpusd['Close'].pct_change().dropna()),
                        calculate_var(data_btc['Close'].pct_change().dropna()),
                        calculate_var(data_eth['Close'].pct_change().dropna())
                    ]
                }
                df_analysis = pd.DataFrame(analysis_data)
                st.table(df_analysis)

                # Annualized metrics
                st.write("**Annualized Metrics:**")
                annualized_data = {
                    "Assets": ['NYSE', 'BSE', 'Gold', 'Oil', 'EURUSD', 'GBPUSD', 'BTC', 'ETH'],
                    "Annualized Return (%)": [
                        ((1 + np.mean(data_nyse['Close'].pct_change())) ** 252 - 1) * 100,
                        ((1 + np.mean(data_bse['Close'].pct_change())) ** 252 - 1) * 100,
                        ((1 + np.mean(data_gold['Close'].pct_change())) ** 252 - 1) * 100,
                        ((1 + np.mean(data_oil['Close'].pct_change())) ** 252 - 1) * 100,
                        ((1 + np.mean(data_eurusd['Close'].pct_change())) ** 252 - 1) * 100,
                        ((1 + np.mean(data_gbpusd['Close'].pct_change())) ** 252 - 1) * 100,
                        ((1 + np.mean(data_btc['Close'].pct_change())) ** 252 - 1) * 100,
                        ((1 + np.mean(data_eth['Close'].pct_change())) ** 252 - 1) * 100
                    ],
                    "Annualized Volatility (%)": [
                        np.std(data_nyse['Close'].pct_change()) * np.sqrt(252) * 100,
                        np.std(data_bse['Close'].pct_change()) * np.sqrt(252) * 100,
                        np.std(data_gold['Close'].pct_change()) * np.sqrt(252) * 100,
                        np.std(data_oil['Close'].pct_change()) * np.sqrt(252) * 100,
                        np.std(data_eurusd['Close'].pct_change()) * np.sqrt(252) * 100,
                        np.std(data_gbpusd['Close'].pct_change()) * np.sqrt(252) * 100,
                        np.std(data_btc['Close'].pct_change()) * np.sqrt(252) * 100,
                        np.std(data_eth['Close'].pct_change()) * np.sqrt(252) * 100
                    ]
                }
                df_annualized = pd.DataFrame(annualized_data)
                st.table(df_annualized)



        elif choice == "Stock Screener":
        # 'Stock Screener' code---------------------------------------------------------------
            st.sidebar.subheader("Screens")
            submenu = st.sidebar.radio("Select Option", ["LargeCap", "MidCap", "SmallCap"])

            # List of stock tickers
            largecap_tickers = [
                "ITC.NS", "JBCHEPHARM.BO", "JWL.BO", "JYOTHYLAB.BO", "KPRMILL.NS", "KAJARIACER.NS", "KEI.BO",
                "LTTS.NS", "LTIM.NS", "MANKIND.NS", "MARICO.NS", "METROBRAND.BO", "MOTILALOFS.NS", "MPHASIS.NS",
                "MUTHOOTFIN.NS", "NH.NS", "NAVINFLUOR.NS", "NAM-INDIA.BO", "NMDC.NS", "OFSS.NS", "PGHH.NS",
                "PIIND.NS", "PAGEIND.NS", "PERSISTENT.NS", "PETRONET.NS", "PFIZER.NS", "PIDILITIND.NS",
                "POLYMED.NS", "POLYCAB.NS", "RRKABEL.NS", "RVNL.NS", "RATNAMANI.NS", "RITES.NS", "SANOFI.NS",
                "SCHAEFFLER.NS", "SKFINDIA.NS", "SOLARINDS.NS", "SONACOMS.NS", "SUMICHEM.NS", "SUNTV.NS",
                "SUNDRMFAST.NS", "SUPREMEIND.BO", "TATAELXSI.NS", "TATATECH.NS", "TCS.NS", "TECHM.NS", "TIMKEN.NS",
                "TITAN.NS", "TRITURBINE.NS", "TIINDIA.NS", "UNITDSPR.BO", "VGUARD.NS", "MANYAVAR.NS",
                "VINATIORGA.NS", "WIPRO.NS", "ZYDUSLIFE.NS", "CUMMINSIND.NS", "CYIENT.NS", "DATAPATTNS.NS", "DEEPAKNTR.NS",
                "DIVISLAB.NS", "LALPATHLAB.NS", "RDY", "ELGIEQUIP.NS", "EMAMILTD.NS", "FIVESTAR.BO", "GRINFRA.NS",
                "GILLETTE.NS", "GLAXO.NS", "GODFRYPHLP.NS", "GRINDWELL.NS", "HAVELLS.NS", "HCLTECH.NS", "HAL.BO",
                "HONAUT.BO", "IRCTC.NS", "ISEC.BO", "INFY.NS", "IPCALAB.BO", "ABBOTINDIA.NS", "ADANIPOWER.NS",
                "AFFLE.BO", "AIAENG.BO", "AJANTPHARM.BO", "APLLTD.BO", "ALKEM.BO", "ARE&M.NS", "ANANDRATHI.BO",
                "APARINDS.BO", "ASIANPAINT.NS", "ASTRAL.NS", "ASTRAZEN.NS", "BAJFINANCE.NS", "BASF.NS", "BAYERCROP.BO",
                "BERGEPAINT.BO", "BDL.NS", "BEL.NS", "BSOFT.BO", "CDSL.NS", "CAMS.NS", "CARBORUNIV.BO", "CASTROLIND.NS",
                "CHAMBLFERT.BO", "COALINDIA.NS", "COFORGE.BO", "COLPAL.NS", "CONCORDBIO.BO", "COROMANDEL.BO", "CREDITACC.BO"
            ]

            midcap_tickers = [
                "PNCINFRA.NS", "INDIASHLTR.NS", "RAYMOND.NS", "KAMAHOLD.BO", "BENGALASM.BO", "CHOICEIN.NS",
                "GRAVITA.NS", "HGINFRA.NS", "JKPAPER.NS", "MTARTECH.NS", "HAPPSTMNDS.NS", "SARDAEN.NS",
                "WELENT.NS", "LTFOODS.NS", "GESHIP.NS", "SHRIPISTON.NS", "SHAREINDIA.NS", "CYIENTDLM.NS", "VTL.NS",
                "EASEMYTRIP.NS", "LLOYDSME.NS", "ROUTE.NS", "VAIBHAVGBL.NS", "GOKEX.NS", "USHAMART.NS", "EIDPARRY.NS",
                "KIRLOSBROS.NS", "MANINFRA.NS", "CMSINFO.NS", "RALLIS.NS", "GHCL.NS", "NEULANDLAB.NS", "SPLPETRO.NS",
                "MARKSANS.NS", "NAVINFLUOR.NS", "ELECON.NS", "TANLA.NS", "KFINTECH.NS", "TIPSINDLTD.NS", "ACI.NS",
                "SURYAROSNI.NS", "GPIL.NS", "GMDCLTD.NS", "MAHSEAMLES.NS", "TDPOWERSYS.NS", "TECHNOE.NS", "JLHL.NS"
            ]

            smallcap_tickers = ["TAPARIA.BO", "LKPFIN.BO", "EQUITAS.NS"]

            # Function to create Plotly figure
            def create_figure(data, indicators, title):
                fig = go.Figure()

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
                        rangeselector=dict(),
                        rangeslider=dict(visible=True),
                        type='date'
                    ),
                    yaxis=dict(fixedrange=False),
                    updatemenus=[dict(
                        type="buttons",
                        buttons=[dict(label="Reset Zoom",
                                    method="relayout",
                                    args=[{"xaxis.range": [None, None],
                                            "yaxis.range": [None, None]}])]
                    )]
                )
                return fig

            # Function to fetch and process stock data
            @st.cache_data(ttl=3600)
            def get_stock_data(ticker_symbols, period):
                try:
                    stock_data = {}
                    for ticker_symbol in ticker_symbols:
                        df = yf.download(ticker_symbol, period=period)
                        if not df.empty:
                            df.interpolate(method='linear', inplace=True)
                            df = calculate_indicators(df)
                            df.dropna(inplace=True)
                            stock_data[ticker_symbol] = df
                    return stock_data
                except Exception as e:
                    print(f"Error fetching data: {e}")
                    return {}

            # Function to calculate technical indicators
            @st.cache_data(ttl=3600)
            def calculate_indicators(df):
                # Calculate Moving Averages
                df['5_MA'] = ta.trend.WMAIndicator(close=df['Close'], window=5).wma()
                df['20_MA'] = ta.trend.WMAIndicator(close=df['Close'], window=20).wma()
                df['50_MA'] = ta.trend.WMAIndicator(close=df['Close'], window=50).wma()

                # Calculate MACD
                macd = ta.trend.MACD(df['Close'])
                df['MACD'] = macd.macd()
                df['MACD_Signal'] = macd.macd_signal()
                df['MACD_Histogram'] = macd.macd_diff()

                # Calculate ADX
                adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'])
                df['ADX'] = adx.adx()

                # Calculate Parabolic SAR
                psar = ta.trend.PSARIndicator(df['High'], df['Low'], df['Close'])
                df['Parabolic_SAR'] = psar.psar()

                # Calculate RSI
                rsi = ta.momentum.RSIIndicator(df['Close'])
                df['RSI'] = rsi.rsi()

                # Calculate Volume Moving Averages
                df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
                df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
                df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()

                # Calculate Bollinger Bands
                bollinger = ta.volatility.BollingerBands(df['Close'])
                df['Bollinger_High'] = bollinger.bollinger_hband()
                df['Bollinger_Low'] = bollinger.bollinger_lband()
                df['Bollinger_Middle'] = bollinger.bollinger_mavg()

                # Calculate Detrended Price Oscillator (DPO)
                df['DPO'] = ta.trend.DPOIndicator(close=df['Close']).dpo()

                # Calculate On-Balance Volume (OBV)
                df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()

                # Calculate Volume Weighted Average Price (VWAP)
                df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).volume_weighted_average_price()

                # Calculate Accumulation/Distribution Line (A/D Line)
                df['A/D Line'] = ta.volume.AccDistIndexIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).acc_dist_index()

                # Calculate Average True Range (ATR)
                df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()

                return df

            # Function to query the stocks
            @st.cache_data(ttl=3600)
            def query_stocks(stock_data, conditions):
                results = []
                for ticker, df in stock_data.items():
                    if df.empty or len(df) < 1:
                        continue
                    condition_met = True
                    for condition in conditions:
                        col1, op, col2 = condition
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
                            'MACD': df['MACD'].iloc[-1],
                            'MACD_Signal': df['MACD_Signal'].iloc[-1],
                            'MACD_Hist': df['MACD_Histogram'].iloc[-1],
                            'RSI': df['RSI'].iloc[-1],
                            'ADX': df['ADX'].iloc[-1],
                            'Close': df['Close'].iloc[-1],
                            '5_MA': df['5_MA'].iloc[-1],
                            '20_MA': df['20_MA'].iloc[-1],
                            'Bollinger_High': df['Bollinger_High'].iloc[-1],
                            'Bollinger_Low': df['Bollinger_Low'].iloc[-1],
                            'Bollinger_Middle': df['Bollinger_Middle'].iloc[-1],
                            'Parabolic_SAR': df['Parabolic_SAR'].iloc[-1],
                            'Volume': df['Volume'].iloc[-1],
                            'Volume_MA_10': df['Volume_MA_10'].iloc[-1],
                            'Volume_MA_20': df['Volume_MA_20'].iloc[-1],
                            'DPO': df['DPO'].iloc[-1]
                        }
                        results.append(row)
                return pd.DataFrame(results)

            # Determine tickers based on submenu selection
            if submenu == "LargeCap":
                st.subheader("LargeCap")
                tickers = largecap_tickers
            elif submenu == "MidCap":
                st.subheader("MidCap")
                tickers = midcap_tickers
            else:
                st.subheader("SmallCap")
                tickers = smallcap_tickers

            # Fetch data and calculate indicators for each stock
            stock_data = get_stock_data(tickers, period='3mo')

            # Define first set of conditions
            first_conditions = [
                ('MACD', '>', 'MACD_Signal'),
                ('Parabolic_SAR', '<', 'Close')

            ]

            # Query stocks based on the first set of conditions
            first_query_df = query_stocks(stock_data, first_conditions)

            # Filter stocks in an uptrend with high volume and positive DPO
            second_query_df = first_query_df[
                (first_query_df['RSI'] < 65) & (first_query_df['RSI'] > 45) & 
                (first_query_df['ADX'] > 25) & (first_query_df['MACD'] > 0)
            ]

            st.write("Stocks in an uptrend with high volume and positive DPO:")
            st.dataframe(second_query_df)

            # Dropdown for analysis type
            col1, col2 = st.columns(2)

            # Set up the start and end date inputs
            with col1:
                selected_stock = st.selectbox("Select Stock", second_query_df['Ticker'].tolist())

            with col2:
                analysis_type = st.selectbox("Select Analysis Type", ["Trend Analysis", "Volume Analysis", "Support & Resistance Levels"])

            # Create two columns
            col1, col2 = st.columns(2)

            # Set up the start and end date inputs
            with col1:
                START = st.date_input('Start Date', pd.to_datetime("2023-06-01"))

            with col2:
                END = st.date_input('End Date', pd.to_datetime("today"))

            # If a stock is selected, plot its data with the selected indicators
            if selected_stock:
                @st.cache_data(ttl=3600)
                def load_data(ticker, start, end):
                    df = yf.download(ticker, start=start, end=end)
                    df.reset_index(inplace=True)
                    return df

                df = load_data(selected_stock, START, END)

                if df.empty:
                    st.write("No data available for the provided ticker.")
                else:
                    df.interpolate(method='linear', inplace=True)
                    df = calculate_indicators(df)

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

                    # Function to generate buy/sell signals
                    def generate_signals(macd, signal, rsi, close):
                        buy_signals = [0] * len(macd)
                        sell_signals = [0] * len(macd)
                        for i in range(1, len(macd)):
                            if macd[i] > signal[i] and macd[i-1] <= signal[i-1]:
                                buy_signals[i] = 1
                            elif macd[i] < signal[i] and macd[i-1] >= signal[i-1]:
                                sell_signals[i] = 1
                        return buy_signals, sell_signals

                    df['Buy_Signal'], df['Sell_Signal'] = generate_signals(df['MACD'], df['MACD_Signal'], df['RSI'], df['Close'])

                    if analysis_type == "Trend Analysis":
                        st.subheader("Trend Analysis")

                        indicators = st.multiselect(
                            "Select Indicators",
                            ['Close', '20_MA', '50_MA', '200_MA', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'RSI', 'Buy_Signal', 'Sell_Signal', 'ADX',
                            'Parabolic_SAR', 'Bollinger_High', 'Bollinger_Low', 'Bollinger_Middle', 'ATR'],
                            default=['Close', 'Buy_Signal', 'Sell_Signal']
                        )
                        timeframe = st.radio(
                            "Select Timeframe",
                            ['15 days', '30 days', '90 days', '180 days', '1 year', 'All'],
                            index=2,
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

                        fig = create_figure(df.set_index('Date'), indicators, f"Trend Analysis for {selected_stock}")

                        colors = {'Close': 'blue', '20_MA': 'orange', '50_MA': 'green', '200_MA': 'red', 'MACD': 'purple',
                                'MACD_Signal': 'brown', 'RSI': 'pink', 'Buy_Signal': 'green', 'Sell_Signal': 'red', 'ADX': 'magenta',
                                'Parabolic_SAR': 'yellow', 'Bollinger_High': 'black', 'Bollinger_Low': 'cyan',
                                'Bollinger_Middle': 'grey', 'ATR': 'darkblue'}

                        for indicator in indicators:
                            if indicator == 'Buy_Signal':
                                fig.add_trace(
                                    go.Scatter(x=df[df[indicator] == 1]['Date'],
                                            y=df[df[indicator] == 1]['Close'], mode='markers', name='Buy Signal',
                                            marker=dict(color='green', symbol='triangle-up')))
                            elif indicator == 'Sell_Signal':
                                fig.add_trace(
                                    go.Scatter(x=df[df[indicator] == 1]['Date'],
                                            y=df[df[indicator] == 1]['Close'], mode='markers', name='Sell Signal',
                                            marker=dict(color='red', symbol='triangle-down')))
                            elif indicator == 'MACD_Histogram':
                                fig.add_trace(
                                    go.Bar(x=df['Date'], y=df[indicator], name=indicator, marker_color='gray'))
                            else:
                                fig.add_trace(
                                    go.Scatter(x=df['Date'], y=df[indicator], mode='lines', name=indicator,
                                            line=dict(color=colors.get(indicator, 'black'))))

                        st.plotly_chart(fig)

                    elif analysis_type == "Volume Analysis":
                        st.subheader("Volume Analysis")
                        volume_indicators = st.multiselect(
                            "Select Volume Indicators",
                            ['Close','Volume', 'Volume_MA_20', 'Volume_MA_10', 'Volume_MA_5', 'OBV', 'VWAP', 'A/D Line'],
                            default=['Close','VWAP']
                        )
                        volume_timeframe = st.radio(
                            "Select Timeframe",
                            ['15 days', '30 days', '90 days', '180 days', '1 year', 'All'],
                            index=2,
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

                        fig = create_figure(df.set_index('Date'), volume_indicators, f"Volume Analysis for {selected_stock}")

                        for indicator in volume_indicators:
                            fig.add_trace(go.Scatter(x=df['Date'], y=df[indicator], mode='lines', name=indicator))

                        st.plotly_chart(fig)

                    elif analysis_type == "Support & Resistance Levels":
                        st.subheader("Support & Resistance Levels")
                        sr_indicators = st.multiselect(
                            "Select Indicators",
                            ['Close', '20_MA', '50_MA', '200_MA', 'Support', 'Resistance', 'Support_Trendline',
                            'Resistance_Trendline', 'Pivot', 'R1', 'S1', 'R2', 'S2'],
                            default=['Close', 'Support', 'Resistance']
                        )
                        sr_timeframe = st.radio(
                            "Select Timeframe",
                            ['15 days', '30 days', '90 days', '180 days', '1 year', 'All'],
                            index=2,
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

                        fig = create_figure(df.set_index('Date'), sr_indicators, f"Support & Resistance Levels for {selected_stock}")

                        for indicator in sr_indicators:
                            fig.add_trace(go.Scatter(x=df['Date'], y=df[indicator], mode='lines', name=indicator))

                        st.plotly_chart(fig) 


        elif choice == "Technical Analysis":
            # 'Technical Analysis' code------------------------------------------------------------------------------------------------

            # Sidebar setup
            st.sidebar.subheader("Interactive Charts")
            # Load stock data
            @st.cache_data
            def load_data(ticker):
                data = yf.download(ticker, period='3mo')
                data.reset_index(inplace=True)
                return data

            # Load index data
            @st.cache_data
            def load_index_data(ticker):
                data = yf.download(ticker, start="2024-02-01", end="2024-06-20")
                data.reset_index(inplace=True)
                return data

            st.title('Stock Technical Analysis')

            # Sidebar for user input

            ticker = st.sidebar.text_input("Enter Stock Symbol", value='RVNL.NS')
            index_ticker = "^NSEI"  # NIFTY 50 index ticker

            # Load data
            data_load_state = st.text('Loading data...')
            data = load_data(ticker).copy()
            index_data = load_index_data(index_ticker).copy()
            data_load_state.text('Loading data...done!')

            # Calculate technical indicators
            def calculate_technical_indicators(df, index_df):
                # Moving averages
                df['20_SMA'] = ta.trend.sma_indicator(df['Close'], window=20)
                df['20_EMA'] = ta.trend.ema_indicator(df['Close'], window=20)
                df['20_WMA'] = ta.trend.wma_indicator(df['Close'], window=20)

                # Momentum Indicators
                df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
                df['%K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
                df['%D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
                df['MACD'] = ta.trend.macd(df['Close'])
                df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
                df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

                # Volume Indicators
                df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
                df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'], window=14)
                df['A/D Line'] = ta.volume.acc_dist_index(df['High'], df['Low'], df['Close'], df['Volume'])

                # Volatility Indicators
                df['BB_High'], df['BB_Middle'], df['BB_Low'] = ta.volatility.bollinger_hband(df['Close']), ta.volatility.bollinger_mavg(df['Close']), ta.volatility.bollinger_lband(df['Close'])
                df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
                df['Std Dev'] = ta.volatility.bollinger_wband(df['Close'])

                # Trend Indicators
                psar = pta.psar(df['High'], df['Low'], df['Close'])
                df['Parabolic_SAR'] = psar['PSARl_0.02_0.2']
                df['Ichimoku_a'] = ta.trend.ichimoku_a(df['High'], df['Low'])
                df['Ichimoku_b'] = ta.trend.ichimoku_b(df['High'], df['Low'])
                df['Ichimoku_base'] = ta.trend.ichimoku_base_line(df['High'], df['Low'])
                df['Ichimoku_conv'] = ta.trend.ichimoku_conversion_line(df['High'], df['Low'])

                # Support and Resistance Levels
                df['Pivot Points'] = (df['High'] + df['Low'] + df['Close']) / 3

                # Price Oscillators
                df['ROC'] = ta.momentum.roc(df['Close'], window=12)
                df['DPO'] = ta.trend.dpo(df['Close'], window=20)
                df['Williams %R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14)

                # Market Breadth Indicators
                df['Advances'] = df['Close'].diff().apply(lambda x: 1 if x > 0 else 0)
                df['Declines'] = df['Close'].diff().apply(lambda x: 1 if x < 0 else 0)
                df['McClellan Oscillator'] = (df['Advances'] - df['Declines']).rolling(window=19).mean() - (df['Advances'] - df['Declines']).rolling(window=39).mean()
                df['TRIN'] = (df['Advances'] / df['Declines']) / (df['Volume'][df['Advances'] > 0].sum() / df['Volume'][df['Declines'] > 0].sum())
                df['Advance-Decline Line'] = df['Advances'].cumsum() - df['Declines'].cumsum()

                # Relative Performance Indicators
                df['Price-to-Volume Ratio'] = df['Close'] / df['Volume']
                df['Relative Strength Comparison'] = df['Close'] / index_df['Close']
                df['Performance Relative to an Index'] = df['Close'].pct_change().cumsum() - index_df['Close'].pct_change().cumsum()

                return df

            data = calculate_technical_indicators(data, index_data)

            # Function to add range buttons to the plot
            def add_range_buttons(fig):
                fig.update_layout(
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=list([
                                dict(count=7, label="7d", step="day", stepmode="backward"),
                                dict(count=14, label="14d", step="day", stepmode="backward"),
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=3, label="3m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(step="all")
                            ])
                        ),
                        rangeslider=dict(visible=True)
                    )
                )

            # Plotly visualization functions
            def plot_indicator(df, indicator, title, yaxis_title='Price', secondary_y=False):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))
                fig.add_trace(go.Scatter(x=df['Date'], y=df[indicator], mode='lines', name=indicator, yaxis="y2" if secondary_y else "y1"))
                
                if secondary_y:
                    fig.update_layout(
                        yaxis2=dict(
                            title=indicator,
                            overlaying='y',
                            side='right'
                        )
                    )
                
                fig.update_layout(title=title, xaxis_title='Date', yaxis_title=yaxis_title)
                add_range_buttons(fig)
                st.plotly_chart(fig)

            # Plotly visualization for trendlines
            def plot_trendlines(df):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))

                # Adding trendlines using linear regression (advanced method)
                x = np.arange(len(df))
                coef = np.polyfit(x, df['Close'], 1)
                trend = np.poly1d(coef)
                fig.add_trace(go.Scatter(x=df['Date'], y=trend(x), mode='lines', name='Trendline', line=dict(color='red', dash='dash')))

                fig.update_layout(title='Trendlines', xaxis_title='Date', yaxis_title='Price')
                add_range_buttons(fig)
                st.plotly_chart(fig)

            # Plotly visualization for Fibonacci retracement levels
            def plot_fibonacci_retracement(df):
                high = df['High'].max()
                low = df['Low'].min()

                diff = high - low
                levels = [high, high - 0.236 * diff, high - 0.382 * diff, high - 0.5 * diff, high - 0.618 * diff, low]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))

                for level in levels:
                    fig.add_trace(go.Scatter(x=[df['Date'].iloc[0], df['Date'].iloc[-1]],
                                            y=[level, level],
                                            mode='lines', name=f'Level {level}', line=dict(dash='dash')))

                fig.update_layout(title='Fibonacci Retracement Levels', xaxis_title='Date', yaxis_title='Price')
                add_range_buttons(fig)
                st.plotly_chart(fig)

            # Plotly visualization for Gann fan lines
            def plot_gann_fan_lines(df):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))

                # Adding Gann fan lines (simple example, for more advanced lines use a proper method)
                for i in range(1, 5):
                    fig.add_trace(go.Scatter(x=[df['Date'].iloc[0], df['Date'].iloc[-1]],
                                            y=[df['Close'].iloc[0], df['Close'].iloc[0] + i * (df['Close'].iloc[-1] - df['Close'].iloc[0]) / 4],
                                            mode='lines', name=f'Gann Fan {i}', line=dict(dash='dash')))

                fig.update_layout(title='Gann Fan Lines', xaxis_title='Date', yaxis_title='Price')
                add_range_buttons(fig)
                st.plotly_chart(fig)

            # Plotly visualization for chart patterns
            def plot_chart_patterns(df, pattern):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))

                # Adding example chart patterns (simple example, for more advanced patterns use a proper method)
                if pattern == 'Head and Shoulders':
                    fig.add_trace(go.Scatter(x=[df['Date'].iloc[0], df['Date'].iloc[len(df)//3], df['Date'].iloc[2*len(df)//3], df['Date'].iloc[-1]],
                                            y=[df['Close'].iloc[0], df['Close'].iloc[len(df)//3], df['Close'].iloc[2*len(df)//3], df['Close'].iloc[-1]],
                                            mode='lines+markers', name='Head and Shoulders', line=dict(color='orange')))

                elif pattern == 'Double Tops and Bottoms':
                    fig.add_trace(go.Scatter(x=[df['Date'].iloc[0], df['Date'].iloc[len(df)//2], df['Date'].iloc[-1]],
                                            y=[df['Close'].iloc[0], df['Close'].iloc[len(df)//2], df['Close'].iloc[-1]],
                                            mode='lines+markers', name='Double Tops and Bottoms', line=dict(color='green')))

                elif pattern == 'Flags and Pennants':
                    fig.add_trace(go.Scatter(x=[df['Date'].iloc[0], df['Date'].iloc[-1]],
                                            y=[df['Close'].iloc[0], df['Close'].iloc[-1]],
                                            mode='lines', name='Flags and Pennants', line=dict(color='purple', dash='dash')))

                fig.update_layout(title=f'{pattern}', xaxis_title='Date', yaxis_title='Price')
                st.plotly_chart(fig)

            # Plotly visualization for McClellan Oscillator
            def plot_mcclellan_oscillator(df):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['McClellan Oscillator'], mode='lines', name='McClellan Oscillator'))
                fig.update_layout(title='McClellan Oscillator', xaxis_title='Date', yaxis_title='Value')
                add_range_buttons(fig)
                st.plotly_chart(fig)

            # Plotly visualization for TRIN
            def plot_trin(df):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['TRIN'], mode='lines', name='TRIN'))
                fig.update_layout(title='Arms Index (TRIN)', xaxis_title='Date', yaxis_title='Value')
                add_range_buttons(fig)
                st.plotly_chart(fig)

            # Detect chart patterns
            def detect_patterns(df):
                patterns = []

                # Head and Shoulders
                hs_pattern = detect_head_and_shoulders(df)
                if hs_pattern:
                    patterns.append(hs_pattern)

                # Double Tops and Bottoms
                dt_pattern = detect_double_tops_and_bottoms(df)
                if dt_pattern:
                    patterns.append(dt_pattern)

                # Flags and Pennants
                fp_pattern = detect_flags_and_pennants(df)
                if fp_pattern:
                    patterns.append(fp_pattern)

                return patterns

            # Placeholder function to detect head and shoulders pattern
            def detect_head_and_shoulders(df):
                # Simplified logic to detect head and shoulders pattern
                pattern_detected = False
                for i in range(2, len(df)-2):
                    if df['High'].iloc[i] > df['High'].iloc[i-1] and df['High'].iloc[i] > df['High'].iloc[i+1] and \
                    df['High'].iloc[i-1] < df['High'].iloc[i-2] and df['High'].iloc[i+1] < df['High'].iloc[i+2]:
                        pattern_detected = True
                        break
                if pattern_detected:
                    return ("Head and Shoulders", "Sell", "Regular Head and Shoulders pattern detected.")
                return None

            # Placeholder function to detect double tops and bottoms
            def detect_double_tops_and_bottoms(df):
                # Simplified logic to detect double tops and bottoms
                pattern_detected = False
                for i in range(1, len(df)-1):
                    if df['High'].iloc[i] == df['High'].iloc[i-1] and df['High'].iloc[i] == df['High'].iloc[i+1]:
                        pattern_detected = True
                        break
                if pattern_detected:
                    return ("Double Tops", "Sell", "Double Tops pattern detected.")
                return None

            # Placeholder function to detect flags and pennants
            def detect_flags_and_pennants(df):
                # Simplified logic to detect flags and pennants
                pattern_detected = False
                for i in range(1, len(df)-1):
                    if df['Close'].iloc[i] > df['Close'].iloc[i-1] and df['Close'].iloc[i] > df['Close'].iloc[i+1]:
                        pattern_detected = True
                        break
                if pattern_detected:
                    return ("Flags and Pennants", "Buy", "Bullish Flag pattern detected.")
                return None

            # Determine buy and sell signals
            def get_signals(df):
                signals = []

                # Example logic for signals (these can be customized)
                if df['Close'].iloc[-1] > df['20_SMA'].iloc[-1]:
                    signals.append(("Simple Moving Average (20_SMA)", "Hold", "Price is above the SMA."))
                else:
                    signals.append(("Simple Moving Average (20_SMA)", "Sell", "Price crossed below the SMA."))

                if df['Close'].iloc[-1] > df['20_EMA'].iloc[-1]:
                    signals.append(("Exponential Moving Average (20_EMA)", "Hold", "Price is above the EMA."))
                else:
                    signals.append(("Exponential Moving Average (20_EMA)", "Sell", "Price crossed below the EMA."))

                if df['Close'].iloc[-1] > df['20_WMA'].iloc[-1]:
                    signals.append(("Weighted Moving Average (20_WMA)", "Hold", "Price is above the WMA."))
                else:
                    signals.append(("Weighted Moving Average (20_WMA)", "Sell", "Price crossed below the WMA."))

                if df['RSI'].iloc[-1] < 30:
                    signals.append(("Relative Strength Index (RSI)", "Buy", "RSI crosses below 30 (oversold)."))
                elif df['RSI'].iloc[-1] > 70:
                    signals.append(("Relative Strength Index (RSI)", "Sell", "RSI crosses above 70 (overbought)."))
                else:
                    signals.append(("Relative Strength Index (RSI)", "Hold", "RSI is between 30 and 70."))

                if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
                    signals.append(("Moving Average Convergence Divergence (MACD)", "Buy", "MACD line crosses above the signal line."))
                else:
                    signals.append(("Moving Average Convergence Divergence (MACD)", "Sell", "MACD line crosses below the signal line."))

                if df['%K'].iloc[-1] < 20 and df['%D'].iloc[-1] < 20 and df['%K'].iloc[-1] > df['%D'].iloc[-1]:
                    signals.append(("Stochastic Oscillator", "Buy", "%K line crosses above %D line and both are below 20."))
                elif df['%K'].iloc[-1] > 80 and df['%D'].iloc[-1] > 80 and df['%K'].iloc[-1] < df['%D'].iloc[-1]:
                    signals.append(("Stochastic Oscillator", "Sell", "%K line crosses below %D line and both are above 80."))
                else:
                    signals.append(("Stochastic Oscillator", "Hold", "No clear buy or sell signal."))

                if df['OBV'].diff().iloc[-1] > 0:
                    signals.append(("On-Balance Volume (OBV)", "Buy", "OBV is increasing."))
                else:
                    signals.append(("On-Balance Volume (OBV)", "Sell", "OBV is decreasing."))

                if df['Close'].iloc[-1] > df['VWAP'].iloc[-1]:
                    signals.append(("Volume Weighted Average Price (VWAP)", "Buy", "Price crosses above the VWAP."))
                else:
                    signals.append(("Volume Weighted Average Price (VWAP)", "Sell", "Price crosses below the VWAP."))

                if df['A/D Line'].diff().iloc[-1] > 0:
                    signals.append(("Accumulation/Distribution Line (A/D Line)", "Buy", "A/D Line is increasing."))
                else:
                    signals.append(("Accumulation/Distribution Line (A/D Line)", "Sell", "A/D Line is decreasing."))

                if df['Close'].iloc[-1] < df['BB_Low'].iloc[-1]:
                    signals.append(("Bollinger Bands", "Buy", "Price crosses below the lower band."))
                elif df['Close'].iloc[-1] > df['BB_High'].iloc[-1]:
                    signals.append(("Bollinger Bands", "Sell", "Price crosses above the upper band."))
                else:
                    signals.append(("Bollinger Bands", "Hold", "Price is within Bollinger Bands."))

                if df['ATR'].iloc[-1] > df['ATR'].rolling(window=14).mean().iloc[-1]:
                    signals.append(("Average True Range (ATR)", "Buy", "ATR is increasing, indicating higher volatility."))
                else:
                    signals.append(("Average True Range (ATR)", "Sell", "ATR is decreasing, indicating lower volatility."))

                if df['Std Dev'].iloc[-1] > df['Close'].rolling(window=20).std().iloc[-1]:
                    signals.append(("Standard Deviation", "Buy", "Price is below the mean minus 2 standard deviations."))
                else:
                    signals.append(("Standard Deviation", "Sell", "Price is above the mean plus 2 standard deviations."))

                if df['Parabolic_SAR'].iloc[-1] < df['Close'].iloc[-1]:
                    signals.append(("Parabolic SAR (Stop and Reverse)", "Buy", "Price crosses above the SAR."))
                else:
                    signals.append(("Parabolic SAR (Stop and Reverse)", "Sell", "Price crosses below the SAR."))

                if df['ROC'].iloc[-1] > 0:
                    signals.append(("Price Rate of Change (ROC)", "Buy", "ROC crosses above zero."))
                else:
                    signals.append(("Price Rate of Change (ROC)", "Sell", "ROC crosses below zero."))

                if df['DPO'].iloc[-1] > 0:
                    signals.append(("Detrended Price Oscillator (DPO)", "Buy", "DPO crosses above zero."))
                else:
                    signals.append(("Detrended Price Oscillator (DPO)", "Sell", "DPO crosses below zero."))

                if df['Williams %R'].iloc[-1] < -80:
                    signals.append(("Williams %R", "Buy", "Williams %R crosses above -80 (indicating oversold)."))
                elif df['Williams %R'].iloc[-1] > -20:
                    signals.append(("Williams %R", "Sell", "Williams %R crosses below -20 (indicating overbought)."))
                else:
                    signals.append(("Williams %R", "Hold", "Williams %R is between -80 and -20."))

                if df['Close'].iloc[-1] > df['Pivot Points'].iloc[-1]:
                    signals.append(("Pivot Points", "Buy", "Price crosses above the pivot point."))
                else:
                    signals.append(("Pivot Points", "Sell", "Price crosses below the pivot point."))

                high = df['High'].max()
                low = df['Low'].min()
                diff = high - low
                fib_levels = [high, high - 0.236 * diff, high - 0.382 * diff, high - 0.5 * diff, high - 0.618 * diff, low]
                for level in fib_levels:
                    if df['Close'].iloc[-1] > level:
                        signals.append(("Fibonacci Retracement Levels", "Buy", "Price crosses above a Fibonacci retracement level."))
                        break
                    elif df['Close'].iloc[-1] < level:
                        signals.append(("Fibonacci Retracement Levels", "Sell", "Price crosses below a Fibonacci retracement level."))
                        break

                gann_fan_line = [df['Close'].iloc[0] + i * (df['Close'].iloc[-1] - df['Close'].iloc[0]) / 4 for i in range(1, 5)]
                for line in gann_fan_line:
                    if df['Close'].iloc[-1] > line:
                        signals.append(("Gann Fan Lines", "Buy", "Price crosses above a Gann fan line."))
                        break
                    elif df['Close'].iloc[-1] < line:
                        signals.append(("Gann Fan Lines", "Sell", "Price crosses below a Gann fan line."))
                        break

                if df['McClellan Oscillator'].iloc[-1] > 0:
                    signals.append(("McClellan Oscillator", "Buy", "Oscillator crosses above zero."))
                else:
                    signals.append(("McClellan Oscillator", "Sell", "Oscillator crosses below zero."))

                if df['TRIN'].iloc[-1] < 1:
                    signals.append(("Arms Index (TRIN)", "Buy", "TRIN below 1.0 (more advancing volume)."))
                else:
                    signals.append(("Arms Index (TRIN)", "Sell", "TRIN above 1.0 (more declining volume)."))

                # Chart Patterns
                patterns = detect_patterns(df)
                signals.extend(patterns)

                # Additional Indicators
                if df['Ichimoku_a'].iloc[-1] > df['Ichimoku_b'].iloc[-1]:
                    signals.append(("Ichimoku Cloud", "Buy", "Ichimoku conversion line above baseline."))
                else:
                    signals.append(("Ichimoku Cloud", "Sell", "Ichimoku conversion line below baseline."))

                if df['Relative Strength Comparison'].iloc[-1] > 1:
                    signals.append(("Relative Strength Comparison", "Buy", "Stock outperforms index."))
                else:
                    signals.append(("Relative Strength Comparison", "Sell", "Stock underperforms index."))

                if df['Performance Relative to an Index'].iloc[-1] > 0:
                    signals.append(("Performance Relative to an Index", "Buy", "Stock outperforms index over time."))
                else:
                    signals.append(("Performance Relative to an Index", "Sell", "Stock underperforms index over time."))

                if df['Advance-Decline Line'].diff().iloc[-1] > 0:
                    signals.append(("Advance-Decline Line", "Buy", "Advances exceed declines."))
                else:
                    signals.append(("Advance-Decline Line", "Sell", "Declines exceed advances."))

                if df['Price-to-Volume Ratio'].iloc[-1] > df['Price-to-Volume Ratio'].rolling(window=14).mean().iloc[-1]:
                    signals.append(("Price-to-Volume Ratio", "Buy", "Price-to-Volume ratio increasing."))
                else:
                    signals.append(("Price-to-Volume Ratio", "Sell", "Price-to-Volume ratio decreasing."))

                return signals

            signals = get_signals(data)

            # Sidebar for technical indicators
            st.sidebar.header('Technical Indicators')
            indicator_category = st.sidebar.radio('Select Indicator Category', [
                'Moving Averages', 'Momentum Indicators', 'Volume Indicators', 'Volatility Indicators', 'Trend Indicators',
                'Support and Resistance Levels', 'Price Oscillators', 'Market Breadth Indicators', 'Chart Patterns', 'Relative Performance Indicators', 'Summary'
            ])

            # Display technical indicators
            
            if indicator_category != 'Summary':
                if indicator_category == 'Moving Averages':
                    indicators = st.selectbox("Select Moving Average", ['20_SMA', '20_EMA', '20_WMA'])
                elif indicator_category == 'Momentum Indicators':
                    indicators = st.selectbox("Select Momentum Indicator", ['RSI', 'Stochastic Oscillator', 'MACD'])
                elif indicator_category == 'Volume Indicators':
                    indicators = st.selectbox("Select Volume Indicator", ['OBV', 'VWAP', 'A/D Line'])
                elif indicator_category == 'Volatility Indicators':
                    indicators = st.selectbox("Select Volatility Indicator", ['Bollinger Bands', 'ATR', 'Standard Deviation'])
                elif indicator_category == 'Trend Indicators':
                    indicators = st.selectbox("Select Trend Indicator", ['Trendlines', 'Parabolic SAR', 'Ichimoku Cloud'])
                elif indicator_category == 'Support and Resistance Levels':
                    indicators = st.selectbox("Select Support and Resistance Level", ['Pivot Points', 'Fibonacci Retracement Levels', 'Gann Fan Lines'])
                elif indicator_category == 'Price Oscillators':
                    indicators = st.selectbox("Select Price Oscillator", ['ROC', 'DPO', 'Williams %R'])
                elif indicator_category == 'Market Breadth Indicators':
                    indicators = st.selectbox("Select Market Breadth Indicator", ['Advance-Decline Line', 'McClellan Oscillator', 'TRIN'])
                elif indicator_category == 'Chart Patterns':
                    indicators = st.selectbox("Select Chart Pattern", ['Head and Shoulders', 'Double Tops and Bottoms', 'Flags and Pennants'])
                elif indicator_category == 'Relative Performance Indicators':
                    indicators = st.selectbox("Select Relative Performance Indicator", ['Price-to-Volume Ratio', 'Relative Strength Comparison', 'Performance Relative to an Index'])

                if indicators == '20_SMA':
                    plot_indicator(data, '20_SMA', 'Simple Moving Average (20_SMA)')
                elif indicators == '20_EMA':
                    plot_indicator(data, '20_EMA', 'Exponential Moving Average (20_EMA)')
                elif indicators == '20_WMA':
                    plot_indicator(data, '20_WMA', 'Weighted Moving Average (20_WMA)')
                elif indicators == 'RSI':
                    plot_indicator(data, 'RSI', 'Relative Strength Index (RSI)', secondary_y=True)
                elif indicators == 'Stochastic Oscillator':
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
                    fig.add_trace(go.Scatter(x=data['Date'], y=data['%K'], mode='lines', name='%K'))
                    fig.add_trace(go.Scatter(x=data['Date'], y=data['%D'], mode='lines', name='%D'))
                    fig.update_layout(title='Stochastic Oscillator', xaxis_title='Date', yaxis_title='Value')
                    add_range_buttons(fig)
                    st.plotly_chart(fig)
                elif indicators == 'MACD':
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
                    fig.add_trace(go.Scatter(x=data['Date'], y=data['MACD'], mode='lines', name='MACD'))
                    fig.add_trace(go.Scatter(x=data['Date'], y=data['MACD_Signal'], mode='lines', name='MACD Signal'))
                    fig.add_trace(go.Bar(x=data['Date'], y=data['MACD_Hist'], name='MACD Histogram', yaxis='y2'))
                    fig.update_layout(
                        title='MACD',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        yaxis2=dict(
                            title='MACD Histogram',
                            overlaying='y',
                            side='right'
                        )
                    )
                    add_range_buttons(fig)
                    st.plotly_chart(fig)
                elif indicators == 'OBV':
                    plot_indicator(data, 'OBV', 'On-Balance Volume (OBV)')
                elif indicators == 'VWAP':
                    plot_indicator(data, 'VWAP', 'Volume Weighted Average Price (VWAP)')
                elif indicators == 'A/D Line':
                    plot_indicator(data, 'A/D Line', 'Accumulation/Distribution Line')
                elif indicators == 'Bollinger Bands':
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
                    fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_High'], mode='lines', name='BB High'))
                    fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_Middle'], mode='lines', name='BB Middle'))
                    fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_Low'], mode='lines', name='BB Low'))
                    fig.update_layout(title='Bollinger Bands', xaxis_title='Date', yaxis_title='Price')
                    add_range_buttons(fig)
                    st.plotly_chart(fig)
                elif indicators == 'ATR':
                    plot_indicator(data, 'ATR', 'Average True Range (ATR)')
                elif indicators == 'Standard Deviation':
                    plot_indicator(data, 'Std Dev', 'Standard Deviation')
                elif indicators == 'Trendlines':
                    plot_trendlines(data)
                elif indicators == 'Parabolic SAR':
                    plot_indicator(data, 'Parabolic_SAR', 'Parabolic SAR')
                elif indicators == 'Ichimoku Cloud':
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
                    fig.add_trace(go.Scatter(x=data['Date'], y=data['Ichimoku_a'], mode='lines', name='Ichimoku A'))
                    fig.add_trace(go.Scatter(x=data['Date'], y=data['Ichimoku_b'], mode='lines', name='Ichimoku B'))
                    fig.add_trace(go.Scatter(x=data['Date'], y=data['Ichimoku_base'], mode='lines', name='Ichimoku Base Line'))
                    fig.add_trace(go.Scatter(x=data['Date'], y=data['Ichimoku_conv'], mode='lines', name='Ichimoku Conversion Line'))
                    fig.update_layout(title='Ichimoku Cloud', xaxis_title='Date', yaxis_title='Value')
                    add_range_buttons(fig)
                    st.plotly_chart(fig)
                elif indicators == 'Pivot Points':
                    plot_indicator(data, 'Pivot Points', 'Pivot Points')
                elif indicators == 'Fibonacci Retracement Levels':
                    plot_fibonacci_retracement(data)
                elif indicators == 'Gann Fan Lines':
                    plot_gann_fan_lines(data)
                elif indicators == 'ROC':
                    plot_indicator(data, 'ROC', 'Rate of Change (ROC)')
                elif indicators == 'DPO':
                    plot_indicator(data, 'DPO', 'Detrended Price Oscillator (DPO)')
                elif indicators == 'Williams %R':
                    plot_indicator(data, 'Williams %R', 'Williams %R')
                elif indicators == 'McClellan Oscillator':
                    plot_mcclellan_oscillator(data)
                elif indicators == 'TRIN':
                    plot_trin(data)
                elif indicators == 'Advance-Decline Line':
                    plot_indicator(data, 'Advance-Decline Line', 'Advance-Decline Line')
                elif indicators == 'Head and Shoulders':
                    plot_chart_patterns(data, 'Head and Shoulders')
                elif indicators == 'Double Tops and Bottoms':
                    plot_chart_patterns(data, 'Double Tops and Bottoms')
                elif indicators == 'Flags and Pennants':
                    plot_chart_patterns(data, 'Flags and Pennants')
                elif indicators == 'Price-to-Volume Ratio':
                    plot_indicator(data, 'Price-to-Volume Ratio', 'Price-to-Volume Ratio', secondary_y=True)
                elif indicators == 'Relative Strength Comparison':
                    plot_indicator(data, 'Relative Strength Comparison', 'Relative Strength Comparison')
                elif indicators == 'Performance Relative to an Index':
                    plot_indicator(data, 'Performance Relative to an Index', 'Performance Relative to an Index')
            else:
                # Display signals in a dataframe with improved visualization
                st.subheader('Technical Indicator Signals')
                signals_df = pd.DataFrame(signals, columns=['Technical Indicator', 'Signal', 'Reason'])
                st.dataframe(signals_df.style.applymap(lambda x: 'background-color: lightgreen' if 'Buy' in x else 'background-color: lightcoral' if 'Sell' in x else '', subset=['Signal']))
                    



        elif choice == "Market Stats":
        # 'Market Stats' code -------------------------------------------------------------------------------------------    
            
            # List of tickers
            tickers = ["ABBOTINDIA.NS", "ADANIPOWER.NS", "AFFLE.BO", "AIAENG.BO", "AJANTPHARM.BO", "APLLTD.BO", "ALKEM.BO",
                    "ARE&M.NS", "ANANDRATHI.BO", "APARINDS.BO", "ASIANPAINT.NS", "ASTRAL.NS", "ASTRAZEN.NS", "BAJFINANCE.NS",
                    "BASF.NS", "BAYERCROP.BO", "BERGEPAINT.BO", "BDL.NS", "BEL.NS", "BSOFT.BO", "CDSL.NS", "CAMS.NS",
                    "CARBORUNIV.BO", "CASTROLIND.NS", "CHAMBLFERT.BO", "COALINDIA.NS", "COFORGE.BO", "COLPAL.NS",
                    "CONCORDBIO.BO", "COROMANDEL.BO", "CREDITACC.BO", "CUMMINSIND.NS", "CYIENT.NS", "DATAPATTNS.NS",
                    "DEEPAKNTR.NS", "DIVISLAB.NS", "LALPATHLAB.NS", "RDY", "ELGIEQUIP.NS", "EMAMILTD.NS", "FIVESTAR.BO",
                    "GRINFRA.NS", "GILLETTE.NS", "GLAXO.NS", "GODFRYPHLP.NS", "GRINDWELL.NS", "HAVELLS.NS", "HCLTECH.NS",
                    "HAL.BO", "HONAUT.BO", "IRCTC.NS", "ISEC.BO", "INFY.NS", "IPCALAB.BO", "ITC.NS", "JBCHEPHARM.BO", "JWL.BO",
                    "JYOTHYLAB.BO", "KPRMILL.NS", "KAJARIACER.NS", "KEI.BO", "LTTS.NS", "LTIM.NS", "MANKIND.NS", "MARICO.NS",
                    "METROBRAND.BO", "MOTILALOFS.NS", "MPHASIS.NS", "MUTHOOTFIN.NS", "NH.NS", "NAVINFLUOR.NS", "NAM-INDIA.BO",
                    "NMDC.NS", "OFSS.NS", "PGHH.NS", "PIIND.NS", "PAGEIND.NS", "PERSISTENT.NS", "PETRONET.NS", "PFIZER.NS",
                    "PIDILITIND.NS", "POLYMED.NS", "POLYCAB.NS", "RRKABEL.NS", "RVNL.NS", "RATNAMANI.NS", "RITES.NS",
                    "SANOFI.NS", "SCHAEFFLER.NS", "SKFINDIA.NS", "SOLARINDS.NS", "SONACOMS.NS", "SUMICHEM.NS", "SUNTV.NS",
                    "SUNDRMFAST.NS", "SUPREMEIND.BO", "TATAELXSI.NS", "TATATECH.NS", "TCS.NS", "TECHM.NS", "TIMKEN.NS",
                    "TITAN.NS", "TRITURBINE.NS", "TIINDIA.NS", "UNITDSPR.BO", "VGUARD.NS", "MANYAVAR.NS", "VINATIORGA.NS",
                    "WIPRO.NS", "ZYDUSLIFE.NS"]

            # Function to fetch data
            def fetch_data(tickers, period='1d', interval='1m'):
                data = yf.download(tickers, period=period, interval=interval)
                return data['Close']

            # Fetch the data for daily, weekly, and monthly periods
            data_daily = fetch_data(tickers, period='1d', interval='1m')
            data_weekly = fetch_data(tickers, period='5d', interval='1d')
            data_monthly = fetch_data(tickers, period='1mo', interval='1d')

            # Drop columns with all NaN values
            data_daily.dropna(axis=1, how='all', inplace=True)
            data_weekly.dropna(axis=1, how='all', inplace=True)
            data_monthly.dropna(axis=1, how='all', inplace=True)

            # Fill missing values with forward fill
            data_daily.fillna(method='ffill', inplace=True)
            data_weekly.fillna(method='ffill', inplace=True)
            data_monthly.fillna(method='ffill', inplace=True)

            # Fill any remaining NaNs with backward fill (in case the first row is NaN)
            data_daily.fillna(method='bfill', inplace=True)
            data_weekly.fillna(method='bfill', inplace=True)
            data_monthly.fillna(method='bfill', inplace=True)

            # Calculate daily, weekly, and monthly changes
            daily_change = data_daily.iloc[-1] - data_daily.iloc[0]
            percent_change_daily = (daily_change / data_daily.iloc[0]) 

            weekly_change = data_weekly.iloc[-1] - data_weekly.iloc[0]
            percent_change_weekly = (weekly_change / data_weekly.iloc[0]) 

            monthly_change = data_monthly.iloc[-1] - data_monthly.iloc[0]
            percent_change_monthly = (monthly_change / data_monthly.iloc[0]) 

            # Create DataFrames
            df_daily = pd.DataFrame({'Ticker': data_daily.columns, 'Last Traded Price': data_daily.iloc[-1].values,
                                    '% Change': percent_change_daily.values})
            df_weekly = pd.DataFrame({'Ticker': data_weekly.columns, 'Last Traded Price': data_weekly.iloc[-1].values,
                                    '% Change': percent_change_weekly.values})
            df_monthly = pd.DataFrame({'Ticker': data_monthly.columns, 'Last Traded Price': data_monthly.iloc[-1].values,
                                    '% Change': percent_change_monthly.values})

            # Remove rows with NaN values
            df_daily.dropna(inplace=True)
            df_weekly.dropna(inplace=True)
            df_monthly.dropna(inplace=True)

            # Top 5 Gainers and Losers for daily, weekly, and monthly
            top_gainers_daily = df_daily.nlargest(5, '% Change')
            top_losers_daily = df_daily.nsmallest(5, '% Change')

            top_gainers_weekly = df_weekly.nlargest(5, '% Change')
            top_losers_weekly = df_weekly.nsmallest(5, '% Change')

            top_gainers_monthly = df_monthly.nlargest(5, '% Change')
            top_losers_monthly = df_monthly.nsmallest(5, '% Change')

            # Function to plot bar charts with gainers and losers on a single line
            def plot_bar_chart(gainers, losers, title):
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=gainers['Ticker'],
                    y=gainers['% Change'],
                    name='Gainers',
                    marker_color='lightseagreen'
                ))

                fig.add_trace(go.Bar(
                    x=losers['Ticker'],
                    y=losers['% Change'],
                    name='Losers',
                    marker_color='lightpink'
                ))

                fig.update_layout(
                    title=title,
                    xaxis_title='Ticker',
                    yaxis_title='% Change',
                    barmode='relative',
                    bargap=0.15,
                    bargroupgap=0.1,
                    yaxis=dict(tickformat='%')
                )

                st.plotly_chart(fig)


            
            plot_bar_chart(top_gainers_daily, top_losers_daily, 'Top 5 Daily Gainers and Losers')

            
            plot_bar_chart(top_gainers_weekly, top_losers_weekly, 'Top 5 Weekly Gainers and Losers')

            
            plot_bar_chart(top_gainers_monthly, top_losers_monthly, 'Top 5 Monthly Gainers and Losers')

        elif choice == "Stock Watch":
        # 'Stock Watch' code -------------------------------------------------------------------------------------------
            st.sidebar.subheader("Screens")
            submenu = st.sidebar.radio("Select Option", ["Bollinger", "Macd"])

            # List of stock tickers
            stock_tickers = [
                "ITC.NS", "JBCHEPHARM.BO", "JWL.BO", "JYOTHYLAB.BO", "KPRMILL.NS", "KAJARIACER.NS", "KEI.BO",
                "LTTS.NS", "LTIM.NS", "MANKIND.NS", "MARICO.NS", "METROBRAND.BO", "MOTILALOFS.NS", "MPHASIS.NS",
                "MUTHOOTFIN.NS", "NH.NS", "NAVINFLUOR.NS", "NAM-INDIA.BO", "NMDC.NS", "OFSS.NS", "PGHH.NS",
                "PIIND.NS", "PAGEIND.NS", "PERSISTENT.NS", "PETRONET.NS", "PFIZER.NS", "PIDILITIND.NS",
                "POLYMED.NS", "POLYCAB.NS", "RRKABEL.NS", "RVNL.NS", "RATNAMANI.NS", "RITES.NS", "SANOFI.NS",
                "SCHAEFFLER.NS", "SKFINDIA.NS", "SOLARINDS.NS", "SONACOMS.NS", "SUMICHEM.NS", "SUNTV.NS",
                "SUNDRMFAST.NS", "SUPREMEIND.BO", "TATAELXSI.NS", "TATATECH.NS", "TCS.NS", "TECHM.NS", "TIMKEN.NS",
                "TITAN.NS", "TRITURBINE.NS", "TIINDIA.NS", "UNITDSPR.BO", "VGUARD.NS", "MANYAVAR.NS",
                "VINATIORGA.NS", "WIPRO.NS", "ZYDUSLIFE.NS", "CUMMINSIND.NS", "CYIENT.NS", "DATAPATTNS.NS", "DEEPAKNTR.NS",
                "DIVISLAB.NS", "LALPATHLAB.NS", "RDY", "ELGIEQUIP.NS", "EMAMILTD.NS", "FIVESTAR.BO", "GRINFRA.NS",
                "GILLETTE.NS", "GLAXO.NS", "GODFRYPHLP.NS", "GRINDWELL.NS", "HAVELLS.NS", "HCLTECH.NS", "HAL.BO",
                "HONAUT.BO", "IRCTC.NS", "ISEC.BO", "INFY.NS", "IPCALAB.BO", "ABBOTINDIA.NS", "ADANIPOWER.NS",
                "AFFLE.BO", "AIAENG.BO", "AJANTPHARM.BO", "APLLTD.BO", "ALKEM.BO", "ARE&M.NS", "ANANDRATHI.BO",
                "APARINDS.BO", "ASIANPAINT.NS", "ASTRAL.NS", "ASTRAZEN.NS", "BAJFINANCE.NS", "BASF.NS", "BAYERCROP.BO",
                "BERGEPAINT.BO", "BDL.NS", "BEL.NS", "BSOFT.BO", "CDSL.NS", "CAMS.NS", "CARBORUNIV.BO", "CASTROLIND.NS",
                "CHAMBLFERT.BO", "COALINDIA.NS", "COFORGE.BO", "COLPAL.NS", "CONCORDBIO.BO", "COROMANDEL.BO", "CREDITACC.BO",
                "PNCINFRA.NS", "INDIASHLTR.NS", "RAYMOND.NS", "KAMAHOLD.BO", "BENGALASM.BO", "CHOICEIN.NS", "GRAVITA.NS",
                "HGINFRA.NS", "JKPAPER.NS", "MTARTECH.NS", "HAPPSTMNDS.NS", "SARDAEN.NS", "WELENT.NS", "LTFOODS.NS",
                "GESHIP.NS", "SHRIPISTON.NS", "SHAREINDIA.NS", "CYIENTDLM.NS", "VTL.NS", "EASEMYTRIP.NS", "LLOYDSME.NS",
                "ROUTE.NS", "VAIBHAVGBL.NS", "GOKEX.NS", "USHAMART.NS", "EIDPARRY.NS", "KIRLOSBROS.NS", "MANINFRA.NS",
                "CMSINFO.NS", "RALLIS.NS", "GHCL.NS", "NEULANDLAB.NS", "SPLPETRO.NS", "MARKSANS.NS", "NAVINFLUOR.NS",
                "ELECON.NS", "TANLA.NS", "KFINTECH.NS", "TIPSINDLTD.NS", "ACI.NS", "SURYAROSNI.NS", "GPIL.NS",
                "GMDCLTD.NS", "MAHSEAMLES.NS", "TDPOWERSYS.NS", "TECHNOE.NS", "JLHL.NS"
            ]

            # Function to fetch and process stock data
            def get_stock_data(ticker_symbols, period):
                try:
                    stock_data = {}
                    for ticker_symbol in ticker_symbols:
                        df = yf.download(ticker_symbol, period=period)
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
                macd = ta.trend.MACD(df['Close'])
                df['MACD'] = macd.macd()
                df['MACD_Signal'] = macd.macd_signal()
                df['MACD_Histogram'] = macd.macd_diff()

                # Calculate ADX
                adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'])
                df['ADX'] = adx.adx()

                # Calculate Parabolic SAR
                psar = pta.psar(df['High'], df['Low'], df['Close'])
                df['Parabolic_SAR'] = psar['PSARl_0.02_0.2']

                # Calculate RSI
                rsi = ta.momentum.RSIIndicator(df['Close'])
                df['RSI'] = rsi.rsi()

                # Calculate Volume Moving Averages
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
                if df is None or df.empty:
                    return pd.DataFrame(results)
                
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

            # Fetch data and calculate indicators for each stock
            stock_data = get_stock_data(stock_tickers, period='3mo')

            if submenu == "Bollinger":
                # Define first set of conditions
                first_conditions = [
                    ('Close', '>', 'Bollinger_Low'),
                    ('Close', '<', 'Bollinger_Middle')
                ]

                # Query stocks based on the first set of conditions
                first_query_df = query_stocks(stock_data, first_conditions, stock_tickers)

                # Generate insights
                if not first_query_df.empty:
                    second_query_df = first_query_df[
                        (first_query_df['RSI'] < 40)
                    ]

                    # Display the final results
                    st.write("Stocks in an uptrend with high volume:")
                    st.dataframe(second_query_df.round(2))
                else:
                    st.write("No stocks met the first set of conditions.")

            elif submenu == "Macd":
                # Define first set of conditions
                first_conditions = [
                    ('MACD', '>', 'MACD_Signal')
                ]

                # Query stocks based on the first set of conditions
                first_query_df = query_stocks(stock_data, first_conditions, stock_tickers)

                # Generate insights
                if not first_query_df.empty:
                    second_query_df = first_query_df[
                        (first_query_df['RSI'] < 65) & (first_query_df['RSI'] > 30) & (first_query_df['ADX'] > 20) & (first_query_df['MACD'] > 0)
                    ]

                    # Display the final results
                    st.write("Stocks in an uptrend with high volume:")
                    st.dataframe(second_query_df.round(2))
                else:
                    st.write("No stocks met the first set of conditions.")

            # Dropdown for analysis type
            # Create two columns for date input
            col1, col2 = st.columns(2)
            with col1:
                
                selected_stock = st.selectbox("Select Stock", second_query_df['Ticker'].tolist())
            with col2:
                analysis_type = st.selectbox("Select Analysis Type", ["Trend Analysis", "Volume Analysis", "Support & Resistance Levels"])

            # Create two columns for date input
            col1, col2 = st.columns(2)

            # Set up the start and end date inputs with default values

            with col1:
                START = st.date_input('Start Date', pd.to_datetime("01-06-23"))

            with col2:
                END = st.date_input('End Date', pd.to_datetime("today"))


            # If a stock is selected, plot its data with the selected indicators
            if selected_stock:
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
                df = load_data(selected_stock)

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

                        # Generate buy/sell signals using advanced methods
                        def advanced_signals(data):
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

                            # Hilbert Transform
                            def apply_hilbert_transform(prices):
                                analytic_signal = hilbert(prices)
                                amplitude_envelope = np.abs(analytic_signal)
                                instantaneous_phase = np.unwrap(np.angle(analytic_signal))
                                instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi)
                                return amplitude_envelope, instantaneous_phase, instantaneous_frequency

                            amplitude_envelope, instantaneous_phase, instantaneous_frequency = apply_hilbert_transform(prices)

                            # Generate Signals
                            def generate_signals(prices, reconstructed_signal, amplitude_envelope,
                                                instantaneous_frequency, df):
                                buy_signals = []
                                sell_signals = []
                                for i in range(2, len(prices) - 1):
                                    if (reconstructed_signal[i] > reconstructed_signal[i - 1] and
                                            reconstructed_signal[i - 1] < reconstructed_signal[i - 2] and
                                            instantaneous_frequency[i - 1] < instantaneous_frequency[i - 2] and
                                            amplitude_envelope[i] > amplitude_envelope[i - 1] and
                                            df['20_MA'][i] > df['50_MA'][i] and df['RSI'][i] < 70):
                                        buy_signals.append((i, prices[i]))
                                    elif (reconstructed_signal[i] < reconstructed_signal[i - 1] and
                                        reconstructed_signal[i - 1] > reconstructed_signal[i - 2] and
                                        instantaneous_frequency[i - 1] > instantaneous_frequency[i - 2] and
                                        amplitude_envelope[i] < amplitude_envelope[i - 1] and
                                        df['20_MA'][i] < df['50_MA'][i] and df['RSI'][i] > 30):
                                        sell_signals.append((i, prices[i]))
                                return buy_signals, sell_signals

                            buy_signals, sell_signals = generate_signals(prices, reconstructed_signal,
                                                                        amplitude_envelope, instantaneous_frequency,
                                                                        data)
                            return buy_signals, sell_signals

                        buy_signals, sell_signals = advanced_signals(df)

                        if analysis_type == "Trend Analysis":
                            st.subheader("Trend Analysis")

                            indicators = st.multiselect(
                                "Select Indicators",
                                ['Close', '20_MA', '50_MA', '200_MA', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'RSI',
                                'Signal', 'ADX',
                                'Parabolic_SAR', 'Bollinger_High', 'Bollinger_Low', 'Bollinger_Middle'],
                                default=['Close', 'Signal']
                            )
                            timeframe = st.radio(
                                "Select Timeframe",
                                ['15 days', '30 days', '90 days', '180 days', '1 year', 'All'],
                                index=2,
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
                            colors = {'Close': 'blue', '20_MA': 'orange', '50_MA': 'green', '200_MA': 'red',
                                    'MACD': 'purple',
                                    'MACD_Signal': 'brown', 'RSI': 'pink', 'Signal': 'black', 'ADX': 'magenta',
                                    'Parabolic_SAR': 'yellow', 'Bollinger_High': 'black', 'Bollinger_Low': 'cyan',
                                    'Bollinger_Middle': 'grey'}

                            for indicator in indicators:
                                if indicator == 'Signal':
                                    # Plot buy and sell signals
                                    fig.add_trace(
                                        go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close',
                                                line=dict(color=colors['Close'])))
                                    buy_signal_points = [df.iloc[bs[0]] for bs in buy_signals if bs[0] < len(df)]
                                    sell_signal_points = [df.iloc[ss[0]] for ss in sell_signals if ss[0] < len(df)]
                                    fig.add_trace(
                                        go.Scatter(x=[point['Date'] for point in buy_signal_points],
                                                y=[point['Close'] for point in buy_signal_points], mode='markers',
                                                name='Buy Signal',
                                                marker=dict(color='green', symbol='triangle-up')))
                                    fig.add_trace(
                                        go.Scatter(x=[point['Date'] for point in sell_signal_points],
                                                y=[point['Close'] for point in sell_signal_points], mode='markers',
                                                name='Sell Signal',
                                                marker=dict(color='red', symbol='triangle-down')))
                                elif indicator == 'MACD_Histogram':
                                    fig.add_trace(
                                        go.Bar(x=df['Date'], y=df[indicator], name=indicator, marker_color='gray'))
                                else:
                                    fig.add_trace(
                                        go.Scatter(x=df['Date'], y=df[indicator], mode='lines', name=indicator,
                                                line=dict(color=colors.get(indicator, 'black'))))

                            st.plotly_chart(fig)

                        elif analysis_type == "Volume Analysis":
                            st.subheader("Volume Analysis")
                            volume_indicators = st.multiselect(
                                "Select Volume Indicators",
                                ['Volume', 'Volume_MA_20', 'OBV', 'Volume_Oscillator', 'CMF'],
                                default=['Volume']
                            )
                            volume_timeframe = st.radio(
                                "Select Timeframe",
                                ['15 days', '30 days', '90 days', '180 days', '1 year', 'All'],
                                index=2,
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

                        elif analysis_type == "Support & Resistance Levels":
                            st.subheader("Support & Resistance Levels")
                            sr_indicators = st.multiselect(
                                "Select Indicators",
                                ['Close', '20_MA', '50_MA', '200_MA', 'Support', 'Resistance', 'Support_Trendline',
                                'Resistance_Trendline', 'Pivot', 'R1', 'S1', 'R2', 'S2'],
                                default=['Close']
                            )
                            sr_timeframe = st.radio(
                                "Select Timeframe",
                                ['15 days', '30 days', '90 days', '180 days', '1 year', 'All'],
                                index=2,
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

                    else:
                        st.write("Not enough data points for technical analysis.")
                    

        

        elif choice == f"{st.session_state.username}'s Watchlist":
        # 'watchlist' code -------------------------------------------------------------------------------------------
            st.header(f"{st.session_state.username}'s Watchlist")
            user_id = session.query(User.id).filter_by(email=st.session_state.email).first()[0]
            watchlist = session.query(Watchlist).filter_by(user_id=user_id).all()

            # Add new ticker to watchlist
            new_ticker = st.text_input("Add a new ticker to your watchlist")
            if st.button("Add Ticker"):
                if not session.query(Watchlist).filter_by(user_id=user_id, ticker=new_ticker).first():
                    new_watchlist_entry = Watchlist(user_id=user_id, ticker=new_ticker)
                    session.add(new_watchlist_entry)
                    session.commit()
                    st.success(f"{new_ticker} added to your watchlist!")
                    # Refresh watchlist data
                    watchlist = session.query(Watchlist).filter_by(user_id=user_id).all()
                else:
                    st.warning(f"{new_ticker} is already in your watchlist.")

            # Display watchlist
            if watchlist:
                watchlist_data = {entry.ticker: yf.download(entry.ticker, period='1d').iloc[-1]['Close'] for entry in
                                  watchlist}
                watchlist_df = pd.DataFrame(list(watchlist_data.items()), columns=['Ticker', 'Close'])
                st.write("Your Watchlist:")
                st.dataframe(watchlist_df)

                # Option to remove ticker from watchlist
                ticker_to_remove = st.selectbox("Select a ticker to remove", [entry.ticker for entry in watchlist])
                if st.button("Remove Ticker"):
                    session.query(Watchlist).filter_by(user_id=user_id, ticker=ticker_to_remove).delete()
                    session.commit()
                    st.success(f"{ticker_to_remove} removed from your watchlist.")
                    st.experimental_rerun()  # Refresh the app to reflect changes
            else:
                st.write("Your watchlist is empty.")
        elif choice == f"{st.session_state.username}'s Portfolio":
        # 'Portifolio' code -------------------------------------------------------------------------------------------
            st.header(f"{st.session_state.username}'s Portfolio")
            user_id = session.query(User.id).filter_by(email=st.session_state.email).first()[0]
            portfolio = session.query(Portfolio).filter_by(user_id=user_id).all()

            # Add new stock to portfolio
            st.subheader("Add to Portfolio")
            # Create three columns
            col1, col2, col3 = st.columns(3)
            with col1:
                new_ticker = st.text_input("Ticker Symbol")
            with col2:
                shares = st.number_input("Number of Shares", min_value=0.0, step=0.01)
            with col3:
                bought_price = st.number_input("Bought Price per Share", min_value=0.0, step=0.01)
            if st.button("Add to Portfolio"):
                if not session.query(Portfolio).filter_by(user_id=user_id, ticker=new_ticker).first():
                    new_portfolio_entry = Portfolio(user_id=user_id, ticker=new_ticker, shares=shares,
                                                    bought_price=bought_price)
                    session.add(new_portfolio_entry)
                    session.commit()
                    st.success(f"{new_ticker} added to your portfolio!")
                    # Refresh portfolio data
                    portfolio = session.query(Portfolio).filter_by(user_id=user_id).all()
                else:
                    st.warning(f"{new_ticker} is already in your portfolio.")

            # Display portfolio
            if portfolio:
                portfolio_data = []
                for entry in portfolio:
                    current_data = yf.download(entry.ticker, period='1d')
                    last_price = current_data['Close'].iloc[-1]
                    invested_value = entry.shares * entry.bought_price
                    current_value = entry.shares * last_price
                    p_l = current_value - invested_value
                    p_l_percent = (p_l / invested_value) * 100
                    portfolio_data.append({
                        "Ticker": entry.ticker,
                        "Shares": entry.shares,
                        "Bought Price": entry.bought_price,
                        "Invested Value": invested_value,
                        "Last Traded Price": last_price,
                        "Current Value": current_value,
                        "P&L (%)": p_l_percent
                    })
                portfolio_df = pd.DataFrame(portfolio_data)
                st.write("Your Portfolio:")
                st.dataframe(portfolio_df)

                # Generate donut chart
                labels = portfolio_df['Ticker']
                values = portfolio_df['Current Value']
                fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
                fig.update_layout(title_text="Portfolio Distribution")
                st.plotly_chart(fig)

                # Option to remove stock from portfolio
                ticker_to_remove = st.selectbox("Select a ticker to remove", [entry.ticker for entry in portfolio])
                if st.button("Remove from Portfolio"):
                    session.query(Portfolio).filter_by(user_id=user_id, ticker=ticker_to_remove).delete()
                    session.commit()
                    st.success(f"{ticker_to_remove} removed from your portfolio.")
                    st.experimental_rerun()  # Refresh the app to reflect changes
            else:
                st.write("Your portfolio is empty.")
       
        elif choice == "Stock Comparison":
        # 'Stock Comparison' code -------------------------------------------------------------------------------------------


            # List of stock tickers
            tickers = [
                "ITC.NS", "JBCHEPHARM.BO", "JWL.BO", "JYOTHYLAB.BO", "KPRMILL.NS", "KAJARIACER.NS", "KEI.BO",
                "LTTS.NS", "LTIM.NS", "MANKIND.NS", "MARICO.NS", "METROBRAND.BO", "MOTILALOFS.NS", "MPHASIS.NS",
                "MUTHOOTFIN.NS", "NH.NS", "NAVINFLUOR.NS", "NAM-INDIA.BO", "NMDC.NS", "OFSS.NS", "PGHH.NS",
                "PIIND.NS", "PAGEIND.NS", "PERSISTENT.NS", "PETRONET.NS", "PFIZER.NS", "PIDILITIND.NS",
                "POLYMED.NS", "POLYCAB.NS", "RRKABEL.NS", "RVNL.NS", "RATNAMANI.NS", "RITES.NS", "SANOFI.NS",
                "SCHAEFFLER.NS", "SKFINDIA.NS", "SOLARINDS.NS", "SONACOMS.NS", "SUMICHEM.NS", "SUNTV.NS",
                "SUNDRMFAST.NS", "SUPREMEIND.BO", "TATAELXSI.NS", "TATATECH.NS", "TCS.NS", "TECHM.NS", "TIMKEN.NS",
                "TITAN.NS", "TRITURBINE.NS", "TIINDIA.NS", "UNITDSPR.BO", "VGUARD.NS", "MANYAVAR.NS",
                "VINATIORGA.NS", "WIPRO.NS", "ZYDUSLIFE.NS", "CUMMINSIND.NS", "CYIENT.NS", "DATAPATTNS.NS", "DEEPAKNTR.NS",
                "DIVISLAB.NS", "LALPATHLAB.NS", "RDY", "ELGIEQUIP.NS", "EMAMILTD.NS", "FIVESTAR.BO", "GRINFRA.NS",
                "GILLETTE.NS", "GLAXO.NS", "GODFRYPHLP.NS", "GRINDWELL.NS", "HAVELLS.NS", "HCLTECH.NS", "HAL.BO",
                "HONAUT.BO", "IRCTC.NS", "ISEC.BO", "INFY.NS", "IPCALAB.BO", "ABBOTINDIA.NS", "ADANIPOWER.NS",
                "AFFLE.BO", "AIAENG.BO", "AJANTPHARM.BO", "APLLTD.BO", "ALKEM.BO", "ARE&M.NS", "ANANDRATHI.BO",
                "APARINDS.BO", "ASIANPAINT.NS", "ASTRAL.NS", "ASTRAZEN.NS", "BAJFINANCE.NS", "BASF.NS", "BAYERCROP.BO",
                "BERGEPAINT.BO", "BDL.NS", "BEL.NS", "BSOFT.BO", "CDSL.NS", "CAMS.NS", "CARBORUNIV.BO", "CASTROLIND.NS",
                "CHAMBLFERT.BO", "COALINDIA.NS", "COFORGE.BO", "COLPAL.NS", "CONCORDBIO.BO", "COROMANDEL.BO", "CREDITACC.BO",
                "PNCINFRA.NS", "INDIASHLTR.NS", "RAYMOND.NS", "KAMAHOLD.BO", "BENGALASM.BO", "CHOICEIN.NS", "GRAVITA.NS",
                "HGINFRA.NS", "JKPAPER.NS", "MTARTECH.NS", "HAPPSTMNDS.NS", "SARDAEN.NS", "WELENT.NS", "LTFOODS.NS",
                "GESHIP.NS", "SHRIPISTON.NS", "SHAREINDIA.NS", "CYIENTDLM.NS", "VTL.NS", "EASEMYTRIP.NS", "LLOYDSME.NS",
                "ROUTE.NS", "VAIBHAVGBL.NS", "GOKEX.NS", "USHAMART.NS", "EIDPARRY.NS", "KIRLOSBROS.NS", "MANINFRA.NS",
                "CMSINFO.NS", "RALLIS.NS", "GHCL.NS", "NEULANDLAB.NS", "SPLPETRO.NS", "MARKSANS.NS", "NAVINFLUOR.NS",
                "ELECON.NS", "TANLA.NS", "KFINTECH.NS", "TIPSINDLTD.NS", "ACI.NS", "SURYAROSNI.NS", "GPIL.NS",
                "GMDCLTD.NS", "MAHSEAMLES.NS", "TDPOWERSYS.NS", "TECHNOE.NS", "JLHL.NS"
            ]

            # Load stock data
            @st.cache_data
            def load_data(ticker):
                data = yf.download(ticker, period='3mo')
                data.reset_index(inplace=True)
                return data

            # Calculate technical indicators
            def calculate_technical_indicators(df, index_df):
                # Moving averages
                df['20_SMA'] = ta.trend.sma_indicator(df['Close'], window=20)
                df['20_EMA'] = ta.trend.ema_indicator(df['Close'], window=20)
                df['20_WMA'] = ta.trend.wma_indicator(df['Close'], window=20)

                # Momentum Indicators
                df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
                df['%K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
                df['%D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
                df['MACD'] = ta.trend.macd(df['Close'])
                df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
                df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

                # Volume Indicators
                df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
                df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'], window=14)
                df['A/D Line'] = ta.volume.acc_dist_index(df['High'], df['Low'], df['Close'], df['Volume'])

                # Volatility Indicators
                df['BB_High'] = ta.volatility.bollinger_hband(df['Close'])
                df['BB_Middle'] = ta.volatility.bollinger_mavg(df['Close'])
                df['BB_Low'] = ta.volatility.bollinger_lband(df['Close'])
                df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
                df['Std Dev'] = ta.volatility.bollinger_wband(df['Close'])

                # Trend Indicators
                psar = pta.psar(df['High'], df['Low'], df['Close'])
                df['Parabolic_SAR'] = psar['PSARl_0.02_0.2']
                df['Ichimoku_a'] = ta.trend.ichimoku_a(df['High'], df['Low'])
                df['Ichimoku_b'] = ta.trend.ichimoku_b(df['High'], df['Low'])
                df['Ichimoku_base'] = ta.trend.ichimoku_base_line(df['High'], df['Low'])
                df['Ichimoku_conv'] = ta.trend.ichimoku_conversion_line(df['High'], df['Low'])

                # Support and Resistance Levels
                df['Pivot Points'] = (df['High'] + df['Low'] + df['Close']) / 3

                # Price Oscillators
                df['ROC'] = ta.momentum.roc(df['Close'], window=12)
                df['DPO'] = ta.trend.dpo(df['Close'], window=20)
                df['Williams %R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14)

                # Market Breadth Indicators
                df['Advances'] = df['Close'].diff().apply(lambda x: 1 if x > 0 else 0)
                df['Declines'] = df['Close'].diff().apply(lambda x: 1 if x < 0 else 0)
                df['McClellan Oscillator'] = (df['Advances'] - df['Declines']).rolling(window=19).mean() - (df['Advances'] - df['Declines']).rolling(window=39).mean()
                df['TRIN'] = (df['Advances'] / df['Declines']) / (df['Volume'][df['Advances'] > 0].sum() / df['Volume'][df['Declines'] > 0].sum())
                df['Advance-Decline Line'] = df['Advances'].cumsum() - df['Declines'].cumsum()

                # Relative Performance Indicators
                df['Price-to-Volume Ratio'] = df['Close'] / df['Volume']
                df['Relative Strength Comparison'] = df['Close'] / index_df['Close']
                df['Performance Relative to an Index'] = df['Close'].pct_change().cumsum() - index_df['Close'].pct_change().cumsum()

                return df

            # Function to add range buttons to the plot
            def add_range_buttons(fig):
                fig.update_layout(
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=list([
                                dict(count=7, label="7d", step="day", stepmode="backward"),
                                dict(count=14, label="14d", step="day", stepmode="backward"),
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=3, label="3m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(step="all")
                            ])
                        ),
                        rangeslider=dict(visible=True)
                    )
                )

            # Plotly visualization functions
            def plot_indicator(df, indicator, title, yaxis_title='Price', secondary_y=False):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))
                fig.add_trace(go.Scatter(x=df['Date'], y=df[indicator], mode='lines', name=indicator, yaxis="y2" if secondary_y else "y1"))
                
                if secondary_y:
                    fig.update_layout(
                        yaxis2=dict(
                            title=indicator,
                            overlaying='y',
                            side='right'
                        )
                    )
                
                fig.update_layout(title=title, xaxis_title='Date', yaxis_title=yaxis_title)
                add_range_buttons(fig)
                st.plotly_chart(fig)

            # Plotly visualization for trendlines
            def plot_trendlines(df):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))

                # Adding trendlines using linear regression (advanced method)
                x = np.arange(len(df))
                coef = np.polyfit(x, df['Close'], 1)
                trend = np.poly1d(coef)
                fig.add_trace(go.Scatter(x=df['Date'], y=trend(x), mode='lines', name='Trendline', line=dict(color='red', dash='dash')))

                fig.update_layout(title='Trendlines', xaxis_title='Date', yaxis_title='Price')
                add_range_buttons(fig)
                st.plotly_chart(fig)

            # Plotly visualization for Fibonacci retracement levels
            def plot_fibonacci_retracement(df):
                high = df['High'].max()
                low = df['Low'].min()

                diff = high - low
                levels = [high, high - 0.236 * diff, high - 0.382 * diff, high - 0.5 * diff, high - 0.618 * diff, low]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))

                for level in levels:
                    fig.add_trace(go.Scatter(x=[df['Date'].iloc[0], df['Date'].iloc[-1]],
                                            y=[level, level],
                                            mode='lines', name=f'Level {level}', line=dict(dash='dash')))

                fig.update_layout(title='Fibonacci Retracement Levels', xaxis_title='Date', yaxis_title='Price')
                add_range_buttons(fig)
                st.plotly_chart(fig)

            # Plotly visualization for Gann fan lines
            def plot_gann_fan_lines(df):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))

                # Adding Gann fan lines (simple example, for more advanced lines use a proper method)
                for i in range(1, 5):
                    fig.add_trace(go.Scatter(x=[df['Date'].iloc[0], df['Date'].iloc[-1]],
                                            y=[df['Close'].iloc[0], df['Close'].iloc[0] + i * (df['Close'].iloc[-1] - df['Close'].iloc[0]) / 4],
                                            mode='lines', name=f'Gann Fan {i}', line=dict(dash='dash')))

                fig.update_layout(title='Gann Fan Lines', xaxis_title='Date', yaxis_title='Price')
                add_range_buttons(fig)
                st.plotly_chart(fig)

            # Plotly visualization for chart patterns
            def plot_chart_patterns(df, pattern):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))

                # Adding example chart patterns (simple example, for more advanced patterns use a proper method)
                if pattern == 'Head and Shoulders':
                    fig.add_trace(go.Scatter(x=[df['Date'].iloc[0], df['Date'].iloc[len(df)//3], df['Date'].iloc[2*len(df)//3], df['Date'].iloc[-1]],
                                            y=[df['Close'].iloc[0], df['Close'].iloc[len(df)//3], df['Close'].iloc[2*len(df)//3], df['Close'].iloc[-1]],
                                            mode='lines+markers', name='Head and Shoulders', line=dict(color='orange')))

                elif pattern == 'Double Tops and Bottoms':
                    fig.add_trace(go.Scatter(x=[df['Date'].iloc[0], df['Date'].iloc[len(df)//2], df['Date'].iloc[-1]],
                                            y=[df['Close'].iloc[0], df['Close'].iloc[len(df)//2], df['Close'].iloc[-1]],
                                            mode='lines+markers', name='Double Tops and Bottoms', line=dict(color='green')))

                elif pattern == 'Flags and Pennants':
                    fig.add_trace(go.Scatter(x=[df['Date'].iloc[0], df['Date'].iloc[-1]],
                                            y=[df['Close'].iloc[0], df['Close'].iloc[-1]],
                                            mode='lines', name='Flags and Pennants', line=dict(color='purple', dash='dash')))

                fig.update_layout(title=f'{pattern}', xaxis_title='Date', yaxis_title='Price')
                st.plotly_chart(fig)

            # Plotly visualization for McClellan Oscillator
            def plot_mcclellan_oscillator(df):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['McClellan Oscillator'], mode='lines', name='McClellan Oscillator'))
                fig.update_layout(title='McClellan Oscillator', xaxis_title='Date', yaxis_title='Value')
                add_range_buttons(fig)
                st.plotly_chart(fig)

            # Plotly visualization for TRIN
            def plot_trin(df):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['TRIN'], mode='lines', name='TRIN'))
                fig.update_layout(title='Arms Index (TRIN)', xaxis_title='Date', yaxis_title='Value')
                add_range_buttons(fig)
                st.plotly_chart(fig)

            # Detect chart patterns
            def detect_patterns(df):
                patterns = []

                # Head and Shoulders
                hs_pattern = detect_head_and_shoulders(df)
                if hs_pattern:
                    patterns.append(hs_pattern)

                # Double Tops and Bottoms
                dt_pattern = detect_double_tops_and_bottoms(df)
                if dt_pattern:
                    patterns.append(dt_pattern)

                # Flags and Pennants
                fp_pattern = detect_flags_and_pennants(df)
                if fp_pattern:
                    patterns.append(fp_pattern)

                return patterns

            # Placeholder function to detect head and shoulders pattern
            def detect_head_and_shoulders(df):
                # Simplified logic to detect head and shoulders pattern
                pattern_detected = False
                for i in range(2, len(df)-2):
                    if df['High'].iloc[i] > df['High'].iloc[i-1] and df['High'].iloc[i] > df['High'].iloc[i+1] and \
                    df['High'].iloc[i-1] < df['High'].iloc[i-2] and df['High'].iloc[i+1] < df['High'].iloc[i+2]:
                        pattern_detected = True
                        break
                if pattern_detected:
                    return ("Head and Shoulders", "Sell", "Regular Head and Shoulders pattern detected.")
                return None

            # Placeholder function to detect double tops and bottoms
            def detect_double_tops_and_bottoms(df):
                # Simplified logic to detect double tops and bottoms
                pattern_detected = False
                for i in range(1, len(df)-1):
                    if df['High'].iloc[i] == df['High'].iloc[i-1] and df['High'].iloc[i] == df['High'].iloc[i+1]:
                        pattern_detected = True
                        break
                if pattern_detected:
                    return ("Double Tops", "Sell", "Double Tops pattern detected.")
                return None

            # Placeholder function to detect flags and pennants
            def detect_flags_and_pennants(df):
                # Simplified logic to detect flags and pennants
                pattern_detected = False
                for i in range(1, len(df)-1):
                    if df['Close'].iloc[i] > df['Close'].iloc[i-1] and df['Close'].iloc[i] > df['Close'].iloc[i+1]:
                        pattern_detected = True
                        break
                if pattern_detected:
                    return ("Flags and Pennants", "Buy", "Bullish Flag pattern detected.")
                return None

            # Streamlit UI
            st.title('Stock Technical Indicators Comparison')

            # Sidebar for user input
            st.sidebar.header('User Input')
            selected_tickers = st.sidebar.multiselect('Select Tickers', tickers)

            if selected_tickers:
                index_ticker = st.sidebar.selectbox('Select Index Ticker for Relative Performance', tickers, index=0)
                index_data = load_data(index_ticker).copy()
                
                all_data = {}

                # Load and process data for each selected ticker
                for ticker in selected_tickers:
                    data_load_state = st.text(f'Loading data for {ticker}...')
                    data = load_data(ticker).copy()
                    data = calculate_technical_indicators(data, index_data)
                    all_data[ticker] = data
                    data_load_state.text(f'Loading data for {ticker}...done!')

                # Display raw data and technical indicators side-by-side
                st.subheader('Technical Indicators Comparison')
                indicators = st.selectbox("Select Technical Indicator", 
                                        ['20_SMA', '20_EMA', '20_WMA', 'RSI', 'Stochastic Oscillator', 'MACD', 'OBV', 'VWAP', 'A/D Line', 
                                        'Bollinger Bands', 'ATR', 'Standard Deviation', 'Parabolic SAR', 'Pivot Points', 'ROC', 'DPO', 
                                        'Williams %R', 'Ichimoku Cloud', 'McClellan Oscillator', 'TRIN', 'Advance-Decline Line', 
                                        'Price-to-Volume Ratio', 'Relative Strength Comparison', 'Performance Relative to an Index'])

                cols = st.columns(2)  # Create two columns for the layout
                
                for idx, ticker in enumerate(selected_tickers):
                    col = cols[idx % 2]  # Alternate between columns
                    data = all_data[ticker]
                    fig = go.Figure()
                    
                    # Add close price to the primary y-axis
                    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name=f'{ticker} Close', yaxis='y1'))

                    if indicators in data.columns:
                        fig.add_trace(go.Scatter(x=data['Date'], y=data[indicators], mode='lines', name=f'{ticker} {indicators}', yaxis='y2'))
                    elif indicators == 'Stochastic Oscillator':
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['%K'], mode='lines', name=f'{ticker} %K', yaxis='y2'))
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['%D'], mode='lines', name=f'{ticker} %D', yaxis='y2'))
                    elif indicators == 'MACD':
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['MACD'], mode='lines', name=f'{ticker} MACD', yaxis='y2'))
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['MACD_Signal'], mode='lines', name=f'{ticker} MACD Signal', yaxis='y2'))
                        fig.add_trace(go.Bar(x=data['Date'], y=data['MACD_Hist'], name=f'{ticker} MACD Hist', marker_color='gray', opacity=0.5, yaxis='y2'))
                    elif indicators == 'Bollinger Bands':
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_High'], mode='lines', name=f'{ticker} BB High', yaxis='y2'))
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_Middle'], mode='lines', name=f'{ticker} BB Middle', yaxis='y2'))
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_Low'], mode='lines', name=f'{ticker} BB Low', yaxis='y2'))
                    elif indicators == 'Ichimoku Cloud':
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['Ichimoku_a'], mode='lines', name=f'{ticker} Ichimoku_a', yaxis='y2'))
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['Ichimoku_b'], mode='lines', name=f'{ticker} Ichimoku_b', yaxis='y2'))
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['Ichimoku_base'], mode='lines', name=f'{ticker} Ichimoku_base', yaxis='y2'))
                        fig.add_trace(go.Scatter(x=data['Date'], y=data['Ichimoku_conv'], mode='lines', name=f'{ticker} Ichimoku_conv', yaxis='y2'))

                    # Update layout with dual y-axes and time range selector
                    fig.update_layout(
                        title=f'{indicators} for {ticker}',
                        xaxis_title='Date',
                        yaxis=dict(title='Close Price', side='left'),
                        yaxis2=dict(title=indicators, side='right', overlaying='y'),
                        legend_title='Indicators',
                        xaxis=dict(
                            rangeselector=dict(
                                buttons=list([
                                    dict(count=7, label='7d', step='day', stepmode='backward'),
                                    dict(count=14, label='14d', step='day', stepmode='backward'),
                                    dict(count=1, label='1m', step='month', stepmode='backward'),
                                    dict(count=2, label='2m', step='month', stepmode='backward'),
                                    dict(step='all', label='All Time')
                                ])
                            ),
                            rangeslider=dict(visible=True),
                            type='date',
                            range=[data['Date'].iloc[-7], data['Date'].iloc[-1]]  # Default to last 7 days
                        )
                    )
                    
                    col.plotly_chart(fig)

