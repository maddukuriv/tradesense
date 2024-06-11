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
    menu_options = ["Markets", "Stock Screener", "Technical Analysis", "Stock Price Forecasting", "Stock Watch",
                    "Strategy Backtesting", f"{st.session_state.username}'s Watchlist",
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

 


    st.subheader("Major Indices")

    # Create three columns
    col1, col2, col3 = st.columns(3)

    # Set up the start and end date inputs
    with col1:
        # List of stock symbols
        stock_symbols = ["^BSESN", "BSE-500.BO", "^BSEMD", "^BSESMLCAP", "^NSEI", "^NSMIDCP", "^NSEMDCP", "^NSESCP"]
        
        # Auto-suggestion using selectbox
        ticker = st.selectbox("Enter Stock symbol", stock_symbols)
        st.write(f"You selected: {ticker}")

    with col2:
        START = st.date_input('Start Date', pd.to_datetime("2020-01-01"))

    with col3:
        END = st.date_input('End Date', pd.to_datetime("today"))

    # Function to get stock data and calculate moving averages
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
            plot_bgcolor='dark grey',  # Changed from 'dark grey' to 'darkgrey'
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

    # Get data and create the figure
    if ticker and START and END:
        data = get_stock_data(ticker, START, END)
        fig = create_figure(data, ['Close', 'MA_15', 'MA_50'], f"{ticker} Stock Prices")
        st.plotly_chart(fig)

    st.subheader("Top Gainers and Losers")
    # List of tickers
    tickers = ["ABBOTINDIA.NS", "ADANIPOWER.NS", "AFFLE.BO", "AIAENG.BO", "AJANTPHARM.BO", "APLLTD.BO", "ALKEM.BO",
               "ARE&M.NS", "ANANDRATHI.BO", "APARINDS.BO", "ASIANPAINT.NS", "ASTRAL.NS", "ASTRAZEN.NS", "BAJFINANCE.NS",
               "BASF.NS", "BAYERCROP.BO", "BERGEPAINT.BO", "BDL.NS", "BEL.NS", "BSOFT.BO", "CDSL.NS", "CAMS.NS",
               "CARBORUNIV.BO", "CASTROLIND.NS", "CHAMBLFERT.BO", "COALINDIA.NS", "COFORGE.BO", "COLPAL.NS",
               "CONCORDBIO.BO", "COROMANDEL.BO", "CREDITACC.BO", "CUMMINSIND.NS", "CYIENT.NS", "DATAPATTNS.NS",
               "DEEPAKNTR.NS", "DIVISLAB.NS", "LALPATHLAB.NS", "RDY", "ELGIEQUIP.NS", "EMAMILTD.NS", "FIVESTAR.BO",
               "GRINFRA.NS", "GILLETTE.NS", "GLAXO.NS", "GODFRYPHLP.NS", "GRINDWELL.NS", "HAVELLS.NS", "HCLTECH.NS",
               "HAL.BO", "HONAUT.BO", "IRCTC.NS", "ISEC.BO", "INFY.NS", "IPCALAB.BO", "ITC.NS", "JBCHEPHARM.BO",
               "JWL.BO",
               "JYOTHYLAB.BO", "KPRMILL.NS", "KAJARIACER.NS", "KEI.BO", "LTTS.NS", "LTIM.NS", "MANKIND.NS", "MARICO.NS",
               "METROBRAND.BO", "MOTILALOFS.NS", "MPHASIS.NS", "MUTHOOTFIN.NS", "NH.NS", "NAVINFLUOR.NS",
               "NAM-INDIA.BO",
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
    percent_change_daily = (daily_change / data_daily.iloc[0]) * 100

    weekly_change = data_weekly.iloc[-1] - data_weekly.iloc[0]
    percent_change_weekly = (weekly_change / data_weekly.iloc[0]) * 100

    monthly_change = data_monthly.iloc[-1] - data_monthly.iloc[0]
    percent_change_monthly = (monthly_change / data_monthly.iloc[0]) * 100

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

    # Round off the % Change values
    df_daily['% Change'] = df_daily['% Change'].round(2)
    df_weekly['% Change'] = df_weekly['% Change'].round(2)
    df_monthly['% Change'] = df_monthly['% Change'].round(2)

    # Sort the DataFrames by '% Change'
    df_daily_sorted = df_daily.sort_values(by='% Change', ascending=True)
    df_weekly_sorted = df_weekly.sort_values(by='% Change', ascending=True)
    df_monthly_sorted = df_monthly.sort_values(by='% Change', ascending=True)


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




    # Dropdown menu to select the period
    heatmap_option = st.selectbox('Select to view:',
                                  ['Daily Gainers/Losers', 'Weekly Gainers/Losers', 'Monthly Gainers/Losers'])

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

    

    st.subheader("Volume Chart")
    # Function to fetch stock data and volume
    def get_volume_data(ticker, start_date, end_date):
        data = yf.download(ticker, start=start_date, end=end_date)
        return data['Volume'].sum()

    # Get start and end date inputs from user
    # Create two columns
    col1, col2 = st.columns(2)

    # Set up the start and end date inputs
    with col1:
        start_date = st.date_input('Start Date', datetime(2020, 1, 1), key='start_date')
    with col2:
        end_date = st.date_input('End Date', datetime.today(), key='end_date')
    # Fetch volume data for each stock
    volume_data = {}
    for ticker in tickers:
        volume = get_volume_data(ticker, start_date, end_date)
        volume_data[ticker] = volume

    # Convert the volume data into a DataFrame for visualization
    volume_df = pd.DataFrame(list(volume_data.items()), columns=['Ticker', 'Volume'])

    # Create a bar chart using Plotly
    fig = px.bar(volume_df, x='Ticker', y='Volume', title='Trading Volume of Stocks',
                labels={'Volume': 'Total Volume'}, color='Volume',
                color_continuous_scale=px.colors.sequential.Viridis)

    # Display the chart
    st.plotly_chart(fig)

    st.subheader("Sector Performance Chart")

    # List of sector indices (NIFTY and BSE sectors as an example)
    sector_indices = {
        'NIFTY_BANK': '^NSEBANK',
        'NIFTY_IT': '^CNXIT',
        'NIFTY_AUTO': '^CNXAUTO',
        'NIFTY_FMCG': '^CNXFMCG',
        'NIFTY_PHARMA': '^CNXPHARMA',
        'BSE_TECK': 'BSE-TECK.BO',
        'BSE_HEALTHCARE': 'BSE-HEALTH.BO',
        'BSE_FINANCE': 'BSE-FINANCE.BO',
        'BSE_POWER': 'BSE-POWER.BO',
    }

    # Function to fetch sector data
    def get_sector_data(ticker_symbol, start_date, end_date):
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
        return data

    # Define the date range for the slider
    min_date = datetime(2018, 1, 1)
    max_date = datetime.today()

    # Get start and end date inputs from user using a slider
    date_range = st.slider("Select Date Range", min_value=min_date, max_value=max_date, value=(min_date, max_date), format="YYYY-MM-DD")
    start_date, end_date = date_range

    # Debug prints to check the selected dates
    st.write(f"Selected start date: {start_date}")
    st.write(f"Selected end date: {end_date}")

    # Function to calculate sector performance
    def calculate_performance(data):
        if not data.empty:
            performance = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
            return performance
        return None

    # Fetch data and calculate performance for each sector
    sector_performance = {}
    for sector, ticker in sector_indices.items():
        data = get_sector_data(ticker, start_date, end_date)
        performance = calculate_performance(data)
        if performance is not None:
            sector_performance[sector] = performance

    # Convert the performance data into a DataFrame for visualization
    performance_df = pd.DataFrame(list(sector_performance.items()), columns=['Sector', 'Performance'])

    # Create a bar chart using Plotly
    fig = px.bar(performance_df, x='Sector', y='Performance', title='Sector Performance',
                labels={'Performance': 'Performance (%)'}, color='Performance',
                color_continuous_scale=px.colors.sequential.Viridis)

    # Display the chart
    st.plotly_chart(fig)


    st.subheader("Market Performance ")

    # List of market indices (Equities, Commodities, Forex, Crypto)
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

    # Function to fetch market data
    def get_market_data(ticker_symbol, start_date, end_date):
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
        return data

    # Define the date range for the slider
    min_date = datetime(2020, 1, 1)
    max_date = datetime.today()

    # Get start and end date inputs from user using a slider
    date_range = st.slider("Select Date Range", min_value=min_date, max_value=max_date, value=(min_date, max_date), format="YYYY-MM-DD")
    start_date, end_date = date_range

    # Debug prints to check the selected dates
    st.write(f"Selected start date: {start_date}")
    st.write(f"Selected end date: {end_date}")

    # Function to calculate market performance
    def calculate_performance(data):
        if not data.empty:
            performance = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
            return performance
        return None

    # Fetch data and calculate performance for each market
    market_performance = {}
    for market, ticker in market_indices.items():
        data = get_market_data(ticker, start_date, end_date)
        performance = calculate_performance(data)
        if performance is not None:
            market_performance[market] = performance

    # Convert the performance data into a DataFrame for visualization
    performance_df = pd.DataFrame(list(market_performance.items()), columns=['Market', 'Performance'])

    # Create a bar chart using Plotly
    fig = px.bar(performance_df, x='Market', y='Performance', title='Market Performance',
                labels={'Performance': 'Performance (%)'}, color='Performance',
                color_continuous_scale=px.colors.diverging.RdYlGn)

    # Display the chart
    st.plotly_chart(fig)

    st.markdown("-----------------------------------------------------------------------------------------------------------------------")
    st.subheader("Unlock your trading potential. Join TradeSense today!")
    st.write("An ultimate platform for smart trading insights. Please log in or sign up to get started.")  
else:
    if choice:
        if choice == "Markets":
            #'Markets' code

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


            # Create two columns
            col1, col2 = st.columns(2)

            # Set up the start and end date inputs
            with col1:
                START = st.date_input('Start Date', pd.to_datetime("2022-01-01"))
            with col2:
                END = st.date_input('End Date', pd.to_datetime("today"))

            if submenu == "Equities":
                st.subheader("Equity Markets")
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
                st.subheader("Commodities")
                tickers = ["GC=F", "CL=F", "NG=F", "SI=F", "HG=F"]
                selected_tickers = st.multiselect("Select stock tickers to visualize", tickers,
                                                  default=["GC=F", "CL=F"])
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
                st.subheader("Currencies")
                tickers = ["EURUSD=X", "GBPUSD=X", "CNYUSD=X", "INRUSD=X"]
                selected_tickers = st.multiselect("Select currency pairs to visualize", tickers,
                                                  default=["INRUSD=X", "CNYUSD=X"])
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
                st.subheader("Cryptocurrencies")
                tickers = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "DOGE-USD"]
                selected_tickers = st.multiselect("Select cryptocurrencies to visualize", tickers,
                                                  default=["BTC-USD", "ETH-USD"])
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
            # Your existing 'Stock Screener' code
            st.sidebar.subheader("Screens")
            submenu = st.sidebar.radio("Select Option",
                                       ["LargeCap-1", "LargeCap-2", "LargeCap-3", "MidCap", "SmallCap"])

            # Define ticker symbols for different market caps
            largecap3_tickers = ["ITC.NS", "JBCHEPHARM.BO", "JWL.BO", "JYOTHYLAB.BO", "KPRMILL.NS", "KAJARIACER.NS",
                                 "KEI.BO",
                                 "LTTS.NS", "LTIM.NS", "MANKIND.NS", "MARICO.NS", "METROBRAND.BO", "MOTILALOFS.NS",
                                 "MPHASIS.NS",
                                 "MUTHOOTFIN.NS", "NH.NS", "NAVINFLUOR.NS", "NAM-INDIA.BO", "NMDC.NS", "OFSS.NS",
                                 "PGHH.NS",
                                 "PIIND.NS", "PAGEIND.NS", "PERSISTENT.NS", "PETRONET.NS", "PFIZER.NS", "PIDILITIND.NS",
                                 "POLYMED.NS", "POLYCAB.NS", "RRKABEL.NS", "RVNL.NS", "RATNAMANI.NS", "RITES.NS",
                                 "SANOFI.NS",
                                 "SCHAEFFLER.NS", "SKFINDIA.NS", "SOLARINDS.NS", "SONACOMS.NS", "SUMICHEM.NS",
                                 "SUNTV.NS",
                                 "SUNDRMFAST.NS", "SUPREMEIND.BO", "TATAELXSI.NS", "TATATECH.NS", "TCS.NS", "TECHM.NS",
                                 "TIMKEN.NS",
                                 "TITAN.NS", "TRITURBINE.NS", "TIINDIA.NS", "UNITDSPR.BO", "VGUARD.NS", "MANYAVAR.NS",
                                 "VINATIORGA.NS", "WIPRO.NS", "ZYDUSLIFE.NS"]
            largecap2_tickers = ["CUMMINSIND.NS", "CYIENT.NS", "DATAPATTNS.NS", "DEEPAKNTR.NS", "DIVISLAB.NS",
                                 "LALPATHLAB.NS",
                                 "RDY", "ELGIEQUIP.NS", "EMAMILTD.NS", "FIVESTAR.BO", "GRINFRA.NS", "GILLETTE.NS",
                                 "GLAXO.NS",
                                 "GODFRYPHLP.NS", "GRINDWELL.NS", "HAVELLS.NS", "HCLTECH.NS", "HAL.BO", "HONAUT.BO",
                                 "IRCTC.NS",
                                 "ISEC.BO", "INFY.NS", "IPCALAB.BO"]
            largecap1_tickers = ["ABBOTINDIA.NS", "ADANIPOWER.NS", "AFFLE.BO", "AIAENG.BO", "AJANTPHARM.BO",
                                 "APLLTD.BO",
                                 "ALKEM.BO", "ARE&M.NS", "ANANDRATHI.BO", "APARINDS.BO", "ASIANPAINT.NS", "ASTRAL.NS",
                                 "ASTRAZEN.NS", "BAJFINANCE.NS", "BASF.NS", "BAYERCROP.BO", "BERGEPAINT.BO", "BDL.NS",
                                 "BEL.NS",
                                 "BSOFT.BO", "CDSL.NS", "CAMS.NS", "CARBORUNIV.BO", "CASTROLIND.NS", "CHAMBLFERT.BO",
                                 "COALINDIA.NS", "COFORGE.BO", "COLPAL.NS", "CONCORDBIO.BO", "COROMANDEL.BO",
                                 "CREDITACC.BO"]
            smallcap_tickers = ["TAPARIA.BO", "LKPFIN.BO", "EQUITAS.NS"]
            midcap_tickers = ["PNCINFRA.NS", "INDIASHLTR.NS", "RAYMOND.NS", "KAMAHOLD.BO", "BENGALASM.BO",
                              "CHOICEIN.NS",
                              "GRAVITA.NS", "HGINFRA.NS", "JKPAPER.NS", "MTARTECH.NS", "HAPPSTMNDS.NS", "SARDAEN.NS",
                              "WELENT.NS",
                              "LTFOODS.NS", "GESHIP.NS", "SHRIPISTON.NS", "SHAREINDIA.NS", "CYIENTDLM.NS", "VTL.NS",
                              "EASEMYTRIP.NS", "LLOYDSME.NS", "ROUTE.NS", "VAIBHAVGBL.NS", "GOKEX.NS", "USHAMART.NS",
                              "EIDPARRY.NS",
                              "KIRLOSBROS.NS", "MANINFRA.NS", "CMSINFO.NS", "RALLIS.NS", "GHCL.NS", "NEULANDLAB.NS",
                              "SPLPETRO.NS",
                              "MARKSANS.NS", "NAVINFLUOR.NS", "ELECON.NS", "TANLA.NS", "KFINTECH.NS", "TIPSINDLTD.NS",
                              "ACI.NS",
                              "SURYAROSNI.NS", "GPIL.NS", "GMDCLTD.NS", "MAHSEAMLES.NS", "TDPOWERSYS.NS", "TECHNOE.NS",
                              "JLHL.NS"]


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
                    combined_df.columns = ['_'.join([ticker, col]).strip() for ticker, df in stock_data.items() for col
                                           in
                                           df.columns]
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
                st.subheader("LargeCap-1")
                tickers = largecap1_tickers

            if submenu == "LargeCap-2":
                st.subheader("LargeCap-2")
                tickers = largecap2_tickers

            if submenu == "LargeCap-3":
                st.subheader("LargeCap-3")
                tickers = largecap3_tickers

            if submenu == "MidCap":
                st.subheader("MidCap")
                tickers = midcap_tickers
            if submenu == "SmallCap":
                st.subheader("SmallCap")
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

            # st.dataframe(first_query_df.round(2))
            # Generate insights
            second_query_df = first_query_df[
                (first_query_df['RSI'] < 70) & (first_query_df['RSI'] > 55) & (first_query_df['ADX'] > 20) & (
                        first_query_df['MACD'] > 0)]
            st.write("Stocks in an uptrend with high volume:")
            st.dataframe(second_query_df)



            # Create two columns
            col1, col2 = st.columns(2)
            # Dropdown for analysis type
            with col1:
                selected_stock = st.selectbox("Select Stock", second_query_df['Ticker'].tolist())
            with col2:
                analysis_type = st.selectbox("Select Analysis Type",
                                             ["Trend Analysis", "Volume Analysis", "Support & Resistance Levels"])

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
                        df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'],
                                                                        df['Volume']).chaikin_money_flow()


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

                            amplitude_envelope, instantaneous_phase, instantaneous_frequency = apply_hilbert_transform(
                                prices)

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

                    else:
                        st.write("Not enough data points for technical analysis.")

        elif choice == "Technical Analysis":
            # 'Technical Analysis' code
            st.sidebar.subheader("Interactive Charts")
            submenu = st.sidebar.radio("Select Option",
                                       ["Trend Analysis", "Volume Analysis", "Support & Resistance Levels"])

            # Create three columns
            col1, col2, col3 = st.columns(3)

            # Set up the start and end date inputs
            with col1:
                # List of stock symbols
                stock_symbols = ["ABBOTINDIA.NS", "ADANIPOWER.NS", "AFFLE.BO", "AIAENG.BO", "AJANTPHARM.BO",
                                 "APLLTD.BO", "ALKEM.BO", "ARE&M.NS", "ANANDRATHI.BO", "APARINDS.BO", "ASIANPAINT.NS",
                                 "ASTRAL.NS", "ASTRAZEN.NS", "BAJFINANCE.NS", "BASF.NS", "BAYERCROP.BO",
                                 "BERGEPAINT.BO", "BDL.NS", "BEL.NS", "BSOFT.BO", "CDSL.NS", "CAMS.NS", "CARBORUNIV.BO",
                                 "CASTROLIND.NS", "CHAMBLFERT.BO", "COALINDIA.NS", "COFORGE.BO", "COLPAL.NS",
                                 "CONCORDBIO.BO", "COROMANDEL.BO", "CREDITACC.BO", "CUMMINSIND.NS", "CYIENT.NS",
                                 "DATAPATTNS.NS", "DEEPAKNTR.NS", "DIVISLAB.NS", "LALPATHLAB.NS", "RDY", "ELGIEQUIP.NS",
                                 "EMAMILTD.NS", "FIVESTAR.BO", "GRINFRA.NS", "GILLETTE.NS", "GLAXO.NS", "GODFRYPHLP.NS",
                                 "GRINDWELL.NS", "HAVELLS.NS", "HCLTECH.NS", "HAL.BO", "HONAUT.BO", "IRCTC.NS",
                                 "ISEC.BO", "INFY.NS", "IPCALAB.BO", "ITC.NS", "JBCHEPHARM.BO", "JWL.BO",
                                 "JYOTHYLAB.BO", "KPRMILL.NS", "KAJARIACER.NS", "KEI.BO", "LTTS.NS", "LTIM.NS",
                                 "MANKIND.NS", "MARICO.NS", "METROBRAND.BO", "MOTILALOFS.NS", "MPHASIS.NS",
                                 "MUTHOOTFIN.NS", "NH.NS", "NAVINFLUOR.NS", "NAM-INDIA.BO", "NMDC.NS", "OFSS.NS",
                                 "PGHH.NS", "PIIND.NS", "PAGEIND.NS", "PERSISTENT.NS", "PETRONET.NS", "PFIZER.NS",
                                 "PIDILITIND.NS", "POLYMED.NS", "POLYCAB.NS", "RRKABEL.NS", "RVNL.NS", "RATNAMANI.NS",
                                 "RITES.NS", "SANOFI.NS", "SCHAEFFLER.NS", "SKFINDIA.NS", "SOLARINDS.NS", "SONACOMS.NS",
                                 "SUMICHEM.NS", "SUNTV.NS", "SUNDRMFAST.NS", "SUPREMEIND.BO", "TATAELXSI.NS",
                                 "TATATECH.NS", "TCS.NS", "TECHM.NS", "TIMKEN.NS", "TITAN.NS", "TRITURBINE.NS",
                                 "TIINDIA.NS", "UNITDSPR.BO", "VGUARD.NS", "MANYAVAR.NS", "VINATIORGA.NS", "WIPRO.NS",
                                 "ZYDUSLIFE.NS"]
                # ticker = st.text_input("Enter Stock symbol", '^BSESN')
                # Auto-suggestion using selectbox
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
                    df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'],
                                                                    df['Volume']).chaikin_money_flow()


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
                        st.subheader("Trend Analysis")

                        indicators = st.multiselect(
                            "Select Indicators",
                            ['Close', '20_MA', '50_MA', '200_MA', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'RSI',
                             'Signal', 'ADX', 'Parabolic_SAR', 'Bollinger_High', 'Bollinger_Low', 'Bollinger_Middle'],
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
                        colors = {'Close': 'blue', '20_MA': 'orange', '50_MA': 'green', '200_MA': 'red',
                                  'MACD': 'purple', 'MACD_Signal': 'brown', 'RSI': 'pink', 'Signal': 'black',
                                  'ADX': 'magenta', 'Parabolic_SAR': 'yellow', 'Bollinger_High': 'black',
                                  'Bollinger_Low': 'cyan', 'Bollinger_Middle': 'grey'}

                        for indicator in indicators:
                            if indicator == 'Signal':
                                # Plot buy and sell signals
                                buy_signals = df[df['Signal'] == 'Buy']
                                sell_signals = df[df['Signal'] == 'Sell']
                                fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Close'], mode='markers',
                                                         name='Buy Signal',
                                                         marker=dict(color='green', symbol='triangle-up')))
                                fig.add_trace(
                                    go.Scatter(x=sell_signals['Date'], y=sell_signals['Close'], mode='markers',
                                               name='Sell Signal', marker=dict(color='red', symbol='triangle-down')))
                            elif indicator == 'MACD_Histogram':
                                fig.add_trace(
                                    go.Bar(x=df['Date'], y=df[indicator], name=indicator, marker_color='gray'))
                            else:
                                fig.add_trace(go.Scatter(x=df['Date'], y=df[indicator], mode='lines', name=indicator,
                                                         line=dict(color=colors.get(indicator, 'black'))))

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
                        st.subheader("Volume Analysis")
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
                                st.markdown(
                                    "Current volume is above the 20-day average - Increased buying/selling interest")
                            else:
                                st.markdown(
                                    "Current volume is below the 20-day average - Decreased buying/selling interest")

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
            # 'Stock Price Forecasting' code
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
            data['ATR'] = ta.volatility.AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close'],
                                                         window=14).average_true_range()

            # Calculate Volume Moving Average
            data['Volume_MA_20'] = data['Volume'].rolling(window=20).mean()
            data['Volume_MA_10'] = data['Volume'].rolling(window=10).mean()

            # On-Balance Volume (OBV)
            data['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=data['Close'],
                                                             volume=data['Volume']).on_balance_volume()

            # Fourier Transform
            data['FFT'] = np.abs(fft(data['Close'].values))


            # Wavelet Transform
            def wavelet_transform(data, wavelet='db1'):
                coeffs = pywt.wavedec(data, wavelet)
                return np.concatenate(coeffs)


            data['Wavelet'] = data['Close'].apply(lambda x: wavelet_transform(data['Close'].values)).apply(
                lambda x: x[0])

            # Hilbert Transform
            data['Hilbert'] = np.abs(hilbert(data['Close'].values))


            # Define custom candlestick pattern detection functions
            def is_hammer(data):
                return (data['Close'] > data['Open']) & (
                            (data['High'] - data['Close']) <= (data['Open'] - data['Low']) * 2)


            def is_doji(data):
                return (abs(data['Close'] - data['Open']) <= (data['High'] - data['Low']) * 0.1)


            def is_engulfing(data):
                return ((data['Open'] < data['Close'].shift(1)) & (data['Close'] > data['Open'].shift(1))) | (
                            (data['Open'] > data['Close'].shift(1)) & (data['Close'] < data['Open'].shift(1)))


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

            st.subheader("Next 5 days forecast:")
            st.dataframe(forecast_df)

            forecast_df.set_index('Day')[['Forecasted_Price', 'Lower Bound', 'Upper Bound']].plot()
            plt.fill_between(forecast_df.index, forecast_df['Lower Bound'], forecast_df['Upper Bound'], color='gray',
                             alpha=0.2)
            plt.title('Next 5 Days Forecast')
            plt.xlabel('Day')
            plt.ylabel('Price')
            plt.show()

            # Plot close using plotly with a time bar
            forecast_fig = go.Figure()

            # Prices plot
            forecast_fig.add_trace(
                go.Scatter(x=forecast_labels, y=forecasted_values, mode='lines', name='Next 5 Days Forecast'))
            forecast_fig.add_trace(
                go.Scatter(x=forecast_labels, y=forecast_df['Lower Bound'], mode='lines', name='Lower Bound',
                           line=dict(dash='dash')))
            forecast_fig.add_trace(
                go.Scatter(x=forecast_labels, y=forecast_df['Upper Bound'], mode='lines', name='Upper Bound',
                           line=dict(dash='dash')))
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
            lstm_fig.add_trace(
                go.Scatter(x=data.index[-len(y_test):], y=y_test_inv.flatten(), mode='lines', name='Actual Price'))
            lstm_fig.add_trace(
                go.Scatter(x=data.index[-len(y_test):], y=y_pred_lstm.flatten(), mode='lines', name='Predicted Price'))
            lstm_fig.update_layout(title='LSTM Model Predictions', xaxis_title='Date', yaxis_title='Price')
            st.plotly_chart(lstm_fig)

            # Ensemble method: Combine predictions
            ensemble_pred = (forecasted_values.values + y_pred_lstm.flatten()[:forecast_steps]) / 2

            ensemble_fig = go.Figure()
            ensemble_fig.add_trace(
                go.Scatter(x=forecast_labels, y=forecasted_values.values, mode='lines', name='SARIMA Forecast'))
            ensemble_fig.add_trace(go.Scatter(x=forecast_labels, y=y_pred_lstm.flatten()[:forecast_steps], mode='lines',
                                              name='LSTM Forecast'))
            ensemble_fig.add_trace(
                go.Scatter(x=forecast_labels, y=ensemble_pred, mode='lines', name='Ensemble Forecast'))
            st.plotly_chart(ensemble_fig)

        elif choice == "Stock Watch":
            # Your existing 'Stock Watch' code
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

        elif choice == "Strategy Backtesting":
            # 'Strategy Backtesting' code
            # List of tickers
            tickers = ["ABBOTINDIA.NS", "ADANIPOWER.NS", "AFFLE.BO", "AIAENG.BO", "AJANTPHARM.BO", "APLLTD.BO",
                       "ALKEM.BO", "ARE&M.NS", "ANANDRATHI.BO",
                       "APARINDS.BO", "ASIANPAINT.NS", "ASTRAL.NS", "ASTRAZEN.NS", "BAJFINANCE.NS", "BASF.NS",
                       "BAYERCROP.BO", "BERGEPAINT.BO", "BDL.NS",
                       "BEL.NS", "BSOFT.BO", "CDSL.NS", "CAMS.NS", "CARBORUNIV.BO", "CASTROLIND.NS", "CHAMBLFERT.BO",
                       "COALINDIA.NS", "COFORGE.BO", "COLPAL.NS",
                       "CONCORDBIO.BO", "COROMANDEL.BO", "CREDITACC.BO", "CUMMINSIND.NS", "CYIENT.NS", "DATAPATTNS.NS",
                       "DEEPAKNTR.NS", "DIVISLAB.NS",
                       "LALPATHLAB.NS", "RDY", "ELGIEQUIP.NS", "EMAMILTD.NS", "FIVESTAR.BO", "GRINFRA.NS",
                       "GILLETTE.NS", "GLAXO.NS", "GODFRYPHLP.NS",
                       "GRINDWELL.NS", "HAVELLS.NS", "HCLTECH.NS", "HAL.BO", "HONAUT.BO", "IRCTC.NS", "ISEC.BO",
                       "INFY.NS", "IPCALAB.BO", "ITC.NS",
                       "JBCHEPHARM.BO", "JWL.BO", "JYOTHYLAB.BO", "KPRMILL.NS", "KAJARIACER.NS", "KEI.BO", "LTTS.NS",
                       "LTIM.NS", "MANKIND.NS",
                       "MARICO.NS", "METROBRAND.BO", "MOTILALOFS.NS", "MPHASIS.NS", "MUTHOOTFIN.NS", "NH.NS",
                       "NAVINFLUOR.NS", "NAM-INDIA.BO", "NMDC.NS",
                       "OFSS.NS", "PGHH.NS", "PIIND.NS", "PAGEIND.NS", "PERSISTENT.NS", "PETRONET.NS", "PFIZER.NS",
                       "PIDILITIND.NS", "POLYMED.NS",
                       "POLYCAB.NS", "RRKABEL.NS", "RVNL.NS", "RATNAMANI.NS", "RITES.NS", "SANOFI.NS", "SCHAEFFLER.NS",
                       "SKFINDIA.NS", "SOLARINDS.NS",
                       "SONACOMS.NS", "SUMICHEM.NS", "SUNTV.NS", "SUNDRMFAST.NS", "SUPREMEIND.BO", "TATAELXSI.NS",
                       "TATATECH.NS", "TCS.NS", "TECHM.NS",
                       "TIMKEN.NS", "TITAN.NS", "TRITURBINE.NS", "TIINDIA.NS", "UNITDSPR.BO", "VGUARD.NS",
                       "MANYAVAR.NS", "VINATIORGA.NS", "WIPRO.NS", "ZYDUSLIFE.NS"]

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
                vwap = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data[
                    'Volume'].cumsum()
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
                    if data['Close'][i - 1] > psar:
                        psar = min(psar + af * (ep - psar), data['Low'][i - 1], data['Low'][i - 2])
                    else:
                        psar = max(psar - af * (psar - ep), data['High'][i - 1], data['High'][i - 2])
                    ep = max(ep, data['High'][i]) if data['Close'][i] > psar else min(ep, data['Low'][i])
                    psar_list.append(psar)
                return pd.Series(psar_list, index=data.index)


            def calculate_ichimoku(data):
                nine_period_high = data['High'].rolling(window=9).max()
                nine_period_low = data['Low'].rolling(window=9).min()
                senkou_span_a = ((nine_period_high + nine_period_low) / 2).shift(26)
                senkou_span_b = (
                            (data['High'].rolling(window=52).max() + data['Low'].rolling(window=52).min()) / 2).shift(
                    26)
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
            data['WMA'] = data['Close'].rolling(window=50).apply(
                lambda x: np.dot(x, np.arange(1, 51)) / np.arange(1, 51).sum(), raw=True)
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

            data['Buy_Signal_Stochastic'] = (
                        (data['Stochastic_K'] < 20) & (data['Stochastic_K'] > data['Stochastic_D'])).astype(int)
            data['Sell_Signal_Stochastic'] = (
                        (data['Stochastic_K'] > 80) & (data['Stochastic_K'] < data['Stochastic_D'])).astype(int)

            data['Buy_Signal_MACD'] = (data['MACD'] > data['MACD_Signal']).astype(int)
            data['Sell_Signal_MACD'] = (data['MACD'] < data['MACD_Signal']).astype(int)

            data['Buy_Signal_OBV'] = (
                        (data['OBV'] > data['OBV'].shift(1)) & (data['Close'] > data['Close'].shift(1))).astype(int)
            data['Sell_Signal_OBV'] = (
                        (data['OBV'] < data['OBV'].shift(1)) & (data['Close'] < data['Close'].shift(1))).astype(int)

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
            data['Buy_Signal_Combined_1'] = ((data['RSI'] < 30) & (data['MACD'] > data['MACD_Signal']) & (
                        data['Close'] > data['SMA'])).astype(int)
            data['Sell_Signal_Combined_1'] = ((data['RSI'] > 70) & (data['MACD'] < data['MACD_Signal']) & (
                        data['Close'] < data['SMA'])).astype(int)

            data['Buy_Signal_Combined_2'] = ((data['Close'] > data['EMA']) & (data['MACD'] > data['MACD_Signal']) & (
                        data['Close'] > data['VWAP'])).astype(int)
            data['Sell_Signal_Combined_2'] = ((data['Close'] < data['EMA']) & (data['MACD'] < data['MACD_Signal']) & (
                        data['Close'] < data['VWAP'])).astype(int)

            data['Buy_Signal_Combined_3'] = ((data['Close'] > data['WMA']) & (data['RSI'] < 30) & (
                        data['Stochastic_K'] > data['Stochastic_D'])).astype(int)
            data['Sell_Signal_Combined_3'] = ((data['Close'] < data['WMA']) & (data['RSI'] > 70) & (
                        data['Stochastic_K'] < data['Stochastic_D'])).astype(int)

            data['Buy_Signal_Combined_4'] = (
                        (data['OBV'] > data['OBV'].shift(1)) & (data['Close'] > data['Ichimoku_Cloud_Top']) & (
                            data['MACD'] > data['MACD_Signal'])).astype(int)
            data['Sell_Signal_Combined_4'] = (
                        (data['OBV'] < data['OBV'].shift(1)) & (data['Close'] < data['Ichimoku_Cloud_Bottom']) & (
                            data['MACD'] < data['MACD_Signal'])).astype(int)

            data['Buy_Signal_Combined_5'] = ((data['Close'] > data['Lower Band']) & (data['RSI'] < 30) & (
                        data['ATR'] > data['ATR'].shift(1))).astype(int)
            data['Sell_Signal_Combined_5'] = ((data['Close'] < data['Upper Band']) & (data['RSI'] > 70) & (
                        data['ATR'] < data['ATR'].shift(1))).astype(int)


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
                final_balance_balance_based = backtest_strategy_balance_based(data, f'Buy_Signal_{strategy}',
                                                                              f'Sell_Signal_{strategy}')
                return_percentage_balance_based = ((final_balance_balance_based - 100000) / 100000) * 100

                cumulative_return, final_balance_cumulative = backtest_strategy_cumulative(data,
                                                                                           f'Buy_Signal_{strategy}',
                                                                                           f'Sell_Signal_{strategy}')
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
            indicator = st.selectbox('Select Indicator',
                                     ['SMA', 'EMA', 'WMA', 'RSI', 'MACD', 'OBV', 'VWAP', 'Bollinger Bands', 'Aroon'])

            if indicator == 'Bollinger Bands':
                st.line_chart(data[['Close', 'Lower Band', 'Upper Band']])
            elif indicator == 'Aroon':
                st.line_chart(data[['Aroon_Up', 'Aroon_Down']])
            else:
                st.line_chart(data[['Close', indicator]])

            st.subheader('Strategy Performance')
            st.table(results_df)

        elif choice == f"{st.session_state.username}'s Watchlist":
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
