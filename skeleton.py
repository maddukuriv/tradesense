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
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import numpy as np


# Set wide mode as default layout
st.set_page_config(layout="wide", page_title="TradeSense")

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
    st.subheader("Please login or sign up to access the e-Trade platform.")

else:
    if choice:
        if choice == "Markets":
            # Your existing 'Markets' code
            pass
        elif choice == "Stock Screener":
            # Your existing 'Stock Screener' code
            pass
        elif choice == "Technical Analysis":
            # Your existing 'Technical Analysis' code
            pass
        elif choice == "Stock Price Forecasting":
            # Your existing 'Stock Price Forecasting' code
            pass
        elif choice == "Stock Watch":
            # Your existing 'Stock Watch' code
            pass
        elif choice == "Strategy Backtesting":
            # Your existing 'Strategy Backtesting' code
            pass
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
            new_ticker = st.text_input("Ticker Symbol")
            shares = st.number_input("Number of Shares", min_value=0.0, step=0.01)
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
