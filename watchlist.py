import streamlit as st
from datetime import datetime
import yfinance as yf
import pandas as pd
import ta
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import hashlib
import plotly.graph_objs as go

# Set wide mode as default layout
st.set_page_config(layout="wide", page_title="e-Trade")

# Database setup
DATABASE_URL = "sqlite:///etrade.db"
Base = declarative_base()


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
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


# Create a new database session
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)  # Create the tables with the new schema
Session = sessionmaker(bind=engine)
session = Session()

# Initialize session state for login status
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'username' not in st.session_state:
    st.session_state.username = ""


# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# Function to handle user signup
def signup():
    st.subheader("Sign Up")
    username = st.text_input("Enter a new username", key='signup_username')
    password = st.text_input("Enter a new password", type="password", key='signup_password')
    if st.button("Sign Up"):
        if session.query(User).filter_by(username=username).first():
            st.error("Username already exists. Try a different username.")
        else:
            new_user = User(username=username, password=hash_password(password))
            session.add(new_user)
            session.commit()
            st.success("User registered successfully!")


# Function to handle user login
def login():
    st.subheader("Login")
    username = st.text_input("Enter your username", key='login_username')
    password = st.text_input("Enter your password", type="password", key='login_password')
    if st.button("Login"):
        user = session.query(User).filter_by(username=username, password=hash_password(password)).first()
        if user:
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
    menu_options = ["Markets", "Stock Screener", "Technical Analysis", "Stock Price Forecasting", "Stock Watch",
                    "Strategy Backtesting", "Watchlist", "My Portfolio"]
    choice = st.selectbox("Select an option", menu_options)
    return choice


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
        elif choice == "Watchlist":
            st.header(f"{st.session_state.username}'s Watchlist")
            user_id = session.query(User.id).filter_by(username=st.session_state.username).first()[0]
            watchlist = session.query(Watchlist).filter_by(user_id=user_id).all()

            # Add new ticker to watchlist
            new_ticker = st.text_input("Add a new ticker to your watchlist")
            if st.button("Add Ticker"):
                if not session.query(Watchlist).filter_by(user_id=user_id, ticker=new_ticker).first():
                    new_watchlist_entry = Watchlist(user_id=user_id, ticker=new_ticker)
                    session.add(new_watchlist_entry)
                    session.commit()
                    st.success(f"{new_ticker} added to your watchlist!")
                else:
                    st.warning(f"{new_ticker} is already in your watchlist.")

            # Display watchlist
            if watchlist:
                watchlist_data = {entry.ticker: get_stock_data(entry.ticker) for entry in watchlist}
                watchlist_df = pd.DataFrame(watchlist_data).T  # Transpose to have tickers as rows
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
        elif choice == "My Portfolio":
            st.header(f"{st.session_state.username}'s Portfolio")
            user_id = session.query(User.id).filter_by(username=st.session_state.username).first()[0]
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
