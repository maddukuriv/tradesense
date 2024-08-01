import os
import streamlit as st
import random
import string
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.exc import NoResultFound


import bcrypt  # for password hashing
from dotenv import load_dotenv
from password_validator import PasswordValidator
import pandas_ta as ta
import yfinance as yf
import pandas as pd
import pandas_ta as pta
import numpy as np
import plotly.graph_objs as go
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots




# technical analysis
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta



# news
import streamlit as st
from newsapi.newsapi_client import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# time series 
import yfinance as yf
import pandas as pd
import numpy as np
import itertools
from ta import add_all_ta_features
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
import streamlit as st
import plotly.graph_objects as go
from pmdarima import auto_arima
from datetime import timedelta, datetime
from scipy.signal import cwt, ricker, hilbert


#Database
from utils.mongodb import users_collection, watchlists_collection, portfolios_collection, init_db
from bson.objectid import ObjectId


#Authentication
from authentication import login, signup, forgot_password

#pages 
from components import my_account, my_portfolio, my_watchlist, markets, stock_screener, stock_analysis, admin,  home_page
# Initialize MongoDB collections
init_db()





# Set wide mode as default layout
st.set_page_config(layout="wide", page_title="TradeSense", page_icon="📈",initial_sidebar_state="expanded")

# Load environment variables from .env file
load_dotenv()



# Initialize session state for login status and reset code
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'email' not in st.session_state:
    st.session_state.email = ""
if 'user_id' not in st.session_state:
    st.session_state.user_id = None






# Logout function
def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.email = ""
    st.session_state.user_id = None





# Main menu function
def main_menu():
    st.subheader("Main Menu")
    menu_options = [f"{st.session_state.username}'s Portfolio",f"{st.session_state.username}'s Watchlist", "Stock Screener", "Stock Analysis",
                    "Markets", "My Account", "Database Admin Page"]
    choice = st.selectbox("Select an option", menu_options)
    return choice

# Sidebar menu
with st.sidebar:
    st.title("TradeSense")
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        st.write(f"Logged in as: {st.session_state.username}")
        if st.button("Logout"):
            logout()
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

######################################################### Main content area ######################################################################

if not st.session_state.logged_in:
   home_page.home_page_app()
else:
    if choice:
        if choice == "My Account":
            my_account.my_account()
        elif choice == f"{st.session_state.username}'s Watchlist":
            my_watchlist.display_watchlist()

        elif choice == f"{st.session_state.username}'s Portfolio":
            my_portfolio.display_portfolio()

        elif choice == "Markets":
                 markets.markets_app()  


        elif choice == "Stock Screener":
   
                       stock_screener.stock_screener_app()
        elif choice == "Stock Analysis":
            stock_analysis.stock_analysis_app()
        elif choice == "Database Admin Page":
              admin.display_tables()