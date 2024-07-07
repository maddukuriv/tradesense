import os
import streamlit as st
import random
import string
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import ta

from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.exc import NoResultFound

import bcrypt  # for password hashing
from dotenv import load_dotenv
from password_validator import PasswordValidator

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
    dob = Column(Date, nullable=False)  # Date of birth
    pob = Column(String, nullable=False)  # Place of birth

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
DATABASE_URL = "sqlite:///new_etrade.db"  # Change the database name here
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)  # This will create the new database

Session = sessionmaker(bind=engine)
session = Session()

# Initialize session state for login status and reset code
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'email' not in st.session_state:
    st.session_state.email = ""
if 'user_id' not in st.session_state:
    st.session_state.user_id = None

# Password validation schema
password_schema = PasswordValidator()
password_schema \
    .min(8) \
    .max(100) \
    .has().uppercase() \
    .has().lowercase() \
    .has().digits() \
    .has().no().spaces()

# Function to hash password using bcrypt
def hash_password(password):
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password.decode('utf-8')

# Function to verify password using bcrypt
def verify_password(hashed_password, plain_password):
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except ValueError as e:
        print(f"Error verifying password: {e}")
        return False

# Function to send email (if needed)
def send_email(to_email, subject, body):
    from_email = os.getenv('EMAIL_USER')
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

# Signup function
def signup():
    st.subheader("Sign Up")
    name = st.text_input("Enter your name", key='signup_name')
    email = st.text_input("Enter your email", key='signup_email')
    dob = st.date_input("Enter your date of birth", key='signup_dob')
    pob = st.text_input("Enter your place of birth", key='signup_pob')
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
            hashed_password = hash_password(password)
            new_user = User(name=name, email=email, password=hashed_password, dob=dob, pob=pob)
            session.add(new_user)
            session.commit()
            st.success("User registered successfully!")

# Login function
def login():
    st.subheader("Login")
    email = st.text_input("Enter your email", key='login_email')
    password = st.text_input("Enter your password", type="password", key='login_password')

    if st.button("Login"):
        try:
            user = session.query(User).filter_by(email=email).one()
            if verify_password(user.password, password):
                st.success("Login successful!")
                st.session_state.logged_in = True
                st.session_state.username = user.name
                st.session_state.email = user.email
                st.session_state.user_id = user.id
            else:
                st.error("Invalid email or password.")
        except NoResultFound:
            st.error("Invalid email or password.")
        except Exception as e:
            st.error(f"Error during login: {e}")

# Forgot password function with security questions
def forgot_password():
    st.subheader("Forgot Password")
    email = st.text_input("Enter your email", key='forgot_email')
    dob = st.date_input("Enter your date of birth", key='forgot_dob')
    pob = st.text_input("Enter your place of birth", key='forgot_pob')

    if 'identity_verified' not in st.session_state:
        st.session_state.identity_verified = False

    if st.button("Submit"):
        try:
            user = session.query(User).filter_by(email=email, dob=dob, pob=pob).one()
            st.session_state.email = email
            st.session_state.user_id = user.id
            st.session_state.identity_verified = True
            st.success("Identity verified. Please reset your password.")
        except NoResultFound:
            st.error("Invalid details provided.")
        except Exception as e:
            st.error(f"Error during password reset: {e}")

    if st.session_state.identity_verified:
        new_password = st.text_input("Enter a new password", type="password", key='reset_new_password')
        confirm_new_password = st.text_input("Confirm your new password", type="password", key='reset_confirm_new_password')

        if st.button("Reset Password"):
            if new_password != confirm_new_password:
                st.error("Passwords do not match.")
            elif not password_schema.validate(new_password):
                st.error("Password does not meet the requirements.")
            else:
                user = session.query(User).filter_by(id=st.session_state.user_id).one()
                user.password = hash_password(new_password)
                session.commit()
                st.success("Password reset successfully. You can now log in with the new password.")
                st.session_state.identity_verified = False


# My Account function to edit details and change password
def my_account():
    st.subheader("My Account")

    if st.session_state.logged_in:
        user = session.query(User).filter_by(id=st.session_state.user_id).one()

        new_name = st.text_input("Update your name", value=user.name, key='account_name')
        new_dob = st.date_input("Update your date of birth", value=user.dob, key='account_dob')
        new_pob = st.text_input("Update your place of birth", value=user.pob, key='account_pob')

        if st.button("Update Details"):
            user.name = new_name
            user.dob = new_dob
            user.pob = new_pob
            session.commit()
            st.success("Details updated successfully!")

        st.subheader("Change Password")
        current_password = st.text_input("Enter your current password", type="password", key='account_current_password')
        new_password = st.text_input("Enter a new password", type="password", key='account_new_password')
        confirm_new_password = st.text_input("Confirm your new password", type="password", key='account_confirm_new_password')

        if st.button("Change Password"):
            if verify_password(user.password, current_password):
                if new_password != confirm_new_password:
                    st.error("Passwords do not match.")
                elif not password_schema.validate(new_password):
                    st.error("Password does not meet the requirements.")
                else:
                    user.password = hash_password(new_password)
                    session.commit()
                    st.success("Password changed successfully!")
            else:
                st.error("Current password is incorrect.")

# Logout function
def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.email = ""
    st.session_state.user_id = None

# Main menu function
def main_menu():
    st.subheader("Main Menu")
    menu_options = ["Markets", "Stock Screener","Stock Watch" ,"Technical Analysis", "Stock Price Forecasting",
                    "Stock Comparison", "Market Stats", "My Account",
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
    st.subheader("Please login or sign up to access the e-Trade platform.")
else:
    if choice:

        if choice == "My Account":
            my_account()
        elif choice == f"{st.session_state.username}'s Watchlist":
            # Your existing 'Watchlist' code
            pass
        elif choice == f"{st.session_state.username}'s Portfolio":
            # Your existing 'Portfolio' code
            pass
        elif choice == "Markets":
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
        elif choice == "Stock Comparison":
            # Your existing 'Stock Comparison' code
            pass
        
        # Add your other options here

# Debugging function to check users in the database
def debug_check_users():
    st.subheader("Debug: Check Users in Database")
    if st.button("Show Users"):
        users = session.query(User).all()
        for user in users:
            st.write(f"ID: {user.id}, Name: {user.name}, Email: {user.email}, Password: {user.password}, DOB: {user.dob}, POB: {user.pob}")

# Add this function call to your Streamlit app somewhere to use it
debug_check_users()

# Add a form in your Streamlit app to verify user
def verify_user(email, plain_password):
    try:
        user = session.query(User).filter_by(email=email).one()
        st.write(f"Debug: Stored hashed password: {user.password}")  # Debugging line
        if verify_password(user.password, plain_password):
            st.success(f"User {email} exists and the password is correct!")
        else:
            st.error(f"User {email} exists but the password is incorrect.")
    except NoResultFound:
        st.error(f"User {email} does not exist.")
    except Exception as e:
        st.error(f"Error verifying user: {e}")

st.subheader("Debug: Verify User")
debug_email = st.text_input("Enter email to verify", key='debug_email')
debug_password = st.text_input("Enter password to verify", type="password", key='debug_password')
if st.button("Verify User"):
    verify_user(debug_email, debug_password)
