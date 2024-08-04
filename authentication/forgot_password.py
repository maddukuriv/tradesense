import streamlit as st
from utils.mongodb import users_collection
from utils.hash_utils import hash_password
from bson.objectid import ObjectId
import re

# Password validation function
def is_valid_password(password):
    if len(password) < 8:
        return False
    if not re.search(r'[A-Z]', password):
        return False
    if not re.search(r'[a-z]', password):
        return False
    if not re.search(r'[0-9]', password):
        return False
    if not re.search(r'[@$!%*?&]', password):
        return False
    return True

# Forgot password function
def forgot_password():
    st.subheader("Forgot Password")
    email = st.text_input("Enter your email", key='forgot_email')
    pob = st.text_input("Enter your place of birth", key='forgot_pob')

    if 'identity_verified' not in st.session_state:
        st.session_state.identity_verified = False

    if st.button("Submit"):
        user = users_collection.find_one({"email": email, "pob": pob})
        if user:
            st.session_state.email = email
            st.session_state.user_id = str(user['_id'])  # Ensure user_id is a string
            st.session_state.identity_verified = True
            st.success("Identity verified. Please reset your password.")
        else:
            st.error("Invalid details provided.")

    if st.session_state.identity_verified:
        new_password = st.text_input("Enter a new password", type="password", key='reset_new_password')
        confirm_new_password = st.text_input("Confirm your new password", type="password", key='reset_confirm_new_password')

        if st.button("Reset Password"):
            if new_password != confirm_new_password:
                st.error("Passwords do not match.")
            elif not is_valid_password(new_password):
                st.error("Password must be at least 8 characters long and include an uppercase letter, a lowercase letter, a number, and a special character.")
            else:
                hashed_password = hash_password(new_password)
                users_collection.update_one({"_id": ObjectId(st.session_state.user_id)}, {"$set": {"password": hashed_password}})
                st.success("Password reset successfully. You can now log in with the new password.")
                st.session_state.identity_verified = False
                st.experimental_rerun()

# Main Page
import streamlit as st
from dotenv import load_dotenv
from utils.mongodb import init_db
from authentication import login, signup, forgot_password
from components import my_account, my_portfolio, my_watchlist, markets, stock_screener, stock_analysis, admin, home_page

# Initialize MongoDB collections
init_db()

# Set wide mode as default layout
st.set_page_config(layout="wide", page_title="TradeSense", page_icon="📈", initial_sidebar_state="expanded")

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
if 'identity_verified' not in st.session_state:
    st.session_state.identity_verified = False

# Logout function
def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.email = ""
    st.session_state.user_id = None
    st.experimental_rerun()

# Main menu function
def main_menu():
    st.subheader("Main Menu")
    menu_options = [f"{st.session_state.username}'s Portfolio", f"{st.session_state.username}'s Watchlist", "Stock Screener", "Stock Analysis",
                    "Markets", "My Account", "Database Admin Page"]
    choice = st.selectbox("Select an option", menu_options)
    return choice

# Sidebar menu
with st.sidebar:
    st.title("TradeSense")
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
