import streamlit as st
from dotenv import load_dotenv
from utils.mongodb import init_db
from authentication import login, signup, forgot_password
from components import my_account, my_portfolio, my_watchlist, markets, stock_screener, stock_analysis, admin, home_page,stock_comparision

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
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False  # Default to non-admin until proven otherwise

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
    # Define the options based on admin status
    menu_options = [
        f"{st.session_state.username}'s Portfolio",
        f"{st.session_state.username}'s Watchlist",
        "Stock Screener",
        "Stock Analysis",
        "Stock Comparision",
        "Markets",
        "My Account"

    ]

    # Add Database Admin Page only if the user is an admin
    if st.session_state.is_admin:
        menu_options.append("Database Admin Page")

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
        elif choice == "Stock Comparision":
            stock_comparision.display_stock_comparison()
        elif choice == "Database Admin Page" and st.session_state.is_admin:
            admin.display_tables()
        else:
            st.error("You don't have permission to access this page.")
