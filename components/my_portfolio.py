import streamlit as st
from utils.mongodb import trades_collection, users_collection
from utils.constants import bse_largecap, bse_midcap
import yfinance as yf
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

# Set Streamlit page configuration
st.set_page_config(page_title="Real-Time Trading Portfolio", layout="wide")

# Cache company names to reduce API calls
@st.cache_data(ttl=600)
def fetch_company_names(tickers):
    ticker_to_name = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        name = stock.info.get('shortName', ticker)
        ticker_to_name[ticker] = name
    return ticker_to_name

# Helper functions
def get_user_id(email):
    user = users_collection.find_one({"email": email})
    return user['_id'] if user else None

def get_company_name(ticker, ticker_to_company):
    return ticker_to_company.get(ticker, ticker)

# Initialize tickers and company names
all_tickers = bse_largecap + bse_midcap
ticker_to_company = fetch_company_names(all_tickers)
company_names = list(ticker_to_company.values())

def display_portfolio():
    st.header(f"{st.session_state.username}'s Portfolio")
    user_id = get_user_id(st.session_state.email)

    # Sidebar: Add to Portfolio
    st.sidebar.header("Portfolio Management")
    st.sidebar.subheader("Trade Stock")
    selected_company = st.sidebar.selectbox('Select Company:', [""] + company_names)
    ticker = next((t for t, name in ticker_to_company.items() if name == selected_company), None) if selected_company else None
    
    trade_type = st.sidebar.selectbox("Trade Type", ["Buy", "Sell"])
    shares = st.sidebar.number_input("Number of Shares", min_value=0.01, step=0.01, key="trade_shares")
    price_per_share = st.sidebar.number_input("Price per Share", min_value=0.01, step=0.01, key="trade_price")
    brokerage = st.sidebar.number_input("Brokerage Charges", min_value=0.0, step=0.01, key="trade_brokerage")
    trade_date = st.sidebar.date_input("Trade Date", datetime.today(), key="trade_date")

    # Global in-memory trade book and portfolio for the current user session
    if 'trade_book' not in st.session_state:
        st.session_state.trade_book = pd.DataFrame(columns=['Date', 'Stock', 'Action', 'Quantity', 'Price'])
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = pd.DataFrame(columns=['Stock', 'Quantity', 'Average Cost'])
    if 'pnl_statement' not in st.session_state:
        st.session_state.pnl_statement = pd.DataFrame(columns=['Date', 'Stock', 'Gross Profit', 'Net Profit', 'Gross Profit %', 'Net Profit %'])

    def register_trade(stock, action, quantity, price, trade_date):
        trade_book = st.session_state.trade_book
        portfolio = st.session_state.portfolio
        pnl_statement = st.session_state.pnl_statement

        # Record the trade in the trade book with the provided trade date
        new_trade = {
            'user_id': get_user_id(st.session_state.email),
            'Date': pd.Timestamp(trade_date),
            'Stock': stock,
            'Action': action,
            'Quantity': quantity,
            'Price': price
        }
        trades_collection.insert_one(new_trade)
        trade_book = pd.concat([trade_book, pd.DataFrame([new_trade])], ignore_index=True)
        
        if action == 'BUY':
            # Update Portfolio
            if stock in portfolio['Stock'].values:
                portfolio.loc[portfolio['Stock'] == stock, 'Quantity'] += quantity
                portfolio.loc[portfolio['Stock'] == stock, 'Average Cost'] = (
                    (portfolio.loc[portfolio['Stock'] == stock, 'Average Cost'] * (portfolio.loc[portfolio['Stock'] == stock, 'Quantity'] - quantity)) + (price * quantity)
                ) / portfolio.loc[portfolio['Stock'] == stock, 'Quantity']
            else:
                new_portfolio_entry = pd.DataFrame([{
                    'Stock': stock,
                    'Quantity': quantity,
                    'Average Cost': price
                }])
                portfolio = pd.concat([portfolio, new_portfolio_entry], ignore_index=True)
        
        elif action == 'SELL':
            # Update Portfolio
            if stock in portfolio['Stock'].values:
                portfolio.loc[portfolio['Stock'] == stock, 'Quantity'] -= quantity
                
                # Calculate Realized P&L
                avg_cost = portfolio.loc[portfolio['Stock'] == stock, 'Average Cost'].values[0]
                gross_profit = (price - avg_cost) * quantity
                net_profit = gross_profit - brokerage
                gross_profit_pct = (gross_profit / (avg_cost * quantity)) * 100
                net_profit_pct = (net_profit / (avg_cost * quantity)) * 100
                
                new_pnl_entry = pd.DataFrame([{
                    'Date': pd.Timestamp(trade_date),
                    'Stock': stock,
                    'Gross Profit': gross_profit,
                    'Net Profit': net_profit,
                    'Gross Profit %': gross_profit_pct,
                    'Net Profit %': net_profit_pct
                }])
                pnl_statement = pd.concat([pnl_statement, new_pnl_entry], ignore_index=True)

                # Ensure that all required columns are present
                for column in ['Date', 'Stock', 'Gross Profit', 'Net Profit', 'Gross Profit %', 'Net Profit %']:
                    if column not in pnl_statement.columns:
                        pnl_statement[column] = pd.NaT if column == 'Date' else 0
                
                # If all shares sold, remove from portfolio
                if portfolio.loc[portfolio['Stock'] == stock, 'Quantity'].values[0] == 0:
                    portfolio = portfolio[portfolio['Stock'] != stock]

        # Debugging output to check for Date column presence
        if 'Date' not in pnl_statement.columns:
            st.write("Debugging: 'Date' column is not in pnl_statement")

        # Save the updates back to session state
        st.session_state.trade_book = trade_book
        st.session_state.portfolio = portfolio
        st.session_state.pnl_statement = pnl_statement

    if st.sidebar.button("Execute Trade"):
        if ticker and shares > 0 and price_per_share > 0:
            try:
                action = 'BUY' if trade_type == 'Buy' else 'SELL'
                register_trade(ticker, action, shares, price_per_share, trade_date)
                st.sidebar.success(f"{trade_type} {shares} shares of {selected_company} ({ticker}) at ₹{price_per_share:.2f} per share.")
                st.experimental_rerun()
            except Exception as e:
                st.sidebar.error(f"Error executing trade: {e}")
        else:
            st.sidebar.error("Please select a company and enter valid trade details.")

    

    # Display Portfolio
    if not st.session_state.portfolio.empty:
        #st.subheader("Portfolio")
        portfolio_df = st.session_state.portfolio.copy()
        portfolio_df['Company Name'] = portfolio_df['Stock'].apply(lambda x: get_company_name(x, ticker_to_company))
        portfolio_df['Last Traded Price'] = portfolio_df['Stock'].apply(lambda x: yf.Ticker(x).history(period="1d")['Close'].iloc[-1])
        portfolio_df['Current Value'] = portfolio_df['Quantity'] * portfolio_df['Last Traded Price']
        portfolio_df['P&L'] = portfolio_df['Current Value'] - (portfolio_df['Quantity'] * portfolio_df['Average Cost'])
        portfolio_df['P&L (%)'] = (portfolio_df['P&L'] / (portfolio_df['Quantity'] * portfolio_df['Average Cost'])) * 100
        st.dataframe(portfolio_df, use_container_width=True)
        # Display summary metrics
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Portfolio Distribution")
            fig1 = go.Figure(go.Pie(labels=portfolio_df['Company Name'], values=portfolio_df['Current Value'], hole=.3))
            fig1.update_layout(showlegend=True, legend_title_text='Companies')
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            st.subheader("Current P&L(%)")
            fig2 = go.Figure(go.Bar(x=portfolio_df['Company Name'], y=portfolio_df['P&L (%)']))
            fig2.update_layout(title_text='Profit/Loss Percentage per Stock', xaxis_title='Company', yaxis_title='P&L (%)', showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No current holdings in your portfolio.")

    st.divider()
    # Display P&L Statement
    st.subheader("P&L Statement")
    pnl_df = st.session_state.pnl_statement.copy()
    if not pnl_df.empty:
        if 'Date' in pnl_df.columns:
            pnl_df['Date'] = pd.to_datetime(pnl_df['Date']).dt.date
            st.dataframe(pnl_df[['Date', 'Stock', 'Gross Profit', 'Net Profit', 'Gross Profit %', 'Net Profit %']], use_container_width=True)

            #st.subheader("Net Profit/Loss Over Time")
            pnl_df['Cumulative Net Profit'] = pnl_df['Net Profit'].cumsum()
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=pnl_df['Date'], y=pnl_df['Cumulative Net Profit'], name='Net Profit', marker_color='green'))
            fig3.update_layout(title='Net Profit/Loss Over Time', xaxis_title='Date', yaxis_title='Cumulative Net Profit', template='plotly_dark')
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("No 'Date' column found in P&L statement.")
    else:
        st.info("No profit/loss data available.")
    st.divider()
    # Display Trade Book
    st.subheader("Trade Book")
    st.dataframe(st.session_state.trade_book, use_container_width=True)

# User Session Management
if 'username' not in st.session_state:
    st.session_state.username = 'Guest'
if 'email' not in st.session_state:
    st.session_state.email = 'guest@example.com'
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.sidebar.header("Login")
    email = st.sidebar.text_input("Email", key="login_email")
    password = st.sidebar.text_input("Password", type="password", key="login_password")
    if st.sidebar.button("Login"):
        user = users_collection.find_one({"email": email, "password": password})  # Ensure password is hashed in production
        if user:
            st.session_state.logged_in = True
            st.session_state.username = user.get('username', 'User')
            st.session_state.email = email
            st.success(f"Logged in as {st.session_state.username}")
            st.experimental_rerun()
        else:
            st.sidebar.error("Invalid credentials")

if st.session_state.logged_in:
    display_portfolio()
else:
    login()
    st.write("Please log in to view your portfolio.")
