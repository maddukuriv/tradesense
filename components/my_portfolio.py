import streamlit as st
from utils.mongodb import trades_collection, users_collection
from utils.constants import ticker_to_company_dict
from supabase import create_client
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Supabase client setup 

SUPABASE_URL= 'https://kbhdeynmboawkjtxvlek.supabase.co'
SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtiaGRleW5tYm9hd2tqdHh2bGVrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDA2NDg3NjAsImV4cCI6MjA1NjIyNDc2MH0.T3L5iIn1FiBlBo5HZMqysgokD8cfOw2n3u_YCJV0DkQ'


supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_last_traded_price(ticker):
    try:
        response = supabase.table("stock_data").select("*").filter("ticker", "eq", ticker).order("date", desc=True).limit(1).execute()
        
        if response.data:
            return response.data[0]['close']  # Assuming 'close' column holds the last traded price
        else:
            return None  # Return None if no data found
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Function to get company name from the dictionary
def get_company_name(ticker, ticker_to_company):
    return ticker_to_company.get(ticker, ticker)


# Updated get_user_id function
def get_user_id(email):
    user = users_collection.find_one({"email": email})
    return user['_id'] if user else None

# Initialize tickers and company names
all_tickers = list(ticker_to_company_dict.keys())
company_names = list(ticker_to_company_dict.values())

# Load user trades from the database
def load_user_trades(user_id):
    trades = list(trades_collection.find({"user_id": user_id}))
    
    if trades:
        trade_book = pd.DataFrame(trades)
        trade_book['Date'] = pd.to_datetime(trade_book['Date'])
        st.session_state.trade_book = trade_book
    else:
        st.session_state.trade_book = pd.DataFrame(columns=['Date', 'Stock', 'Action', 'Quantity', 'Price', 'Brokerage'])
    
    portfolio = pd.DataFrame(columns=['Stock', 'Quantity', 'Average Cost'])
    pnl_statement = pd.DataFrame(columns=['Date', 'Stock', 'Gross Profit', 'Net Profit', 'Gross Profit %', 'Net Profit %'])
    
    if not st.session_state.trade_book.empty:
        for _, trade in st.session_state.trade_book.iterrows():
            stock = trade['Stock']
            action = trade['Action']
            quantity = trade['Quantity']
            price = trade['Price']
            brokerage = trade.get('Brokerage', 0)
            date = trade['Date']
            
            if action == 'BUY':
                if stock in portfolio['Stock'].values:
                    existing_quantity = portfolio.loc[portfolio['Stock'] == stock, 'Quantity'].values[0]
                    existing_avg_cost = portfolio.loc[portfolio['Stock'] == stock, 'Average Cost'].values[0]
                    
                    new_quantity = existing_quantity + quantity
                    new_avg_cost = ((existing_avg_cost * existing_quantity) + (price * quantity)) / new_quantity
                    
                    portfolio.loc[portfolio['Stock'] == stock, 'Quantity'] = new_quantity
                    portfolio.loc[portfolio['Stock'] == stock, 'Average Cost'] = new_avg_cost
                else:
                    new_portfolio_entry = pd.DataFrame([{
                        'Stock': stock,
                        'Quantity': quantity,
                        'Average Cost': price
                    }])
                    portfolio = pd.concat([portfolio, new_portfolio_entry], ignore_index=True)
            
            elif action == 'SELL':
                if stock in portfolio['Stock'].values:
                    existing_quantity = portfolio.loc[portfolio['Stock'] == stock, 'Quantity'].values[0]
                    avg_cost = portfolio.loc[portfolio['Stock'] == stock, 'Average Cost'].values[0]
                    
                    new_quantity = existing_quantity - quantity
                    gross_profit = (price - avg_cost) * quantity
                    net_profit = gross_profit - brokerage
                    gross_profit_pct = (gross_profit / (avg_cost * quantity)) * 100
                    net_profit_pct = (net_profit / (avg_cost * quantity)) * 100
                    
                    pnl_entry = pd.DataFrame([{
                        'Date': date,
                        'Stock': stock,
                        'Gross Profit': gross_profit,
                        'Net Profit': net_profit,
                        'Gross Profit %': gross_profit_pct,
                        'Net Profit %': net_profit_pct
                    }])
                    pnl_statement = pd.concat([pnl_statement, pnl_entry], ignore_index=True)
                    
                    if new_quantity > 0:
                        portfolio.loc[portfolio['Stock'] == stock, 'Quantity'] = new_quantity
                    else:
                        portfolio = portfolio[portfolio['Stock'] != stock]

    st.session_state.portfolio = portfolio
    st.session_state.pnl_statement = pnl_statement

def display_portfolio():
    st.header(f"{st.session_state.username}'s Portfolio")
    user_id = get_user_id(st.session_state.email)

    if user_id is not None and 'trade_book' not in st.session_state:
        load_user_trades(user_id)

    st.sidebar.header("Portfolio Management")
    st.sidebar.subheader("Trade Stock")
    selected_company = st.sidebar.selectbox('Select Company:', [""] + company_names)
    ticker = next((t for t, name in ticker_to_company_dict.items() if name == selected_company), None) if selected_company else None
    
    trade_type = st.sidebar.selectbox("Trade Type", ["Buy", "Sell"])
    shares = st.sidebar.number_input("Number of Shares", min_value=0.01, step=0.01, key="trade_shares")
    price_per_share = st.sidebar.number_input("Price per Share", min_value=0.01, step=0.01, key="trade_price")
    brokerage = st.sidebar.number_input("Brokerage Charges", min_value=0.0, step=0.01, key="trade_brokerage")
    trade_date = st.sidebar.date_input("Trade Date", datetime.today(), key="trade_date")

    def register_trade(stock, action, quantity, price, trade_date):
        trade_book = st.session_state.trade_book
        portfolio = st.session_state.portfolio
        pnl_statement = st.session_state.pnl_statement

        if user_id is None:
            st.error("User ID could not be found. Ensure the user is logged in.")
            return

        new_trade = {
            'user_id': user_id,
            'Date': pd.Timestamp(trade_date),
            'Stock': stock,
            'Action': action,
            'Quantity': quantity,
            'Price': price,
            'Brokerage': brokerage
        }

        try:
            result = trades_collection.insert_one(new_trade)
            st.write("Trade inserted with ID:", result.inserted_id)
            
            trade_book = pd.concat([trade_book, pd.DataFrame([new_trade])], ignore_index=True)
            
            if action == 'BUY':
                if stock in portfolio['Stock'].values:
                    existing_quantity = portfolio.loc[portfolio['Stock'] == stock, 'Quantity'].values[0]
                    existing_avg_cost = portfolio.loc[portfolio['Stock'] == stock, 'Average Cost'].values[0]
                    
                    new_quantity = existing_quantity + quantity
                    new_avg_cost = ((existing_avg_cost * existing_quantity) + (price * quantity)) / new_quantity
                    
                    portfolio.loc[portfolio['Stock'] == stock, 'Quantity'] = new_quantity
                    portfolio.loc[portfolio['Stock'] == stock, 'Average Cost'] = new_avg_cost
                else:
                    new_portfolio_entry = pd.DataFrame([{
                        'Stock': stock,
                        'Quantity': quantity,
                        'Average Cost': price
                    }])
                    portfolio = pd.concat([portfolio, new_portfolio_entry], ignore_index=True)
            
            elif action == 'SELL':
                if stock in portfolio['Stock'].values:
                    existing_quantity = portfolio.loc[portfolio['Stock'] == stock, 'Quantity'].values[0]
                    avg_cost = portfolio.loc[portfolio['Stock'] == stock, 'Average Cost'].values[0]
                    
                    new_quantity = existing_quantity - quantity
                    gross_profit = (price - avg_cost) * quantity
                    net_profit = gross_profit - brokerage
                    gross_profit_pct = (gross_profit / (avg_cost * quantity)) * 100
                    net_profit_pct = (net_profit / (avg_cost * quantity)) * 100
                    
                    pnl_entry = pd.DataFrame([{
                        'Date': trade_date,
                        'Stock': stock,
                        'Gross Profit': gross_profit,
                        'Net Profit': net_profit,
                        'Gross Profit %': gross_profit_pct,
                        'Net Profit %': net_profit_pct
                    }])
                    pnl_statement = pd.concat([pnl_statement, pnl_entry], ignore_index=True)
                    
                    if new_quantity > 0:
                        portfolio.loc[portfolio['Stock'] == stock, 'Quantity'] = new_quantity
                    else:
                        portfolio = portfolio[portfolio['Stock'] != stock]

            st.session_state.trade_book = trade_book
            st.session_state.portfolio = portfolio
            st.session_state.pnl_statement = pnl_statement

        except Exception as e:
            st.error(f"Error inserting trade into the database: {e}")

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

    if not st.session_state.portfolio.empty:
        portfolio_df = st.session_state.portfolio.copy()
        portfolio_df['Company Name'] = portfolio_df['Stock'].apply(lambda x: get_company_name(x, ticker_to_company_dict))
        portfolio_df['Last Traded Price'] = portfolio_df['Stock'].apply(lambda x: get_last_traded_price(x))
        portfolio_df['Invested Value'] = portfolio_df['Quantity'] * portfolio_df['Average Cost']
        portfolio_df['Current Value'] = portfolio_df['Quantity'] * portfolio_df['Last Traded Price']
        portfolio_df['P&L'] = portfolio_df['Current Value'] - (portfolio_df['Quantity'] * portfolio_df['Average Cost'])
        portfolio_df['P&L (%)'] = (portfolio_df['P&L'] / (portfolio_df['Quantity'] * portfolio_df['Average Cost'])) * 100
      

        # Split into two DataFrames
        portfolio_table_1 = portfolio_df[['Stock','Quantity','Average Cost','Last Traded Price']]
        portfolio_table_2 = portfolio_df[['Company Name','Invested Value' , 'Current Value', 'P&L', 'P&L (%)']]

        # Streamlit layout with two columns
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Investment Details")
            st.dataframe(portfolio_table_1, use_container_width=True)

        with col2:
            st.subheader("Performance")
            st.dataframe(portfolio_table_2, use_container_width=True)


        # Streamlit layout with two columns
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Portfolio Distribution")
            
            # Pie chart using Plotly Express
            fig1 = px.pie(portfolio_df, 
                        names='Company Name', 
                        values='Current Value', 
                        #title="Portfolio Distribution",
                        hole=0.3)
            fig1.update_layout(showlegend=True, legend_title_text='Companies')
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            # Total values
            total_invested = portfolio_df['Invested Value'].sum()
            total_current = portfolio_df['Current Value'].sum()

            st.subheader("Current Holdings P&L")
            
            # Create a DataFrame for the data
            data = pd.DataFrame({
                'Metric': ['Invested Value', 'Current Value'],
                'Value': [total_invested, total_current]
            })
            
            # Bar chart for current holdings P&L
            fig3 = px.bar(data, 
                        x='Metric', 
                        y='Value', 
                        color='Metric',
                        color_discrete_sequence=['gray', 'blue'],
                        #title="Current Holdings P&L"
                        )
            
            fig3.update_layout(
                xaxis_title='Metric',
                yaxis_title='Value',
                showlegend=False,  # Hide legend since color already represents metrics
                bargap=0.2         # Add gap between bars
            )
            st.plotly_chart(fig3, use_container_width=True)

        # Sort the DataFrame by 'P&L (%)' in ascending order
        portfolio_df_sorted = portfolio_df.sort_values(by='P&L (%)', ascending=True)

        # Create the bar chart using the sorted DataFrame
        st.subheader("Current Holdings - Individual P&L(%)")
        fig2 = px.bar(portfolio_df_sorted, 
                    x='Company Name', 
                    y='P&L (%)', 
                    #title="Current Holdings - Individual P&L(%)",
                    labels={'Company Name': 'Company', 'P&L (%)': 'P&L (%)'},
                    color='Company Name')  # Optional: color for variety
        fig2.update_layout(
            xaxis_title='Company',
            yaxis_title='P&L (%)',
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.info("No current holdings in your portfolio.")

    st.divider()

    st.subheader("P&L Statement")
    pnl_df = st.session_state.pnl_statement.copy()
    if not pnl_df.empty:
        if 'Date' in pnl_df.columns:
            pnl_df['Date'] = pd.to_datetime(pnl_df['Date']).dt.date
            st.dataframe(pnl_df[['Date', 'Stock', 'Gross Profit', 'Net Profit', 'Gross Profit %', 'Net Profit %']], use_container_width=True)

            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=pnl_df['Date'], y=pnl_df['Net Profit'], name='Net Profit', marker_color='green'))
            fig3.update_layout(title='Net Profit/Loss Over Time', xaxis_title='Date', yaxis_title='Net Profit', template='plotly_dark')
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("No 'Date' column found in P&L statement.")
    else:
        st.info("No profit/loss data available.")
    
    st.divider()

    st.subheader("Trade Book")
    st.dataframe(st.session_state.trade_book, use_container_width=True)

# User Session Management
if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    display_portfolio()
else:
    st.write("Please log in to view your portfolio.")