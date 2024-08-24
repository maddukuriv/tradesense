import streamlit as st
from utils.mongodb import portfolios_collection, users_collection, buy_trades_collection, sell_trades_collection
from utils.constants import bse_largecap, bse_smallcap, bse_midcap, sp500_tickers, ftse100_tickers
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta, datetime

# Helper function to get user ID from email
def get_user_id(email):
    user = users_collection.find_one({"email": email})
    return user['_id'] if user else None

# Helper function to get company info
def get_company_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info.get('longName', 'N/A'), info.get('sector', 'N/A'), info.get('industry', 'N/A')
    except Exception as e:
        return 'N/A', 'N/A', 'N/A'

# Function to get company names from tickers
def get_company_name(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.info.get('shortName', ticker)
    except:
        return ticker  # Return ticker if company name not found

# Assuming a list of tickers (this could be from an index or predefined list)
all_tickers = bse_largecap + bse_smallcap + bse_midcap
ticker_to_company = {ticker: get_company_name(ticker) for ticker in all_tickers}
company_names = list(ticker_to_company.values())

# Portfolio feature
def display_portfolio():
    st.header(f"{st.session_state.username}'s Portfolio")
    user_id = get_user_id(st.session_state.email)

    # Refresh portfolio data
    portfolio = list(portfolios_collection.find({"user_id": user_id}))
    print(portfolio)
    # Add new stock to portfolio
    st.sidebar.subheader("Add to Portfolio")
    selected_company = st.sidebar.selectbox('Select or Enter Company Name:', company_names)
    ticker = [ticker for ticker, company in ticker_to_company.items() if company == selected_company][0]
    shares = st.sidebar.number_input("Number of Shares", min_value=0.0, step=0.01)
    bought_price = st.sidebar.number_input("Bought Price per Share", min_value=0.0, step=0.01)
    buy_brokerage = st.sidebar.number_input("Buy Brokerage Charges", min_value=0.0, step=0.01)
    buy_date = st.sidebar.date_input("Buy Date", pd.Timestamp.now(), key="buy_date")
    if st.sidebar.button("Add to Portfolio"):
        try:
            current_data = yf.download(ticker, period='1d')
            if current_data.empty:
                raise ValueError("Ticker not found")

            # Log the buy trade
            buy_trades_collection.insert_one({
                "user_id": user_id,
                "ticker": ticker,
                "shares": shares,
                "bought_price": bought_price,
                "brokerage": buy_brokerage,
                "date": pd.Timestamp(buy_date)
            })

            # Add to or update portfolio
            portfolios_collection.insert_one({
                "user_id": user_id,
                "ticker": ticker,
                "shares": shares,
                "bought_price": bought_price,
                "brokerage": buy_brokerage,
                "date_added": pd.Timestamp(buy_date)
            })
            st.success(f"{selected_company} ({ticker}) added to your portfolio!")
            st.experimental_rerun()  # Refresh the app to reflect changes
        except Exception as e:
            st.error(f"Error adding stock: {e}")

    # Display portfolio
    if portfolio:
        portfolio_data = []
        for entry in portfolio:
            try:
                current_data = yf.download(entry['ticker'], period='1d')
                if current_data.empty:
                    raise ValueError(f"Ticker {entry['ticker']} not found")

                last_price = current_data['Close'].iloc[-1]
                invested_value = entry['shares'] * entry['bought_price']
                current_value = entry['shares'] * last_price
                p_l = current_value - invested_value
                p_l_percent = (p_l / invested_value) * 100
                company_name, sector, industry = get_company_info(entry['ticker'])
                portfolio_data.append({
                    "Ticker": entry['ticker'],
                    "Company Name": company_name,
                    "Sector": sector,
                    "Industry": industry,
                    "Shares": entry['shares'],
                    "Bought Price": entry['bought_price'],
                    "Invested Value": invested_value,
                    "Last Traded Price": last_price,
                    "Current Value": current_value,
                    "P&L (%)": p_l_percent,
                    "Date Added": entry['date_added']
                })
            except Exception as e:
                st.error(f"Error retrieving data for {entry['ticker']}: {e}")

        portfolio_df = pd.DataFrame(portfolio_data)

        st.write("Your Portfolio:")
        st.dataframe(portfolio_df)

        col1, col2 = st.columns(2)

        with col1:
            labels = portfolio_df['Company Name']
            values = portfolio_df['Current Value']
            fig1 = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
            fig1.update_layout(title_text="Portfolio Distribution")
            st.plotly_chart(fig1)

        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=portfolio_df['Company Name'], y=portfolio_df['P&L (%)']))
            fig2.update_layout(title_text='Profit Percentage of Each Stock', xaxis_title='Company', yaxis_title='P&L (%)')
            st.plotly_chart(fig2)

        # Edit stock in portfolio
        st.sidebar.subheader("Edit Portfolio")
        company_names_in_portfolio = [entry['Company Name'] for entry in portfolio_data]
        company_to_edit = st.sidebar.selectbox("Select a company to edit", company_names_in_portfolio)
        ticker_to_edit = [entry['Ticker'] for entry in portfolio_data if entry['Company Name'] == company_to_edit][0]

        # Fetch current values for the selected company
        entry_to_edit = next(item for item in portfolio if item['ticker'] == ticker_to_edit)
        current_shares = entry_to_edit['shares']
        current_bought_price = entry_to_edit['bought_price']
        current_date_added = entry_to_edit.get('date_added', pd.Timestamp.now())

        # Display current values and allow editing
        new_shares = st.sidebar.number_input("Number of Shares", value=current_shares, min_value=0.0, step=0.01)
        new_bought_price = st.sidebar.number_input("Bought Price per Share", value=current_bought_price, min_value=0.0, step=0.01)
        new_date_added = st.sidebar.date_input("Buy Date", current_date_added, key="edit_date")

        if st.sidebar.button("Save Changes"):
            portfolios_collection.update_one(
                {"user_id": user_id, "ticker": ticker_to_edit, "date_added": current_date_added},
                {"$set": {"shares": new_shares, "bought_price": new_bought_price, "date_added": pd.Timestamp(new_date_added)}}
            )
            st.success(f"{company_to_edit} ({ticker_to_edit}) updated in your portfolio.")
            st.experimental_rerun()  # Refresh the app to reflect changes

        # Sell stock from portfolio
        st.sidebar.subheader("Sell from Portfolio")
        company_to_sell = st.sidebar.selectbox("Select a company to sell", company_names_in_portfolio)
        ticker_to_sell = [entry['Ticker'] for entry in portfolio_data if entry['Company Name'] == company_to_sell][0]

        # Fetch current values for the selected company
        entry_to_sell = next(item for item in portfolio if item['ticker'] == ticker_to_sell)
        sell_shares = st.sidebar.number_input("Number of Shares to Sell", min_value=0.0, max_value=entry_to_sell['shares'], step=0.01)
        sell_price = st.sidebar.number_input("Sell Price per Share", min_value=0.0, step=0.01)
        sell_brokerage = st.sidebar.number_input("Sell Brokerage Charges", min_value=0.0, step=0.01)
        sell_date = st.sidebar.date_input("Sell Date", pd.Timestamp.now(), key="sell_date")

        if st.sidebar.button("Sell Stock"):
            if sell_shares > entry_to_sell['shares']:
                st.error(f"Cannot sell more shares than you own for {company_to_sell}.")
            else:
                # Log the sell trade
                sell_trades_collection.insert_one({
                    "user_id": user_id,
                    "ticker": ticker_to_sell,
                    "shares": sell_shares,
                    "sell_price": sell_price,
                    "brokerage": sell_brokerage,
                    "date": pd.Timestamp(sell_date)
                })

                # Calculate net profit/loss
                bought_price = entry_to_sell['bought_price']
                buy_brokerage = entry_to_sell.get('brokerage', 0)  # Use .get to provide a default value of 0 if 'brokerage' is missing

                # Calculate the brokerage cost per share
                brokerage_per_share = buy_brokerage / entry_to_sell['shares']

                total_sell_value = sell_shares * sell_price - sell_brokerage
                total_invested_value = sell_shares * (bought_price + brokerage_per_share)
                net_profit_loss = total_sell_value - total_invested_value
                
                st.success(f"Sold {sell_shares} shares of {company_to_sell} for a net {'profit' if net_profit_loss >= 0 else 'loss'} of {net_profit_loss:.2f}.")

                # Update the portfolio
                remaining_shares = entry_to_sell['shares'] - sell_shares
                if remaining_shares > 0:
                    portfolios_collection.update_one(
                        {"user_id": user_id, "ticker": ticker_to_sell, "date_added": entry_to_sell['date_added']},
                        {"$set": {"shares": remaining_shares}}
                    )
                else:
                    portfolios_collection.delete_one({"user_id": user_id, "ticker": ticker_to_sell, "date_added": entry_to_sell['date_added']})
                    st.success(f"{company_to_sell} removed from your portfolio.")

                st.experimental_rerun()  # Refresh the app to reflect changes

        # Display P&L table for trades
        st.header("P&L Statement")
        trades_data = []
        sell_trades = list(sell_trades_collection.find({"user_id": user_id}))
        for trade in sell_trades:
            # Retrieve corresponding buy trade details
            buy_trades = list(buy_trades_collection.find({"user_id": user_id, "ticker": trade['ticker']}))
            if buy_trades:
                total_buy_value = sum([bt['shares'] * (bt['bought_price'] + (bt['brokerage'] / bt['shares'])) for bt in buy_trades])
                avg_buy_price = total_buy_value / sum([bt['shares'] for bt in buy_trades])
                total_sell_value = trade['shares'] * trade['sell_price'] - trade['brokerage']
                net_p_l = total_sell_value - (trade['shares'] * avg_buy_price)
                trades_data.append({
                    "Ticker": trade['ticker'],
                    "Shares Sold": trade['shares'],
                    "Buy Price (Avg)": avg_buy_price,
                    "Sell Price": trade['sell_price'],
                    "Buy Brokerage (Per Share)": sum([bt['brokerage'] for bt in buy_trades]) / sum([bt['shares'] for bt in buy_trades]),
                    "Sell Brokerage": trade['brokerage'],
                    "Net P&L": net_p_l,
                    "Sell Date": trade['date']
                })

        p_l_df = pd.DataFrame(trades_data)
        if not p_l_df.empty:
            p_l_df['Sell Date'] = pd.to_datetime(p_l_df['Sell Date'])
            p_l_df.set_index('Sell Date', inplace=True)

        st.write("Editable P&L Statement:")
        edited_p_l_df = st.data_editor(p_l_df, num_rows="dynamic")

        # Generate plots for P&L with dates on the x-axis
        if not p_l_df.empty:
            col3, col4 = st.columns(2)
            with col3:
                fig3 = go.Figure()
                fig3.add_trace(go.Bar(x=edited_p_l_df.index, y=edited_p_l_df['Net P&L'], name='Net P&L'))
                fig3.update_layout(title_text='Net Profit/Loss Over Time', xaxis_title='Date', yaxis_title='Net P&L')
                st.plotly_chart(fig3)

            with col4:
                total_pl = edited_p_l_df['Net P&L'].sum()
                st.metric("Total Net P&L", f"{total_pl:.2f}")

    else:
        st.write("Your portfolio is empty.")

# Call the function to display the portfolio
if 'username' not in st.session_state:
    st.session_state.username = 'Guest'  # or handle the case where username is not set
if 'email' not in st.session_state:
    st.session_state.email = 'guest@example.com'  # or handle the case where email is not set
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False  # or handle the case where logged_in is not set

if st.session_state.logged_in:
    display_portfolio()
else:
    st.write("Please log in to view your portfolio.")
