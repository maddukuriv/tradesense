import streamlit as st
from utils.mongodb import portfolios_collection, users_collection
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go



def get_user_id(email):
    user = users_collection.find_one({"email": email})
    return user['_id'] if user else None


# Portfolio feature
def display_portfolio():
    st.header(f"{st.session_state.username}'s Portfolio")
    user_id = get_user_id(st.session_state.email)
    portfolio = list(portfolios_collection.find({"user_id": user_id}))

    # Add new stock to portfolio
    st.sidebar.subheader("Add to Portfolio")
    new_ticker = st.text_input("Ticker Symbol")
    shares = st.number_input("Number of Shares", min_value=0.0, step=0.01)
    bought_price = st.number_input("Bought Price per Share", min_value=0.0, step=0.01)
    if st.sidebar.button("Add to Portfolio"):
        try:
            current_data = yf.download(new_ticker, period='1d')
            if current_data.empty:
                raise ValueError("Ticker not found")

            if not portfolios_collection.find_one({"user_id": user_id, "ticker": new_ticker}):
                portfolios_collection.insert_one({
                    "user_id": user_id,
                    "ticker": new_ticker,
                    "shares": shares,
                    "bought_price": bought_price,
                    "date_added": pd.Timestamp.now()
                })
                st.success(f"{new_ticker} added to your portfolio!")
            else:
                st.warning(f"{new_ticker} is already in your portfolio.")
        except Exception as e:
            st.error(f"Error adding ticker: {e}")

    # Display portfolio
    if portfolio:
        portfolio_data = []
        invested_values = []
        current_values = []
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
                portfolio_data.append({
                    "Ticker": entry['ticker'],
                    "Shares": entry['shares'],
                    "Bought Price": entry['bought_price'],
                    "Invested Value": invested_value,
                    "Last Traded Price": last_price,
                    "Current Value": current_value,
                    "P&L (%)": p_l_percent
                })
                invested_values.append(invested_value)
                current_values.append(current_value)
            except Exception as e:
                st.error(f"Error retrieving data for {entry['ticker']}: {e}")

        portfolio_df = pd.DataFrame(portfolio_data)

        st.write("Your Portfolio:")
        st.dataframe(portfolio_df)

        col1, col2 = st.columns(2)

        with col1:
            labels = portfolio_df['Ticker']
            values = portfolio_df['Current Value']
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
            fig.update_layout(title_text="Portfolio Distribution")
            st.plotly_chart(fig)

        with col2:
            total_invested_value = sum(invested_values)
            total_current_value = sum(current_values)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=['Total Invested Value', 'Total Current Value'], y=[total_invested_value, total_current_value]))
            fig.update_layout(title_text='Profit/Loss', xaxis_title='Value', yaxis_title='Sum')
            st.plotly_chart(fig)

        ticker_to_remove = st.sidebar.selectbox("Select a ticker to remove", [entry['ticker'] for entry in portfolio])
        if st.sidebar.button("Remove from Portfolio"):
            portfolios_collection.delete_one({"user_id": user_id, "ticker": ticker_to_remove})
            st.success(f"{ticker_to_remove} removed from your portfolio.")
            st.experimental_rerun()  # Refresh the app to reflect changes
    else:
        st.write("Your portfolio is empty.")
