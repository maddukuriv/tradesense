import streamlit as st
import yfinance as yf
import pandas as pd
import threading
import time
from supabase import create_client
from datetime import datetime, timedelta
from utils.constants import Largecap, Midcap, Smallcap,sp500_tickers,ftse100_tickers,crypto_largecap,crypto_midcap,Indices,Commodities,Currencies,SUPABASE_URL,SUPABASE_KEY
from utils.mongodb import users_collection, watchlists_collection, trades_collection


# Supabase credentials
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Stock ticker categories
ticker_categories = {
    "Largecap": Largecap,
    "Midcap": Midcap,
    "Smallcap": Smallcap,
    "Indices": Indices
}

# ✅ Function to identify missing dates per ticker
def get_missing_dates_per_ticker(ticker, existing_data):
    ticker_dates = {date for t, date in existing_data if t == ticker}
    today = datetime.now().strftime('%Y-%m-%d')

    if ticker_dates:
        last_date = max(ticker_dates)
        start_date = (datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    all_dates = pd.date_range(start=start_date, end=today).strftime('%Y-%m-%d').tolist()
    missing_dates = sorted(set(all_dates) - ticker_dates)

    return missing_dates

# ✅ Function to fetch stock data and fill missing gaps
def fetch_stock_data(tickers_batch, results, existing_data):
    for ticker in tickers_batch:
        missing_dates = get_missing_dates_per_ticker(ticker, existing_data)

        if not missing_dates:
            print(f"✅ No missing data for {ticker}. Skipping...")
            continue

        start_date, end_date = missing_dates[0], missing_dates[-1]
        retries = 3

        while retries > 0:
            try:
                time.sleep(1.5)
                stock = yf.Ticker(ticker)
                stock_data = stock.history(start=start_date, end=end_date)

                if stock_data.empty:
                    print(f"⚠️ No data found for {ticker} ({start_date} to {end_date}). Skipping...")
                    break

                stock_data.reset_index(inplace=True)

                if 'Date' in stock_data.columns:
                    stock_data.rename(columns={'Date': 'date'}, inplace=True)
                elif 'date' not in stock_data.columns:
                    print(f"⚠️ 'date' column missing in data for {ticker}. Skipping...")
                    break

                stock_data['ticker'] = ticker
                stock_data['date'] = stock_data['date'].dt.strftime('%Y-%m-%d')
                stock_data.columns = [col.lower() for col in stock_data.columns]

                required_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']
                stock_data = stock_data[required_columns]

                filtered_data = [row for row in stock_data.to_dict(orient="records") if (row["ticker"], row["date"]) not in existing_data]

                if filtered_data:
                    results.extend(filtered_data)

                print(f"✅ Fetched data for {ticker} ({start_date} to {end_date})")
                break

            except Exception as e:
                print(f"⚠️ Failed to fetch data for {ticker}: {e}")
                retries -= 1
                time.sleep(3)

# ✅ Streamlit application
def etl_app():
    st.title("📈 Stock Data ETL")

    selected_category = st.selectbox("Select Stock Category", list(ticker_categories.keys()))
    tickers = ticker_categories[selected_category]

    if st.button("Load Data"):
        # ✅ Re-fetch existing data to keep it fresh per run
        existing_data_query = supabase.table("stock_data").select("ticker", "date").execute()
        existing_data = {(row["ticker"], row["date"]) for row in existing_data_query.data} if existing_data_query.data else set()

        # ✅ Check for latest date in Supabase
        latest_date_query = supabase.table("stock_data").select("date").order("date", desc=True).limit(1).execute()
        latest_date = latest_date_query.data[0]['date'] if latest_date_query.data else None

        today = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d') if not latest_date else (datetime.strptime(latest_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')

        if start_date >= today:
            st.warning("⚠️ No new stock data to fetch.")
            return

        updated_data = []
        threads = []
        batch_size = 50

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            thread = threading.Thread(target=fetch_stock_data, args=(batch, updated_data, existing_data))
            threads.append(thread)
            thread.start()
            time.sleep(2)

        for thread in threads:
            thread.join()

        if updated_data:
            try:
                supabase.table("stock_data").insert(updated_data).execute()
                st.success("✅ Missing stock data successfully updated in Supabase.")
            except Exception as e:
                st.error(f"⚠️ Error inserting stock data: {e}")
        else:
            st.info("ℹ️ No new stock data was added.")

        # ✅ Clean-up old data
        one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        try:
            supabase.table("stock_data").delete().filter("date", "lt", one_year_ago).execute()
            print("✅ Old data deleted (older than 1 year).")
        except Exception as e:
            print(f"⚠️ Error deleting old data: {e}")

if __name__ == "__main__":
    etl_app()