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

# ✅ Fetch last available date for each ticker in Supabase
existing_data_query = supabase.table("stock_data").select("ticker", "date").execute()
existing_data = {(row["ticker"], row["date"]) for row in existing_data_query.data} if existing_data_query.data else set()

# ✅ Define today's date
today = datetime.now().strftime('%Y-%m-%d')

# ✅ Identify missing dates per ticker
def get_missing_dates_per_ticker(ticker):
    ticker_dates = {date for t, date in existing_data if t == ticker}
    
    if ticker_dates:
        last_date = max(ticker_dates)  # Latest date for this ticker
        start_date = (datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # Fetch 1 year if no data exists

    all_dates = pd.date_range(start=start_date, end=today).strftime('%Y-%m-%d').tolist()
    missing_dates = sorted(set(all_dates) - ticker_dates)
    
    return missing_dates

# ✅ Function to fetch stock data and fill missing gaps
def fetch_stock_data(tickers_batch, results):
    for ticker in tickers_batch:
        missing_dates = get_missing_dates_per_ticker(ticker)
        
        if not missing_dates:
            print(f"✅ No missing data for {ticker}. Skipping...")
            continue
        
        start_date, end_date = missing_dates[0], missing_dates[-1]

        retries = 3  # Retry up to 3 times if request fails
        while retries > 0:
            try:
                time.sleep(1.5)  # Avoid hitting API rate limits
                stock = yf.Ticker(ticker)
                stock_data = stock.history(start=start_date, end=end_date)

                if stock_data.empty:
                    print(f"⚠️ No data found for {ticker} ({start_date} to {end_date}). Skipping...")
                    break  

                # ✅ Reset index and rename correctly
                stock_data.reset_index(inplace=True)

                # ✅ Ensure 'Date' column exists and convert it properly
                if 'Date' in stock_data.columns:
                    stock_data.rename(columns={'Date': 'date'}, inplace=True)
                elif 'date' not in stock_data.columns:
                    print(f"⚠️ 'date' column missing in data for {ticker}. Skipping...")
                    break  

                stock_data['ticker'] = ticker
                stock_data['date'] = stock_data['date'].dt.strftime('%Y-%m-%d')  # Convert datetime to string format
                
                stock_data.columns = [col.lower() for col in stock_data.columns]  # Convert all column names to lowercase
                
                required_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']
                stock_data = stock_data[required_columns]  # Select only relevant columns

                # ✅ Filter out duplicates before inserting
                filtered_data = [row for row in stock_data.to_dict(orient="records") if (row["ticker"], row["date"]) not in existing_data]

                if filtered_data:
                    results.extend(filtered_data)

                print(f"✅ Fetched data for {ticker} ({start_date} to {end_date})")
                break  # Success, exit retry loop

            except Exception as e:
                print(f"⚠️ Failed to fetch data for {ticker}: {e}")
                retries -= 1
                time.sleep(3)  # Wait 3 seconds before retrying

# Streamlit application
def etl_app():
    st.title("Stock Data ETL App")

    selected_category = st.selectbox("Select Stock Category", list(ticker_categories.keys()))
    tickers = ticker_categories[selected_category]

    if st.button("Load Data"):
        # Fetch the latest date from the database
        latest_date_query = supabase.table("stock_data").select("date").order("date", desc=True).limit(1).execute()
        latest_date = latest_date_query.data[0]['date'] if latest_date_query.data else None
        
        today = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d') if not latest_date else (datetime.strptime(latest_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')

        if start_date >= today:
            st.warning("No new stock data to fetch.")
            return
        
        updated_data = []
        threads = []
        batch_size = 50  # Adjust batch size for performance
        

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            thread = threading.Thread(target=fetch_stock_data, args=(batch, updated_data))
            threads.append(thread)
            thread.start()
            time.sleep(2)  # Short delay to prevent excessive API calls

        for thread in threads:
            thread.join()

        # ✅ Insert only missing data into Supabase
        if updated_data:
            try:
                supabase.table("stock_data").insert(updated_data).execute()
                print("✅ Missing stock data successfully updated in Supabase.")
            except Exception as e:
                print(f"⚠️ Error inserting stock data: {e}")

        # ✅ Delete data older than 1 year
        one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        try:
            supabase.table("stock_data").delete().filter("date", "lt", one_year_ago).execute()
            print("✅ Old data deleted (older than 1 year).")
        except Exception as e:
            print(f"⚠️ Error deleting old data: {e}")

if __name__ == "__main__":
    etl_app()
