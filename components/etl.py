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
supabase = create_client(SUPABASE_URL,SUPABASE_KEY)

# Stock ticker categories
ticker_categories = {
    "Largecap": Largecap,
    "Midcap": Midcap,
    "Smallcap": Smallcap
}

def fetch_stock_data(tickers_batch, start_date, end_date, results):
    for ticker in tickers_batch:
        retries = 3
        while retries > 0:
            try:
                time.sleep(1.5)
                stock = yf.Ticker(ticker)
                stock_data = stock.history(start=start_date, end=end_date)

                if stock_data.empty:
                    print(f"Warning: No data found for {ticker}")
                    break

                stock_data['ticker'] = ticker
                stock_data.columns = [col.lower() for col in stock_data.columns]
                stock_data = stock_data.reset_index()
                stock_data.rename(columns={"Date": "date"}, inplace=True)
                stock_data['date'] = stock_data['date'].astype(str)

                required_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']
                results.extend(stock_data[required_columns].to_dict(orient="records"))
                print(f"✅ Fetched data for {ticker} ({start_date} to {end_date})")
                break
            except Exception as e:
                print(f"⚠️ Failed to fetch data for {ticker}: {e}")
                retries -= 1
                time.sleep(3)

def etl_app():
    st.title("Stock Data ETL App")
    
    selected_category = st.selectbox("Select Stock Category", list(ticker_categories.keys()))
    tickers = ticker_categories[selected_category]
    
    if st.button("Load Data"):
        latest_date_query = supabase.table("stock_data").select("date").order("date", desc=True).limit(1).execute()
        latest_date = latest_date_query.data[0]['date'] if latest_date_query.data else None
        
        today = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d') if not latest_date else (datetime.strptime(latest_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        
        if start_date >= today:
            st.warning("No new stock data to fetch.")
            return
        
        updated_data = []
        threads = []
        batch_size = 50

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            thread = threading.Thread(target=fetch_stock_data, args=(batch, start_date, today, updated_data))
            threads.append(thread)
            thread.start()
            time.sleep(2)

        for thread in threads:
            thread.join()
        
        if updated_data:
            try:
                supabase.table("stock_data").insert(updated_data).execute()
                st.success("✅ Stock data successfully updated in Supabase.")
            except Exception as e:
                st.error(f"⚠️ Error inserting stock data: {e}")
        
        one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        try:
            supabase.table("stock_data").delete().filter("date", "lt", one_year_ago).execute()
            st.success("✅ Old data deleted (older than 1 year).")
        except Exception as e:
            st.error(f"⚠️ Error deleting old data: {e}")

if __name__ == "__main__":
    etl_app()
