# pages/markets.py
import streamlit as st
from supabase import create_client
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.constants import Largecap, Midcap, Smallcap,sp500_tickers,ftse100_tickers,crypto_largecap,crypto_midcap,Indices,Commodities,Currencies,SUPABASE_URL,SUPABASE_KEY
from datetime import datetime, timedelta
import requests
    
# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_stock_data(ticker):
    """Fetch stock data from Supabase."""
    try:
        response = supabase.table("stock_data").select("*").filter("ticker", "eq", ticker).execute()
        if response.data:
            data = pd.DataFrame(response.data)
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data.set_index('date', inplace=True)
                data = data.sort_index()
            return data
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def get_stock_info(ticker):
    """Fetch stock sector and industry info from Supabase."""
    try:
        response = supabase.table("stock_info").select("sector", "industry").filter("ticker", "eq", ticker).execute()
        if response.data:
            return response.data[0]  # Return sector and industry info
        else:
            return {"sector": "N/A", "industry": "N/A"}
    except Exception as e:
        st.error(f"Error fetching stock info: {e}")
        return {"sector": "N/A", "industry": "N/A"}

@st.cache_data(ttl=60)
def get_sector_industry_price_changes(tickers, include_sector_industry=False):
    """Calculate price changes for given tickers."""
    data = {
        'Ticker': [], 'Company Name': [], 'Last Traded Price': [],
        '1D % Change': [], '3D % Change': [], '5D % Change': [], '15D % Change': [], '30D % Change': [], '6M % Change': [], '12M % Change': [],
        '1D volume': []
    }
    if include_sector_industry:
        data['Sector'] = []
        data['Industry'] = []
    
    for ticker in tickers:
        try:
            price_data_1y = get_stock_data(ticker)
            stock_info = get_stock_info(ticker) if include_sector_industry else {}
            
            if price_data_1y is not None and not price_data_1y.empty:
                last_traded_price = price_data_1y['close'].iloc[-1]
                one_day_volume = price_data_1y['volume'].iloc[-1]
                price_changes = price_data_1y['close'].pct_change() * 100
                one_day_change = price_changes.iloc[-1]
                three_day_change = price_changes.iloc[-3:].sum()
                five_day_change = price_changes.iloc[-5:].sum()
                fifteen_day_change = price_changes.iloc[-15:].sum()
                thirty_day_change = price_changes.iloc[-30:].sum()
                six_month_change = price_changes.iloc[-126:].sum()
                twelve_month_change = price_changes.sum()
            else:
                last_traded_price = None
                one_day_volume = None
                one_day_change = None
                three_day_change = None
                five_day_change = None
                fifteen_day_change = None
                thirty_day_change = None
                six_month_change = None
                twelve_month_change = None

            data['Ticker'].append(ticker)
            data['Company Name'].append('N/A')  # Placeholder for company info
            data['Last Traded Price'].append(last_traded_price)
            data['1D % Change'].append(one_day_change)
            data['3D % Change'].append(three_day_change)
            data['5D % Change'].append(five_day_change)
            data['15D % Change'].append(fifteen_day_change)
            data['30D % Change'].append(thirty_day_change)
            data['6M % Change'].append(six_month_change)
            data['12M % Change'].append(twelve_month_change)
            data['1D volume'].append(one_day_volume)
            
            if include_sector_industry:
                data['Sector'].append(stock_info.get('sector', 'N/A'))
                data['Industry'].append(stock_info.get('industry', 'N/A'))

        except Exception as e:
            st.error(f"Error processing data for {ticker}: {e}")
    
    df = pd.DataFrame(data)

    # Convert numeric columns to float and handle NaNs
    numeric_columns = ['1D % Change', '3D % Change', '5D % Change', '15D % Change', '30D % Change', '6M % Change', '12M % Change']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    return df

def markets_app():
    submenu = st.sidebar.selectbox("Select Option", ["Indices", "Stocks", "Commodities", "Currencies", "Cryptocurrencies"])
    
    ticker_mapping = {
        "Indices": Indices,
        "Stocks": Largecap + Midcap + Smallcap,
        "Commodities": Commodities,
        "Currencies": Currencies,
        "Cryptocurrencies": crypto_largecap + crypto_midcap
    }
    
    tickers = ticker_mapping.get(submenu, [])
    include_sector_industry = submenu == "Stocks"
    
    if tickers:
        st.subheader(f"{submenu} Market Overview")
        df = get_sector_industry_price_changes(tickers, include_sector_industry)
        st.dataframe(df)

        time_frame = st.radio("Select Time Frame", ['1D % Change', '3D % Change', '5D % Change', '15D % Change', '30D % Change', '6M % Change', '12M % Change'],horizontal=True)
        
        st.subheader('Price Changes')
        fig_price = px.bar(df, x='Ticker', y=time_frame, title=f'{time_frame} Price Change', color=time_frame)
        st.plotly_chart(fig_price)

        
        if include_sector_industry:
            st.subheader('Sector Performance')
            
            # Ensure numeric columns are converted to float before aggregation
            numeric_columns = ['1D % Change', '3D % Change', '5D % Change', '15D % Change', '30D % Change', '6M % Change', '12M % Change']
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

            df_sector = df.groupby('Sector')[numeric_columns].mean().reset_index()
            fig_sector = px.bar(df_sector, x='Sector', y=time_frame, title=f'Sector Performance - {time_frame}', color=time_frame)
            st.plotly_chart(fig_sector)

            st.subheader('Industry Performance')
            df_industry = df.groupby('Industry')[numeric_columns].mean().reset_index()
            fig_industry = px.bar(df_industry, x='Industry', y=time_frame, title=f'Industry Performance - {time_frame}', color=time_frame)
            st.plotly_chart(fig_industry)


if __name__ == "__main__":
    markets_app()



