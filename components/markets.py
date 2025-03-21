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
    """Fetch stock data from Supabase and ensure it is sorted by date."""
    try:
        response = supabase.table("stock_data").select("date, close, volume").filter("ticker", "eq", ticker).execute()
        if response.data:
            data = pd.DataFrame(response.data)
            data['date'] = pd.to_datetime(data['date'])
            data = data.sort_values(by='date', ascending=True)  # Ensure sorting by date
            data.set_index('date', inplace=True)
            return data
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def get_stock_info(ticker):
    """Fetch stock sector and industry info from Supabase."""
    try:
        response = supabase.table("stock_info").select("sector, industry","longname","marketcap").filter("ticker", "eq", ticker).execute()
        return response.data[0] if response.data else {"sector": "N/A", "industry": "N/A","longname": "N/A","marketcap": "N/A"}
    except Exception as e:
        st.error(f"Error fetching stock info: {e}")
        return {"sector": "N/A", "industry": "N/A","longname": "N/A","marketcap": "N/A"}

@st.cache_data(ttl=300)
def get_sector_industry_price_changes(tickers, include_sector_industry=False):
    """Calculate price changes for given tickers, ensuring date sorting for accuracy."""
    data = {
        'Ticker': [], 'Last Traded Price': [],
        '1D % Change': [], '2D % Change': [], '3D % Change': [], '5D % Change': [], '10D % Change': [],
        '15D % Change': [], '30D % Change': [], '3M % Change': [], '6M % Change': [], '12M % Change': [],
        '1D volume': []
    }
    if include_sector_industry:
        data['Sector'] = []
        data['Industry'] = []
        data['MarketCap'] = []
        data['CompanyName'] = []

    for ticker in tickers:
        try:
            price_data_1y = get_stock_data(ticker)
            stock_info = get_stock_info(ticker) if include_sector_industry else {}

            if price_data_1y is not None and not price_data_1y.empty:
                price_data_1y = price_data_1y.sort_index()  # Ensure chronological order
                last_traded_price = price_data_1y['close'].iloc[-1]
                one_day_volume = price_data_1y['volume'].iloc[-1]

                def get_price_change(days):
                    if len(price_data_1y) >= days + 1:
                        return ((last_traded_price / price_data_1y['close'].iloc[-(days + 1)]) - 1) * 100
                    return 0

                one_day_change = get_price_change(1)
                two_day_change = get_price_change(2)
                three_day_change = get_price_change(3)
                five_day_change = get_price_change(5)
                ten_day_change = get_price_change(10)
                fifteen_day_change = get_price_change(15)
                thirty_day_change = get_price_change(30)
                three_month_change = get_price_change(63)  # ~63 trading days in 3 months
                six_month_change = get_price_change(126)  # ~126 trading days in 6 months
                twelve_month_change = get_price_change(252)  # ~252 trading days in a year

            else:
                last_traded_price = None
                one_day_volume = None
                one_day_change = 0
                two_day_change = 0
                three_day_change = 0
                five_day_change = 0
                ten_day_change = 0
                fifteen_day_change = 0
                thirty_day_change = 0
                three_month_change = 0
                six_month_change = 0
                twelve_month_change = 0

            data['Ticker'].append(ticker)
            data['Last Traded Price'].append(last_traded_price)
            data['1D % Change'].append(one_day_change)
            data['2D % Change'].append(two_day_change)
            data['3D % Change'].append(three_day_change)
            data['5D % Change'].append(five_day_change)
            data['10D % Change'].append(ten_day_change)
            data['15D % Change'].append(fifteen_day_change)
            data['30D % Change'].append(thirty_day_change)
            data['3M % Change'].append(three_month_change)
            data['6M % Change'].append(six_month_change)
            data['12M % Change'].append(twelve_month_change)
            data['1D volume'].append(one_day_volume)

            if include_sector_industry:
                data['Sector'].append(stock_info.get('sector', 'N/A'))
                data['Industry'].append(stock_info.get('industry', 'N/A'))
                data['MarketCap'].append(stock_info.get('marketcap', 'N/A'))
                data['CompanyName'].append(stock_info.get('longname', 'N/A'))

        except Exception as e:
            print(f"Error processing data for {ticker}: {e}")

    df = pd.DataFrame(data)
    df.fillna(0, inplace=True)
    return df


def markets_app():
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Indices", "Stocks", "Commodities", "Currencies", "Cryptocurrencies"])
    
    ticker_mapping = {
        "Indices": Indices,
        "Stocks": {
            "Largecap": Largecap,
            "Midcap": Midcap,
            "Smallcap": Smallcap,
            "Multicap": Largecap + Midcap + Smallcap
        },
        "Commodities": Commodities,
        "Currencies": Currencies,
        "Cryptocurrencies": crypto_largecap + crypto_midcap
    }

    for tab, market in zip([tab1, tab2, tab3, tab4, tab5], ticker_mapping.keys()):
        with tab:
            if market == "Stocks":
                stock_category = st.radio("Select Stock Category", ["Largecap", "Midcap", "Smallcap", "Multicap"],horizontal=True)
                tickers = ticker_mapping[market][stock_category]
            else:
                tickers = ticker_mapping[market]
            include_sector_industry = market == "Stocks"

            st.subheader(f"{market} Market Overview")
            df = get_sector_industry_price_changes(tickers, include_sector_industry)
            st.dataframe(df)

            time_frame = st.radio(f"Select Time Frame for {market}", 
                ['1D % Change','2D % Change', '3D % Change', '5D % Change', '10D % Change','15D % Change', '30D % Change', '3M % Change', '6M % Change', '12M % Change'],horizontal=True
            )

            if time_frame in df.columns:
                df_sorted = df.sort_values(by=time_frame, ascending=False).reset_index(drop=True)
                fig_price = px.bar(df_sorted, x='Ticker', y=time_frame, title=f'{time_frame} Price Change', color=time_frame,
                                   color_continuous_scale=px.colors.diverging.RdYlGn)
                st.plotly_chart(fig_price)
            else:
                st.warning(f"No data available for {time_frame}.")

            if include_sector_industry:
                st.subheader('Sector and Industry Performance')
                df_treemap = df[df[time_frame] > 0]
                if not df_treemap.empty:
                    fig_treemap = px.treemap(df_treemap, path=['Sector', 'Industry'], values=time_frame,
                                             title=f'Treemap of {time_frame} Performance', color=time_frame)
                    st.plotly_chart(fig_treemap)
                else:
                    st.warning("No meaningful data available.")

if __name__ == "__main__":
    markets_app()






