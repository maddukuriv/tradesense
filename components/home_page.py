# home_page.py

import streamlit as st

import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
from supabase import create_client

def home_page_app():
    st.title("TradeSense")
    st.write("An ultimate platform for smart trading insights. Please log in or sign up to get started.")
       
    # Supabase Credentials

    SUPABASE_URL= 'https://kbhdeynmboawkjtxvlek.supabase.co'
    SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtiaGRleW5tYm9hd2tqdHh2bGVrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDA2NDg3NjAsImV4cCI6MjA1NjIyNDc2MH0.T3L5iIn1FiBlBo5HZMqysgokD8cfOw2n3u_YCJV0DkQ'


    # Initialize Supabase client
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Function to fetch stock data
    def get_stock_data(ticker):
        try:
            response = supabase.table("stock_data").select("*").filter("ticker", "eq", ticker).execute()
            
            if response.data:
                data = pd.DataFrame(response.data)
                
                # Convert date column to datetime and set as index if present
                if 'date' in data.columns:
                    data['date'] = pd.to_datetime(data['date'])
                    data.set_index('date', inplace=True)
                    data = data.sort_index()
                
                return data
            else:
                st.warning(f"No data found for ticker: {ticker}")
                return None
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None

    # Function to calculate technical indicators
    def calculate_technical_indicators(data):
        if data is None or 'close' not in data.columns:
            return data
        
        data['MA_10'] = data['close'].rolling(window=10).mean()
        data['MA_20'] = data['close'].rolling(window=20).mean()
        data['MA_50'] = data['close'].rolling(window=50).mean()
        
        return data

    # Function to create Plotly figure
    def create_figure(data, indicators, ticker):
        if data is None or len(data) == 0:
            return go.Figure()
        
        fig = go.Figure()

        # Add candlestick chart
        if {'open', 'high', 'low', 'close'}.issubset(data.columns):
            fig.add_trace(go.Candlestick(x=data.index,
                                        open=data['open'],
                                        high=data['high'],
                                        low=data['low'],
                                        close=data['close'],
                                        name='Candlesticks'))
        
        # Add moving averages
        for indicator in indicators:
            if indicator in data.columns:
                fig.add_trace(go.Scatter(x=data.index, y=data[indicator], mode='lines', name=indicator))

        if  'close' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['close'], mode='lines', name='Close Price', line=dict(color='blue')))
        
        # Add volume
        if 'volume' in data.columns:
            fig.add_trace(go.Bar(x=data.index, y=data['volume'], name='Volume', yaxis='y2', marker_color='rgba(255, 0, 255, 0.2)'))
        
        fig.update_layout(
            title={
            'text': f'{ticker} Price and Technical Indicators',
            'y': 0.97,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
            height=600,
            yaxis=dict(title='Price'),
            yaxis2=dict(title='Volume', overlaying='y', side='right'),
            xaxis=dict(
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label='7d', step='day', stepmode='backward'),
                    dict(count=14, label='14d', step='day', stepmode='backward'),
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(count=3, label='3m', step='month', stepmode='backward'),
                    dict(count=6, label='6m', step='month', stepmode='backward'),
                    dict(count=1, label='1y', step='year', stepmode='backward'),
                    dict(step='all')
                ])
            ),
            type='date'
        ),
            hovermode='x unified',
            hoverlabel=dict(bgcolor="sky blue", font_size=12, font_family="Rockwell")
        )
        
        return fig



    # Dictionary of stock indices
    indices = {
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones Industrial Average",
        "^IXIC": "NASDAQ Composite",
        "^NYA": "NYSE Composite",
        "^XAX": "NYSE AMEX Composite Index",
        "^RUT": "Russell 2000",
        "^VIX": "CBOE Volatility Index",
        "^FTSE": "FTSE 100 (UK)",
        "^GDAXI": "DAX (Germany)",
        "^FCHI": "CAC 40 (France)",
        "^STOXX50E": "EURO STOXX 50",
        "^N100": "Euronext 100",
        "^BFX": "BEL 20 (Belgium)",
        "^GSPTSE": "S&P/TSX Composite Index (Canada)",
        "^HSI": "Hang Seng Index (Hong Kong)",
        "^STI": "STI Index (Singapore)",
        "^AXJO": "S&P/ASX 200 (Australia)",
        "^AORD": "All Ordinaries (Australia)",
        "^BSESN": "S&P BSE SENSEX (India)",
        "^JKSE": "IDX Composite (Indonesia)",
        "^KLSE": "FTSE Bursa Malaysia KLCI",
        "^NZ50": "S&P/NZX 50 Index (New Zealand)",
        "^KS11": "KOSPI Composite Index (South Korea)",
        "^TWII": "TWSE Capitalization Weighted Stock Index (Taiwan)",
        "000001.SS": "SSE Composite Index (China)",
        "^N225": "Nikkei 225 (Japan)",
        "^BVSP": "IBOVESPA (Brazil)",
        "^MXX": "IPC Mexico",
        "^IPSA": "S&P IPSA (Chile)",
        "^MERV": "MERVAL (Argentina)",
        "^TA125.TA": "TA-125 (Israel)",
        "^CASE30": "EGX 30 (Egypt)",
        "^JN0U.JO": "Top 40 USD Net TRI Index (South Africa)",
        "DX-Y.NYB": "US Dollar Index",
        "^XDB": "British Pound Currency Index",
        "^XDE": "Euro Currency Index",
        "^XDN": "Japanese Yen Currency Index"
    }

    


    # Time periods for performance calculation
    periods = {
        252: "1Y",  # ~252 trading days (~1 year)
        126: "6M",  # ~126 trading days (~6 months)
        63: "3M",   # ~63 trading days (~3 months)
        21: "1M",   # ~21 trading days (~1 month)
        15: "15D",  # 15 trading days
        5: "5D",    # 5 trading days
        3: "3D",    # 3 trading days
        1: "1D"     # 1 trading day
    }

    # Function to compute performance for different timeframes
    def compute_performance(data):
        performance = {}
        latest_price = data['close'].iloc[-1]

        for period, label in periods.items():
            if len(data) >= period:
                if period == 1 and len(data) > 1:
                    past_price = data['close'].iloc[-2]  # Ensure it picks the previous trading day
                else:
                    past_price = data['close'].iloc[-period]
                
                if past_price != 0:  # Avoid division by zero
                    performance[label] = ((latest_price - past_price) / past_price) * 100
                else:
                    performance[label] = None
            else:
                performance[label] = None  # Not enough data
        
        return performance

    # Function to fetch performance of all indices
    @st.cache_data
    def get_indices_performance():
        performance_data = []

        for ticker, name in indices.items():
            data = get_stock_data(ticker)
            
            if data is not None and 'close' in data.columns and not data.empty:
                performance = compute_performance(data)
                performance["Index"] = name
                performance_data.append(performance)
        
        if performance_data:
            return pd.DataFrame(performance_data).set_index("Index")
        else:
            return pd.DataFrame(columns=["Index"] + list(periods.values()))

    # Function to create a bar chart
    def create_performance_chart(performance_df, period):
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=performance_df.index,
            x=performance_df[period],
            orientation='h'
        ))
        
        fig.update_layout(
            title=f"Stock Index Performance ({period})",
            xaxis_title="Performance %",
            yaxis_title="Index",
            height=700
        )
        
        return fig


    

    tab1, tab2 = st.tabs(["📈Trend", "📊Performance"])

    with tab1:
        # Create two columns
        col1, col2 = st.columns(2)

        # User input for ticker
        with col1:
            ticker = st.selectbox("Select Stock Index:", options=list(indices.keys()), format_func=lambda x: indices[x])

        # Select indicators
        with col2:
            indicators = st.multiselect("Select Indicators", ["MA_10", "MA_20", "MA_50"], default=["MA_10", "MA_20", "MA_50"])

        if ticker:
            stock_data = get_stock_data(ticker)
            stock_data = calculate_technical_indicators(stock_data)
            
            if stock_data is not None:
                #st.subheader("Stock Data Preview")
                #st.dataframe(stock_data.tail())
                
                # Plot chart
                fig = create_figure(stock_data, indicators, ticker)
                st.plotly_chart(fig)


        st.divider()

    with tab2:
        # Radio buttons for timeframes
        selected_period = st.radio("Select a timeframe to view performance trends", list(periods.values()), index=7,horizontal=True)

        # Fetch performance data
        performance_df = get_indices_performance()

        if not performance_df.empty:
            # Display table of performance
            #st.write("### Performance Data")
            #st.dataframe(performance_df)

            # Display selected period chart
            # st.write(f"### {selected_period} Performance Chart")
            fig = create_performance_chart(performance_df, selected_period)
            st.plotly_chart(fig)
        else:
            st.warning("No performance data available.")
        
        st.divider()

    
# To run the app
if __name__ == "__main__":
    home_page_app()
