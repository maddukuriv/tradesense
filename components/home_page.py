# home_page.py

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta

def home_page_app():
    st.title("TradeSense")
    st.write("An ultimate platform for smart trading insights. Please log in or sign up to get started.")
    
    # Function to get stock data and calculate moving averages
    @st.cache_data(ttl=60)
    def get_stock_data(ticker_symbol, start_date, end_date):
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
        data['MA_10'] = data['Close'].rolling(window=10).mean()
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data.dropna(inplace=True)
        return data

    # Function to create Plotly figure with volume histogram
    def create_figure(data, indicators, ticker):
        fig = go.Figure()

        # Add candlestick chart
        fig.add_trace(go.Candlestick(x=data.index,
                                    open=data['Open'],
                                    high=data['High'],
                                    low=data['Low'],
                                    close=data['Close'],
                                    name='Candlesticks'))

        if 'Close' in indicators:
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
        if 'MA_10' in indicators:
            fig.add_trace(go.Scatter(x=data.index, y=data['MA_10'], mode='lines', name='10-day MA'))
        if 'MA_20' in indicators:
            fig.add_trace(go.Scatter(x=data.index, y=data['MA_20'], mode='lines', name='20-day MA'))

        # Add volume histogram
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', yaxis='y2', marker_color='rgba(255, 0, 255, 0.2)'))

        fig.update_layout(
            title={
                'text': f'{ticker} Price and Technical Indicators',
                'y': 0.97,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            height=500,
            margin=dict(t=100, b=10, l=50, r=50),
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
            legend=dict(x=0.5, y=-0.02, orientation='h', xanchor='center', yanchor='top'),
            hovermode='x unified',
            hoverlabel=dict( font_size=16, font_family="Rockwell")
        )
        return fig

    col1, col2, col3 = st.columns(3)

    with col1:
        stock_symbols = {
            "NIFTY 50": "^NSEI",
            "BSE 500": "BSE-500.BO",
            "S&P 500": "^GSPC",
            "FTSE 100": "^FTSE",
            "SSE Composite (China)": "000001.SS",
            "Nikkei 225 (Japan)": "^N225",
            "ASX 200 (Australia)": "^AXJO",
            "S&P/TSX (Canada)": "^GSPTSE",
            "Bitcoin": "BTC-USD",
            "EUR/USD": "EURUSD=X",
            "Gold Futures": "GC=F",
            "Crude Oil Futures": "CL=F"
        }
        stock_name = st.selectbox("Select Index", list(stock_symbols.keys()))
        ticker = stock_symbols[stock_name]
        st.write(f"You selected: {stock_name}")

    with col2:
        START = st.date_input('Start Date', value=datetime.now() - timedelta(days=365))

    with col3:
        END = st.date_input('End Date', value=datetime.now() + timedelta(days=1))

    if ticker and START and END:
        try:
            data = get_stock_data(ticker, START, END)
            fig = create_figure(data, ['Close', 'MA_10', 'MA_20'], ticker)
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error fetching data: {e}")

    st.divider()
    
    # Market Performance
    st.subheader("Market's Performance")

    market_indices = {
        # Market Indices
        'S&P 500': '^GSPC',
        'Dow Jones': '^DJI',
        'NASDAQ': '^IXIC',
        'Nikkei 225': '^N225',
        'FTSE 100': '^FTSE',
        'DAX': '^GDAXI',
        'CAC 40': '^FCHI',
        'Shanghai Composite': '000001.SS',
        'Hang Seng Index': '^HSI',
        'Sensex': '^BSESN',

        # Commodities
        'Gold': 'GC=F',
        'Silver': 'SI=F',
        'Oil': 'CL=F',
        'Natural Gas': 'NG=F',
        'Copper': 'HG=F',
        'Corn': 'ZC=F',
        'Soybeans': 'ZS=F',
        'Wheat': 'ZW=F',
        'Cotton': 'CT=F',
        'Coffee': 'KC=F',

        # Currencies
        'EUR/USD': 'EURUSD=X',
        'GBP/USD': 'GBPUSD=X',
        'USD/JPY': 'JPY=X',
        'USD/CHF': 'CHF=X',
        'USD/CAD': 'CAD=X',
        'AUD/USD': 'AUDUSD=X',
        'NZD/USD': 'NZDUSD=X',
        'USD/CNY': 'CNY=X',
        'USD/SEK': 'SEK=X',
        'USD/INR': 'INR=X',

        # Cryptocurrencies
        'Bitcoin': 'BTC-USD',
        'Ethereum': 'ETH-USD',
        'Tether': 'USDT-USD',
        'Binance Coin': 'BNB-USD',
        'USD Coin': 'USDC-USD',
        'XRP': 'XRP-USD',
        'Cardano': 'ADA-USD',
        'Solana': 'SOL-USD',
        'Dogecoin': 'DOGE-USD',
        'Polygon': 'MATIC-USD'
    }

    # Define date ranges
    date_ranges = {
        "1 day": timedelta(days=1),
        "2 days": timedelta(days=2),
        "3 days": timedelta(days=3),
        "5 days": timedelta(days=5),
        "10 days": timedelta(days=10),
        "1 month": timedelta(days=30),
        "3 months": timedelta(days=90),
        "6 months": timedelta(days=180),
        "1 year": timedelta(days=365),
        "2 years": timedelta(days=730),
        "3 years": timedelta(days=1095),
        "5 years": timedelta(days=1825)
    }

    selected_range = st.select_slider(
        "Select Date Range for Market Performance",
        options=list(date_ranges.keys()),
        value="1 year"
    )
    END = datetime.now()
    START = END - date_ranges[selected_range]

    @st.cache_data(ttl=60)
    def get_market_data(ticker_symbol, start_date, end_date):
        try:
            data = yf.download(ticker_symbol, start=start_date, end=end_date)
            if data.empty:
                raise ValueError(f"No data found for ticker {ticker_symbol}")
            return data
        except Exception as e:
            st.error(f"Error downloading data for {ticker_symbol}: {e}")
            return None

    def calculate_performance(data):
        if data is not None and not data.empty:
            performance = (data['Close'][-1] - data['Close'][0]) / data['Close'][0] * 100
            return performance
        return None

    market_performance = {
        market: calculate_performance(get_market_data(ticker, START, END))
        for market, ticker in market_indices.items()
        if get_market_data(ticker, START, END) is not None
    }

    performance_df = pd.DataFrame(list(market_performance.items()), columns=['Market', 'Performance'])
    fig2 = px.bar(performance_df, x='Market', y='Performance', title='Market Performance',
                labels={'Performance': 'Performance (%)'}, color='Performance',
                color_continuous_scale=px.colors.diverging.RdYlGn)
    st.plotly_chart(fig2)

# To run the app
if __name__ == "__main__":
    home_page_app()
