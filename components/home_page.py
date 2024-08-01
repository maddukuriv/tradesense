# home_page.py

import streamlit as st
import yfinance as yf
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta


def home_page_app():
  st.title("TradeSense")
  st.write("An ultimate platform for smart trading insights. Please log in or sign up to get started.")

  st.subheader("Market Performance")

  # Function to get stock data and calculate moving averages
  @st.cache_data
  def get_stock_data(ticker_symbol, start_date, end_date):
      data = yf.download(ticker_symbol, start=start_date, end=end_date)
      data['MA_10'] = data['Close'].rolling(window=10).mean()
      data['MA_20'] = data['Close'].rolling(window=20).mean()
      data.dropna(inplace=True)
      return data

  # Function to create Plotly figure with volume histogram
  def create_figure(data, indicators, title):
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
      fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', yaxis='y2', marker_color='rgba(0, 0, 100, 0.5)'))

      def update_layout(fig, title, mode='light'):
        if mode == 'dark':
            bgcolor = 'rgb(17, 17, 17)'
            font_color = 'white'
            grid_color = 'gray'
        else:
            bgcolor = 'white'
            font_color = 'black'
            grid_color = 'lightgray'

        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=True,
            plot_bgcolor=bgcolor,
            paper_bgcolor=bgcolor,
            font=dict(color=font_color),
            hovermode='x',
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type='date',
                gridcolor=grid_color
            ),
            yaxis=dict(
                title='Price',
                fixedrange=False,
                gridcolor=grid_color
            ),
            yaxis2=dict(
                title='Volume',
                overlaying='y',
                side='right',
                gridcolor=grid_color
            ),
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Reset Zoom",
                            method="relayout",
                            args=[{"xaxis.range": [None, None],
                                    "yaxis.range": [None, None]}])]
            )]
        )
        return fig

  col1, col2, col3 = st.columns(3)

  with col1:
      stock_symbols = {
          "BSE 500": "BSE-500.BO",
          "NIFTY 50": "^NSEI",
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
      stock_name = st.selectbox("Select Stock", list(stock_symbols.keys()))
      ticker = stock_symbols[stock_name]
      st.write(f"You selected: {stock_name}")

  with col2:
      START = st.date_input('Start Date', value=datetime.now() - timedelta(days=365))

  with col3:
      END = st.date_input('End Date', value=datetime.now() + timedelta(days=1))

  if ticker and START and END:
      data = get_stock_data(ticker, START, END)
      fig = create_figure(data, ['Close', 'MA_10', 'MA_20'], f"{stock_name} Stock Prices")
      st.plotly_chart(fig)

  # Market Performance

  market_indices = {
      'S&P 500': '^GSPC',
      'Dow Jones': '^DJI',
      'NASDAQ': '^IXIC',
      'Gold': 'GC=F',
      'Silver': 'SI=F',
      'Oil': 'CL=F',
      'EUR/USD': 'EURUSD=X',
      'GBP/USD': 'GBPUSD=X',
      'Bitcoin': 'BTC-USD',
      'Ethereum': 'ETH-USD'
  }

  # Define date ranges
  date_ranges = {
      "1 day": timedelta(days=1),
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

  def get_market_data(ticker_symbol, start_date, end_date):
      return yf.download(ticker_symbol, start=start_date, end=end_date)

  def calculate_performance(data):
      if data is not None and not data.empty:
          performance = (data['Close'][-1] - data['Close'][0]) / data['Close'][0] * 100
          return performance
      return None

  market_performance = {
      market: calculate_performance(get_market_data(ticker, START, END))
      for market, ticker in market_indices.items()
      if calculate_performance(get_market_data(ticker, START, END)) is not None
  }

  performance_df = pd.DataFrame(list(market_performance.items()), columns=['Market', 'Performance'])
  fig = px.bar(performance_df, x='Market', y='Performance', title='Market Performance',
              labels={'Performance': 'Performance (%)'}, color='Performance',
              color_continuous_scale=px.colors.diverging.RdYlGn)
  st.plotly_chart(fig)
