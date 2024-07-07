import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import ta
from datetime import datetime

# Function to get stock data and calculate moving averages and indicators
@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.warning(f"No data found for {ticker}.")
            return pd.DataFrame()
        data['MA_15'] = data['Close'].rolling(window=15).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
        data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close']).adx()
        data.dropna(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

# Function to create Plotly figure
def create_figure(data, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA_15'], mode='lines', name='15-day MA'))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA_50'], mode='lines', name='50-day MA'))
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=True,
        plot_bgcolor='dark grey',
        paper_bgcolor='white',
        font=dict(color='black'),
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
            type='date'
        ),
        yaxis=dict(fixedrange=False),
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Reset Zoom",
                          method="relayout",
                          args=[{"xaxis.range": [None, None], "yaxis.range": [None, None]}])]
        )]
    )
    return fig

# Function to fetch data for multiple tickers
@st.cache_data
def fetch_data(tickers, period='1d', interval='1m'):
    try:
        data = yf.download(tickers, period=period, interval=interval)
        return data['Close']
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Function to reshape data for heatmap
def reshape_for_heatmap(df, num_columns=10):
    num_rows = int(np.ceil(len(df) / num_columns))
    reshaped_data = np.zeros((num_rows, num_columns))
    reshaped_tickers = np.empty((num_rows, num_columns), dtype=object)
    reshaped_data[:] = np.nan
    reshaped_tickers[:] = ''
    index = 0
    for y in range(num_rows):
        for x in range(num_columns):
            if index < len(df):
                reshaped_data[y, x] = df['% Change'].values[index]
                reshaped_tickers[y, x] = df['Ticker'].values[index]
                index += 1
    return reshaped_data, reshaped_tickers

# Create annotated heatmaps using Plotly
def create_horizontal_annotated_heatmap(df, title, num_columns=10):
    reshaped_data, tickers = reshape_for_heatmap(df, num_columns)
    annotations = []
    for y in range(reshaped_data.shape[0]):
        for x in range(reshaped_data.shape[1]):
            text = f'<b>{tickers[y, x]}</b><br>{reshaped_data[y, x]}%'
            annotations.append(
                go.layout.Annotation(
                    text=text,
                    x=x,
                    y=y,
                    xref='x',
                    yref='y',
                    showarrow=False,
                    font=dict(size=10, color="black", family="Arial, sans-serif"),
                    align="left"
                )
            )
    fig = go.Figure(data=go.Heatmap(
        z=reshaped_data,
        x=list(range(reshaped_data.shape[1])),
        y=list(range(reshaped_data.shape[0])),
        hoverinfo='text',
        colorscale='Blues',
        showscale=False,
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        annotations=annotations,
        autosize=False,
        width=1800,
        height=200 + 50 * len(reshaped_data),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

# Function to fetch stock data and volume
@st.cache_data
def get_volume_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        return data['Volume'].sum()
    except Exception as e:
        st.error(f"Error fetching volume data for {ticker}: {e}")
        return 0

# Function to calculate performance
def calculate_performance(data):
    if not data.empty:
        performance = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
        return performance
    return None

# Function to fetch market/sector data
@st.cache_data
def get_data(ticker_symbol, start_date, end_date):
    try:
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker_symbol}: {e}")
        return pd.DataFrame()

# Main Streamlit App
st.title("TradeSense")
st.write("An ultimate platform for smart trading insights. Please log in or sign up to get started.")

tile_selection = st.selectbox("Select a section", ["Major Indices", "Top Gainers and Losers", "Volume Chart", "Sector Performance Chart", "Market Performance"])

if tile_selection == "Major Indices":
    st.subheader("Major Indices")
    col1, col2, col3 = st.columns(3)
    with col1:
        stock_symbols = ["^BSESN", "BSE-500.BO", "^BSEMD", "^BSESMLCAP", "^NSEI", "^NSMIDCP", "^NSEMDCP", "^NSESCP"]
        ticker = st.selectbox("Enter Stock symbol", stock_symbols)
    with col2:
        start_date = st.date_input('Start Date', pd.to_datetime("2023-06-06"))
    with col3:
        end_date = st.date_input('End Date', pd.to_datetime("today"))
    if ticker and start_date and end_date:
        data = get_stock_data(ticker, start_date, end_date)
        if not data.empty:
            fig = create_figure(data, f"{ticker} Stock Prices")
            st.plotly_chart(fig)

elif tile_selection == "Top Gainers and Losers":
    st.subheader("Top Gainers and Losers")
    data_daily = fetch_data(tickers, period='1d', interval='1m')
    data_weekly = fetch_data(tickers, period='5d', interval='1d')
    data_monthly = fetch_data(tickers, period='1mo', interval='1d')

    for data in [data_daily, data_weekly, data_monthly]:
        data.dropna(axis=1, how='all', inplace=True)
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)

    daily_change = data_daily.iloc[-1] - data_daily.iloc[0]
    percent_change_daily = (daily_change / data_daily.iloc[0]) * 100
    weekly_change = data_weekly.iloc[-1] - data_weekly.iloc[0]
    percent_change_weekly = (weekly_change / data_weekly.iloc[0]) * 100
    monthly_change = data_monthly.iloc[-1] - data_monthly.iloc[0]
    percent_change_monthly = (monthly_change / data_monthly.iloc[0]) * 100

    df_daily = pd.DataFrame({'Ticker': data_daily.columns, 'Last Traded Price': data_daily.iloc[-1].values, '% Change': percent_change_daily.values})
    df_weekly = pd.DataFrame({'Ticker': data_weekly.columns, 'Last Traded Price': data_weekly.iloc[-1].values, '% Change': percent_change_weekly.values})
    df_monthly = pd.DataFrame({'Ticker': data_monthly.columns, 'Last Traded Price': data_monthly.iloc[-1].values, '% Change': percent_change_monthly.values})

    for df in [df_daily, df_weekly, df_monthly]:
        df['% Change'] = df['% Change'].round(2)

    heatmap_option = st.selectbox('Select to view:', ['Daily Gainers/Losers', 'Weekly Gainers/Losers', 'Monthly Gainers/Losers'])

    if heatmap_option == 'Daily Gainers/Losers':
        fig = create_horizontal_annotated_heatmap(df_daily.sort_values(by='% Change', ascending=True), 'Daily Gainers/Losers')
        st.plotly_chart(fig)
    elif heatmap_option == 'Weekly Gainers/Losers':
        fig = create_horizontal_annotated_heatmap(df_weekly.sort_values(by='% Change', ascending=True), 'Weekly Gainers/Losers')
        st.plotly_chart(fig)
    elif heatmap_option == 'Monthly Gainers/Losers':
        fig = create_horizontal_annotated_heatmap(df_monthly.sort_values(by='% Change', ascending=True), 'Monthly Gainers/Losers')
        st.plotly_chart(fig)

elif tile_selection == "Volume Chart":
    st.subheader("Volume Chart")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('Start Date', datetime(2022, 1, 1), key='start_date')
    with col2:
        end_date = st.date_input('End Date', datetime.today(), key='end_date')
    volume_data = {ticker: get_volume_data(ticker, start_date, end_date) for ticker in tickers}
    volume_df = pd.DataFrame(list(volume_data.items()), columns=['Ticker', 'Volume'])
    fig = px.bar(volume_df, x='Ticker', y='Volume', title='Trading Volume of Stocks', labels={'Volume': 'Total Volume'}, color='Volume', color_continuous_scale=px.colors.sequential.Viridis)
    st.plotly_chart(fig)

elif tile_selection == "Sector Performance Chart":
    st.subheader("Sector Performance Chart")
    sector_indices = {
        'NIFTY_BANK': '^NSEBANK',
        'NIFTY_IT': '^CNXIT',
        'NIFTY_AUTO': '^CNXAUTO',
        'NIFTY_FMCG': '^CNXFMCG',
        'NIFTY_PHARMA': '^CNXPHARMA',
        'NIFTY_REALTY': '^CNXREALTY',
        'NIFTY_METAL': '^CNXMETAL',
        'NIFTY_MEDIA': '^CNXMEDIA',
        'NIFTY_PSU_BANK': '^CNXPSUBANK',
        'NIFTY_ENERGY': '^CNXENERGY',
        'NIFTY_COMMODITIES': '^CNXCOMMOD',
        'NIFTY_INFRASTRUCTURE': '^CNXINFRA',
        'NIFTY_SERVICES_SECTOR': '^CNXSERVICE',
        'NIFTY_FINANCIAL_SERVICES': '^CNXFINANCE',
        'NIFTY_MNC': '^CNXMNC',
        'NIFTY_PSE': '^CNXPSE',
        'NIFTY_CPSE': '^CNXCPSE',
        'NIFTY_100': '^CNX100',
        'NIFTY_200': '^CNX200',
        'NIFTY_500': '^CNX500',
        'NIFTY_MIDCAP_50': '^CNXMID50',
        'NIFTY_MIDCAP_100': '^CNXMIDCAP',
        'NIFTY_SMALLCAP_100': '^CNXSMCAP',
        'NIFTY_NEXT_50': '^CNXNIFTY'
    }
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('Start Date', datetime(2022, 1, 1), key='start_date')
    with col2:
        end_date = st.date_input('End Date', datetime.today(), key='end_date')
    sector_performance = {sector: calculate_performance(get_data(ticker, start_date, end_date)) for sector, ticker in sector_indices.items() if calculate_performance(get_data(ticker, start_date, end_date)) is not None}
    performance_df = pd.DataFrame(list(sector_performance.items()), columns=['Sector', 'Performance'])
    fig = px.bar(performance_df, x='Sector', y='Performance', title='Sector Performance', labels={'Performance': 'Performance (%)'}, color='Performance', color_continuous_scale=px.colors.sequential.Viridis)
    st.plotly_chart(fig)

elif tile_selection == "Market Performance":
    st.subheader("Market Performance")
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
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('Start Date', datetime(2022, 1, 1), key='start_date')
    with col2:
        end_date = st.date_input('End Date', datetime.today(), key='end_date')
    market_performance = {market: calculate_performance(get_data(ticker, start_date, end_date)) for market, ticker in market_indices.items() if calculate_performance(get_data(ticker, start_date, end_date)) is not None}
    performance_df = pd.DataFrame(list(market_performance.items()), columns=['Market', 'Performance'])
    fig = px.bar(performance_df, x='Market', y='Performance', title='Market Performance', labels={'Performance': 'Performance (%)'}, color='Performance', color_continuous_scale=px.colors.diverging.RdYlGn)
    st.plotly_chart(fig)

st.markdown("-----------------------------------------------------------------------------------------------------------------------")
st.subheader("Unlock your trading potential. Join TradeSense today!")
