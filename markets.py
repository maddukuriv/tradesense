import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from functools import lru_cache

# Function to download data and calculate moving averages with caching
@lru_cache(maxsize=32)
def get_stock_data(ticker_symbol, start_date, end_date):
    data = yf.download(ticker_symbol, start=start_date, end=end_date)
    data['MA_15'] = data['Close'].rolling(window=15).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['Upper_Band'] = data['Close'].rolling(20).mean() + (data['Close'].rolling(20).std() * 2)
    data['Lower_Band'] = data['Close'].rolling(20).mean() - (data['Close'].rolling(20).std() * 2)
    data.dropna(inplace=True)
    return data

# Function to create Plotly figure
def create_figure(data, indicators, title):
    fig = go.Figure()
    if 'Close' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    if 'MA_15' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=data['MA_15'], mode='lines', name='15-day MA'))
    if 'MA_50' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=data['MA_50'], mode='lines', name='50-day MA'))
    if 'MACD' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD'))
        fig.add_trace(go.Scatter(x=data.index, y=data['Signal_Line'], mode='lines', name='Signal Line'))
    if 'Bollinger Bands' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=data['Upper_Band'], mode='lines', name='Upper Band'))
        fig.add_trace(go.Scatter(x=data.index, y=data['Lower_Band'], mode='lines', name='Lower Band'))

    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price',
                      xaxis_rangeslider_visible=True,
                      plot_bgcolor='dark grey',
                      paper_bgcolor='white',
                      font=dict(color='black'),
                      hovermode='x',
                      xaxis=dict(rangeselector=dict(buttons=list([
                          dict(count=1, label="1m", step="month", stepmode="backward"),
                          dict(count=6, label="6m", step="month", stepmode="backward"),
                          dict(count=1, label="YTD", step="year", stepmode="todate"),
                          dict(count=1, label="1y", step="year", stepmode="backward"),
                          dict(step="all")
                      ])),
                          rangeslider=dict(visible=True),
                          type='date'),
                      yaxis=dict(fixedrange=False),
                      updatemenus=[dict(type="buttons",
                                        buttons=[dict(label="Reset Zoom",
                                                      method="relayout",
                                                      args=[{"xaxis.range": [None, None],
                                                             "yaxis.range": [None, None]}])])])
    return fig

# Function to calculate correlation
def calculate_correlation(data1, data2):
    return data1['Close'].corr(data2['Close'])

# Function to plot correlation matrix
def plot_correlation_matrix(correlation_matrix):
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='Viridis'))
    fig.update_layout(title="Correlation Matrix", xaxis_title='Assets', yaxis_title='Assets')
    return fig

# Function to calculate Sharpe Ratio
def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    return (returns.mean() - risk_free_rate) / returns.std()

# Function to calculate Beta
def calculate_beta(asset_returns, market_returns):
    # Align the series to have the same index
    aligned_returns = pd.concat([asset_returns, market_returns], axis=1).dropna()
    covariance_matrix = np.cov(aligned_returns.iloc[:, 0], aligned_returns.iloc[:, 1])
    beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
    return beta

# Function to calculate Value at Risk (VaR)
def calculate_var(returns, confidence_level=0.05):
    return np.percentile(returns, confidence_level * 100)

# Main application
st.title("Market Insights")

# Date inputs
col1, col2 = st.columns(2)
with col1:
    START = st.date_input('Start Date', pd.to_datetime("2023-06-06"))
with col2:
    END = st.date_input('End Date', pd.to_datetime("today"))

# Markets submenu
submenu = st.sidebar.radio("Select Option", ["Equities", "Commodities", "Currencies", "Cryptocurrencies", "Analysis"])

if submenu == "Equities":
    st.subheader("Equity Markets")
    data_nyse = get_stock_data("^NYA", START, END)
    data_bse = get_stock_data("^BSESN", START, END)
    indicators = st.multiselect("Select Indicators", ['Close', 'MA_15', 'MA_50', 'MACD', 'Bollinger Bands'], default=['Close'])
    fig_nyse = create_figure(data_nyse, indicators, 'NYSE Price')
    fig_bse = create_figure(data_bse, indicators, 'BSE Price')
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_nyse)

    with col2:
        st.plotly_chart(fig_bse)


elif submenu == "Commodities":
    st.subheader("Commodities")
    tickers = ["GC=F", "CL=F", "NG=F", "SI=F", "HG=F"]
    selected_tickers = st.multiselect("Select stock tickers to visualize", tickers, default=["GC=F", "CL=F"])
    indicators = st.multiselect("Select Indicators", ['Close', 'MA_15', 'MA_50', 'MACD', 'Bollinger Bands'], default=['Close'])
    if not selected_tickers:
        st.warning("Please select at least one ticker.")
    else:
        columns = st.columns(len(selected_tickers))
        for ticker, col in zip(selected_tickers, columns):
            data = get_stock_data(ticker, START, END)
            fig = create_figure(data, indicators, f'{ticker} Price')
            col.plotly_chart(fig)

elif submenu == "Currencies":
    st.subheader("Currencies")
    tickers = ["EURUSD=X", "GBPUSD=X", "CNYUSD=X", "INRUSD=X"]
    selected_tickers = st.multiselect("Select currency pairs to visualize", tickers, default=["INRUSD=X", "CNYUSD=X"])
    indicators = st.multiselect("Select Indicators", ['Close', 'MA_15', 'MA_50', 'MACD', 'Bollinger Bands'], default=['Close'])
    if not selected_tickers:
        st.warning("Please select at least one currency pair.")
    else:
        columns = st.columns(len(selected_tickers))
        for ticker, col in zip(selected_tickers, columns):
            data = get_stock_data(ticker, START, END)
            fig = create_figure(data, indicators, f'{ticker} Price')
            col.plotly_chart(fig)

elif submenu == "Cryptocurrencies":
    st.subheader("Cryptocurrencies")
    tickers = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "DOGE-USD"]
    selected_tickers = st.multiselect("Select cryptocurrencies to visualize", tickers, default=["BTC-USD", "ETH-USD"])
    indicators = st.multiselect("Select Indicators", ['Close', 'MA_15', 'MA_50', 'MACD', 'Bollinger Bands'], default=['Close'])
    if not selected_tickers:
        st.warning("Please select at least one cryptocurrency.")
    else:
        columns = st.columns(len(selected_tickers))
        for ticker, col in zip(selected_tickers, columns):
            data = get_stock_data(ticker, START, END)
            fig = create_figure(data, indicators, f'{ticker} Price')
            col.plotly_chart(fig)

elif submenu == "Analysis":
    st.subheader("Detailed Market Analysis")
    st.write("This section provides an in-depth analysis of the markets, commodities, forex, and cryptos.")

    # Get data for all categories
    data_nyse = get_stock_data("^NYA", START, END)
    data_bse = get_stock_data("^BSESN", START, END)
    data_gold = get_stock_data("GC=F", START, END)
    data_oil = get_stock_data("CL=F", START, END)
    data_eurusd = get_stock_data("EURUSD=X", START, END)
    data_gbpusd = get_stock_data("GBPUSD=X", START, END)
    data_btc = get_stock_data("BTC-USD", START, END)
    data_eth = get_stock_data("ETH-USD", START, END)

    # Calculate correlations
    correlation_data = {
        'NYSE': data_nyse['Close'],
        'BSE': data_bse['Close'],
        'Gold': data_gold['Close'],
        'Oil': data_oil['Close'],
        'EURUSD': data_eurusd['Close'],
        'GBPUSD': data_gbpusd['Close'],
        'BTC': data_btc['Close'],
        'ETH': data_eth['Close']
    }
    df_correlation = pd.DataFrame(correlation_data)
    correlation_matrix = df_correlation.corr()

    # Plot correlation matrix
    fig_corr_matrix = plot_correlation_matrix(correlation_matrix)
    st.plotly_chart(fig_corr_matrix)

    # Calculate market returns for beta calculation (assuming S&P 500 as market index)
    data_sp500 = get_stock_data("^GSPC", START, END)
    market_returns = data_sp500['Close'].pct_change().dropna()

    # Trend and Additional Insights Analysis
    st.write("**Trend Analysis and Insights:**")
    analysis_data = {
        "Assets": ['NYSE', 'BSE', 'Gold', 'Oil', 'EURUSD', 'GBPUSD', 'BTC', 'ETH'],
        "Trend": [
            "Bullish" if data_nyse['MA_15'].iloc[-1] > data_nyse['MA_50'].iloc[-1] else "Bearish",
            "Bullish" if data_bse['MA_15'].iloc[-1] > data_bse['MA_50'].iloc[-1] else "Bearish",
            "Bullish" if data_gold['MA_15'].iloc[-1] > data_gold['MA_50'].iloc[-1] else "Bearish",
            "Bullish" if data_oil['MA_15'].iloc[-1] > data_oil['MA_50'].iloc[-1] else "Bearish",
            "Bullish" if data_eurusd['MA_15'].iloc[-1] > data_eurusd['MA_50'].iloc[-1] else "Bearish",
            "Bullish" if data_gbpusd['MA_15'].iloc[-1] > data_gbpusd['MA_50'].iloc[-1] else "Bearish",
            "Bullish" if data_btc['MA_15'].iloc[-1] > data_btc['MA_50'].iloc[-1] else "Bearish",
            "Bullish" if data_eth['MA_15'].iloc[-1] > data_eth['MA_50'].iloc[-1] else "Bearish"
        ],
        "Volatility (Daily)": [
            np.std(data_nyse['Close']),
            np.std(data_bse['Close']),
            np.std(data_gold['Close']),
            np.std(data_oil['Close']),
            np.std(data_eurusd['Close']),
            np.std(data_gbpusd['Close']),
            np.std(data_btc['Close']),
            np.std(data_eth['Close'])
        ],
        "Average Return (%) (Daily)": [
            np.mean(data_nyse['Close'].pct_change()) * 100,
            np.mean(data_bse['Close'].pct_change()) * 100,
            np.mean(data_gold['Close'].pct_change()) * 100,
            np.mean(data_oil['Close'].pct_change()) * 100,
            np.mean(data_eurusd['Close'].pct_change()) * 100,
            np.mean(data_gbpusd['Close'].pct_change()) * 100,
            np.mean(data_btc['Close'].pct_change()) * 100,
            np.mean(data_eth['Close'].pct_change()) * 100
        ],
        "Sharpe Ratio (Daily)": [
            calculate_sharpe_ratio(data_nyse['Close'].pct_change()),
            calculate_sharpe_ratio(data_bse['Close'].pct_change()),
            calculate_sharpe_ratio(data_gold['Close'].pct_change()),
            calculate_sharpe_ratio(data_oil['Close'].pct_change()),
            calculate_sharpe_ratio(data_eurusd['Close'].pct_change()),
            calculate_sharpe_ratio(data_gbpusd['Close'].pct_change()),
            calculate_sharpe_ratio(data_btc['Close'].pct_change()),
            calculate_sharpe_ratio(data_eth['Close'].pct_change())
        ],
        "Max Drawdown (%)": [
            (data_nyse['Close'].max() - data_nyse['Close'].min()) / data_nyse['Close'].max() * 100,
            (data_bse['Close'].max() - data_bse['Close'].min()) / data_bse['Close'].max() * 100,
            (data_gold['Close'].max() - data_gold['Close'].min()) / data_gold['Close'].max() * 100,
            (data_oil['Close'].max() - data_oil['Close'].min()) / data_oil['Close'].max() * 100,
            (data_eurusd['Close'].max() - data_eurusd['Close'].min()) / data_eurusd['Close'].max() * 100,
            (data_gbpusd['Close'].max() - data_gbpusd['Close'].min()) / data_gbpusd['Close'].max() * 100,
            (data_btc['Close'].max() - data_btc['Close'].min()) / data_btc['Close'].max() * 100,
            (data_eth['Close'].max() - data_eth['Close'].min()) / data_eth['Close'].max() * 100
        ],
        "Beta": [
            calculate_beta(data_nyse['Close'].pct_change().dropna(), market_returns),
            calculate_beta(data_bse['Close'].pct_change().dropna(), market_returns),
            calculate_beta(data_gold['Close'].pct_change().dropna(), market_returns),
            calculate_beta(data_oil['Close'].pct_change().dropna(), market_returns),
            calculate_beta(data_eurusd['Close'].pct_change().dropna(), market_returns),
            calculate_beta(data_gbpusd['Close'].pct_change().dropna(), market_returns),
            calculate_beta(data_btc['Close'].pct_change().dropna(), market_returns),
            calculate_beta(data_eth['Close'].pct_change().dropna(), market_returns)
        ],
        "Value at Risk (VaR) 5%": [
            calculate_var(data_nyse['Close'].pct_change().dropna()),
            calculate_var(data_bse['Close'].pct_change().dropna()),
            calculate_var(data_gold['Close'].pct_change().dropna()),
            calculate_var(data_oil['Close'].pct_change().dropna()),
            calculate_var(data_eurusd['Close'].pct_change().dropna()),
            calculate_var(data_gbpusd['Close'].pct_change().dropna()),
            calculate_var(data_btc['Close'].pct_change().dropna()),
            calculate_var(data_eth['Close'].pct_change().dropna())
        ]
    }
    df_analysis = pd.DataFrame(analysis_data)
    st.table(df_analysis)

    # Annualized metrics
    st.write("**Annualized Metrics:**")
    annualized_data = {
        "Assets": ['NYSE', 'BSE', 'Gold', 'Oil', 'EURUSD', 'GBPUSD', 'BTC', 'ETH'],
        "Annualized Return (%)": [
            ((1 + np.mean(data_nyse['Close'].pct_change())) ** 252 - 1) * 100,
            ((1 + np.mean(data_bse['Close'].pct_change())) ** 252 - 1) * 100,
            ((1 + np.mean(data_gold['Close'].pct_change())) ** 252 - 1) * 100,
            ((1 + np.mean(data_oil['Close'].pct_change())) ** 252 - 1) * 100,
            ((1 + np.mean(data_eurusd['Close'].pct_change())) ** 252 - 1) * 100,
            ((1 + np.mean(data_gbpusd['Close'].pct_change())) ** 252 - 1) * 100,
            ((1 + np.mean(data_btc['Close'].pct_change())) ** 252 - 1) * 100,
            ((1 + np.mean(data_eth['Close'].pct_change())) ** 252 - 1) * 100
        ],
        "Annualized Volatility (%)": [
            np.std(data_nyse['Close'].pct_change()) * np.sqrt(252) * 100,
            np.std(data_bse['Close'].pct_change()) * np.sqrt(252) * 100,
            np.std(data_gold['Close'].pct_change()) * np.sqrt(252) * 100,
            np.std(data_oil['Close'].pct_change()) * np.sqrt(252) * 100,
            np.std(data_eurusd['Close'].pct_change()) * np.sqrt(252) * 100,
            np.std(data_gbpusd['Close'].pct_change()) * np.sqrt(252) * 100,
            np.std(data_btc['Close'].pct_change()) * np.sqrt(252) * 100,
            np.std(data_eth['Close'].pct_change()) * np.sqrt(252) * 100
        ]
    }
    df_annualized = pd.DataFrame(annualized_data)
    st.table(df_annualized)
