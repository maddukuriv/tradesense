# pages/markets.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.constants import bse_largecap, bse_midcap, bse_smallcap
from datetime import datetime

# Function to download data and calculate moving averages with caching
def get_stock_data(ticker_symbol):
    data = yf.download(ticker_symbol, period='1y')
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
                      xaxis_rangeslider_visible=True, hovermode='x',
                      xaxis=dict(rangeselector=dict(buttons=list([
                          dict(count=1, label="1m", step="month", stepmode="backward"),
                          dict(count=6, label="6m", step="month", stepmode="backward"),
                          dict(count=1, label="YTD", step="year", stepmode="todate"),
                          dict(count=1, label="1y", step="year", stepmode="backward"),
                          dict(step="all")
                      ])), rangeslider=dict(visible=True), type='date'),
                      yaxis=dict(fixedrange=False),
                      updatemenus=[dict(type="buttons", buttons=[dict(label="Reset Zoom",
                                                                      method="relayout",
                                                                      args=[{"xaxis.range": [None, None],
                                                                             "yaxis.range": [None, None]}])])])
    return fig

# Function to calculate correlation
def calculate_correlation(data1, data2):
    return data1['Close'].corr(data2['Close'])

# Function to plot correlation matrix
def plot_correlation_matrix(correlation_matrix):
    fig = go.Figure(data=go.Heatmap(z=correlation_matrix.values,
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
    aligned_returns = pd.concat([asset_returns, market_returns], axis=1).dropna()
    covariance_matrix = np.cov(aligned_returns.iloc[:, 0], aligned_returns.iloc[:, 1])
    beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
    return beta

# Function to calculate Value at Risk (VaR)
def calculate_var(returns, confidence_level=0.05):
    return np.percentile(returns, confidence_level * 100)

# Main application
def markets_app():
    submenu = st.sidebar.selectbox("Select Option", ["Equities", "Commodities", "Currencies", "Cryptocurrencies", "Insights"])

    if submenu == "Equities":
        ticker_category = st.sidebar.selectbox("Select Index", ["BSE-LargeCap", "BSE-MidCap", "BSE-SmallCap"])
        tickers = {"BSE-LargeCap": bse_largecap, "BSE-MidCap": bse_midcap, "BSE-SmallCap": bse_smallcap}[ticker_category]

        @st.cache_data(ttl=60)
        def get_sector_industry_price_changes(tickers, timestamp):
            data = {
                'Ticker': [], 'Company Name': [], 'Sector': [], 'Industry': [], 'Market Cap': [], 'Last Traded Price': [],
                '1D % Change': [], '2D % Change': [], '3D % Change': [], '5D % Change': [], '2W % Change': [],
                '1M % Change': [], '3M % Change': [], '6M % Change': [], '1Y % Change': [],
                '1D Volume': [], '2D Volume': [], '5D Volume': [], '2W Volume': [],
                '1M Volume': [], '3M Volume': [], '6M Volume': [], '1Y Volume': [],
                'Volume Change %': []
            }
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    price_data_1y = yf.download(ticker, period='1y')

                    if not price_data_1y.empty:
                        last_traded_price = price_data_1y['Close'].iloc[-1]
                        one_day_volume = price_data_1y['Volume'].iloc[-1]
                        two_day_volume = price_data_1y['Volume'].iloc[-2:].mean()
                        five_day_volume = price_data_1y['Volume'].iloc[-5:].mean()
                        two_week_volume = price_data_1y['Volume'].iloc[-10:].mean()
                        one_month_volume = price_data_1y['Volume'].iloc[-21:].mean()
                        three_month_volume = price_data_1y['Volume'].iloc[-63:].mean()
                        six_month_volume = price_data_1y['Volume'].iloc[-126:].mean()
                        one_year_volume = price_data_1y['Volume'].mean()
                        avg_volume = price_data_1y['Volume'].mean()
                        volume_change = ((one_day_volume - avg_volume) / avg_volume) * 100 if avg_volume != 0 else 'N/A'
                        price_changes = price_data_1y['Close'].pct_change() * 100
                        one_day_change = price_changes.iloc[-1]
                        two_day_change = price_changes.iloc[-2:].sum()
                        three_day_change = price_changes.iloc[-3:].sum()
                        five_day_change = price_changes.iloc[-5:].sum()
                        two_week_change = price_changes.iloc[-10:].sum()
                        one_month_change = price_changes.iloc[-21:].sum()
                        three_month_change = price_changes.iloc[-63:].sum()
                        six_month_change = price_changes.iloc[-126:].sum()
                        one_year_change = price_changes.sum()
                    else:
                        last_traded_price = 'N/A'
                        one_day_volume = 'N/A'
                        two_day_volume = 'N/A'
                        five_day_volume = 'N/A'
                        two_week_volume = 'N/A'
                        one_month_volume = 'N/A'
                        three_month_volume = 'N/A'
                        six_month_volume = 'N/A'
                        one_year_volume = 'N/A'
                        volume_change = 'N/A'
                        one_day_change = 'N/A'
                        two_day_change = 'N/A'
                        three_day_change = 'N/A'
                        five_day_change = 'N/A'
                        two_week_change = 'N/A'
                        one_month_change = 'N/A'
                        three_month_change = 'N/A'
                        six_month_change = 'N/A'
                        one_year_change = 'N/A'

                    data['Ticker'].append(ticker)
                    data['Company Name'].append(info.get('longName', 'N/A'))
                    data['Sector'].append(info.get('sector', 'N/A'))
                    data['Industry'].append(info.get('industry', 'N/A'))
                    data['Last Traded Price'].append(last_traded_price)
                    data['Market Cap'].append(info.get('marketCap', 'N/A'))
                    data['1D % Change'].append(one_day_change)
                    data['2D % Change'].append(two_day_change)
                    data['3D % Change'].append(three_day_change)
                    data['5D % Change'].append(five_day_change)
                    data['2W % Change'].append(two_week_change)
                    data['1M % Change'].append(one_month_change)
                    data['3M % Change'].append(three_month_change)
                    data['6M % Change'].append(six_month_change)
                    data['1Y % Change'].append(one_year_change)
                    data['1D Volume'].append(one_day_volume)
                    data['2D Volume'].append(two_day_volume)
                    data['5D Volume'].append(five_day_volume)
                    data['2W Volume'].append(two_week_volume)
                    data['1M Volume'].append(one_month_volume)
                    data['3M Volume'].append(three_month_volume)
                    data['6M Volume'].append(six_month_volume)
                    data['1Y Volume'].append(one_year_volume)
                    data['Volume Change %'].append(volume_change)

                except Exception as e:
                    st.error(f"Error fetching data for {ticker}: {e}")

            df = pd.DataFrame(data)

            # Convert all relevant columns to numeric and fill NaNs with 0
            numeric_columns = [col for col in df.columns if col not in ['Ticker', 'Company Name', 'Sector', 'Industry']]
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

            return df

        sector_industry_price_changes_df = get_sector_industry_price_changes(tickers, datetime.now())

        # Streamlit app
        st.subheader('Market Stats')
        st.dataframe(sector_industry_price_changes_df)
        st.subheader('Price')
        price_chart_option = st.selectbox('Select period to view price changes:', [
            '1D % Change', '2D % Change', '3D % Change', '5D % Change',
            '2W % Change', '1M % Change', '3M % Change', '6M % Change', '1Y % Change'
        ])

        df_price_sorted = sector_industry_price_changes_df[['Ticker', price_chart_option]].copy()
        df_price_sorted[price_chart_option] = pd.to_numeric(df_price_sorted[price_chart_option], errors='coerce')
        df_price_sorted = df_price_sorted.sort_values(by=price_chart_option, ascending=False).reset_index(drop=True)
        df_price_sorted.columns = ['Ticker', '% Change']

        fig_price = px.bar(df_price_sorted, x='Ticker', y='% Change', title=f'{price_chart_option} Gainers/Losers', color='% Change', color_continuous_scale=px.colors.diverging.RdYlGn)
        st.plotly_chart(fig_price)

        st.subheader('Volume')
        volume_chart_option = st.selectbox('Select period to view volume changes:', [
            '1D Volume', '2D Volume', '5D Volume', '2W Volume',
            '1M Volume', '3M Volume', '6M Volume', '1Y Volume'
        ])

        df_volume_sorted = sector_industry_price_changes_df[['Ticker', volume_chart_option]].copy()
        df_volume_sorted[volume_chart_option] = pd.to_numeric(df_volume_sorted[volume_chart_option], errors='coerce')
        df_volume_sorted = df_volume_sorted.sort_values(by=volume_chart_option, ascending=False).reset_index(drop=True)
        df_volume_sorted.columns = ['Ticker', 'Volume']

        fig_volume = px.bar(df_volume_sorted, x='Ticker', y='Volume', title=f'{volume_chart_option} Volume', color='Volume', color_continuous_scale=px.colors.diverging.RdYlGn)
        st.plotly_chart(fig_volume)

        st.subheader('Sector and Industry Performance')
        numeric_columns = [col for col in sector_industry_price_changes_df.columns if col not in ['Ticker', 'Company Name', 'Sector', 'Industry']]

        sector_performance = sector_industry_price_changes_df.groupby('Sector')[numeric_columns].mean().reset_index()
        sector_chart_option = st.selectbox('Select period to view sector performance:', numeric_columns)

        sector_sorted = sector_performance[['Sector', sector_chart_option]].copy()
        sector_sorted = sector_sorted.sort_values(by=sector_chart_option, ascending=False).reset_index(drop=True)
        sector_sorted.columns = ['Sector', '% Change']

        fig_sector = px.bar(sector_sorted, x='Sector', y='% Change', title=f'{sector_chart_option} by Sector', color='% Change', color_continuous_scale=px.colors.diverging.RdYlGn)
        st.plotly_chart(fig_sector)

        industry_performance = sector_industry_price_changes_df.groupby('Industry')[numeric_columns].mean().reset_index()
        industry_chart_option = st.selectbox('Select period to view industry performance:', numeric_columns)

        industry_sorted = industry_performance[['Industry', industry_chart_option]].copy()
        industry_sorted = industry_sorted.sort_values(by=industry_chart_option, ascending=False).reset_index(drop=True)
        industry_sorted.columns = ['Industry', '% Change']

        fig_industry = px.bar(industry_sorted, x='Industry', y='% Change', title=f'{industry_chart_option} by Industry', color='% Change', color_continuous_scale=px.colors.diverging.RdYlGn)
        st.plotly_chart(fig_industry)

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
                data = get_stock_data(ticker)
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
                data = get_stock_data(ticker)
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
                data = get_stock_data(ticker)
                fig = create_figure(data, indicators, f'{ticker} Price')
                col.plotly_chart(fig)

    elif submenu == "Insights":
        st.subheader("Detailed Market Analysis")
        st.write("This section provides an in-depth analysis of the markets, commodities, forex, and cryptos.")

        # Get data for all categories
        data_nyse = get_stock_data("^NYA")
        data_bse = get_stock_data("^BSESN")
        data_gold = get_stock_data("GC=F")
        data_oil = get_stock_data("CL=F")
        data_eurusd = get_stock_data("EURUSD=X")
        data_gbpusd = get_stock_data("GBPUSD=X")
        data_btc = get_stock_data("BTC-USD")
        data_eth = get_stock_data("ETH-USD")

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
        data_sp500 = get_stock_data("^GSPC")
        market_returns = data_sp500['Close'].pct_change().dropna()

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

# Run the app
if __name__ == "__main__":
    markets_app()
