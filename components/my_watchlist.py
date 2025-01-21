import streamlit as st
from components.my_portfolio import get_user_id
from utils.mongodb import watchlists_collection
import yfinance as yf
import pandas as pd
from utils.constants import ticker_to_company_dict 


import plotly.graph_objects as go
import plotly.subplots as sp
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta


# Helper function to calculate RSI
def rsi(series, window=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Helper function to calculate ADX
def calculate_adx(df):
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr = pd.concat([df['High'] - df['Low'], 
                    (df['High'] - df['Close'].shift()).abs(), 
                    (df['Low'] - df['Close'].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1/14).mean() / atr))
    adx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).ewm(alpha=1/14).mean()
    return adx, plus_di, minus_di

# Helper function to calculate Bollinger Bands
def calculate_bollinger_bands(df, window=20):
    df['BB_Middle'] = df['Close'].rolling(window=window).mean()
    df['BB_Std'] = df['Close'].rolling(window=window).std()
    df['Bollinger_High'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['Bollinger_Low'] = df['BB_Middle'] - (df['BB_Std'] * 2)
    return df['Bollinger_High'], df['Bollinger_Low']

# Helper function to calculate indicators
def calculate_indicators(data):
    try:
        data['5_day_EMA'] = data['Close'].ewm(span=5, adjust=False).mean()
        data['15_day_EMA'] = data['Close'].ewm(span=15, adjust=False).mean()
        
        # MACD calculations
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

        # Manual RSI calculation
        data['RSI'] = rsi(data['Close'])

        # Manual ADX calculation
        data['ADX'], data['+DI'], data['-DI'] = calculate_adx(data)

        # Manual Bollinger Bands calculation
        data['Bollinger_High'], data['Bollinger_Low'] = calculate_bollinger_bands(data)

        # Volume MA calculation
        data['20_day_vol_MA'] = data['Volume'].rolling(window=20).mean()
        
        return data
    except Exception as e:
        raise ValueError(f"Error calculating indicators: {str(e)}")

# Helper function to fetch ticker data from yfinance
def fetch_ticker_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1y")
        if data.empty:
            raise ValueError(f"Ticker {ticker} not found")
        return data
    except Exception as e:
        raise ValueError(f"Error fetching data for ticker {ticker}: {str(e)}")

# Helper function to fetch company info from yfinance
def get_company_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info.get('longName', 'N/A'), info.get('sector', 'N/A'), info.get('industry', 'N/A')
    except Exception as e:
        return 'N/A', 'N/A', 'N/A'

# Convert the ticker_to_company_dict dictionary to a list of company names
company_names = list(ticker_to_company_dict.values())

# Watchlist feature
def display_watchlist():
    st.header(f"{st.session_state.username}'s Watchlist")
    user_id = get_user_id(st.session_state.email)
    watchlist = list(watchlists_collection.find({"user_id": user_id}))

    # Replace text input with a selectbox for company name auto-suggestion
    selected_company = st.sidebar.selectbox('Select or Enter Company Name:', company_names)

    # Retrieve the corresponding ticker for the selected company
    ticker = [ticker for ticker, company in ticker_to_company_dict.items() if company == selected_company][0]

    # Add new ticker to watchlist
    if st.sidebar.button("Add Ticker"):
        try:
            fetch_ticker_data(ticker)
            if not watchlists_collection.find_one({"user_id": user_id, "ticker": ticker}):
                watchlists_collection.insert_one({"user_id": user_id, "ticker": ticker})
                st.success(f"{ticker} ({selected_company}) added to your watchlist!")
                st.experimental_rerun()  # Refresh the app to reflect changes
            else:
                st.warning(f"{ticker} ({selected_company}) is already in your watchlist.")
        except ValueError as ve:
            st.error(ve)

    # Display watchlist
    if watchlist:
        watchlist_data = {}
        ticker_to_name_map = {}
        for entry in watchlist:
            ticker = entry['ticker']
            try:
                data = fetch_ticker_data(ticker)
                data = calculate_indicators(data)
                latest_data = data.iloc[-1]
                company_name, sector, industry = get_company_info(ticker)
                watchlist_data[ticker] = {
                    'Company Name': company_name,
                    'Sector': sector,
                    'Industry': industry,
                    'Close': latest_data['Close'],
                    '5_day_EMA': latest_data['5_day_EMA'],
                    '15_day_EMA': latest_data['15_day_EMA'],
                    'MACD': latest_data['MACD'],
                    'MACD_Signal': latest_data['MACD_Signal'],
                    'MACD_Hist': latest_data['MACD_Hist'],
                    'RSI': latest_data['RSI'],
                    'ADX': latest_data['ADX'],
                    'Bollinger_High': latest_data['Bollinger_High'],
                    'Bollinger_Low': latest_data['Bollinger_Low'],
                    'Volume': latest_data['Volume'],
                    '20_day_vol_MA': latest_data['20_day_vol_MA']
                }
                ticker_to_name_map[ticker] = company_name
            except ValueError as ve:
                st.error(f"Error fetching data for {ticker}: {ve}")

        if watchlist_data:
            watchlist_df = pd.DataFrame.from_dict(watchlist_data, orient='index')
            watchlist_df.reset_index(inplace=True)
            watchlist_df.rename(columns={'index': 'Ticker'}, inplace=True)

            # Use Styler to format the DataFrame
            styled_df = watchlist_df.style.format(precision=2)

            st.write("Your Watchlist:")
            st.dataframe(styled_df.set_properties(**{'text-align': 'center'}).set_table_styles(
                [{'selector': 'th', 'props': [('text-align', 'center')]}]
            ))

            # Streamlit App Title
            st.subheader("Technical Indicators and Price Analysis")

            # Sidebar for Inputs
            col1, col2, col3 = st.columns(3)
            with col1:
                stock_symbol = st.selectbox("Select Stock", watchlist_df['Ticker'].tolist())
                #st.text_input("Stock Symbol", "APOLLOHOSP.NS")
                
                watchlist_data[ticker]
            with col2:
                start_date = st.date_input('Start Date', value=datetime.now() - timedelta(days=365))
            with col3:
                end_date = st.date_input('End Date', value=datetime.now() + timedelta(days=1))

            # Step 1: Download Stock Data
            data = yf.download(stock_symbol, start=start_date, end=end_date)

            # Check if data is available
            if data.empty:
                st.warning("No data available for the given ticker and date range. Please try again.")
            else:
                # Step 2: Calculate Technical Indicators
                # VWAP (Volume Weighted Average Price)
                data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()

                # MFI (Money Flow Index)
                data['MFI'] = ta.mfi(data['High'], data['Low'], data['Close'], data['Volume'], length=14)

                # OBV (On-Balance Volume)
                data['OBV'] = ta.obv(data['Close'], data['Volume'])

                # CMF (Chaikin Money Flow)
                data['CMF'] = ta.cmf(data['High'], data['Low'], data['Close'], data['Volume'], length=20)

                # A/D (Accumulation/Distribution)
                data['AD'] = ta.ad(data['High'], data['Low'], data['Close'], data['Volume'])

                # Ichimoku Cloud
                data['Ichimoku_Tenkan'] = (data['High'].rolling(window=9).max() + data['Low'].rolling(window=9).min()) / 2
                data['Ichimoku_Kijun'] = (data['High'].rolling(window=26).max() + data['Low'].rolling(window=26).min()) / 2
                data['Ichimoku_Senkou_Span_A'] = ((data['Ichimoku_Tenkan'] + data['Ichimoku_Kijun']) / 2).shift(26)
                data['Ichimoku_Senkou_Span_B'] = ((data['High'].rolling(window=52).max() + data['Low'].rolling(window=52).min()) / 2).shift(26)

                # MACD (Moving Average Convergence Divergence)
                data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
                data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
                data['MACD'] = data['EMA_12'] - data['EMA_26']
                data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
                data['MACD_hist'] = data['MACD'] - data['MACD_signal']

                # SuperTrend
                supertrend = ta.supertrend(data['High'], data['Low'], data['Close'], length=7, multiplier=3.0)
                data['SuperTrend'] = supertrend['SUPERT_7_3.0']

                # Bollinger Bands
                data['BB_Middle'] = data['Close'].rolling(window=20).mean()
                data['BB_Std'] = data['Close'].rolling(window=20).std()
                data['BB_High'] = data['BB_Middle'] + (data['BB_Std'] * 2)
                data['BB_Low'] = data['BB_Middle'] - (data['BB_Std'] * 2)

                # Parabolic SAR
                def parabolic_sar(high, low, close, af=0.02, max_af=0.2):
                    psar = close.copy()
                    psar.fillna(0, inplace=True)
                    bull = True
                    ep = low[0]
                    hp = high[0]
                    lp = low[0]
                    for i in range(2, len(close)):
                        psar[i] = psar[i - 1] + af * (ep - psar[i - 1])
                        if bull:
                            if low[i] < psar[i]:
                                bull = False
                                psar[i] = hp
                                lp = low[i]
                                af = 0.02
                                ep = low[i]
                            if high[i] > ep:
                                ep = high[i]
                                af = min(af + 0.02, max_af)
                            if low[i - 1] < psar[i]:
                                psar[i] = low[i - 1]
                            if low[i - 2] < psar[i]:
                                psar[i] = low[i - 2]
                        else:
                            if high[i] > psar[i]:
                                bull = True
                                psar[i] = lp
                                hp = high[i]
                                af = 0.02
                                ep = high[i]
                            if low[i] < ep:
                                ep = low[i]
                                af = min(af + 0.02, max_af)
                            if high[i - 1] > psar[i]:
                                psar[i] = high[i - 1]
                            if high[i - 2] > psar[i]:
                                psar[i] = high[i - 2]
                    return psar

                data['PSAR'] = parabolic_sar(data['High'], data['Low'], data['Close'])

                # GMMA (Guppy Multiple Moving Average)
                short_ema = ta.ema(data['Close'], length=3)
                long_ema = ta.ema(data['Close'], length=30)

                # RSI (Relative Strength Index)
                data['RSI'] = ta.rsi(data['Close'], length=14)

                # Stochastic Oscillator
                data['Stochastic_%K'] = (data['Close'] - data['Low'].rolling(window=14).min()) / (data['High'].rolling(window=14).max() - data['Low'].rolling(window=14).min()) * 100
                data['Stochastic_%D'] = data['Stochastic_%K'].rolling(window=3).mean()

                # DMI (Directional Movement Index)
                def calculate_adx(data):
                    # Calculate the Directional Movements
                    plus_dm = data['High'].diff()
                    minus_dm = data['Low'].diff()
                    plus_dm[plus_dm < 0] = 0
                    minus_dm[minus_dm > 0] = 0
                    
                    # Calculate the True Range (TR) and Average True Range (ATR)
                    tr = pd.concat([data['High'] - data['Low'], 
                                    (data['High'] - data['Close'].shift()).abs(), 
                                    (data['Low'] - data['Close'].shift()).abs()], axis=1).max(axis=1)
                    atr = tr.rolling(window=14).mean()
                    
                    # Calculate the Positive Directional Indicator (Plus DI) and Negative Directional Indicator (Minus DI)
                    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
                    minus_di = abs(100 * (minus_dm.ewm(alpha=1/14).mean() / atr))
                    
                    # Calculate the ADX
                    adx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).ewm(alpha=1/14).mean()
                    
                    return adx, plus_di, minus_di

                # Apply ADX to your data
                data['ADX'], data['Plus_DI'], data['Minus_DI'] = calculate_adx(data)

                # Awesome Oscillator
                data['AO'] = ta.ao(data['High'], data['Low'])

                # Bollinger Bands %B
                data['BB%'] =  (data['Close'] - data['BB_Low']) / (data['BB_High'] - data['BB_Low'])

                # Mass Index
                data['Mass_Index'] = (data['High'] - data['Low']).rolling(window=25).sum() / (data['High'] - data['Low']).rolling(window=9).sum()

                # Relative Volatility Index (RVI)
                data['RVI'] = ta.rvi(data['High'], data['Low'], data['Close'], length=14)

                # ZigZag
                def zigzag(close, percentage=5):
                    zz = [0]
                    for i in range(1, len(close)):
                        change = (close[i] - close[zz[-1]]) / close[zz[-1]] * 100
                        if abs(change) > percentage:
                            zz.append(i)
                    zigzag_series = pd.Series(index=close.index, data=np.nan)
                    zigzag_series.iloc[zz] = close.iloc[zz]
                    return zigzag_series.ffill()

                data['ZigZag'] = zigzag(data['Close'])

                # Pivot Points Standard
                data['Pivot'] = (data['High'] + data['Low'] + data['Close']) / 3
                data['R1'] = 2 * data['Pivot'] - data['Low']
                data['S1'] = 2 * data['Pivot'] - data['High']

                # Fibonacci Levels
                max_price = data['High'].max()
                min_price = data['Low'].min()
                diff = max_price - min_price
                data['Fibo_23_6'] = max_price - 0.236 * diff
                data['Fibo_38_2'] = max_price - 0.382 * diff
                data['Fibo_50'] = max_price - 0.5 * diff
                data['Fibo_61_8'] = max_price - 0.618 * diff

                # Step 3: Create Subplots for all indicators
           

                fig = sp.make_subplots(rows=7, cols=3, subplot_titles=[
                    'VWAP', 'MFI', 'OBV', 'CMF', 
                    'A/D', 'Ichimoku Cloud', 'MACD', 'SuperTrend', 
                    'Bollinger Bands', 'Parabolic SAR', 'GMMA', 'RSI', 
                    'Stochastic Oscillator', 'DMI', 'Awesome Oscillator', 'BB%',
                    'Mass Index', 'RVI', 'ZigZag', 'Pivot Points', 
                    'Fibonacci Levels'],
                    vertical_spacing=0.05, horizontal_spacing=0.05)

                # Plot VWAP and Close
                fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], name='VWAP',line={'color': 'red', 'width': 2}), row=1, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close',line={'color': 'blue', 'width': 2}), row=1, col=1)


                # Plot MFI with red lines at 20 and 80
                fig.add_trace(go.Scatter(x=data.index, y=data['MFI'], name='MFI', line=dict(color='blue')), row=1, col=2)
                fig.add_trace(go.Scatter(x=data.index, y=[20] * len(data), name='MFI 20', line={'color':'green', 'width': 2}), row=1, col=2)
                fig.add_trace(go.Scatter(x=data.index, y=[80] * len(data), name='MFI 80', line={'color':'red', 'width': 2}), row=1, col=2)


                # Plot OBV
                fig.add_trace(go.Scatter(x=data.index, y=data['OBV'], name='OBV'), row=1, col=3)

                # Plot CMF
                fig.add_trace(go.Scatter(x=data.index, y=data['CMF'], name='CMF',line={'color': 'blue', 'width': 2}), row=2, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=[0] * len(data), name='CMF 0', line={'color':'red', 'width': 2}), row=2, col=1)

                # Plot A/D
                fig.add_trace(go.Scatter(x=data.index, y=data['AD'], name='A/D'), row=2, col=2)

                # Plot Ichimoku Cloud
                fig.add_trace(go.Scatter(x=data.index, y=data['Ichimoku_Senkou_Span_A'], name='Ichimoku A', fill='tonexty', fillcolor='rgba(0,128,0,0.3)'), row=2, col=3)
                fig.add_trace(go.Scatter(x=data.index, y=data['Ichimoku_Senkou_Span_B'], name='Ichimoku B', fill='tonexty', fillcolor='rgba(255,0,0,0.8)'), row=2, col=3)
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close',line={'color': 'blue', 'width': 2}), row=2, col=3)



                # Plot MACD
                fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='green')), row=3, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data['MACD_signal'], name='MACD Signal', line=dict(color='red')), row=3, col=1)
                fig.add_trace(go.Bar(x=data.index, y=data['MACD_hist'], name='MACD Histogram', marker_color='rgba(255,0,0,2)'), row=3, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=[0] * len(data), name='MACD 0', line={'color': 'black', 'width': 0.5}), row=3, col=1)

                # Plot SuperTrend
                fig.add_trace(go.Scatter(x=data.index, y=data['SuperTrend'], name='SuperTrend', line=dict(color='red')), row=3, col=2)
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close', line={'color': 'blue', 'width': 2}), row=3, col=2)

                # Plot Bollinger Bands
                fig.add_trace(go.Scatter(x=data.index, y=data['BB_High'], name='BB High', line=dict(color='red')), row=3, col=3)
                fig.add_trace(go.Scatter(x=data.index, y=data['BB_Low'], name='BB Low', line=dict(color='green')), row=3, col=3)
                fig.add_trace(go.Scatter(x=data.index, y=data['BB_Middle'], name='BB Middle', line={'dash': 'dot'}), row=3, col=3)
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close', line={'color': 'blue', 'width': 2}), row=3, col=3)

                # Plot Parabolic SAR
                fig.add_trace(go.Scatter(x=data.index, y=data['PSAR'], mode='markers', name='PSAR', marker=dict(color='red', size=3)), row=4, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close', line={'color': 'blue', 'width': 2}), row=4, col=1)

                # Plot GMMA
                fig.add_trace(go.Scatter(x=data.index, y=short_ema, name='Short EMA', line=dict(color='green')), row=4, col=2)
                fig.add_trace(go.Scatter(x=data.index, y=long_ema, name='Long EMA', line=dict(color='red')), row=4, col=2)

                # Plot RSI
                fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI',line={'color': 'blue', 'width': 2}), row=4, col=3)
                fig.add_trace(go.Scatter(x=data.index, y=[30] * len(data), name='RSI 30', line={'color':'green', 'width': 2}), row=4, col=3)
                fig.add_trace(go.Scatter(x=data.index, y=[70] * len(data), name='RSI 70', line={'color':'red', 'width': 2}), row=4, col=3)


                # Plot Stochastic Oscillator
                fig.add_trace(go.Scatter(x=data.index, y=data['Stochastic_%K'], name='Stochastic_%K', line=dict(color='green')), row=5, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=data['Stochastic_%D'], name='Stochastic_%D', line=dict(color='red')), row=5, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=[20] * len(data), name='Stochastic 20', line=dict(color='blue', dash='dash')), row=5, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=[80] * len(data), name='Stochastic 80', line=dict(color='blue', dash='dash')), row=5, col=1)

                # Plot DMI
                fig.add_trace(go.Scatter(x=data.index, y=data['Plus_DI'], name='Plus DI', line=dict(color='green')), row=5, col=2)
                fig.add_trace(go.Scatter(x=data.index, y=data['Minus_DI'], name='Minus DI', line=dict(color='red')), row=5, col=2)
                fig.add_trace(go.Scatter(x=data.index, y=data['ADX'], name='ADX', line=dict(color='blue')), row=5, col=2)

                # Plot Awesome Oscillator
                fig.add_trace(go.Scatter(x=data.index, y=data['AO'], name='Awesome Oscillator', line=dict(color='blue')), row=5, col=3)
                fig.add_trace(go.Scatter(x=data.index, y=[0] * len(data), name='AO 0', line=dict(color='red', dash='dash')), row=5, col=3)

                # Plot BB%
                fig.add_trace(go.Scatter(x=data.index, y=data['BB%'], name='BB%', line=dict(color='blue')), row=6, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=[1] * len(data), name='BB 1', line=dict(color='red', dash='dash')), row=6, col=1)
                fig.add_trace(go.Scatter(x=data.index, y=[0] * len(data), name='BB 0', line=dict(color='red', dash='dash')), row=6, col=1)

                # Plot Mass Index
                fig.add_trace(go.Scatter(x=data.index, y=data['Mass_Index'], name='Mass Index'), row=6, col=2)

                # Plot RVI
                fig.add_trace(go.Scatter(x=data.index, y=data['RVI'], name='RVI', line=dict(color='blue')), row=6, col=3)
                fig.add_trace(go.Scatter(x=data.index, y=[50] * len(data), name='RVI 50', line=dict(color='red', dash='dash')), row=6, col=3)

                # Plot ZigZag
                fig.add_trace(go.Scatter(x=data.index, y=data['ZigZag'], name='ZigZag'), row=7, col=1)

                # Plot Pivot Points
                fig.add_trace(go.Scatter(x=data.index, y=data['Pivot'], name='Pivot'), row=7, col=2)
                fig.add_trace(go.Scatter(x=data.index, y=data['R1'], name='R1', line={'dash': 'dot'}), row=7, col=2)
                fig.add_trace(go.Scatter(x=data.index, y=data['S1'], name='S1', line={'dash': 'dot'}), row=7, col=2)
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close', line={'color': 'blue', 'width': 2}), row=7, col=2)

                # Plot Fibonacci Levels
                fig.add_trace(go.Scatter(x=data.index, y=data['Fibo_23_6'], name='Fibo 23.6%', line={'dash': 'dot'}), row=7, col=3)
                fig.add_trace(go.Scatter(x=data.index, y=data['Fibo_38_2'], name='Fibo 38.2%', line={'dash': 'dot'}), row=7, col=3)
                fig.add_trace(go.Scatter(x=data.index, y=data['Fibo_50'], name='Fibo 50%', line={'dash': 'dot'}), row=7, col=3)
                fig.add_trace(go.Scatter(x=data.index, y=data['Fibo_61_8'], name='Fibo 61.8%', line={'dash': 'dot'}), row=7, col=3)
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close', line={'color': 'blue', 'width': 2}), row=7, col=3)

                # Layout
                fig.update_layout(
                    height=3000, width=1500,
                    title='Technical Indicators and Price Analysis',
                    showlegend=False
                )

                # Display the plot
                st.plotly_chart(fig)

            # Option to remove ticker from watchlist
            company_names_in_watchlist = [ticker_to_name_map[entry['ticker']] for entry in watchlist]
            company_to_remove = st.sidebar.selectbox("Select a company to remove", company_names_in_watchlist)
            ticker_to_remove = [ticker for ticker, name in ticker_to_name_map.items() if name == company_to_remove][0]
            if st.sidebar.button("Remove Ticker"):
                watchlists_collection.delete_one({"user_id": user_id, "ticker": ticker_to_remove})
                st.success(f"{company_to_remove} removed from your watchlist.")
                st.experimental_rerun()  # Refresh the app to reflect changes
        else:
            st.write("No valid data found for the tickers in your watchlist.")
    else:
        st.write("Your watchlist is empty.")









# Call the function to display the watchlist
if 'username' not in st.session_state:
    st.session_state.username = 'Guest'  # or handle the case where username is not set
if 'email' not in st.session_state:
    st.session_state.email = 'guest@example.com'  # or handle the case where email is not set
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False  # or handle the case where logged_in is not set

if st.session_state.logged_in:
    display_watchlist()
else:
    st.write("Please log in to view your watchlist.")