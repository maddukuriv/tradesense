import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objs as go

# List of stock tickers
tickers = [
    "ITC.NS", "JBCHEPHARM.BO", "JWL.BO", "JYOTHYLAB.BO", "KPRMILL.NS", "KAJARIACER.NS", "KEI.BO",
    "LTTS.NS", "LTIM.NS", "MANKIND.NS", "MARICO.NS", "METROBRAND.BO", "MOTILALOFS.NS", "MPHASIS.NS",
    "MUTHOOTFIN.NS", "NH.NS", "NAVINFLUOR.NS", "NAM-INDIA.BO", "NMDC.NS", "OFSS.NS", "PGHH.NS",
    "PIIND.NS", "PAGEIND.NS", "PERSISTENT.NS", "PETRONET.NS", "PFIZER.NS", "PIDILITIND.NS",
    "POLYMED.NS", "POLYCAB.NS", "RRKABEL.NS", "RVNL.NS", "RATNAMANI.NS", "RITES.NS", "SANOFI.NS",
    "SCHAEFFLER.NS", "SKFINDIA.NS", "SOLARINDS.NS", "SONACOMS.NS", "SUMICHEM.NS", "SUNTV.NS",
    "SUNDRMFAST.NS", "SUPREMEIND.BO", "TATAELXSI.NS", "TATATECH.NS", "TCS.NS", "TECHM.NS", "TIMKEN.NS",
    "TITAN.NS", "TRITURBINE.NS", "TIINDIA.NS", "UNITDSPR.BO", "VGUARD.NS", "MANYAVAR.NS",
    "VINATIORGA.NS", "WIPRO.NS", "ZYDUSLIFE.NS", "CUMMINSIND.NS", "CYIENT.NS", "DATAPATTNS.NS", "DEEPAKNTR.NS",
    "DIVISLAB.NS", "LALPATHLAB.NS", "RDY", "ELGIEQUIP.NS", "EMAMILTD.NS", "FIVESTAR.BO", "GRINFRA.NS",
    "GILLETTE.NS", "GLAXO.NS", "GODFRYPHLP.NS", "GRINDWELL.NS", "HAVELLS.NS", "HCLTECH.NS", "HAL.BO",
    "HONAUT.BO", "IRCTC.NS", "ISEC.BO", "INFY.NS", "IPCALAB.BO", "ABBOTINDIA.NS", "ADANIPOWER.NS",
    "AFFLE.BO", "AIAENG.BO", "AJANTPHARM.BO", "APLLTD.BO", "ALKEM.BO", "ARE&M.NS", "ANANDRATHI.BO",
    "APARINDS.BO", "ASIANPAINT.NS", "ASTRAL.NS", "ASTRAZEN.NS", "BAJFINANCE.NS", "BASF.NS", "BAYERCROP.BO",
    "BERGEPAINT.BO", "BDL.NS", "BEL.NS", "BSOFT.BO", "CDSL.NS", "CAMS.NS", "CARBORUNIV.BO", "CASTROLIND.NS",
    "CHAMBLFERT.BO", "COALINDIA.NS", "COFORGE.BO", "COLPAL.NS", "CONCORDBIO.BO", "COROMANDEL.BO", "CREDITACC.BO",
    "PNCINFRA.NS", "INDIASHLTR.NS", "RAYMOND.NS", "KAMAHOLD.BO", "BENGALASM.BO", "CHOICEIN.NS",
    "GRAVITA.NS", "HGINFRA.NS", "JKPAPER.NS", "MTARTECH.NS", "HAPPSTMNDS.NS", "SARDAEN.NS",
    "WELENT.NS", "LTFOODS.NS", "GESHIP.NS", "SHRIPISTON.NS", "SHAREINDIA.NS", "CYIENTDLM.NS", "VTL.NS",
    "EASEMYTRIP.NS", "LLOYDSME.NS", "ROUTE.NS", "VAIBHAVGBL.NS", "GOKEX.NS", "USHAMART.NS", "EIDPARRY.NS",
    "KIRLOSBROS.NS", "MANINFRA.NS", "CMSINFO.NS", "RALLIS.NS", "GHCL.NS", "NEULANDLAB.NS", "SPLPETRO.NS",
    "MARKSANS.NS", "NAVINFLUOR.NS", "ELECON.NS", "TANLA.NS", "KFINTECH.NS", "TIPSINDLTD.NS", "ACI.NS",
    "SURYAROSNI.NS", "GPIL.NS", "GMDCLTD.NS", "MAHSEAMLES.NS", "TDPOWERSYS.NS", "TECHNOE.NS", "JLHL.NS"
]

st.sidebar.subheader("Strategies")
submenu = st.sidebar.radio("Select Option", ["MACD", "Bollinger", "Strategy1", "Strategy2", "Strategy3", "Strategy4", "Resistance"])

if submenu == "MACD":
    st.subheader("MACD")
        
    # Function to calculate MACD, RSI, and ADX
    @st.cache_data(ttl=3600)
    def calculate_indicators(data):
        # Calculate short-term and long-term EMA
        short_ema = data['Close'].ewm(span=6, adjust=False).mean()
        long_ema = data['Close'].ewm(span=13, adjust=False).mean()
        
        # Calculate MACD and Signal line
        macd = short_ema - long_ema
        signal = macd.ewm(span=5, adjust=False).mean()
        
        # Calculate MACD Histogram
        macd_hist = macd - signal
        
        # Calculate RSI
        delta = data['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=9, min_periods=1).mean()
        avg_loss = loss.rolling(window=9, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate ADX
        adx = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=14).adx()
        
        return macd, signal, macd_hist, rsi, adx

    # Function to fetch and process stock data
    @st.cache_data(ttl=3600)
    def get_stock_data(tickers, period):
        try:
            stock_data = {}
            for ticker in tickers:
                df = yf.download(ticker, period=period)
                if not df.empty:
                    df.interpolate(method='linear', inplace=True)
                    macd, signal, macd_hist, rsi, adx = calculate_indicators(df)
                    df['MACD'] = macd
                    df['MACD Signal'] = signal
                    df['MACD Histogram'] = macd_hist
                    df['RSI'] = rsi
                    df['ADX'] = adx
                    df.dropna(inplace=True)  # Ensure we drop rows with NaN values after indicator calculation
                    stock_data[ticker] = df
            return stock_data
        except Exception as e:
            st.write(f"Error fetching data: {e}")
            return {}

    # Function to screen stocks based on MACD, RSI, and ADX
    def screen_stocks(stock_data):
        screened_stocks = []

        for ticker, data in stock_data.items():
            # Check condition for screening stocks
            if data['MACD'].iloc[-1] > data['MACD Signal'].iloc[-1] and data['MACD'].iloc[-2] <= data['MACD Signal'].iloc[-2]:
                screened_stocks.append({
                    'Ticker': ticker,
                    'Close': data['Close'].iloc[-1],
                    'MACD': data['MACD'].iloc[-1],
                    'MACD Signal': data['MACD Signal'].iloc[-1],
                    'MACD Histogram': data['MACD Histogram'].iloc[-1],
                    'RSI': data['RSI'].iloc[-1],
                    'ADX': data['ADX'].iloc[-1]
                })

        return pd.DataFrame(screened_stocks)

    # Fetch data for the last 3 months for all tickers
    stock_data = get_stock_data(tickers, period="3mo")

    # Screen the stocks
    screened_stocks_df = screen_stocks(stock_data)

    st.subheader("Screened Stocks")
    st.write("Stocks that meet the screening criteria based on MACD, RSI, and ADX:")
    st.dataframe(screened_stocks_df)

    # Dropdown to select stock from screened stocks
    selected_stock = st.selectbox("Select Stock", screened_stocks_df['Ticker'].tolist())

    # Radio buttons for timeframes
    timeframe = st.radio("Select Timeframe", ["5 days", "10 days", "15 days", "1 month", "2 months", "3 months"], index=2, horizontal=True)

    # Determine the period based on the selected timeframe
    if timeframe == "5 days":
        period = "5d"
    elif timeframe == "10 days":
        period = "10d"
    elif timeframe == "15 days":
        period = "15d"
    elif timeframe == "1 month":
        period = "1mo"
    elif timeframe == "2 months":
        period = "2mo"
    else:
        period = "3mo"

    # Multi-select options for indicators to display
    indicators_to_display = st.multiselect("Select Indicators", ["Close", "MACD", "MACD Signal", "MACD Histogram", "RSI", "ADX"], default=["Close"])

    # Fetch the selected stock data
    if selected_stock:
        stock_data_for_plot = stock_data[selected_stock].copy()
        
        # Filter data based on the selected period
        if period == "5d":
            stock_data_for_plot = stock_data_for_plot.tail(5)
        elif period == "10d":
            stock_data_for_plot = stock_data_for_plot.tail(10)
        elif period == "15d":
            stock_data_for_plot = stock_data_for_plot.tail(15)
        elif period == "1mo":
            stock_data_for_plot = stock_data_for_plot.tail(30)
        elif period == "2mo":
            stock_data_for_plot = stock_data_for_plot.tail(60)
        elif period == "3mo":
            stock_data_for_plot = stock_data_for_plot

        fig = go.Figure()

        for indicator in indicators_to_display:
            if indicator == "Close":
                fig.add_trace(go.Scatter(x=stock_data_for_plot.index, y=stock_data_for_plot['Close'], mode='lines', name='Close'))
            elif indicator == "MACD Histogram":
                fig.add_trace(go.Bar(x=stock_data_for_plot.index, y=stock_data_for_plot['MACD Histogram'], name='MACD Histogram'))
            else:
                fig.add_trace(go.Scatter(x=stock_data_for_plot.index, y=stock_data_for_plot[indicator], mode='lines', name=indicator))

        fig.update_layout(
            title=f"{selected_stock} Price and Indicators",
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=True
        )

        st.plotly_chart(fig)

elif submenu == "Bollinger":
    st.subheader("Bollinger")

    # Function to calculate Bollinger Bands and RSI
    @st.cache_data(ttl=3600)
    def calculate_indicators(data):
        # Calculate Bollinger Bands
        bollinger = ta.volatility.BollingerBands(data['Close'])
        data['Bollinger High'] = bollinger.bollinger_hband()
        data['Bollinger Low'] = bollinger.bollinger_lband()
        data['Bollinger Middle'] = bollinger.bollinger_mavg()
        
        # Calculate RSI
        rsi = ta.momentum.RSIIndicator(data['Close']).rsi()
        data['RSI'] = rsi
        
        return data

    # Function to fetch and process stock data
    @st.cache_data(ttl=3600)
    def get_stock_data(tickers, period):
        try:
            stock_data = {}
            for ticker in tickers:
                df = yf.download(ticker, period=period)
                if not df.empty:
                    df.interpolate(method='linear', inplace=True)
                    df = calculate_indicators(df)
                    df.dropna(inplace=True)  # Ensure we drop rows with NaN values after indicator calculation
                    stock_data[ticker] = df
            return stock_data
        except Exception as e:
            st.write(f"Error fetching data: {e}")
            return {}

    # Function to screen stocks based on Close < Bollinger Middle and RSI < 40
    def screen_stocks(stock_data):
        screened_stocks = []

        for ticker, data in stock_data.items():
            # Check condition for screening stocks
            if data['Close'].iloc[-1] < data['Bollinger Middle'].iloc[-1] and data['RSI'].iloc[-1] < 40:
                screened_stocks.append({
                    'Ticker': ticker,
                    'Close': data['Close'].iloc[-1],
                    'Bollinger Middle': data['Bollinger Middle'].iloc[-1],
                    'RSI': data['RSI'].iloc[-1]
                })

        return pd.DataFrame(screened_stocks)

    # Fetch data for the last 3 months for all tickers
    stock_data = get_stock_data(tickers, period="3mo")

    # Screen the stocks
    screened_stocks_df = screen_stocks(stock_data)

    st.subheader("Screened Stocks")
    st.write("Stocks that meet the screening criteria based on Close < Bollinger Middle and RSI < 40:")
    st.dataframe(screened_stocks_df)

    # Dropdown to select stock from screened stocks
    selected_stock = st.selectbox("Select Stock", screened_stocks_df['Ticker'].tolist())

    # Radio buttons for timeframes
    timeframe = st.radio("Select Timeframe", ["5 days", "10 days", "15 days", "1 month", "2 months", "3 months"], index=5, horizontal=True)

    # Determine the period based on the selected timeframe
    if timeframe == "5 days":
        period = "5d"
    elif timeframe == "10 days":
        period = "10d"
    elif timeframe == "15 days":
        period = "15d"
    elif timeframe == "1 month":
        period = "1mo"
    elif timeframe == "2 months":
        period = "2mo"
    else:
        period = "3mo"

    # Multi-select options for indicators to display
    indicators_to_display = st.multiselect("Select Indicators", ["Close", "Bollinger High", "Bollinger Low", "Bollinger Middle", "RSI"], default=["Close"])

    # Fetch the selected stock data
    if selected_stock:
        stock_data_for_plot = stock_data[selected_stock].copy()
        
        # Filter data based on the selected period
        if period == "5d":
            stock_data_for_plot = stock_data_for_plot.tail(5)
        elif period == "10d":
            stock_data_for_plot = stock_data_for_plot.tail(10)
        elif period == "15d":
            stock_data_for_plot = stock_data_for_plot.tail(15)
        elif period == "1mo":
            stock_data_for_plot = stock_data_for_plot.tail(30)
        elif period == "2mo":
            stock_data_for_plot = stock_data_for_plot.tail(60)
        elif period == "3mo":
            stock_data_for_plot = stock_data_for_plot

        fig = go.Figure()

        for indicator in indicators_to_display:
            if indicator == "Close":
                fig.add_trace(go.Scatter(x=stock_data_for_plot.index, y=stock_data_for_plot['Close'], mode='lines', name='Close'))
            else:
                fig.add_trace(go.Scatter(x=stock_data_for_plot.index, y=stock_data_for_plot[indicator], mode='lines', name=indicator))

        fig.update_layout(
            title=f"{selected_stock} Price and Indicators",
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=True
        )

        st.plotly_chart(fig)

elif submenu == "Strategy1":
    st.subheader("RSI < 30, Close < Lower Bollinger Band, and MACD Bullish Crossover")

    # Function to calculate indicators
    @st.cache_data(ttl=3600)
    def calculate_indicators(data):
        # Calculate Bollinger Bands
        bollinger = ta.volatility.BollingerBands(data['Close'])
        data['Bollinger High'] = bollinger.bollinger_hband()
        data['Bollinger Low'] = bollinger.bollinger_lband()
        data['Bollinger Middle'] = bollinger.bollinger_mavg()
        
        # Calculate RSI
        rsi = ta.momentum.RSIIndicator(data['Close']).rsi()
        data['RSI'] = rsi

        # Calculate MACD
        macd = ta.trend.MACD(data['Close'])
        data['MACD'] = macd.macd()
        data['MACD Signal'] = macd.macd_signal()
        data['MACD Histogram'] = macd.macd_diff()

        return data

    # Function to fetch and process stock data
    @st.cache_data(ttl=3600)
    def get_stock_data(tickers, period):
        try:
            stock_data = {}
            for ticker in tickers:
                df = yf.download(ticker, period=period)
                if not df.empty:
                    df.interpolate(method='linear', inplace=True)
                    df = calculate_indicators(df)
                    df.dropna(inplace=True)  # Ensure we drop rows with NaN values after indicator calculation
                    stock_data[ticker] = df
            return stock_data
        except Exception as e:
            st.write(f"Error fetching data: {e}")
            return {}

    # Function to screen stocks based on RSI < 30, Close < Lower Bollinger Band, and MACD crossover
    def screen_stocks(stock_data):
        screened_stocks = []

        for ticker, data in stock_data.items():
            # Ensure MACD and other indicators are available
            if 'MACD' not in data.columns or 'MACD Signal' not in data.columns or 'RSI' not in data.columns or 'Bollinger Low' not in data.columns:
                continue

            # Check condition for screening stocks
            if data['RSI'].iloc[-1] < 30 or data['Close'].iloc[-1] < data['Bollinger Low'].iloc[-1] or (data['MACD'].iloc[-1] > data['MACD Signal'].iloc[-1] and data['MACD'].iloc[-2] <= data['MACD Signal'].iloc[-2]):
                screened_stocks.append({
                    'Ticker': ticker,
                    'Close': data['Close'].iloc[-1],
                    'RSI': data['RSI'].iloc[-1],
                    'Bollinger Low': data['Bollinger Low'].iloc[-1],
                    'MACD': data['MACD'].iloc[-1],
                    'MACD Signal': data['MACD Signal'].iloc[-1],
                })

        return pd.DataFrame(screened_stocks)

    # Fetch data for the last 3 months for all tickers
    stock_data = get_stock_data(tickers, period="3mo")

    # Screen the stocks
    screened_stocks_df = screen_stocks(stock_data)

    st.subheader("Screened Stocks")
    st.write("Stocks that meet the screening criteria (RSI < 30 or Close < Lower Bollinger Band or MACD Bullish Crossover):")
    if not screened_stocks_df.empty:
        st.dataframe(screened_stocks_df)
    else:
        st.write("No stocks meet the criteria.")

    # Dropdown to select stock from screened stocks
    if not screened_stocks_df.empty:
        selected_stock = st.selectbox("Select Stock", screened_stocks_df['Ticker'].tolist())

        # Radio buttons for timeframes
        timeframe = st.radio("Select Timeframe", ["5 days", "10 days", "15 days", "1 month", "2 months", "3 months"], index=5)

        # Determine the period based on the selected timeframe
        if timeframe == "5 days":
            period = "5d"
        elif timeframe == "10 days":
            period = "10d"
        elif timeframe == "15 days":
            period = "15d"
        elif timeframe == "1 month":
            period = "1mo"
        elif timeframe == "2 months":
            period = "2mo"
        else:
            period = "3mo"

        # Multi-select options for indicators to display
        indicators_to_display = st.multiselect("Select Indicators", ["Close", "Bollinger High", "Bollinger Low", "Bollinger Middle", "RSI"], default=["Close"])

        # Fetch the selected stock data
        if selected_stock:
            stock_data_for_plot = stock_data[selected_stock].copy()
            
            # Filter data based on the selected period
            if period == "5d":
                stock_data_for_plot = stock_data_for_plot.tail(5)
            elif period == "10d":
                stock_data_for_plot = stock_data_for_plot.tail(10)
            elif period == "15d":
                stock_data_for_plot = stock_data_for_plot.tail(15)
            elif period == "1mo":
                stock_data_for_plot = stock_data_for_plot.tail(30)
            elif period == "2mo":
                stock_data_for_plot = stock_data_for_plot.tail(60)
            elif period == "3mo":
                stock_data_for_plot = stock_data_for_plot

            fig = go.Figure()

            for indicator in indicators_to_display:
                if indicator == "Close":
                    fig.add_trace(go.Scatter(x=stock_data_for_plot.index, y=stock_data_for_plot['Close'], mode='lines', name='Close'))
                else:
                    fig.add_trace(go.Scatter(x=stock_data_for_plot.index, y=stock_data_for_plot[indicator], mode='lines', name=indicator))

            fig.update_layout(
                title=f"{selected_stock} Price and Indicators",
                xaxis_title='Date',
                yaxis_title='Price',
                xaxis_rangeslider_visible=True
            )

            st.plotly_chart(fig)

elif submenu == "Strategy2":
    st.subheader("MACD Bullish Crossover, RSI Above 50, and Close Above 50-day Moving Average")

    # Function to calculate indicators
    @st.cache_data(ttl=3600)
    def calculate_indicators(data):
        # Calculate Bollinger Bands
        bollinger = ta.volatility.BollingerBands(data['Close'])
        data['Bollinger High'] = bollinger.bollinger_hband()
        data['Bollinger Low'] = bollinger.bollinger_lband()
        data['Bollinger Middle'] = bollinger.bollinger_mavg()
        
        # Calculate RSI
        rsi = ta.momentum.RSIIndicator(data['Close']).rsi()
        data['RSI'] = rsi

        # Calculate MACD
        macd = ta.trend.MACD(data['Close'])
        data['MACD'] = macd.macd()
        data['MACD Signal'] = macd.macd_signal()
        data['MACD Histogram'] = macd.macd_diff()

        # Calculate 50-day Moving Average
        data['50MA'] = data['Close'].rolling(window=50).mean()

        return data

    # Function to fetch and process stock data
    @st.cache_data(ttl=3600)
    def get_stock_data(tickers, period):
        try:
            stock_data = {}
            for ticker in tickers:
                df = yf.download(ticker, period=period)
                if not df.empty:
                    df.interpolate(method='linear', inplace=True)
                    df = calculate_indicators(df)
                    df.dropna(inplace=True)  # Ensure we drop rows with NaN values after indicator calculation
                    stock_data[ticker] = df
            return stock_data
        except Exception as e:
            st.write(f"Error fetching data: {e}")
            return {}

    # Function to screen stocks based on MACD Bullish Crossover, RSI > 50, and Close > 50MA
    def screen_stocks(stock_data):
        screened_stocks = []

        for ticker, data in stock_data.items():
            # Ensure MACD and other indicators are available
            if 'MACD' not in data.columns or 'MACD Signal' not in data.columns or 'RSI' not in data.columns or '50MA' not in data.columns:
                continue

            # Check condition for screening stocks
            if (data['MACD'].iloc[-1] > data['MACD Signal'].iloc[-1] and data['MACD'].iloc[-2] <= data['MACD Signal'].iloc[-2] and
                data['RSI'].iloc[-1] > 50 and data['Close'].iloc[-1] > data['50MA'].iloc[-1]):
                screened_stocks.append({
                    'Ticker': ticker,
                    'Close': data['Close'].iloc[-1],
                    'RSI': data['RSI'].iloc[-1],
                    'MACD': data['MACD'].iloc[-1],
                    'MACD Signal': data['MACD Signal'].iloc[-1],
                    '50MA': data['50MA'].iloc[-1]
                })

        return pd.DataFrame(screened_stocks)

    # Fetch data for the last 3 months for all tickers
    stock_data = get_stock_data(tickers, period="3mo")

    # Screen the stocks
    screened_stocks_df = screen_stocks(stock_data)

    st.subheader("Screened Stocks")
    st.write("Stocks that meet the screening criteria (MACD Bullish Crossover, RSI > 50, and Close > 50MA):")
    if not screened_stocks_df.empty:
        st.dataframe(screened_stocks_df)
    else:
        st.write("No stocks meet the criteria.")

    # Dropdown to select stock from screened stocks
    if not screened_stocks_df.empty:
        selected_stock = st.selectbox("Select Stock", screened_stocks_df['Ticker'].tolist())

        # Radio buttons for timeframes
        timeframe = st.radio("Select Timeframe", ["5 days", "10 days", "15 days", "1 month", "2 months", "3 months"], index=5)

        # Determine the period based on the selected timeframe
        if timeframe == "5 days":
            period = "5d"
        elif timeframe == "10 days":
            period = "10d"
        elif timeframe == "15 days":
            period = "15d"
        elif timeframe == "1 month":
            period = "1mo"
        elif timeframe == "2 months":
            period = "2mo"
        else:
            period = "3mo"

        # Multi-select options for indicators to display
        indicators_to_display = st.multiselect("Select Indicators", ["Close", "Bollinger High", "Bollinger Low", "Bollinger Middle", "RSI", "MACD", "MACD Signal", "50MA"], default=["Close"])

        # Fetch the selected stock data
        if selected_stock:
            stock_data_for_plot = stock_data[selected_stock].copy()
            
            # Filter data based on the selected period
            if period == "5d":
                stock_data_for_plot = stock_data_for_plot.tail(5)
            elif period == "10d":
                stock_data_for_plot = stock_data_for_plot.tail(10)
            elif period == "15d":
                stock_data_for_plot = stock_data_for_plot.tail(15)
            elif period == "1mo":
                stock_data_for_plot = stock_data_for_plot.tail(30)
            elif period == "2mo":
                stock_data_for_plot = stock_data_for_plot.tail(60)
            elif period == "3mo":
                stock_data_for_plot = stock_data_for_plot

            fig = go.Figure()

            for indicator in indicators_to_display:
                if indicator == "Close":
                    fig.add_trace(go.Scatter(x=stock_data_for_plot.index, y=stock_data_for_plot['Close'], mode='lines', name='Close'))
                elif indicator == "MACD Histogram":
                    fig.add_trace(go.Bar(x=stock_data_for_plot.index, y=stock_data_for_plot['MACD Histogram'], name='MACD Histogram'))
                else:
                    fig.add_trace(go.Scatter(x=stock_data_for_plot.index, y=stock_data_for_plot[indicator], mode='lines', name=indicator))

            fig.update_layout(
                title=f"{selected_stock} Price and Indicators",
                xaxis_title='Date',
                yaxis_title='Price',
                xaxis_rangeslider_visible=True
            )

            st.plotly_chart(fig)

elif submenu == "Strategy3":
    st.subheader("Support and Resistance Levels with RSI")

    # Function to calculate indicators
    @st.cache_data(ttl=3600)
    def calculate_indicators(data):
        # Calculate RSI
        rsi = ta.momentum.RSIIndicator(data['Close']).rsi()
        data['RSI'] = rsi

        # Calculate support and resistance levels
        data['Support'] = data['Low'].rolling(window=20, center=True).min()
        data['Resistance'] = data['High'].rolling(window=20, center=True).max()

        return data

    # Function to fetch and process stock data
    @st.cache_data(ttl=3600)
    def get_stock_data(tickers, period):
        try:
            stock_data = {}
            for ticker in tickers:
                df = yf.download(ticker, period=period)
                if not df.empty:
                    df.interpolate(method='linear', inplace=True)
                    df = calculate_indicators(df)
                    df.dropna(inplace=True)  # Ensure we drop rows with NaN values after indicator calculation
                    stock_data[ticker] = df
            return stock_data
        except Exception as e:
            st.write(f"Error fetching data: {e}")
            return {}

    # Function to screen stocks based on support/resistance levels and RSI
    def screen_stocks(stock_data):
        screened_stocks = []

        for ticker, data in stock_data.items():
            # Ensure necessary indicators are available
            if 'RSI' not in data.columns or 'Support' not in data.columns or 'Resistance' not in data.columns:
                continue

            # Check condition for screening stocks
            if data['Close'].iloc[-1] <= data['Support'].iloc[-1] or data['Close'].iloc[-1] >= data['Resistance'].iloc[-1] or data['RSI'].iloc[-1] < 30 or data['RSI'].iloc[-1] > 70:
                screened_stocks.append({
                    'Ticker': ticker,
                    'Close': data['Close'].iloc[-1],
                    'RSI': data['RSI'].iloc[-1],
                    'Support': data['Support'].iloc[-1],
                    'Resistance': data['Resistance'].iloc[-1],
                })

        return pd.DataFrame(screened_stocks)

    # Fetch data for the last 3 months for all tickers
    stock_data = get_stock_data(tickers, period="3mo")

    # Screen the stocks
    screened_stocks_df = screen_stocks(stock_data)

    st.subheader("Screened Stocks")
    st.write("Stocks that meet the screening criteria (Support/Resistance levels with RSI):")
    if not screened_stocks_df.empty:
        st.dataframe(screened_stocks_df)
    else:
        st.write("No stocks meet the criteria.")

    # Dropdown to select stock from screened stocks
    if not screened_stocks_df.empty:
        selected_stock = st.selectbox("Select Stock", screened_stocks_df['Ticker'].tolist())

        # Radio buttons for timeframes
        timeframe = st.radio("Select Timeframe", ["5 days", "10 days", "15 days", "1 month", "2 months", "3 months"], index=5)

        # Determine the period based on the selected timeframe
        if timeframe == "5 days":
            period = "5d"
        elif timeframe == "10 days":
            period = "10d"
        elif timeframe == "15 days":
            period = "15d"
        elif timeframe == "1 month":
            period = "1mo"
        elif timeframe == "2 months":
            period = "2mo"
        else:
            period = "3mo"

        # Multi-select options for indicators to display
        indicators_to_display = st.multiselect("Select Indicators", ["Close", "Support", "Resistance", "RSI"], default=["Close"])

        # Fetch the selected stock data
        if selected_stock:
            stock_data_for_plot = stock_data[selected_stock].copy()
            
            # Filter data based on the selected period
            if period == "5d":
                stock_data_for_plot = stock_data_for_plot.tail(5)
            elif period == "10d":
                stock_data_for_plot = stock_data_for_plot.tail(10)
            elif period == "15d":
                stock_data_for_plot = stock_data_for_plot.tail(15)
            elif period == "1mo":
                stock_data_for_plot = stock_data_for_plot.tail(30)
            elif period == "2mo":
                stock_data_for_plot = stock_data_for_plot.tail(60)
            elif period == "3mo":
                stock_data_for_plot = stock_data_for_plot

            fig = go.Figure()

            for indicator in indicators_to_display:
                if indicator == "Close":
                    fig.add_trace(go.Scatter(x=stock_data_for_plot.index, y=stock_data_for_plot['Close'], mode='lines', name='Close'))
                else:
                    fig.add_trace(go.Scatter(x=stock_data_for_plot.index, y=stock_data_for_plot[indicator], mode='lines', name=indicator))

            fig.update_layout(
                title=f"{selected_stock} Price and Indicators",
                xaxis_title='Date',
                yaxis_title='Price',
                xaxis_rangeslider_visible=True
            )

            st.plotly_chart(fig)

elif submenu == "Strategy4":
    st.subheader("Volume Analysis with MACD and RSI")

    # Function to calculate indicators
    @st.cache_data(ttl=3600)
    def calculate_indicators(data):
        # Calculate MACD
        macd = ta.trend.MACD(data['Close'])
        data['MACD'] = macd.macd()
        data['MACD Signal'] = macd.macd_signal()
        data['MACD Histogram'] = macd.macd_diff()
        
        # Calculate RSI
        rsi = ta.momentum.RSIIndicator(data['Close']).rsi()
        data['RSI'] = rsi

        # Calculate Volume Moving Average
        data['Volume MA'] = data['Volume'].rolling(window=20).mean()
        data['Volume Spike'] = data['Volume'] > 1.5 * data['Volume MA']

        return data

    # Function to fetch and process stock data
    @st.cache_data(ttl=3600)
    def get_stock_data(tickers, period):
        try:
            stock_data = {}
            for ticker in tickers:
                df = yf.download(ticker, period=period)
                if not df.empty:
                    df.interpolate(method='linear', inplace=True)
                    df = calculate_indicators(df)
                    df.dropna(inplace=True)  # Ensure we drop rows with NaN values after indicator calculation
                    stock_data[ticker] = df
            return stock_data
        except Exception as e:
            st.write(f"Error fetching data: {e}")
            return {}

    # Function to screen stocks based on Volume Analysis
    def screen_stocks(stock_data):
        screened_stocks = []

        for ticker, data in stock_data.items():
            # Ensure necessary indicators are available
            if 'Volume Spike' not in data.columns or 'MACD' not in data.columns or 'MACD Signal' not in data.columns or 'RSI' not in data.columns:
                continue

            # Check condition for screening stocks
            if data['Volume Spike'].iloc[-1] and (data['MACD'].iloc[-1] > data['MACD Signal'].iloc[-1] or data['RSI'].iloc[-1] < 30 or data['RSI'].iloc[-1] > 70):
                screened_stocks.append({
                    'Ticker': ticker,
                    'Close': data['Close'].iloc[-1],
                    'Volume': data['Volume'].iloc[-1],
                    'Volume MA': data['Volume MA'].iloc[-1],
                    'MACD': data['MACD'].iloc[-1],
                    'MACD Signal': data['MACD Signal'].iloc[-1],
                    'RSI': data['RSI'].iloc[-1]
                })

        return pd.DataFrame(screened_stocks)

    # Fetch data for the last 3 months for all tickers
    stock_data = get_stock_data(tickers, period="3mo")

    # Screen the stocks based on Volume Analysis strategy
    screened_stocks_df = screen_stocks(stock_data)

    st.subheader("Screened Stocks (Volume Analysis)")
    st.write("Stocks that meet the screening criteria (Volume Spike, MACD, and RSI):")
    if not screened_stocks_df.empty:
        st.dataframe(screened_stocks_df)
    else:
        st.write("No stocks meet the criteria.")

    # Dropdown to select stock from screened stocks
    if not screened_stocks_df.empty:
        selected_stock = st.selectbox("Select Stock", screened_stocks_df['Ticker'].tolist())

        # Radio buttons for timeframes
        timeframe = st.radio("Select Timeframe", ["5 days", "10 days", "15 days", "1 month", "2 months", "3 months"], index=5)

        # Determine the period based on the selected timeframe
        if timeframe == "5 days":
            period = "5d"
        elif timeframe == "10 days":
            period = "10d"
        elif timeframe == "15 days":
            period = "15d"
        elif timeframe == "1 month":
            period = "1mo"
        elif timeframe == "2 months":
            period = "2mo"
        else:
            period = "3mo"

        # Multi-select options for indicators to display
        indicators_to_display = st.multiselect("Select Indicators", ["Close", "Volume", "Volume MA", "MACD", "MACD Signal", "RSI"], default=["Close"])

        # Fetch the selected stock data
        if selected_stock:
            stock_data_for_plot = stock_data[selected_stock].copy()
            
            # Filter data based on the selected period
            if period == "5d":
                stock_data_for_plot = stock_data_for_plot.tail(5)
            elif period == "10d":
                stock_data_for_plot = stock_data_for_plot.tail(10)
            elif period == "15d":
                stock_data_for_plot = stock_data_for_plot.tail(15)
            elif period == "1mo":
                stock_data_for_plot = stock_data_for_plot.tail(30)
            elif period == "2mo":
                stock_data_for_plot = stock_data_for_plot.tail(60)
            elif period == "3mo":
                stock_data_for_plot = stock_data_for_plot

            fig = go.Figure()

            for indicator in indicators_to_display:
                if indicator == "Close":
                    fig.add_trace(go.Scatter(x=stock_data_for_plot.index, y=stock_data_for_plot['Close'], mode='lines', name='Close'))
                elif indicator == "MACD Histogram":
                    fig.add_trace(go.Bar(x=stock_data_for_plot.index, y=stock_data_for_plot['MACD Histogram'], name='MACD Histogram'))
                else:
                    fig.add_trace(go.Scatter(x=stock_data_for_plot.index, y=stock_data_for_plot[indicator], mode='lines', name=indicator))

            fig.update_layout(
                title=f"{selected_stock} Price and Indicators",
                xaxis_title='Date',
                yaxis_title='Price',
                xaxis_rangeslider_visible=True
            )

            st.plotly_chart(fig)

else:
    st.subheader("Resistance")
    # Define and process Resistance conditions here
    pass
