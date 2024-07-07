import streamlit as st
import yfinance as yf
import ta
import pandas as pd

st.sidebar.subheader("Strategies")

submenu = st.sidebar.selectbox("Select Strategy", ["MACD", "Moving Average", "RSI", "Bollinger Bands", "Stochastic Oscillator",
    "Ichimoku Cloud", "ADX", "Fibonacci Retracement", "Parabolic SAR",
    "Candlestick Patterns", "Pivot Points"])
# List of stock tickers
largecap_tickers = [
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
    "CHAMBLFERT.BO", "COALINDIA.NS", "COFORGE.BO", "COLPAL.NS", "CONCORDBIO.BO", "COROMANDEL.BO", "CREDITACC.BO"
]
midcap_tickers = [
    "PNCINFRA.NS", "INDIASHLTR.NS", "RAYMOND.NS", "KAMAHOLD.BO", "BENGALASM.BO", "CHOICEIN.NS",
    "GRAVITA.NS", "HGINFRA.NS", "JKPAPER.NS", "MTARTECH.NS", "HAPPSTMNDS.NS", "SARDAEN.NS",
    "WELENT.NS", "LTFOODS.NS", "GESHIP.NS", "SHRIPISTON.NS", "SHAREINDIA.NS", "CYIENTDLM.NS", "VTL.NS",
    "EASEMYTRIP.NS", "LLOYDSME.NS", "ROUTE.NS", "VAIBHAVGBL.NS", "GOKEX.NS", "USHAMART.NS", "EIDPARRY.NS",
    "KIRLOSBROS.NS", "MANINFRA.NS", "CMSINFO.NS", "RALLIS.NS", "GHCL.NS", "NEULANDLAB.NS", "SPLPETRO.NS",
    "MARKSANS.NS", "NAVINFLUOR.NS", "ELECON.NS", "TANLA.NS", "KFINTECH.NS", "TIPSINDLTD.NS", "ACI.NS",
    "SURYAROSNI.NS", "GPIL.NS", "GMDCLTD.NS", "MAHSEAMLES.NS", "TDPOWERSYS.NS", "TECHNOE.NS", "JLHL.NS"
]

# Dropdown for selecting ticker category
ticker_category = st.sidebar.selectbox("Select Ticker Category", ["Large Cap", "Mid Cap"])

# Set tickers based on selected category
if ticker_category == "Large Cap":
    tickers = largecap_tickers
else:
    tickers = midcap_tickers

# Function to calculate MACD and related values
def calculate_macd(data, slow=26, fast=12, signal=9):
    data['EMA_fast'] = data['Close'].ewm(span=fast, min_periods=fast).mean()
    data['EMA_slow'] = data['Close'].ewm(span=slow, min_periods=slow).mean()
    data['MACD'] = data['EMA_fast'] - data['EMA_slow']
    data['MACD_signal'] = data['MACD'].ewm(span=signal, min_periods=signal).mean()
    data['MACD_histogram'] = data['MACD'] - data['MACD_signal']
    return data

# Function to calculate RSI
def calculate_rsi(data, period=14):
    rsi_indicator = ta.momentum.RSIIndicator(close=data['Close'], window=period)
    data['RSI'] = rsi_indicator.rsi()
    return data

# Function to calculate ADX and related values
def calculate_adx(data, period=14):
    adx_indicator = ta.trend.ADXIndicator(high=data['High'], low=data['Low'], close=data['Close'], window=period)
    data['ADX'] = adx_indicator.adx()
    data['+DI'] = adx_indicator.adx_pos()
    data['-DI'] = adx_indicator.adx_neg()
    return data

# Function to check MACD < MACD signal
def check_macd_signal(data):
    return data['MACD'].iloc[-1] < data['MACD_signal'].iloc[-1]

# Function to check the second criteria
def check_negative_histogram_and_price(data):
    histogram_increasing = (data['MACD_histogram'].iloc[-3] <= data['MACD_histogram'].iloc[-2] <= data['MACD_histogram'].iloc[-1])
    histogram_negative = data['MACD_histogram'].iloc[-1] < 0
    price_increasing = (data['Close'].iloc[-1] >= data['Close'].iloc[-2] >= data['Close'].iloc[-3] >= data['Close'].iloc[-4])
    return histogram_increasing, histogram_negative, price_increasing

# Function to fetch and process stock data
@st.cache_data(ttl=3600)
def get_stock_data(ticker_symbols, period, interval):
    try:
        stock_data = {}
        progress_bar = st.progress(0)
        for idx, ticker_symbol in enumerate(ticker_symbols):
            df = yf.download(ticker_symbol, period=period, interval=interval)
            if not df.empty:
                df.interpolate(method='linear', inplace=True)
                df = calculate_indicators(df)
                df.dropna(inplace=True)
                stock_data[ticker_symbol] = df
            progress_bar.progress((idx + 1) / len(ticker_symbols))
        return stock_data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return {}

# Function to calculate essential technical indicators
@st.cache_data(ttl=3600)
def calculate_indicators(df):
    # Calculate Moving Averages
    df['20_MA'] = ta.trend.WMAIndicator(close=df['Close'], window=20).wma()
    df['50_MA'] = ta.trend.WMAIndicator(close=df['Close'], window=50).wma()

    # Calculate MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Histogram'] = macd.macd_diff()

    # Calculate RSI
    rsi = ta.momentum.RSIIndicator(df['Close'])
    df['RSI'] = rsi.rsi()

    # Calculate Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['Bollinger_High'] = bollinger.bollinger_hband()
    df['Bollinger_Low'] = bollinger.bollinger_lband()
    df['Bollinger_Middle'] = bollinger.bollinger_mavg()

    # Calculate Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['%K'] = stoch.stoch()
    df['%D'] = stoch.stoch_signal()

    # Calculate Ichimoku Cloud
    ichimoku = ta.trend.IchimokuIndicator(df['High'], df['Low'])
    df['Tenkan_Sen'] = ichimoku.ichimoku_conversion_line()
    df['Kijun_Sen'] = ichimoku.ichimoku_base_line()
    df['Senkou_Span_A'] = ichimoku.ichimoku_a()
    df['Senkou_Span_B'] = ichimoku.ichimoku_b()
    # Chikou Span (Lagging Span) is the close price shifted by 26 periods
    df['Chikou_Span'] = df['Close'].shift(-26)

    # Calculate Parabolic SAR
    psar = ta.trend.PSARIndicator(df['High'], df['Low'], df['Close'])
    df['Parabolic_SAR'] = psar.psar()

    return df

# Calculate exponential moving averages
def calculate_ema(data, short_window, long_window):
    data['Short_EMA'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['Long_EMA'] = data['Close'].ewm(span=long_window, adjust=False).mean()
    return data

# Check if 50-day EMA crossed above 200-day EMA in the last 5 days
def check_moving_average_crossover(data):
    recent_data = data[-5:]
    for i in range(1, len(recent_data)):
        if (recent_data['Short_EMA'].iloc[i] > recent_data['Long_EMA'].iloc[i] and
            recent_data['Short_EMA'].iloc[i-1] <= recent_data['Long_EMA'].iloc[i-1]):
            return True
    return False

# Check if price crossed below Bollinger Low in the last 5 days
def check_bollinger_low_cross(data):
    recent_data = data[-5:]
    for i in range(1, len(recent_data)):
        if (recent_data['Close'].iloc[i] < recent_data['Bollinger_Low'].iloc[i] and
            recent_data['Close'].iloc[i-1] >= recent_data['Bollinger_Low'].iloc[i-1]):
            return True
    return False

# Check if RSI is below 30
def check_rsi(data):
    return data['RSI'].iloc[-1] < 30

# Check if Stochastic %K crossed above %D from below 20 in the last 5 days
def check_stochastic(data):
    recent_data = data[-5:]
    for i in range(1, len(recent_data)):
        if (recent_data['%K'].iloc[i] > recent_data['%D'].iloc[i] and
            recent_data['%K'].iloc[i-1] <= recent_data['%D'].iloc[i-1] and
            recent_data['%K'].iloc[i] < 20):
            return True
    return False

# Check if price is above Ichimoku Cloud
def check_ichimoku(data):
    return data['Close'].iloc[-1] > data['Senkou_Span_A'].iloc[-1] and data['Close'].iloc[-1] > data['Senkou_Span_B'].iloc[-1]

# Check if +DI crossed above -DI in the last 5 days
def check_adx(data):
    recent_data = data[-5:]
    for i in range(1, len(recent_data)):
        if (recent_data['+DI'].iloc[i] > recent_data['-DI'].iloc[i] and
            recent_data['+DI'].iloc[i-1] <= recent_data['-DI'].iloc[i-1]):
            return True
    return False

# Check if price crossed above Parabolic SAR
def check_parabolic_sar(data):
    recent_data = data[-5:]
    for i in range(1, len(recent_data)):
        if (recent_data['Close'].iloc[i] > recent_data['Parabolic_SAR'].iloc[i] and
            recent_data['Close'].iloc[i-1] <= recent_data['Parabolic_SAR'].iloc[i-1]):
            return True
    return False

# Calculate pivot points and their respective support and resistance levels
def calculate_pivot_points(data, period='daily'):
    if period == 'daily':
        data['Pivot'] = (data['High'].shift(1) + data['Low'].shift(1) + data['Close'].shift(1)) / 3
    elif period == 'weekly':
        data['Pivot'] = (data['High'].resample('W').max().shift(1) + data['Low'].resample('W').min().shift(1) + data['Close'].resample('W').last().shift(1)) / 3
    elif period == 'monthly':
        data['Pivot'] = (data['High'].resample('M').max().shift(1) + data['Low'].resample('M').min().shift(1) + data['Close'].resample('M').last().shift(1)) / 3

    data['R1'] = 2 * data['Pivot'] - data['Low']
    data['S1'] = 2 * data['Pivot'] - data['High']
    data['R2'] = data['Pivot'] + (data['High'] - data['Low'])
    data['S2'] = data['Pivot'] - (data['High'] - data['Low'])
    data['R3'] = data['Pivot'] + 2 * (data['High'] - data['Low'])
    data['S3'] = data['Pivot'] - 2 * (data['High'] - data['Low'])
    
    return data

# Check for trading breakouts above resistance levels
def check_pivot_breakouts(data):
    recent_data = data[-5:]
    for i in range(1, len(recent_data)):
        if (recent_data['Close'].iloc[i] > recent_data['R1'].iloc[i] and recent_data['Close'].iloc[i-1] <= recent_data['R1'].iloc[i-1]) or \
           (recent_data['Close'].iloc[i] > recent_data['R2'].iloc[i] and recent_data['Close'].iloc[i-1] <= recent_data['R2'].iloc[i-1]) or \
           (recent_data['Close'].iloc[i] > recent_data['R3'].iloc[i] and recent_data['Close'].iloc[i-1] <= recent_data['R3'].iloc[i-1]):
            return True
    return False

macd_signal_list = []
negative_histogram_tickers = []
moving_average_tickers = []
bollinger_low_cross_tickers = []
rsi_tickers = []
stochastic_tickers = []
ichimoku_tickers = []
adx_tickers = []
parabolic_sar_tickers = []
pivot_point_tickers = []

progress_bar = st.progress(0)
progress_step = 1 / len(tickers)

for i, ticker in enumerate(tickers):
    progress_bar.progress((i + 1) * progress_step)
    data = yf.download(ticker, period="1y", interval="1d")
    if data.empty:
        continue
    data = calculate_indicators(data)
    if submenu == "MACD":
        data = calculate_macd(data)
        data = calculate_rsi(data)
        data = calculate_adx(data)
        if check_macd_signal(data):
            macd_signal_list.append(ticker)
        histogram_increasing, histogram_negative, price_increasing = check_negative_histogram_and_price(data)
        if histogram_increasing and histogram_negative and price_increasing:
            negative_histogram_tickers.append(ticker)
    elif submenu == "Moving Average":
        data = calculate_ema(data, short_window=50, long_window=200)
        if check_moving_average_crossover(data):
            moving_average_tickers.append(ticker)
    elif submenu == "Bollinger Bands":
        if check_bollinger_low_cross(data):
            bollinger_low_cross_tickers.append(ticker)
    elif submenu == "RSI":
        if check_rsi(data):
            rsi_tickers.append(ticker)
    elif submenu == "Stochastic Oscillator":
        if check_stochastic(data):
            stochastic_tickers.append(ticker)
    elif submenu == "Ichimoku Cloud":
        if check_ichimoku(data):
            ichimoku_tickers.append(ticker)
    elif submenu == "ADX":
        data = calculate_adx(data)
        if check_adx(data):
            adx_tickers.append(ticker)
    elif submenu == "Parabolic SAR":
        if check_parabolic_sar(data):
            parabolic_sar_tickers.append(ticker)
    elif submenu == "Pivot Points":
        data = calculate_pivot_points(data, period='daily')
        if check_pivot_breakouts(data):
            pivot_point_tickers.append(ticker)

# Fetch latest data and indicators for the selected stocks
def fetch_latest_data(tickers):
    technical_data = []
    for ticker in tickers:
        data = yf.download(ticker, period='1y')
        if data.empty:
            continue
        data['5_day_EMA'] = ta.trend.ema_indicator(data['Close'], window=5)
        data['15_day_EMA'] = ta.trend.ema_indicator(data['Close'], window=15)
        data['MACD'] = ta.trend.macd(data['Close'])
        data['MACD_Hist'] = ta.trend.macd_diff(data['Close'])
        data['RSI'] = ta.momentum.rsi(data['Close'])
        data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'])
        data['Bollinger_High'] = ta.volatility.bollinger_hband(data['Close'])
        data['Bollinger_Low'] = ta.volatility.bollinger_lband(data['Close'])
        data['20_day_vol_MA'] = data['Volume'].rolling(window=20).mean()

        latest_data = data.iloc[-1]
        technical_data.append({
            'Ticker': ticker,
            'Close': latest_data['Close'],
            '5_day_EMA': latest_data['5_day_EMA'],
            '15_day_EMA': latest_data['15_day_EMA'],
            'MACD': latest_data['MACD'],
            'MACD_Hist': latest_data['MACD_Hist'],
            'RSI': latest_data['RSI'],
            'ADX': latest_data['ADX'],
            'Bollinger_High': latest_data['Bollinger_High'],
            'Bollinger_Low': latest_data['Bollinger_Low'],
            'Volume': latest_data['Volume'],
            '20_day_vol_MA': latest_data['20_day_vol_MA']
        })
    return pd.DataFrame(technical_data)

# Generate dataframes for the selected strategies
df_macd_signal = fetch_latest_data(macd_signal_list)
df_negative_histogram = fetch_latest_data(negative_histogram_tickers)
df_moving_average_signal = fetch_latest_data(moving_average_tickers)
df_bollinger_low_cross_signal = fetch_latest_data(bollinger_low_cross_tickers)
df_rsi_signal = fetch_latest_data(rsi_tickers)
df_stochastic_signal = fetch_latest_data(stochastic_tickers)
df_ichimoku_signal = fetch_latest_data(ichimoku_tickers)
df_adx_signal = fetch_latest_data(adx_tickers)
df_parabolic_sar_signal = fetch_latest_data(parabolic_sar_tickers)
df_pivot_point_signal = fetch_latest_data(pivot_point_tickers)

st.title("Stock Analysis Based on Selected Strategy")

if submenu == "Moving Average":
    st.write("Stocks with 50-day EMA crossing above 200-day EMA in the last 5 days:")
    st.dataframe(df_moving_average_signal)

elif submenu == "MACD":
    st.write("Stocks with Negative MACD Histogram Increasing and Price Increasing for 3 Consecutive Days:")
    st.dataframe(df_negative_histogram)

elif submenu == "Bollinger Bands":
    st.write("Stocks with price crossing below Bollinger Low in the last 5 days:")
    st.dataframe(df_bollinger_low_cross_signal)

elif submenu == "RSI":
    st.write("Stocks with RSI below 30:")
    st.dataframe(df_rsi_signal)

elif submenu == "Stochastic Oscillator":
    st.write("Stocks with %K crossing above %D from below 20 in the last 5 days:")
    st.dataframe(df_stochastic_signal)

elif submenu == "Ichimoku Cloud":
    st.write("Stocks with price above Ichimoku Cloud:")
    st.dataframe(df_ichimoku_signal)

elif submenu == "ADX":
    st.write("Stocks with +DI crossing above -DI in the last 5 days:")
    st.dataframe(df_adx_signal)

elif submenu == "Parabolic SAR":
    st.write("Stocks with price crossing above Parabolic SAR in the last 5 days:")
    st.dataframe(df_parabolic_sar_signal)

elif submenu == "Pivot Points":
    st.write("Stocks with price breaking above resistance levels in the last 5 days:")
    st.dataframe(df_pivot_point_signal)

elif submenu == "Fibonacci Retracement":
    st.subheader("Fibonacci Retracement Strategies")

    pass

elif submenu == "Candlestick Patterns":
    st.subheader("Candlestick Patterns")

    pass


