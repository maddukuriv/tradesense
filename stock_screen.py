import streamlit as st
from datetime import date
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import ta
import pandas_ta as pta
from scipy.stats import linregress
from scipy.fftpack import fft, ifft
from scipy.signal import hilbert
import pywt

st.sidebar.subheader("Screens")
submenu = st.sidebar.radio("Select Option", ["LargeCap-1", "LargeCap-2", "LargeCap-3", "MidCap", "SmallCap"])

# Define ticker symbols for different market caps
largecap3_tickers = ["ITC.NS", "JBCHEPHARM.BO", "JWL.BO", "JYOTHYLAB.BO", "KPRMILL.NS", "KAJARIACER.NS", "KEI.BO",
                     "LTTS.NS", "LTIM.NS", "MANKIND.NS", "MARICO.NS", "METROBRAND.BO", "MOTILALOFS.NS", "MPHASIS.NS",
                     "MUTHOOTFIN.NS", "NH.NS", "NAVINFLUOR.NS", "NAM-INDIA.BO", "NMDC.NS", "OFSS.NS", "PGHH.NS",
                     "PIIND.NS", "PAGEIND.NS", "PERSISTENT.NS", "PETRONET.NS", "PFIZER.NS", "PIDILITIND.NS",
                     "POLYMED.NS", "POLYCAB.NS", "RRKABEL.NS", "RVNL.NS", "RATNAMANI.NS", "RITES.NS", "SANOFI.NS",
                     "SCHAEFFLER.NS", "SKFINDIA.NS", "SOLARINDS.NS", "SONACOMS.NS", "SUMICHEM.NS", "SUNTV.NS",
                     "SUNDRMFAST.NS", "SUPREMEIND.BO", "TATAELXSI.NS", "TATATECH.NS", "TCS.NS", "TECHM.NS", "TIMKEN.NS",
                     "TITAN.NS", "TRITURBINE.NS", "TIINDIA.NS", "UNITDSPR.BO", "VGUARD.NS", "MANYAVAR.NS",
                     "VINATIORGA.NS", "WIPRO.NS", "ZYDUSLIFE.NS"]
largecap2_tickers = ["CUMMINSIND.NS", "CYIENT.NS", "DATAPATTNS.NS", "DEEPAKNTR.NS", "DIVISLAB.NS", "LALPATHLAB.NS",
                     "RDY", "ELGIEQUIP.NS", "EMAMILTD.NS", "FIVESTAR.BO", "GRINFRA.NS", "GILLETTE.NS", "GLAXO.NS",
                     "GODFRYPHLP.NS", "GRINDWELL.NS", "HAVELLS.NS", "HCLTECH.NS", "HAL.BO", "HONAUT.BO", "IRCTC.NS",
                     "ISEC.BO", "INFY.NS", "IPCALAB.BO"]
largecap1_tickers = ["ABBOTINDIA.NS", "ADANIPOWER.NS", "AFFLE.BO", "AIAENG.BO", "AJANTPHARM.BO", "APLLTD.BO",
                     "ALKEM.BO", "ARE&M.NS", "ANANDRATHI.BO", "APARINDS.BO", "ASIANPAINT.NS", "ASTRAL.NS",
                     "ASTRAZEN.NS", "BAJFINANCE.NS", "BASF.NS", "BAYERCROP.BO", "BERGEPAINT.BO", "BDL.NS", "BEL.NS",
                     "BSOFT.BO", "CDSL.NS", "CAMS.NS", "CARBORUNIV.BO", "CASTROLIND.NS", "CHAMBLFERT.BO",
                     "COALINDIA.NS", "COFORGE.BO", "COLPAL.NS", "CONCORDBIO.BO", "COROMANDEL.BO", "CREDITACC.BO"]
smallcap_tickers = ["TAPARIA.BO", "LKPFIN.BO", "EQUITAS.NS"]
midcap_tickers = ["PNCINFRA.NS", "INDIASHLTR.NS", "RAYMOND.NS", "KAMAHOLD.BO", "BENGALASM.BO", "CHOICEIN.NS",
                  "GRAVITA.NS", "HGINFRA.NS", "JKPAPER.NS", "MTARTECH.NS", "HAPPSTMNDS.NS", "SARDAEN.NS", "WELENT.NS",
                  "LTFOODS.NS", "GESHIP.NS", "SHRIPISTON.NS", "SHAREINDIA.NS", "CYIENTDLM.NS", "VTL.NS",
                  "EASEMYTRIP.NS", "LLOYDSME.NS", "ROUTE.NS", "VAIBHAVGBL.NS", "GOKEX.NS", "USHAMART.NS", "EIDPARRY.NS",
                  "KIRLOSBROS.NS", "MANINFRA.NS", "CMSINFO.NS", "RALLIS.NS", "GHCL.NS", "NEULANDLAB.NS", "SPLPETRO.NS",
                  "MARKSANS.NS", "NAVINFLUOR.NS", "ELECON.NS", "TANLA.NS", "KFINTECH.NS", "TIPSINDLTD.NS", "ACI.NS",
                  "SURYAROSNI.NS", "GPIL.NS", "GMDCLTD.NS", "MAHSEAMLES.NS", "TDPOWERSYS.NS", "TECHNOE.NS", "JLHL.NS"]

# Function to fetch and process stock data
def get_stock_data(ticker_symbols, start_date, end_date):
    try:
        stock_data = {}
        for ticker_symbol in ticker_symbols:
            df = yf.download(ticker_symbol, start=start_date, end=end_date)
            print(f"Downloaded data for {ticker_symbol}: Shape = {df.shape}")
            df.interpolate(method='linear', inplace=True)
            df = calculate_indicators(df)
            df.dropna(inplace=True)
            print(f"Processed data for {ticker_symbol}: Shape = {df.shape}")
            stock_data[ticker_symbol] = df
        combined_df = pd.concat(stock_data.values(), axis=1)
        combined_df.columns = ['_'.join([ticker, col]).strip() for ticker, df in stock_data.items() for col in
                               df.columns]
        return combined_df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

# Function to calculate technical indicators
def calculate_indicators(df):
    # Calculate Moving Averages
    df['5_MA'] = df['Close'].rolling(window=5).mean()
    df['20_MA'] = df['Close'].rolling(window=20).mean()
    df['50_MA'] = df['Close'].rolling(window=50).mean()

    # Calculate MACD
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

    # Calculate ADX
    df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()

    # Calculate Parabolic SAR
    psar = pta.psar(df['High'], df['Low'], df['Close'])
    df['Parabolic_SAR'] = psar['PSARl_0.02_0.2']

    # Calculate RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()

    # Calculate Volume Moving Average (20 days)
    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()

    # Calculate Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['Bollinger_High'] = bollinger.bollinger_hband()
    df['Bollinger_Low'] = bollinger.bollinger_lband()
    df['Bollinger_Middle'] = bollinger.bollinger_mavg()

    return df

# Function to query the stocks
def query_stocks(df, conditions, tickers):
    results = []
    for ticker in tickers:
        condition_met = True
        for condition in conditions:
            col1, op, col2 = condition
            col1 = f"{ticker}_{col1}"
            col2 = f"{ticker}_{col2}"
            if col1 not in df.columns or col2 not in df.columns:
                condition_met = False
                break
            if op == '>':
                if not (df[col1] > df[col2]).iloc[-1]:
                    condition_met = False
                    break
            elif op == '<':
                if not (df[col1] < df[col2]).iloc[-1]:
                    condition_met = False
                    break
            elif op == '>=':
                if not (df[col1] >= df[col2]).iloc[-1]:
                    condition_met = False
                    break
            elif op == '<=':
                if not (df[col1] <= df[col2]).iloc[-1]:
                    condition_met = False
                    break
        if condition_met:
            row = {
                'Ticker': ticker,
                'MACD': df[f"{ticker}_MACD"].iloc[-1],
                'MACD_Signal': df[f"{ticker}_MACD_Signal"].iloc[-1],
                'RSI': df[f"{ticker}_RSI"].iloc[-1],
                'ADX': df[f"{ticker}_ADX"].iloc[-1],
                'Close': df[f"{ticker}_Close"].iloc[-1],
                '5_MA': df[f"{ticker}_5_MA"].iloc[-1],
                '20_MA': df[f"{ticker}_20_MA"].iloc[-1],
                'Bollinger_High': df[f"{ticker}_Bollinger_High"].iloc[-1],
                'Bollinger_Low': df[f"{ticker}_Bollinger_Low"].iloc[-1],
                'Bollinger_Middle': df[f"{ticker}_Bollinger_Middle"].iloc[-1],
                'Parabolic_SAR': df[f"{ticker}_Parabolic_SAR"].iloc[-1],
                'Volume': df[f"{ticker}_Volume"].iloc[-1],
                'Volume_MA_20': df[f"{ticker}_Volume_MA_20"].iloc[-1]
            }
            results.append(row)
    return pd.DataFrame(results)

# Create two columns
col1, col2 = st.columns(2)

# Set up the start and end date inputs
with col1:
    START = st.date_input('Start Date', pd.to_datetime("2015-01-01"))

with col2:
    END = st.date_input('End Date', pd.to_datetime("today"))

if submenu == "LargeCap-1":
    st.header("LargeCap-1")
    tickers = largecap1_tickers

if submenu == "LargeCap-2":
    st.header("LargeCap-2")
    tickers = largecap2_tickers

if submenu == "LargeCap-3":
    st.header("LargeCap-3")
    tickers = largecap3_tickers

if submenu == "MidCap":
    st.header("MidCap")
    tickers = midcap_tickers
if submenu == "SmallCap":
    st.header("SmallCap")
    tickers = smallcap_tickers

# Fetch data and calculate indicators for each stock
stock_data = get_stock_data(tickers, START, END)

# Define first set of conditions
first_conditions = [
    ('Volume', '>', 'Volume_MA_20'),
    ('MACD', '>', 'MACD_Signal')
]

# Query stocks based on the first set of conditions
first_query_df = query_stocks(stock_data, first_conditions, tickers)

# Display the final results

# st.dataframe(first_query_df.round(2))
# Generate insights
second_query_df = first_query_df[
    (first_query_df['RSI'] < 70) & (first_query_df['RSI'] > 55) & (first_query_df['ADX'] > 20) & (
                first_query_df['MACD'] > 0)]
st.write("Stocks in an uptrend with high volume:")
st.dataframe(second_query_df)

# Dropdown for stock selection
st.subheader("Analysis:")

# Create two columns
col1, col2 = st.columns(2)
# Dropdown for analysis type
with col1:
    selected_stock = st.selectbox("Select Stock", second_query_df['Ticker'].tolist())
with col2:
    analysis_type = st.selectbox("Select Analysis Type",["Trend Analysis", "Volume Analysis", "Support & Resistance Levels"])

# If a stock is selected, plot its data with the selected indicators
if selected_stock:
    # Load data
    def load_data(ticker):
        df = yf.download(ticker, START, END)
        df.reset_index(inplace=True)
        return df

    # Handle null values
    def interpolate_dataframe(df):
        if df.isnull().values.any():
            df = df.interpolate()
        return df

    # Load data for the given ticker
    df = load_data(selected_stock)

    if df.empty:
        st.write("No data available for the provided ticker.")
    else:
        df = interpolate_dataframe(df)

        # Ensure enough data points for the calculations
        if len(df) > 200:  # 200 is the maximum window size used in calculations
            # Calculate Moving Averages
            df['15_MA'] = df['Close'].rolling(window=15).mean()
            df['20_MA'] = df['Close'].rolling(window=20).mean()
            df['50_MA'] = df['Close'].rolling(window=50).mean()
            df['200_MA'] = df['Close'].rolling(window=200).mean()

            # Calculate MACD
            df['MACD'] = ta.trend.MACD(df['Close']).macd()
            df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

            # Calculate ADX
            df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()

            # Calculate Parabolic SAR using 'pandas_ta' library
            psar = pta.psar(df['High'], df['Low'], df['Close'])
            df['Parabolic_SAR'] = psar['PSARl_0.02_0.2']

            # Calculate RSI
            df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()

            # Calculate Volume Moving Average (20 days)
            df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()

            # Calculate On-Balance Volume (OBV)
            df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()

            # Calculate Volume Oscillator (20-day EMA and 50-day EMA)
            df['Volume_EMA_20'] = ta.trend.EMAIndicator(df['Volume'], window=20).ema_indicator()
            df['Volume_EMA_50'] = ta.trend.EMAIndicator(df['Volume'], window=50).ema_indicator()
            df['Volume_Oscillator'] = df['Volume_EMA_20'] - df['Volume_EMA_50']

            # Calculate Chaikin Money Flow (20 days)
            df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'],
                                                            df['Volume']).chaikin_money_flow()

            # Identify Horizontal Support and Resistance
            def find_support_resistance(df, window=20):
                df['Support'] = df['Low'].rolling(window, center=True).min()
                df['Resistance'] = df['High'].rolling(window, center=True).max()
                return df

            df = find_support_resistance(df)

            # Draw Trendlines
            def calculate_trendline(df, kind='support'):
                if kind == 'support':
                    prices = df['Low']
                elif kind == 'resistance':
                    prices = df['High']
                else:
                    raise ValueError("kind must be either 'support' or 'resistance'")

                indices = np.arange(len(prices))
                slope, intercept, _, _, _ = linregress(indices, prices)
                trendline = slope * indices + intercept
                return trendline

            df['Support_Trendline'] = calculate_trendline(df, kind='support')
            df['Resistance_Trendline'] = calculate_trendline(df, kind='resistance')

            # Calculate Fibonacci Retracement Levels
            def fibonacci_retracement_levels(high, low):
                diff = high - low
                levels = {
                    'Level_0': high,
                    'Level_0.236': high - 0.236 * diff,
                    'Level_0.382': high - 0.382 * diff,
                    'Level_0.5': high - 0.5 * diff,
                    'Level_0.618': high - 0.618 * diff,
                    'Level_1': low
                }
                return levels

            recent_high = df['High'].max()
            recent_low = df['Low'].min()
            fib_levels = fibonacci_retracement_levels(recent_high, recent_low)

            # Calculate Pivot Points
            def pivot_points(df):
                df['Pivot'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
                df['R1'] = 2 * df['Pivot'] - df['Low'].shift(1)
                df['S1'] = 2 * df['Pivot'] - df['High'].shift(1)
                df['R2'] = df['Pivot'] + (df['High'].shift(1) - df['Low'].shift(1))
                df['S2'] = df['Pivot'] - (df['High'].shift(1) - df['Low'].shift(1))
                return df

            df = pivot_points(df)

            # Calculate Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['Close'])
            df['Bollinger_High'] = bollinger.bollinger_hband()
            df['Bollinger_Low'] = bollinger.bollinger_lband()
            df['Bollinger_Middle'] = bollinger.bollinger_mavg()  # Middle band is typically the SMA

            # Generate buy/sell signals using advanced methods
            def advanced_signals(data):
                prices = data['Close'].values

                # Fourier Transform
                def apply_fft(prices):
                    N = len(prices)
                    T = 1.0  # Assuming daily data, T=1 day
                    yf = fft(prices)
                    xf = np.fft.fftfreq(N, T)[:N//2]
                    return xf, 2.0/N * np.abs(yf[0:N//2])

                # Apply FFT
                frequencies, magnitudes = apply_fft(prices)

                # Inverse FFT for noise reduction (optional)
                def inverse_fft(yf, threshold=0.1):
                    yf_filtered = yf.copy()
                    yf_filtered[np.abs(yf_filtered) < threshold] = 0
                    return ifft(yf_filtered)

                yf_transformed = fft(prices)
                filtered_signal = inverse_fft(yf_transformed)

                # Wavelet Transform
                def apply_wavelet(prices, wavelet='db4', level=4):
                    coeffs = pywt.wavedec(prices, wavelet, level=level)
                    return coeffs

                # Apply Wavelet Transform
                coeffs = apply_wavelet(prices)
                reconstructed_signal = pywt.waverec(coeffs, 'db4')

                # Hilbert Transform
                def apply_hilbert_transform(prices):
                    analytic_signal = hilbert(prices)
                    amplitude_envelope = np.abs(analytic_signal)
                    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
                    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi)
                    return amplitude_envelope, instantaneous_phase, instantaneous_frequency

                amplitude_envelope, instantaneous_phase, instantaneous_frequency = apply_hilbert_transform(prices)

                # Generate Signals
                def generate_signals(prices, reconstructed_signal, amplitude_envelope, instantaneous_frequency, df):
                    buy_signals = []
                    sell_signals = []
                    for i in range(2, len(prices) - 1):
                        if (reconstructed_signal[i] > reconstructed_signal[i-1] and
                            reconstructed_signal[i-1] < reconstructed_signal[i-2] and
                            instantaneous_frequency[i-1] < instantaneous_frequency[i-2] and
                            amplitude_envelope[i] > amplitude_envelope[i-1] and
                            df['20_MA'][i] > df['50_MA'][i] and df['RSI'][i] < 70):
                            buy_signals.append((i, prices[i]))
                        elif (reconstructed_signal[i] < reconstructed_signal[i-1] and
                              reconstructed_signal[i-1] > reconstructed_signal[i-2] and
                              instantaneous_frequency[i-1] > instantaneous_frequency[i-2] and
                              amplitude_envelope[i] < amplitude_envelope[i-1] and
                              df['20_MA'][i] < df['50_MA'][i] and df['RSI'][i] > 30):
                            sell_signals.append((i, prices[i]))
                    return buy_signals, sell_signals

                buy_signals, sell_signals = generate_signals(prices, reconstructed_signal, amplitude_envelope, instantaneous_frequency, data)
                return buy_signals, sell_signals

            buy_signals, sell_signals = advanced_signals(df)

            if analysis_type == "Trend Analysis":
                st.header("Trend Analysis")

                indicators = st.multiselect(
                    "Select Indicators",
                    ['Close', '20_MA', '50_MA', '200_MA', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'RSI', 'Signal', 'ADX',
                     'Parabolic_SAR', 'Bollinger_High', 'Bollinger_Low', 'Bollinger_Middle'],
                    default=['Close', 'Signal']
                )
                timeframe = st.radio(
                    "Select Timeframe",
                    ['15 days', '30 days', '90 days', '180 days', '1 year', 'All'],
                    index=4,
                    horizontal=True
                )

                if timeframe == '15 days':
                    df = df[-15:]
                elif timeframe == '30 days':
                    df = df[-30:]
                elif timeframe == '90 days':
                    df = df[-90:]
                elif timeframe == '180 days':
                    df = df[-180:]
                elif timeframe == '1 year':
                    df = df[-365:]

                fig = go.Figure()
                colors = {'Close': 'blue', '20_MA': 'orange', '50_MA': 'green', '200_MA': 'red', 'MACD': 'purple',
                          'MACD_Signal': 'brown', 'RSI': 'pink', 'Signal': 'black', 'ADX': 'magenta',
                          'Parabolic_SAR': 'yellow', 'Bollinger_High': 'black', 'Bollinger_Low': 'cyan',
                          'Bollinger_Middle': 'grey'}

                for indicator in indicators:
                    if indicator == 'Signal':
                        # Plot buy and sell signals
                        fig.add_trace(
                            go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close',
                                       line=dict(color=colors['Close'])))
                        buy_signal_points = [df.iloc[bs[0]] for bs in buy_signals if bs[0] < len(df)]
                        sell_signal_points = [df.iloc[ss[0]] for ss in sell_signals if ss[0] < len(df)]
                        fig.add_trace(
                            go.Scatter(x=[point['Date'] for point in buy_signal_points], y=[point['Close'] for point in buy_signal_points], mode='markers', name='Buy Signal',
                                       marker=dict(color='green', symbol='triangle-up')))
                        fig.add_trace(
                            go.Scatter(x=[point['Date'] for point in sell_signal_points], y=[point['Close'] for point in sell_signal_points], mode='markers', name='Sell Signal',
                                       marker=dict(color='red', symbol='triangle-down')))
                    elif indicator == 'MACD_Histogram':
                        fig.add_trace(go.Bar(x=df['Date'], y=df[indicator], name=indicator, marker_color='gray'))
                    else:
                        fig.add_trace(go.Scatter(x=df['Date'], y=df[indicator], mode='lines', name=indicator,
                                                 line=dict(color=colors.get(indicator, 'black'))))

                st.plotly_chart(fig)

            elif analysis_type == "Volume Analysis":
                st.header("Volume Analysis")
                volume_indicators = st.multiselect(
                    "Select Volume Indicators",
                    ['Volume', 'Volume_MA_20', 'OBV', 'Volume_Oscillator', 'CMF'],
                    default=['Volume']
                )
                volume_timeframe = st.radio(
                    "Select Timeframe",
                    ['15 days', '30 days', '90 days', '180 days', '1 year', 'All'],
                    index=4,
                    horizontal=True
                )

                if volume_timeframe == '15 days':
                    df = df[-15:]
                elif volume_timeframe == '30 days':
                    df = df[-30:]
                elif volume_timeframe == '90 days':
                    df = df[-90:]
                elif volume_timeframe == '180 days':
                    df = df[-180:]
                elif volume_timeframe == '1 year':
                    df = df[-365:]

                fig = go.Figure()
                for indicator in volume_indicators:
                    fig.add_trace(go.Scatter(x=df['Date'], y=df[indicator], mode='lines', name=indicator))

                st.plotly_chart(fig)

            elif analysis_type == "Support & Resistance Levels":
                st.header("Support & Resistance Levels")
                sr_indicators = st.multiselect(
                    "Select Indicators",
                    ['Close', '20_MA', '50_MA', '200_MA', 'Support', 'Resistance', 'Support_Trendline',
                     'Resistance_Trendline', 'Pivot', 'R1', 'S1', 'R2', 'S2'],
                    default=['Close']
                )
                sr_timeframe = st.radio(
                    "Select Timeframe",
                    ['15 days', '30 days', '90 days', '180 days', '1 year', 'All'],
                    index=4,
                    horizontal=True
                )

                if sr_timeframe == '15 days':
                    df = df[-15:]
                elif sr_timeframe == '30 days':
                    df = df[-30:]
                elif sr_timeframe == '90 days':
                    df = df[-90:]
                elif sr_timeframe == '180 days':
                    df = df[-180:]
                elif sr_timeframe == '1 year':
                    df = df[-365:]

                fig = go.Figure()
                for indicator in sr_indicators:
                    fig.add_trace(go.Scatter(x=df['Date'], y=df[indicator], mode='lines', name=indicator))

                st.plotly_chart(fig)

        else:
            st.write("Not enough data points for technical analysis.")
