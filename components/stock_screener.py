import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from utils.constants import bse_largecap, bse_smallcap, bse_midcap, sp500_tickers, ftse100_tickers
import pandas_ta as ta
from plotly.subplots import make_subplots
import plotly.graph_objs as go


def stock_screener_app():
    st.sidebar.subheader("Stock Screener")

    # Dropdown for selecting ticker category
    ticker_category = st.sidebar.selectbox("Select Index", ["BSE-LargeCap", "BSE-MidCap", "BSE-SmallCap", "S&P 500", "FTSE 100"])

    # Dropdown for Strategies
    submenu = st.sidebar.selectbox("Select Strategy", ["MACD", "Moving Average", "Bollinger Bands", "Volume"])

    # Date inputs
    start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", value=datetime.now() + timedelta(days=1))

    # Set tickers based on selected category
    tickers = {
        "BSE-LargeCap": bse_largecap,
        "BSE-MidCap": bse_midcap,
        "BSE-SmallCap": bse_smallcap,
        "S&P 500": sp500_tickers,
        "FTSE 100": ftse100_tickers
    }[ticker_category]

    def atr(high, low, close, window=14):
        tr = pd.concat([high.diff(), low.diff().abs(), (high - low).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr

    def aroon_up_down(high, low, window=14):
        aroon_up = 100 * (window - high.rolling(window=window).apply(np.argmax)) / window
        aroon_down = 100 * (window - low.rolling(window=window).apply(np.argmin)) / window
        return aroon_up, aroon_down

    def rsi(series, window=14):
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def hull_moving_average(series, window=9):
        half_length = int(window / 2)
        sqrt_length = int(np.sqrt(window))
        wma_half = series.rolling(window=half_length).mean()
        wma_full = series.rolling(window=window).mean()
        raw_hma = 2 * wma_half - wma_full
        hma = raw_hma.rolling(window=sqrt_length).mean()
        return hma

    def lsma(series, window=25):
        return series.rolling(window).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] * len(x) + np.polyfit(np.arange(len(x)), x, 1)[1])

    def parabolic_sar(high, low, close, af=0.02, max_af=0.2):
        psar = close.copy()
        psar.fillna(0, inplace=True)
        bull = True
        af = af
        max_af = max_af
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
                return adx
    
    def volume_profile_fixed_range(df, start_idx, end_idx, bins=50):
        data = df.iloc[start_idx:end_idx]
        price_min, price_max = data['Close'].min(), data['Close'].max()
        price_range = np.linspace(price_min, price_max, bins)
        volume_profile = np.zeros(len(price_range))

        for i in range(len(price_range) - 1):
            mask = (data['Close'] >= price_range[i]) & (data['Close'] < price_range[i + 1])
            volume_profile[i] = data['Volume'][mask].sum()

        df['VPFR'] = 0
        df.iloc[start_idx:end_idx, df.columns.get_loc('VPFR')] = volume_profile[np.searchsorted(price_range, data['Close']) - 1]

        return df['VPFR']

    def volume_profile_visible_range(df, visible_range=100, bins=50):
        end_idx = len(df)
        start_idx = max(0, end_idx - visible_range)
        return volume_profile_fixed_range(df, start_idx, end_idx, bins)
    
    def calculate_accelerator_oscillator(df):
        # Awesome Oscillator
        df['AO'] = df['High'].rolling(window=5).mean() - df['Low'].rolling(window=34).mean()
        # 5-period SMA of AO
        df['AO_SMA_5'] = df['AO'].rolling(window=5).mean()
        # Accelerator Oscillator
        df['AC'] = df['AO'] - df['AO_SMA_5']
        return df['AC']

    def calculate_awesome_oscillator(df):
        midpoint = (df['High'] + df['Low']) / 2
        df['AO'] = midpoint.rolling(window=5).mean() - midpoint.rolling(window=34).mean()
        return df['AO']

    def moving_average_channel(series, window, offset=2):
        mac_upper = series.rolling(window).mean() + offset * series.rolling(window).std()
        mac_lower = series.rolling(window).mean() - offset * series.rolling(window).std()
        return mac_upper, mac_lower

    def price_channel(high, low, window):
        price_channel_upper = high.rolling(window).max()
        price_channel_lower = low.rolling(window).min()
        return price_channel_upper, price_channel_lower

    def triple_ema(series, window):
        ema1 = series.ewm(span=window, adjust=False).mean()
        ema2 = ema1.ewm(span=window, adjust=False).mean()
        ema3 = ema2.ewm(span=window, adjust=False).mean()
        tema = 3 * (ema1 - ema2) + ema3
        return tema
    
    def volatility_close_to_close(df, window):
        # Close-to-Close Volatility
        df['Vol_CtC'] = df['Close'].pct_change().rolling(window=window).std() * np.sqrt(window)
        return df['Vol_CtC']

    def volatility_zero_trend_close_to_close(df, window):
        # Zero Trend Close-to-Close Volatility
        returns = df['Close'].pct_change()
        zero_returns = returns[returns == 0]
        df['Vol_ZtC'] = zero_returns.rolling(window=window).std() * np.sqrt(window)
        df['Vol_ZtC'].fillna(0, inplace=True)  # Handle NaNs
        return df['Vol_ZtC']

    def volatility_ohlc(df, window):
        # OHLC Volatility
        df['HL'] = df['High'] - df['Low']
        df['OC'] = np.abs(df['Close'] - df['Open'])
        df['Vol_OHLC'] = df[['HL', 'OC']].max(axis=1).rolling(window=window).mean()
        return df['Vol_OHLC']

    def volatility_index(df, window):
        # Volatility Index (standard deviation of returns)
        df['Vol_Index'] = df['Close'].pct_change().rolling(window=window).std() * np.sqrt(window)
        return df['Vol_Index']
    
    def historical_volatility(df, window=252):
        # Calculate the daily returns
        df['Returns'] = df['Close'].pct_change()
        
        # Calculate the rolling standard deviation of returns over the specified window
        df['Hist_Vol'] = df['Returns'].rolling(window=window).std()
        
        # Annualize the historical volatility (assuming 252 trading days in a year)
        df['Hist_Vol_Annualized'] = df['Hist_Vol'] * np.sqrt(window)
        
        return df['Hist_Vol_Annualized']    

    def williams_fractal(df):
        def fractal_high(df, n):
            return df['High'][(df['High'] == df['High'].rolling(window=n, center=True).max()) & (df['High'].shift(-n // 2) < df['High']) & (df['High'].shift(n // 2) < df['High'])]

        def fractal_low(df, n):
            return df['Low'][(df['Low'] == df['Low'].rolling(window=n, center=True).min()) & (df['Low'].shift(-n // 2) > df['Low']) & (df['Low'].shift(n // 2) > df['Low'])]

        n = 5  # Number of periods, typical value for Williams Fractal
        df['Fractal_Up'] = fractal_high(df, n)
        df['Fractal_Down'] = fractal_low(df, n)

        # Fill NaNs with 0 for convenience, indicating no fractal at these points
        df['Fractal_Up'] = df['Fractal_Up'].fillna(0)
        df['Fractal_Down'] = df['Fractal_Down'].fillna(0)

        return df[['Fractal_Up', 'Fractal_Down']]


    def calculate_indicators(df):

        df['ALMA'] = ta.alma(df['Close'])

        # Simple Moving Average (SMA)
        df['SMA_20'] = df['Close'].rolling(window=20).mean()

        # Exponential Moving Average (EMA)
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_High'] = df['BB_Middle'] + (df['BB_Std'] * 2)
        df['BB_Low'] = df['BB_Middle'] - (df['BB_Std'] * 2)

        # MACD
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        # ADX calculation
        df['ADX'] = calculate_adx(df)
        
        # On-Balance Volume (OBV)
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()

        # Average True Range (ATR)
        df['ATR'] = atr(df['High'], df['Low'], df['Close'])

        # Aroon Indicator
        df['Aroon_Up'], df['Aroon_Down'] = aroon_up_down(df['High'], df['Low'])

        # Double Exponential Moving Average (DEMA)
        df['DEMA'] = 2 * df['EMA_20'] - df['EMA_20'].ewm(span=20, adjust=False).mean()

        # Envelopes
        df['Envelope_High'] = df['Close'].rolling(window=20).mean() * 1.02
        df['Envelope_Low'] = df['Close'].rolling(window=20).mean() * 0.98

        # Guppy Multiple Moving Average (GMMA)
        df['GMMA_Short'] = df['Close'].ewm(span=3, adjust=False).mean()
        df['GMMA_Long'] = df['Close'].ewm(span=30, adjust=False).mean()

        # Hull Moving Average (HMA)
        df['HMA'] = hull_moving_average(df['Close'])

        # Ichimoku Cloud
        df['Ichimoku_Tenkan'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
        df['Ichimoku_Kijun'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
        df['Ichimoku_Senkou_Span_A'] = ((df['Ichimoku_Tenkan'] + df['Ichimoku_Kijun']) / 2).shift(26)
        df['Ichimoku_Senkou_Span_B'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)

        # Keltner Channels
        df['KC_Middle'] = df['Close'].rolling(window=20).mean()
        df['ATR_10'] = atr(df['High'], df['Low'], df['Close'], window=10)
        df['KC_High'] = df['KC_Middle'] + (df['ATR_10'] * 2)
        df['KC_Low'] = df['KC_Middle'] - (df['ATR_10'] * 2)

        # Least Squares Moving Average (LSMA)
        df['LSMA'] = lsma(df['Close'])

        # Parabolic SAR
        df['Parabolic_SAR'] = parabolic_sar(df['High'], df['Low'], df['Close'])
        
        # SuperTrend
        supertrend = ta.supertrend(df['High'], df['Low'], df['Close'], length=7, multiplier=3.0)
        df['SuperTrend'] = supertrend['SUPERT_7_3.0']

        # Moving Average Channel (MAC)
        df['MAC_Upper'], df['MAC_Lower'] = moving_average_channel(df['Close'], window=20, offset=2)

        # Price Channel
        df['Price_Channel_Upper'], df['Price_Channel_Lower'] = price_channel(df['High'], df['Low'], window=20)

        # Triple EMA (TEMA)
        df['TEMA_20'] = triple_ema(df['Close'], window=20)

        # Momentum Indicators
        df['AO'] = calculate_awesome_oscillator(df)
        df['AC'] = calculate_accelerator_oscillator(df)
        df['CMO'] = rsi(df['Close'], window=14) - 50
        df['CCI'] = (df['Close'] - df['Close'].rolling(window=20).mean()) / (0.015 * df['Close'].rolling(window=20).std())
        df['CRSI'] = (rsi(df['Close'], window=3) + rsi(df['Close'], window=2) + rsi(df['Close'], window=5)) / 3
        df['Coppock'] = df['Close'].diff(14).ewm(span=10, adjust=False).mean() + df['Close'].diff(11).ewm(span=10, adjust=False).mean()
        df['DPO'] = df['Close'].shift(int(20 / 2 + 1)) - df['Close'].rolling(window=20).mean()
        df['KST'] = df['Close'].rolling(window=10).mean() + df['Close'].rolling(window=15).mean() + df['Close'].rolling(window=20).mean() + df['Close'].rolling(window=30).mean()
        df['Momentum'] = df['Close'] - df['Close'].shift(10)
        df['RSI'] = rsi(df['Close'])
        df['ROC'] = df['Close'].pct_change(12)
        df['Stochastic_%K'] = (df['Close'] - df['Low'].rolling(window=14).min()) / (df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min()) * 100
        df['Stochastic_%D'] = df['Stochastic_%K'].rolling(window=3).mean()
        df['Stochastic_RSI'] = (rsi(df['Close'], window=14) - rsi(df['Close'], window=14).rolling(window=14).min()) / (rsi(df['Close'], window=14).rolling(window=14).max() - rsi(df['Close'], window=14).rolling(window=14).min())
        df['TRIX'] = df['Close'].ewm(span=15, adjust=False).mean().pct_change(1)
        df['TSI'] = df['Close'].diff(1).ewm(span=25, adjust=False).mean() / df['Close'].diff(1).abs().ewm(span=13, adjust=False).mean()
        df['Ultimate_Oscillator'] = (4 * (df['Close'] - df['Low']).rolling(window=7).sum() + 2 * (df['Close'] - df['Low']).rolling(window=14).sum() + (df['Close'] - df['Low']).rolling(window=28).sum()) / ((df['High'] - df['Low']).rolling(window=7).sum() + (df['High'] - df['Low']).rolling(window=14).sum() + (df['High'] - df['Low']).rolling(window=28).sum()) * 100

        # Volume Indicators
        df['AD'] = (df['Close'] - df['Low'] - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
        df['BoP'] = (df['Close'] - df['Open']) / (df['High'] - df['Low'])
        df['CMF'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
        df['CO'] = df['Close'].diff(3).ewm(span=10, adjust=False).mean()
        df['EMV'] = (df['High'] - df['Low']) / df['Volume']
        df['EFI'] = df['Close'].diff(1) * df['Volume']
        df['KVO'] = (df['High'] - df['Low']).ewm(span=34, adjust=False).mean() - (df['High'] - df['Low']).ewm(span=55, adjust=False).mean()
        df['MFI'] = (df['Close'].diff(1) / df['Close'].shift(1) * df['Volume']).rolling(window=14).mean()
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        df['PVT'] = (df['Close'].pct_change(1) * df['Volume']).cumsum()
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        df['VO'] = df['Volume'].pct_change(12)
        df['Vortex_Pos'] = df['High'].diff(1).abs().rolling(window=14).sum() / atr(df['High'], df['Low'], df['Close'])
        df['Vortex_Neg'] = df['Low'].diff(1).abs().rolling(window=14).sum() / atr(df['High'], df['Low'], df['Close'])
        df['Volume'] = df['Volume']

        # Volume Weighted Moving Average (VWMA)
        df['VWMA'] = ta.vwma(df['Close'], df['Volume'], length=20)

        # Net Volume
        df['Net_Volume'] = df['Volume'] * (df['Close'].diff() / df['Close'].shift(1))

        # Volume Profile Fixed Range (VPFR)
        df['VPFR'] = volume_profile_fixed_range(df, start_idx=0, end_idx=len(df)-1)

        # Volume Profile Visible Range (VPVR)
        df['VPVR'] = volume_profile_visible_range(df, visible_range=100)

        # Volatility Indicators
        df['BB_%B'] = (df['Close'] - df['BB_Low']) / (df['BB_High'] - df['BB_Low'])
        df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['Close']
        df['Chaikin_Volatility'] = (df['High'] - df['Low']).ewm(span=10, adjust=False).mean()
        df['Choppiness_Index'] = np.log10((df['High'] - df['Low']).rolling(window=14).sum() / (df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min())) * 100
        df['Hist_Vol_Annualized'] = historical_volatility(df, window=252)
        df['Mass_Index'] = (df['High'] - df['Low']).rolling(window=25).sum() / (df['High'] - df['Low']).rolling(window=9).sum()
        df['RVI'] = df['Close'].rolling(window=10).mean() / df['Close'].rolling(window=10).std()
        df['Standard_Deviation'] = df['Close'].rolling(window=20).std()

        # Volatility Close-to-Close
        df['Vol_CtC'] = volatility_close_to_close(df, window=20)

        # Volatility Zero Trend Close-to-Close
        df['Vol_ZtC'] = volatility_zero_trend_close_to_close(df, window=20)

        # Volatility O-H-L-C
        df['Vol_OHLC'] = volatility_ohlc(df, window=20)

        # Volatility Index
        df['Vol_Index'] = volatility_index(df, window=20)

        # Williams Fractal
        fractals = williams_fractal(df)
        df['Fractal_Up'] = fractals['Fractal_Up']
        df['Fractal_Down'] = fractals['Fractal_Down']


        # Support and Resistance Indicators
        def pivot_points(high, low, close):
            pp = (high + low + close) / 3
            r1 = 2 * pp - low
            s1 = 2 * pp - high
            r2 = pp + (high - low)
            s2 = pp - (high - low)
            r3 = high + 2 * (pp - low)
            s3 = low - 2 * (high - pp)
            return pp, r1, s1, r2, s2, r3, s3

        df['Pivot_Point'], df['Resistance_1'], df['Support_1'], df['Resistance_2'], df['Support_2'], df['Resistance_3'], df['Support_3'] = pivot_points(df['High'], df['Low'], df['Close'])

        
        return df

    def fetch_company_info(ticker):
        try:
            ticker_info = yf.Ticker(ticker).info
            return ticker_info.get('longName'), ticker_info.get('sector'), ticker_info.get('industry')
        except Exception as e:
            st.error(f"Error fetching company info for ticker '{ticker}': {e}")
            return None, None, None

    def fetch_latest_data(tickers_with_dates):
        technical_data = []
        for ticker, occurrence_date in tickers_with_dates:
            try:
                data = yf.download(ticker, start=start_date, end=end_date)
                if data.empty:
                    continue

                data = calculate_indicators(data)
                latest_data = data.iloc[-1]
                company_name, sector, industry = fetch_company_info(ticker)
                technical_data.append({
                    
                    'Date of Occurrence': occurrence_date,
                    'Ticker': ticker,
                    'Company Name': company_name,
                    'Sector': sector,
                    'Industry': industry,
                    'Close': latest_data['Close'],
                    'Volume': latest_data['Volume'],
                    'SMA_20': latest_data['SMA_20'],
                    'EMA_20': latest_data['EMA_20'],
                    'BB_High': latest_data['BB_High'],
                    'BB_Low': latest_data['BB_Low'],
                    'MACD': latest_data['MACD'],
                    'MACD_signal': latest_data['MACD_signal'],
                    'MACD_hist': latest_data['MACD_hist'],
                    'RSI': latest_data['RSI'],
                    'OBV': latest_data['OBV'],
                    'ATR': latest_data['ATR'],
                    'ALMA': latest_data['ALMA'],
                    'Aroon_Up': latest_data['Aroon_Up'],
                    'Aroon_Down': latest_data['Aroon_Down'],
                    'ADX':latest_data['ADX'],
                    'DEMA': latest_data['DEMA'],
                    'Envelope_High': latest_data['Envelope_High'],
                    'Envelope_Low': latest_data['Envelope_Low'],
                    'GMMA_Short': latest_data['GMMA_Short'],
                    'GMMA_Long': latest_data['GMMA_Long'],
                    'HMA': latest_data['HMA'],
                    'Ichimoku_Tenkan': latest_data['Ichimoku_Tenkan'],
                    'Ichimoku_Kijun': latest_data['Ichimoku_Kijun'],
                    'Ichimoku_Senkou_Span_A': latest_data['Ichimoku_Senkou_Span_A'],
                    'Ichimoku_Senkou_Span_B': latest_data['Ichimoku_Senkou_Span_B'],
                    'KC_High': latest_data['KC_High'],
                    'KC_Low': latest_data['KC_Low'],
                    'LSMA': latest_data['LSMA'],
                    'Parabolic_SAR': latest_data['Parabolic_SAR'],
                    'SuperTrend': latest_data['SuperTrend'],
                    'AO': latest_data['AO'],
                    'AC': latest_data['AC'],
                    'CMO': latest_data['CMO'],
                    'CCI': latest_data['CCI'],
                    'CRSI': latest_data['CRSI'],
                    'Coppock': latest_data['Coppock'],
                    'DPO': latest_data['DPO'],
                    'KST': latest_data['KST'],
                    'Momentum': latest_data['Momentum'],
                    'ROC': latest_data['ROC'],
                    'Stochastic_%K': latest_data['Stochastic_%K'],
                    'Stochastic_%D': latest_data['Stochastic_%D'],
                    'Stochastic_RSI': latest_data['Stochastic_RSI'],
                    'TRIX': latest_data['TRIX'],
                    'TSI': latest_data['TSI'],
                    'Ultimate_Oscillator': latest_data['Ultimate_Oscillator'],
                    'AD': latest_data['AD'],
                    'BoP': latest_data['BoP'],
                    'CMF': latest_data['CMF'],
                    'CO': latest_data['CO'],
                    'EMV': latest_data['EMV'],
                    'EFI': latest_data['EFI'],
                    'KVO': latest_data['KVO'],
                    'MFI': latest_data['MFI'],
                    'PVT': latest_data['PVT'],
                    'VWAP': latest_data['VWAP'],
                    'VO': latest_data['VO'],
                    'Vortex_Pos': latest_data['Vortex_Pos'],
                    'Vortex_Neg': latest_data['Vortex_Neg'],
                    'Volume': latest_data['Volume'],
                    'BB_%B': latest_data['BB_%B'],
                    'BB_Width': latest_data['BB_Width'],
                    'Chaikin_Volatility': latest_data['Chaikin_Volatility'],
                    'Choppiness_Index': latest_data['Choppiness_Index'],
                    'Hist_Vol_Annualized': latest_data['Hist_Vol_Annualized'],
                    'Mass_Index': latest_data['Mass_Index'],
                    'RVI': latest_data['RVI'],
                    'Standard_Deviation': latest_data['Standard_Deviation'],
                    'Pivot_Point': latest_data['Pivot_Point'],
                    'Resistance_1': latest_data['Resistance_1'],
                    'Support_1': latest_data['Support_1'],
                    'Resistance_2': latest_data['Resistance_2'],
                    'Support_2': latest_data['Support_2'],
                    'Resistance_3': latest_data['Resistance_3'],
                    'Support_3': latest_data['Support_3'],
                    'VWMA':latest_data['VWMA'],
                    'Net_Volume':latest_data['Net_Volume'],
                    'VPFR':latest_data['VPFR'],
                    'VPVR':latest_data['VPVR'],
                    'MAC_Upper':latest_data['MAC_Upper'],
                    'MAC_Lower':latest_data['MAC_Lower'],
                    'Price_Channel_Upper':latest_data['Price_Channel_Upper'],
                    'Price_Channel_Lower':latest_data['Price_Channel_Lower'],
                    'TEMA_20':latest_data['TEMA_20'],
                    'Vol_CtC':latest_data['Vol_CtC'],
                    'Vol_ZtC':latest_data['Vol_ZtC'],
                    'Vol_OHLC':latest_data['Vol_OHLC'],
                    'Vol_Index':latest_data['Vol_Index'],
                    'Fractal_Up':latest_data['Fractal_Up'],
                    'Fractal_Down':latest_data['Fractal_Down'],


                })
            except Exception as e:
                st.error(f"Error fetching latest data for ticker '{ticker}': {e}")

        return pd.DataFrame(technical_data)

    def check_signal(data, strategy):
        recent_data = data[-5:]
        if strategy == "MACD":
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data['MACD_hist'] = data['MACD'] - data['MACD_signal']
            for i in range(1, len(recent_data)):
                if (recent_data['MACD'].iloc[i] > recent_data['MACD_signal'].iloc[i] and
                    recent_data['MACD'].iloc[i-1] < recent_data['MACD_signal'].iloc[i-1] and
                    recent_data['MACD'].iloc[i] > 0 and
                    recent_data['MACD_hist'].iloc[i] > 0 and
                    recent_data['MACD_hist'].iloc[i-1] < 0 and
                    recent_data['MACD_hist'].iloc[i] > recent_data['MACD_hist'].iloc[i-1] > recent_data['MACD_hist'].iloc[i-2]):
                    return recent_data.index[i]
        elif strategy == "Moving Average":
            data['Short_EMA'] = data['Close'].ewm(span=10, adjust=False).mean()
            data['Long_EMA'] = data['Close'].ewm(span=20, adjust=False).mean()
            for i in range(1, len(recent_data)):
                if (recent_data['Short_EMA'].iloc[i] > recent_data['Long_EMA'].iloc[i] and
                    recent_data['Short_EMA'].iloc[i-1] <= recent_data['Long_EMA'].iloc[i-1]):
                    return recent_data.index[i]
        elif strategy == "Bollinger Bands":
            for i in range(1, len(recent_data)):
                if (recent_data['Close'].iloc[i] > recent_data['BB_Low'].iloc[i] and
                    recent_data['Close'].iloc[i-1] <= recent_data['BB_Low'].iloc[i-1]):
                    return recent_data.index[i]
        elif strategy == "Volume":
            data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
            for i in range(1, len(recent_data)):
                if (recent_data['Volume'].iloc[i] > recent_data['Volume_MA'].iloc[i] and
                    recent_data['Volume'].iloc[i-1] <= recent_data['Volume_MA'].iloc[i-1]):
                    return recent_data.index[i]
        return None

    tickers_with_signals = []
    progress_bar = st.progress(0)
    progress_step = 1 / len(tickers)

    for i, ticker in enumerate(tickers):
        progress_bar.progress((i + 1) * progress_step)
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                continue
            data = calculate_indicators(data)
            occurrence_date = check_signal(data, submenu)
            if occurrence_date:
                tickers_with_signals.append((ticker, occurrence_date))
        except KeyError as e:
            st.error(f"KeyError: {e} - Check if the ticker symbol '{ticker}' is valid.")
        except Exception as e:
            st.error(f"Error processing data for ticker '{ticker}': {e}")

    df_signals = fetch_latest_data(tickers_with_signals)

    def add_scoring(df):
        # Customize scoring for each indicator
        volume_score_conditions = {
            'AD': df['AD'] > df['AD'].shift(1),
            'BoP': df['BoP'] > 0,
            'CMF': df['CMF'] > 0,
            'CO': df['CO'] > df['CO'].shift(1),
            'EMV': df['EMV'] > 0,
            'EFI': (df['EFI'] > 0) & (df['EFI'] > df['EFI'].shift(1)) ,
            'KVO': (df['KVO'] > 0) & (df['KVO'] > df['KVO'].shift(1)),
            'MFI': df['MFI'] > 50,
            'Net_Volume': (df['Net_Volume'] > 0) & (df['Net_Volume'] > df['Net_Volume'].shift(1)),
            'OBV': df['OBV'] > df['OBV'].shift(1),
            'PVT': df['PVT'] > df['PVT'].shift(1),
            'VWAP': df['Close'] > df['VWAP'],
            'VWMA': df['Close'] > df['VWMA'],
            'VO': df['VO'] > 0,
            'VPFR': df['VPFR'] > 0,
            'VPVR': df['VPVR'] > 0,
            'Vortex_Pos': df['Vortex_Pos'] > df['Vortex_Neg']

            
        }
        trend_score_conditions = {
            'ALMA': df['ALMA'] < df['Close'],
            'Aroon_Up': df['Aroon_Up'] > 70,
            'Aroon_Down': df['Aroon_Down'] < 30,
            'ADX': df['ADX'] > 25,
            'BB_High': df['Close'] < df['BB_High'],
            'SMA_20': df['Close'] > df['SMA_20'],
            'DEMA': df['Close'] > df['DEMA'],
            'Envelope_High': df['Close'] < df['Envelope_High'],
            'GMMA_Short': df['GMMA_Short'] > df['GMMA_Long'],
            'HMA': df['HMA'] < df['Close'],
            'Ichimoku_Tenkan': df['Close'] > df['Ichimoku_Tenkan'],
            'Ichimoku_Kijun': df['Close'] > df['Ichimoku_Kijun'],
            'Ichimoku_Senkou_Span_A': df['Close'] > df['Ichimoku_Senkou_Span_A'],
            'Ichimoku_Senkou_Span_B': df['Close'] > df['Ichimoku_Senkou_Span_B'],
            'KC_High': df['Close'] < df['KC_High'],
            'LSMA': df['Close'] > df['LSMA'],
            'EMA_20': df['Close'] > df['EMA_20'],
            'MACD': df['MACD'] > df['MACD_signal'],
            'MACD_hist': df['MACD_hist'] > 0,
            'Parabolic_SAR': df['Close'] > df['Parabolic_SAR'],
            'SuperTrend': df['Close'] > df['SuperTrend'],
            'MAC_Upper': df['Close'] > df['MAC_Upper'],
            'Price_Channel_Upper': df['Close'] > df['Price_Channel_Upper'],
            'TEMA_20': df['Close'] > df['TEMA_20']
        }
        momentum_score_conditions = {
            'AO': df['AO'] > 0,
            'AC': df['AC'] > 0,
            'CMO': df['CMO'] > 0,
            'CCI': df['CCI'] > 100,
            'CRSI': (df['CRSI'] > 30) & (df['CRSI'] > df['CRSI'].shift(1)),
            'Coppock': (df['Coppock'] > 0) & (df['Coppock'] > df['Coppock'].shift(1)),
            'DPO': df['DPO'] > 0,
            'KST': df['KST'] > 0,
            'Momentum': df['Momentum'] > 0,
            'RSI': (df['RSI'] > 30) & (df['RSI'] < 70),
            'ROC': df['ROC'] > 0,
            'Stochastic_%K': df['Stochastic_%K'] > 20,
            'Stochastic_%D': df['Stochastic_%D'] > 20,
            'Stochastic_RSI': df['Stochastic_RSI'] < 80,
            'TRIX': df['TRIX'] > 0,
            'TSI': df['TSI'] > 0,
            'Ultimate_Oscillator': df['Ultimate_Oscillator'] > 50
        }
        volatility_score_conditions = {
            'BB_%B': df['BB_%B'] < 1,
            'BB_Width': df['BB_Width'] > df['BB_Width'].mean(),
            'Chaikin_Volatility': df['Chaikin_Volatility'] > df['Chaikin_Volatility'].mean(),
            'Choppiness_Index': df['Choppiness_Index'] < 61.8,
            'Hist_Vol_Annualized': df['Hist_Vol_Annualized'] > df['Hist_Vol_Annualized'].mean(),
            'Mass_Index': df['Mass_Index'] > 27,
            'RVI': df['RVI'] > 50,
            'Standard_Deviation': df['Standard_Deviation'] > df['Standard_Deviation'].mean(),
            'Vol_CtC': df['Vol_CtC'] > df['Vol_CtC'].mean(),
            'Vol_ZtC': df['Vol_ZtC'] > df['Vol_ZtC'].mean(),
            'Vol_OHLC': df['Vol_OHLC'] > df['Vol_OHLC'].mean(),
            'Vol_Index': df['Vol_Index'] > df['Vol_Index'].mean()
        }
        support_resistance_score_conditions = {
            'Pivot_Point': df['Close'] > df['Pivot_Point'],
            'Resistance_1': df['Close'] > df['Resistance_1'],
            'Support_1': df['Close'] > df['Support_1'],
            'Resistance_2': df['Close'] > df['Resistance_2'],
            'Support_2': df['Close'] > df['Support_2'],
            'Resistance_3': df['Close'] > df['Resistance_3'],
            'Support_3': df['Close'] > df['Support_3'],
            'Fractal_Up': df['Fractal_Up'] > 0,
            'Fractal_Down': df['Fractal_Down'] > 0
        }

        # Calculate scores based on conditions
        df['Volume_Score'] = sum([cond.astype(int) for cond in volume_score_conditions.values()])
        df['Trend_Score'] = sum([cond.astype(int) for cond in trend_score_conditions.values()])
        df['Momentum_Score'] = sum([cond.astype(int) for cond in momentum_score_conditions.values()])
        df['Volatility_Score'] = sum([cond.astype(int) for cond in volatility_score_conditions.values()])
        df['Support_Resistance_Score'] = sum([cond.astype(int) for cond in support_resistance_score_conditions.values()])

        # Normalize scores to be between -1 and 1
        df['Volume_Score'] = df['Volume_Score'] / df['Volume_Score'].abs().max()
        df['Trend_Score'] = df['Trend_Score'] / df['Trend_Score'].abs().max()
        df['Momentum_Score'] = df['Momentum_Score'] / df['Momentum_Score'].abs().max()
        df['Volatility_Score'] = df['Volatility_Score'] / df['Volatility_Score'].abs().max()
        df['Support_Resistance_Score'] = df['Support_Resistance_Score'] / df['Support_Resistance_Score'].abs().max()

        # Overall score as an average of the five categories
        df['Overall_Score'] = (df['Volume_Score'] + df['Trend_Score'] + df['Momentum_Score'] + df['Volatility_Score'] + df['Support_Resistance_Score']) / 5

        return df

    if not df_signals.empty:
        df_signals = add_scoring(df_signals)
        table1_columns = ['Ticker', 'Date of Occurrence', 'Company Name', 'Sector', 'Industry', 'Close', 'Volume']
        trend_columns = ['Ticker', 'Close','ALMA','Aroon_Up', 'Aroon_Down','ADX', 'BB_High', 'SMA_20','BB_Low', 'DEMA', 'Envelope_High', 'Envelope_Low', 'GMMA_Short', 'GMMA_Long', 'HMA', 'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'Ichimoku_Senkou_Span_A', 'Ichimoku_Senkou_Span_B', 'KC_High', 'KC_Low', 'LSMA','EMA_20','MACD', 'MACD_signal', 'MACD_hist', 'Parabolic_SAR', 'SuperTrend','MAC_Upper','MAC_Lower','Price_Channel_Upper','Price_Channel_Lower','TEMA_20']
        momentum_columns = ['Ticker','AO', 'AC', 'CMO', 'CCI', 'CRSI', 'Coppock', 'DPO', 'KST', 'Momentum', 'RSI','ROC', 'Stochastic_%K', 'Stochastic_%D', 'Stochastic_RSI', 'TRIX', 'TSI', 'Ultimate_Oscillator']
        volume_columns = ['Ticker','AD', 'BoP', 'CMF', 'CO', 'EMV', 'EFI', 'KVO', 'MFI', 'Net_Volume','OBV', 'PVT', 'VWAP','VWMA', 'VO','VPFR','VPVR', 'Vortex_Pos', 'Vortex_Neg', 'Volume']
        volatility_columns = ['Ticker','ATR','BB_%B', 'BB_Width', 'Chaikin_Volatility', 'Choppiness_Index', 'Hist_Vol_Annualized', 'Mass_Index', 'RVI', 'Standard_Deviation','Vol_CtC','Vol_ZtC','Vol_OHLC','Vol_Index']
        support_resistance_columns = ['Ticker', 'Close','Pivot_Point', 'Resistance_1', 'Support_1', 'Resistance_2', 'Support_2', 'Resistance_3', 'Support_3','Fractal_Up','Fractal_Down']
        
        st.title("Stocks Based on Selected Strategy")
        st.write(f"Stocks with {submenu} signal in the last 5 days:")

        
        st.dataframe(df_signals[table1_columns])

        st.subheader("Volume Indicators")
        st.dataframe(df_signals[volume_columns])

        st.subheader("Momentum Indicators")
        st.dataframe(df_signals[momentum_columns])

        st.subheader("Trend Indicators")
        st.dataframe(df_signals[trend_columns])

        st.subheader("Volatility Indicators")
        st.dataframe(df_signals[volatility_columns])

        st.subheader("Support and Resistance Levels")
        st.dataframe(df_signals[support_resistance_columns])

        st.subheader("Scoring System")
        st.dataframe(df_signals[['Ticker', 'Volume_Score', 'Trend_Score', 'Momentum_Score', 'Volatility_Score', 'Support_Resistance_Score', 'Overall_Score']])

        # Dropdown for stock selection
        st.subheader("Interactive Plots")
        selected_stock = st.selectbox("Select Stock", df_signals['Ticker'].tolist())

        # Define technical indicator categories
        indicator_groups = {
            "Trend Indicators": ['ALMA','Aroon_Up', 'Aroon_Down','ADX', 'BB_High', 'SMA_20','BB_Low', 'DEMA', 'Envelope_High', 'Envelope_Low', 'GMMA_Short', 'GMMA_Long', 'HMA', 'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'Ichimoku_Senkou_Span_A', 'Ichimoku_Senkou_Span_B', 'KC_High', 'KC_Low', 'LSMA','EMA_20','MACD', 'MACD_signal', 'MACD_hist', 'Parabolic_SAR', 'SuperTrend','MAC_Upper','MAC_Lower','Price_Channel_Upper','Price_Channel_Lower','TEMA_20'],
            "Momentum Indicators": ['AO', 'AC', 'CMO', 'CCI', 'CRSI', 'Coppock', 'DPO', 'KST', 'Momentum', 'RSI','ROC', 'Stochastic_%K', 'Stochastic_%D', 'Stochastic_RSI', 'TRIX', 'TSI', 'Ultimate_Oscillator'],
            "Volume Indicators": ['AD', 'BoP', 'CMF', 'CO', 'EMV', 'EFI', 'KVO', 'MFI', 'Net_Volume','OBV', 'PVT', 'VWAP','VWMA', 'VO','VPFR','VPVR', 'Vortex_Pos', 'Vortex_Neg', 'Volume'],
            "Volatility Indicators": ['ATR','BB_%B', 'BB_Width', 'Chaikin_Volatility', 'Choppiness_Index', 'Hist_Vol_Annualized', 'Mass_Index', 'RVI', 'Standard_Deviation','Vol_CtC','Vol_ZtC','Vol_OHLC','Vol_Index'],
            "Support & Resistance Indicators": ['Pivot_Point', 'Resistance_1', 'Support_1', 'Resistance_2', 'Support_2', 'Resistance_3', 'Support_3','Fractal_Up','Fractal_Down']
        }
        # Create multiselect options for each indicator group
        selected_indicators = []
        for group_name, indicators in indicator_groups.items():
            with st.expander(group_name):
                selected_indicators.extend(st.multiselect(f'Select {group_name}', indicators))

        show_candlestick = st.sidebar.checkbox('Heikin-Ashi Candles')

        def get_macd_hist_colors(macd_hist):
            colors = []
            for i in range(1, len(macd_hist)):
                if macd_hist.iloc[i] > 0:
                    color = 'green' if macd_hist.iloc[i] > macd_hist.iloc[i - 1] else 'lightgreen'
                else:
                    color = 'red' if macd_hist.iloc[i] < macd_hist.iloc[i - 1] else 'lightcoral'
                colors.append(color)
            return colors

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{}]], 
                            row_heights=[0.5, 0.3, 0.2], vertical_spacing=0.02)

        data = yf.download(selected_stock, start=start_date, end=end_date)
        data = calculate_indicators(data)

        if show_candlestick:
            fig.add_trace(go.Candlestick(x=data.index,
                                        open=data['Open'],
                                        high=data['High'],
                                        low=data['Low'],
                                        close=data['Close'],
                                        name='Candlestick'), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'), row=1, col=1)

        for i, indicator in enumerate(selected_indicators):
            if indicator == 'MACD_hist':
                macd_hist_colors = get_macd_hist_colors(data[indicator])
                fig.add_trace(go.Bar(x=data.index[1:], y=data[indicator][1:], name='MACD Histogram', marker_color=macd_hist_colors), row=1, col=1, secondary_y=True)
            elif 'Fib' in indicator or 'Gann' in indicator:
                fig.add_trace(go.Scatter(x=data.index, y=data[indicator], mode='lines', name=indicator, line=dict(dash='dash')), row=1, col=1)
            else:
                fig.add_trace(go.Scatter(x=data.index, y=data[indicator], mode='lines', name=indicator), row=1, col=1, secondary_y=True)

            fig.update_layout(**{
                f'yaxis{i+2}': go.layout.YAxis(
                    title=indicator,
                    overlaying='y',
                    side='right',
                    position=1 - (i * 0.05)
                )
            })

        fig.update_layout(
            title={
                'text': f'{selected_stock} Price and Technical Indicators',
                'y': 0.97,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            height=800,
            margin=dict(t=100, b=10, l=50, r=50),
            yaxis=dict(title='Price'),
            yaxis2=dict(title='Indicators', overlaying='y', side='right'),
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
            legend=dict(x=0.5, y=0.3, orientation='h', xanchor='center', yanchor='top')
        )

        fig.update_layout(
            hovermode='x unified',
            hoverlabel=dict(bgcolor="light blue", font_size=16, font_family="Rockwell")
        )

        st.plotly_chart(fig)

    else:
        st.write("No data available for the selected tickers and date range.")


if __name__ == "__main__":
    stock_screener_app()
