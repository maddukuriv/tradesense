import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress
from newsapi.newsapi_client import NewsApiClient
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import requests
from utils.constants import ticker_to_company_dict
import nltk
import plotly.express as px

nltk.download('vader_lexicon')



# Initialize NewsApiClient with your API key
newsapi = NewsApiClient(api_key='252b2075083945dfbed8945ddc240a2b')
analyzer = SentimentIntensityAnalyzer()

# Helper functions

# Generate the ticker to company mapping using the predefined dictionary
ticker_to_company = {ticker: ticker_to_company_dict.get(ticker, ticker) for ticker in ticker_to_company_dict.keys()}

# Convert the ticker_to_company dictionary to a list of company names
company_names = list(ticker_to_company.values())




@st.cache_data(ttl=300)
def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    return data



# manual technical formaulas
def atr(high, low, close, window=14):
    tr = pd.concat([(high - low).abs(), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
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
    return adx, plus_di, minus_di

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
        return df['High'][(df['High'] == df['High'].rolling(window=n, center=True).max()) &
                        (df['High'] > df['High'].shift(1)) &
                        (df['High'] > df['High'].shift(2)) &
                        (df['High'] > df['High'].shift(-1)) &
                        (df['High'] > df['High'].shift(-2))]

    def fractal_low(df, n):
        return df['Low'][(df['Low'] == df['Low'].rolling(window=n, center=True).min()) &
                        (df['Low'] < df['Low'].shift(1)) &
                        (df['Low'] < df['Low'].shift(2)) &
                        (df['Low'] < df['Low'].shift(-1)) &
                        (df['Low'] < df['Low'].shift(-2))]

    n = 5  # Number of periods, typical value for Williams Fractal
    df['Fractal_Up'] = fractal_high(df, n)
    df['Fractal_Down'] = fractal_low(df, n)

    # Replace NaN with 0, indicating no fractal at these points
    df['Fractal_Up'] = df['Fractal_Up'].fillna(0)
    df['Fractal_Down'] = df['Fractal_Down'].fillna(0)

    return df[['Fractal_Up', 'Fractal_Down']]

# Calculate Correlation Coefficient
def correlation_coefficient(series1, series2, n=14):
    return series1.rolling(window=n).corr(series2)

# Calculate Log Correlation
def log_correlation(series1, series2, n=14):
    log_series1 = np.log(series1)
    log_series2 = np.log(series2)
    return correlation_coefficient(log_series1, log_series2, n)

# Calculate Linear Regression Curve
def linear_regression_curve(series, n=14):
    lr_curve = pd.Series(index=series.index, dtype='float64')
    for i in range(n, len(series)):
        y = series[i-n:i]
        x = np.arange(n)
        slope, intercept, _, _, _ = linregress(x, y)
        lr_curve[i] = intercept + slope * (n-1)
    return lr_curve

# Calculate Linear Regression Slope
def linear_regression_slope(series, n=14):
    lr_slope = pd.Series(index=series.index, dtype='float64')
    for i in range(n, len(series)):
        y = series[i-n:i]
        x = np.arange(n)
        slope, _, _, _, _ = linregress(x, y)
        lr_slope[i] = slope
    return lr_slope

# Calculate Standard Error
def standard_error(series, n=14):
    std_err = pd.Series(index=series.index, dtype='float64')
    for i in range(n, len(series)):
        y = series[i-n:i]
        x = np.arange(n)
        _, _, _, _, stderr = linregress(x, y)
        std_err[i] = stderr
    return std_err


# Calculate Standard Error Bands
def standard_error_bands(series, n=14, num_std=2):
    lr_curve = linear_regression_curve(series, n)
    stderr = standard_error(series, n)
    upper_band = lr_curve + num_std * stderr
    lower_band = lr_curve - num_std * stderr
    return upper_band, lower_band


# Calculate Choppiness Index
def choppiness_index(high, low, close, n=14):
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(n).sum()
    high_low_range = high.rolling(n).max() - low.rolling(n).min()
    chop = 100 * np.log10(atr / high_low_range) / np.log10(n)
    return chop

# Calculate Chande Kroll Stop
def chande_kroll_stop(high, low, close, n=10, m=1):
    atr = (high - low).rolling(n).mean()
    long_stop = close - (m * atr)
    short_stop = close + (m * atr)
    return long_stop, short_stop

# Calculate Fisher Transform
def fisher_transform(price, n=10):
    median_price = price.rolling(window=n).median()
    min_low = price.rolling(window=n).min()
    max_high = price.rolling(window=n).max()
    value = 2 * ((median_price - min_low) / (max_high - min_low) - 0.5)
    fish = 0.5 * np.log((1 + value) / (1 - value))
    fish_signal = fish.shift(1)
    return fish, fish_signal

# Calculate Relative Vigor Index (RVI)
def relative_vigor_index(open, high, low, close, n=10):
    numerator = close - open
    denominator = high - low
    rvi = (numerator.rolling(n).sum()) / (denominator.rolling(n).sum())
    return rvi

# Calculate SMI Ergodic Indicator/Oscillator
def smi_ergodic(close, n=14, m=5, signal_n=3):
    smi = (close - close.shift(n)) / (close.rolling(n).std())
    signal = smi.rolling(signal_n).mean()
    return smi, signal

# Calculate Williams %R
def williams_r(high, low, close, n=14):
    highest_high = high.rolling(n).max()
    lowest_low = low.rolling(n).min()
    r = (highest_high - close) / (highest_high - lowest_low) * -100
    return r

# Calculate Williams Alligator
def alligator(high, low, close, jaw_n=13, teeth_n=8, lips_n=5):
    jaw = close.rolling(window=jaw_n).mean().shift(8)
    teeth = close.rolling(window=teeth_n).mean().shift(5)
    lips = close.rolling(window=lips_n).mean().shift(3)
    return jaw, teeth, lips

# Calculate ZigZag
def zigzag(close, percentage=5):
    zz = [0]
    for i in range(1, len(close)):
        change = (close[i] - close[zz[-1]]) / close[zz[-1]] * 100
        if abs(change) > percentage:
            zz.append(i)
    zigzag_series = pd.Series(index=close.index, data=np.nan)
    zigzag_series.iloc[zz] = close.iloc[zz]
    return zigzag_series.ffill()

def calculate_technical_indicators(df):
    ##Trend Indicators--------------------------------------------------------
    # Moving Averages
    df['5_day_EMA'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['10_day_EMA'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['20_day_EMA'] = df['Close'].ewm(span=20, adjust=False).mean()
    # Arnaud Legoux Moving Average (ALMA)
    df['ALMA'] = ta.alma(df['Close'])
    # Aroon Indicator
    df['Aroon_Up'], df['Aroon_Down'] = aroon_up_down(df['High'], df['Low'])
    # ADX calculation
    df['ADX'], df['Plus_DI'], df['Minus_DI'] = calculate_adx(df) 
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_High'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Low'] = df['BB_Middle'] - (df['BB_Std'] * 2)
    # Double Exponential Moving Average (DEMA)
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
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
    # Moving Average Channel (MAC)
    df['MAC_Upper'], df['MAC_Lower'] = moving_average_channel(df['Close'], window=20, offset=2)
    # MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    # Parabolic SAR
    df['Parabolic_SAR'] = parabolic_sar(df['High'], df['Low'], df['Close'])
    # SuperTrend
    supertrend = ta.supertrend(df['High'], df['Low'], df['Close'], length=7, multiplier=3.0)
    df['SuperTrend'] = supertrend['SUPERT_7_3.0']
    # Price Channel
    df['Price_Channel_Upper'], df['Price_Channel_Lower'] = price_channel(df['High'], df['Low'], window=20)
    # Triple EMA (TEMA)
    df['TEMA_20'] = triple_ema(df['Close'], window=20)
    # Calculate Advance/Decline
    df['Advance_Decline'] = df['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)).cumsum()
    # Chande kroll stop
    df['Chande_Kroll_Stop_Long'], df['Chande_Kroll_Stop_Short'] = chande_kroll_stop(df['High'], df['Low'], df['Close'])
    # William Alligator
    df['Williams_Alligator_Jaw'], df['Williams_Alligator_Teeth'], df['Williams_Alligator_Lips'] = alligator(df['High'], df['Low'], df['Close'])
    # Donchian Channels
    donchian = ta.donchian(df['High'], df['Low'])
    df['Donchian_High'] = donchian['DCU_20_20']
    df['Donchian_Low'] = donchian['DCL_20_20']


    ## Momentum Indicators----------------------------------
    # Awesome Oscillator (AO)
    df['AO'] = calculate_awesome_oscillator(df)
    # Accelerator Oscillator (AC)
    df['AC'] = calculate_accelerator_oscillator(df)
    # Chande Momentum Oscillator (CMO):
    df['CMO'] = rsi(df['Close'], window=14) - 50
    # Commodity Channel Index (CCI)
    df['CCI'] = (df['Close'] - df['Close'].rolling(window=20).mean()) / (0.015 * df['Close'].rolling(window=20).std())
    # Connors RSI
    df['CRSI'] = (rsi(df['Close'], window=3) + rsi(df['Close'], window=2) + rsi(df['Close'], window=5)) / 3
    # Coppock Curve
    df['Coppock'] = df['Close'].diff(14).ewm(span=10, adjust=False).mean() + df['Close'].diff(11).ewm(span=10, adjust=False).mean()
    # Detrended Price Oscillator (DPO):
    df['DPO'] = df['Close'].shift(int(20 / 2 + 1)) - df['Close'].rolling(window=20).mean()
    # Directional Movement Index (DMI)

    # Know Sure Thing (KST)
    df['KST'] = df['Close'].rolling(window=10).mean() + df['Close'].rolling(window=15).mean() + df['Close'].rolling(window=20).mean() + df['Close'].rolling(window=30).mean()
    df['KST_Signal'] = df['KST'].rolling(window=9).mean()
    # Momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    # Relative Strength Index (RSI)
    df['RSI'] = rsi(df['Close'])
    # Rate of Change (ROC)
    df['ROC'] = df['Close'].pct_change(12)
    # Stochastic Oscillator
    df['Stochastic_%K'] = (df['Close'] - df['Low'].rolling(window=14).min()) / (df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min()) * 100
    df['Stochastic_%D'] = df['Stochastic_%K'].rolling(window=3).mean()
    # Stochastic RSI:
    df['Stochastic_RSI'] = (rsi(df['Close'], window=14) - rsi(df['Close'], window=14).rolling(window=14).min()) / (rsi(df['Close'], window=14).rolling(window=14).max() - rsi(df['Close'], window=14).rolling(window=14).min())
    # TRIX
    df['TRIX'] = df['Close'].ewm(span=15, adjust=False).mean().pct_change(1)
    trix = ta.trix(df['Close'])
    df['TRIX'] = trix['TRIX_30_9']
    df['TRIX_Signal'] = trix['TRIXs_30_9']
    # True Strength Index (TSI)
    df['TSI'] = df['Close'].diff(1).ewm(span=25, adjust=False).mean() / df['Close'].diff(1).abs().ewm(span=13, adjust=False).mean()
    df['TSI_Signal'] = df['TSI'].ewm(span=9, adjust=False).mean()
    # Ultimate Oscillator
    df['Ultimate_Oscillator'] = (4 * (df['Close'] - df['Low']).rolling(window=7).sum() + 2 * (df['Close'] - df['Low']).rolling(window=14).sum() + (df['Close'] - df['Low']).rolling(window=28).sum()) / ((df['High'] - df['Low']).rolling(window=7).sum() + (df['High'] - df['Low']).rolling(window=14).sum() + (df['High'] - df['Low']).rolling(window=28).sum()) * 100
    # Relative Vigor Index (RVI)
    df['Relative_Vigor_Index'] = relative_vigor_index(df['Open'], df['High'], df['Low'], df['Close'])
    df['RVI_Signal'] = df['Relative_Vigor_Index'].ewm(span=14, adjust=False).mean()
    # SMI Ergodic Indicator/Oscillator:
    df['SMI_Ergodic'], df['SMI_Ergodic_Signal'] = smi_ergodic(df['Close'])
    # Fisher Transform
    df['Fisher_Transform'], df['Fisher_Transform_Signal'] = fisher_transform(df['Close'])
    # William %R
    df['Williams_%R'] = williams_r(df['High'], df['Low'], df['Close'])
  


    ## Volume Indicators--------------------------------------------------------------
    # Accumulation/Distribution Line (A/D)
    df['AD'] = (df['Close'] - df['Low'] - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
    # Balance of Power (BOP)
    df['BoP'] = (df['Close'] - df['Open']) / (df['High'] - df['Low'])
    # Chaikin Money Flow (CMF)
    df['CMF'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
    # Chaikin Oscillator
    df['CO'] = df['Close'].diff(3).ewm(span=10, adjust=False).mean()
    # Ease of Movement (EMV)
    df['EMV'] = (df['High'] - df['Low']) / df['Volume']
    # Elder's Force Index (EFI)
    df['EFI'] = df['Close'].diff(1) * df['Volume']
    # Klinger Oscillator
    df['KVO'] = (df['High'] - df['Low']).ewm(span=34, adjust=False).mean() - (df['High'] - df['Low']).ewm(span=55, adjust=False).mean()
    df['KVO_Signal'] = df['KVO'].ewm(span=13, adjust=False).mean()
    # Money Flow Index (MFI)
    df['MFI'] = (df['Close'].diff(1) / df['Close'].shift(1) * df['Volume']).rolling(window=14).mean()
    # Net Volume
    df['Net_Volume'] = df['Volume'] * (df['Close'].diff() / df['Close'].shift(1))
    # On Balance Volume (OBV):
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
    # Price Volume Trend (PVT)
    df['PVT'] = (df['Close'].pct_change(1) * df['Volume']).cumsum()
    # VWAP (Volume Weighted Average Price)
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    # Volume Oscillator
    df['VO'] = df['Volume'].pct_change(12)
    # Vortex Indicator:
    df['Vortex_Pos'] = df['High'].diff(1).abs().rolling(window=14).sum() / atr(df['High'], df['Low'], df['Close'])
    df['Vortex_Neg'] = df['Low'].diff(1).abs().rolling(window=14).sum() / atr(df['High'], df['Low'], df['Close'])
    # Volume
    df['Volume'] = df['Volume']
    # Volume Weighted Moving Average (VWMA)
    df['VWMA'] = ta.vwma(df['Close'], df['Volume'], length=20)
    # Volume Profile Fixed Range (VPFR)
    df['VPFR'] = volume_profile_fixed_range(df, start_idx=0, end_idx=len(df)-1)
    # Volume Profile Visible Range (VPVR)
    df['VPVR'] = volume_profile_visible_range(df, visible_range=100)
    # Spread
    df['Spread'] = df['High'] - df['Low']
    # Elder-Ray Bull Power and Bear Power
    bull_power = df['High'] - ta.ema(df['Close'], length=13)
    bear_power = df['Low'] - ta.ema(df['Close'], length=13)
    df['Elder_Ray_Bull'] = bull_power
    df['Elder_Ray_Bear'] = bear_power
    # Volume profile
    df['Volume_Profile'] = df.groupby(pd.cut(df['Close'], bins=20))['Volume'].transform('sum')
    # Price to Volume
    df['Price_to_Volume'] = df['Close'] / df['Volume']
    # McClellan Oscillator
    df['McClellan_Oscillator'] = df['Close'].ewm(span=19, adjust=False).mean() - df['Close'].ewm(span=39, adjust=False).mean()
    # TRIN
    df['TRIN'] = (df['Close'].rolling(window=14).mean() / df['Volume'].rolling(window=14).mean())
    # Williams Accumulation/Distribution
    wad = ta.ad(df['High'], df['Low'], df['Close'], df['Volume'])
    df['Williams_AD'] = wad
    # Ease of Movement
    distance_moved = ((df['High'] + df['Low']) / 2) - ((df['High'].shift(1) + df['Low'].shift(1)) / 2)
    box_ratio = (df['Volume'] / 1e8) / (df['High'] - df['Low'])
    emv = distance_moved / box_ratio
    df['Ease_of_Movement'] = emv.rolling(window=14).mean()


    ## Volatility Indicators----------------------------------------------------
    # Average True Range (ATR)
    df['ATR'] = atr(df['High'], df['Low'], df['Close'])
    # Bollinger Bands %B:    
    df['BB_%B'] = (df['Close'] - df['BB_Low']) / (df['BB_High'] - df['BB_Low'])
    # Bollinger Bands Width:
    df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['Close']
    # Chaikin Volatility:
    df['Chaikin_Volatility'] = (df['High'] - df['Low']).ewm(span=10, adjust=False).mean()
    # Choppiness Index:
    df['Choppiness_Index'] = np.log10((df['High'] - df['Low']).rolling(window=14).sum() / (df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min())) * 100
    # Historical Volatility
    df['Hist_Vol_Annualized'] = historical_volatility(df)
    # Mass Index:
    df['Mass_Index'] = (df['High'] - df['Low']).rolling(window=25).sum() / (df['High'] - df['Low']).rolling(window=9).sum()
    # Relative Volatility Index (RVI):
    df['RVI'] = df['Close'].rolling(window=10).mean() / df['Close'].rolling(window=10).std()
    # Standard Deviation:
    df['Standard_Deviation'] = df['Close'].rolling(window=20).std()
    # Volatility Close-to-Close
    df['Vol_CtC'] = volatility_close_to_close(df, window=20)
    # Volatility Zero Trend Close-to-Close
    df['Vol_ZtC'] = volatility_zero_trend_close_to_close(df, window=20)
    # Volatility O-H-L-C
    df['Vol_OHLC'] = volatility_ohlc(df, window=20)
    # Volatility Index
    df['Vol_Index'] = volatility_index(df, window=20)
    # Chop Zone
    df['Chop_Zone'] = choppiness_index(df['High'], df['Low'], df['Close'])
    # ZigZag
    df['ZigZag'] = zigzag(df['Close'])
    # Keltner 
    keltner = ta.kc(df['High'], df['Low'], df['Close'])
    df['Keltner_High'] = keltner['KCUe_20_2']
    df['Keltner_Low'] = keltner['KCLe_20_2']


    ## Support and Resistance Indicators---------------------------------------------
    # Williams Fractal
    fractals = williams_fractal(df)
    df['Fractal_Up'] = fractals['Fractal_Up']
    df['Fractal_Down'] = fractals['Fractal_Down']
    # Pivot points standard
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
    # Typical Price
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    # Darvas Box Theory
    df['Darvas_High'] = df['High'].rolling(window=20).max()
    df['Darvas_Low'] = df['Low'].rolling(window=20).min()
    # Fibonacci_levels
    high = df['High'].max()
    low = df['Low'].min()
    diff = high - low
    df['Fib_0.0'] = high
    df['Fib_0.236'] = high - 0.236 * diff
    df['Fib_0.382'] = high - 0.382 * diff
    df['Fib_0.5'] = high - 0.5 * diff
    df['Fib_0.618'] = high - 0.618 * diff
    df['Fib_1.0'] = low


    ## Statistical indicators--------------------------------------------------------
    # Correlation Coefficient:
    df['Correlation_Coefficient'] = correlation_coefficient(df['Close'], df['Close'].shift(1))
    # Correlation - Log
    df['Log_Correlation'] = log_correlation(df['Close'], df['Close'].shift(1))
    # Linear Regression Curve
    df['Linear_Regression_Curve'] = linear_regression_curve(df['Close'])
    # Linear Regression Slope
    df['Linear_Regression_Slope'] = linear_regression_slope(df['Close'])
    # Standard Error:
    df['Standard_Error'] = standard_error(df['Close'])
    # Standard Error Bands:
    df['Standard_Error_Band_Upper'], df['Standard_Error_Band_Lower'] = standard_error_bands(df['Close'])
    # Median Price
    df['Median_Price'] = (df['High'] + df['Low']) / 2


    
    # Simple Moving Average (SMA)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
        # Exponential Moving Average (EMA)
        
    df = df.loc[:, ~df.columns.duplicated()]  # Remove any duplicate columns
    return df

def calculate_scores(df):
    scores = {
        'Trend': 0,
        'Momentum': 0,
        'Volatility': 0,
        'Volume': 0,
        'Support_Resistance': 0
    }
    details = {
        'Trend': "",
        'Momentum': "",
        'Volatility': "",
        'Volume': "",
        'Support_Resistance': ""
    }

    ## Trend Indicators-------------------------------
    trend_score = 0

    # 1. EMA Crossover (5-day, 10-day, 20-day)
    if df['5_day_EMA'].iloc[-1] > df['10_day_EMA'].iloc[-1] > df['20_day_EMA'].iloc[-1]:
        trend_score += 1
        details['Trend'] += "EMA: Strong Bullish; "
    elif df['5_day_EMA'].iloc[-1] > df['10_day_EMA'].iloc[-1]:
        trend_score += 0.5
        details['Trend'] += "EMA: Moderate Bullish; "
    elif df['5_day_EMA'].iloc[-1] < df['10_day_EMA'].iloc[-1] < df['20_day_EMA'].iloc[-1]:
        trend_score += 0
        details['Trend'] += "EMA: Bearish; "

    # 2. ALMA (Arnaud Legoux Moving Average)
    if df['Close'].iloc[-1] > df['ALMA'].iloc[-1]:
        trend_score += 1
        details['Trend'] += "ALMA: Bullish; "
    else:
        trend_score += 0
        details['Trend'] += "ALMA: Bearish; "

    # 3. Aroon
    if df['Aroon_Up'].iloc[-1] > df['Aroon_Down'].iloc[-1]:
        trend_score += 1
        details['Trend'] += "Aroon: Bullish; "
    else:
        trend_score += 0
        details['Trend'] += "Aroon: Bearish; "

    # 4. ADX (Average Directional Index)
    adx = df['ADX'].iloc[-1]
    if adx > 25 and adx > df['ADX'].iloc[-2]:
        trend_score += 1
        details['Trend'] += "ADX: Strong Trend; "
    elif adx > 25:
        trend_score += 0.75
        details['Trend'] += "ADX: Moderate Trend; "
    elif adx > 20:
        trend_score += 0.5
        details['Trend'] += "ADX: Building Trend; "
    else:
        trend_score += 0.25
        details['Trend'] += "ADX: Weak Trend; "

    # 5. Bollinger Bands
    current_price = df['Close'].iloc[-1]
    if current_price > df['BB_High'].iloc[-1]:
        trend_score += 0
        details['Trend'] += "Bollinger Bands: Overbought (Bearish); "
    elif current_price < df['BB_Low'].iloc[-1]:
        trend_score += 1
        details['Trend'] += "Bollinger Bands: Oversold (Bullish); "
    else:
        trend_score += 0.5
        details['Trend'] += "Bollinger Bands: Neutral; "

    # 6. DEMA (Double Exponential Moving Average)
    if current_price > df['DEMA'].iloc[-1]:
        trend_score += 1
        details['Trend'] += "DEMA: Bullish; "
    else:
        trend_score += 0
        details['Trend'] += "DEMA: Bearish; "

    # 7. Envelope Indicator
    if current_price > df['Envelope_High'].iloc[-1]:
        trend_score += 0
        details['Trend'] += "Envelope: Overbought (Bearish); "
    elif current_price < df['Envelope_Low'].iloc[-1]:
        trend_score += 1
        details['Trend'] += "Envelope: Oversold (Bullish); "
    else:
        trend_score += 0.5
        details['Trend'] += "Envelope: Neutral; "

    # 8. GMMA (Guppy Multiple Moving Average)
    if df['GMMA_Short'].iloc[-1] > df['GMMA_Long'].iloc[-1]:
        trend_score += 1
        details['Trend'] += "GMMA: Bullish; "
    else:
        trend_score += 0
        details['Trend'] += "GMMA: Bearish; "

    # 9. Hull Moving Average (HMA)
    if current_price > df['HMA'].iloc[-1]:
        trend_score += 1
        details['Trend'] += "HMA: Bullish; "
    else:
        trend_score += 0
        details['Trend'] += "HMA: Bearish; "

    # 10. Ichimoku Cloud
    if df['Ichimoku_Tenkan'].iloc[-1] > df['Ichimoku_Kijun'].iloc[-1] and current_price > df['Ichimoku_Senkou_Span_A'].iloc[-1] and current_price > df['Ichimoku_Senkou_Span_B'].iloc[-1]:
        trend_score += 1
        details['Trend'] += "Ichimoku: Strong Bullish; "
    elif df['Ichimoku_Tenkan'].iloc[-1] > df['Ichimoku_Kijun'].iloc[-1]:
        trend_score += 0.75
        details['Trend'] += "Ichimoku: Moderate Bullish; "
    elif df['Ichimoku_Tenkan'].iloc[-1] < df['Ichimoku_Kijun'].iloc[-1] and (current_price > df['Ichimoku_Senkou_Span_A'].iloc[-1] or current_price > df['Ichimoku_Senkou_Span_B'].iloc[-1]):
        trend_score += 0.5
        details['Trend'] += "Ichimoku: Neutral; "
    else:
        trend_score += 0.25
        details['Trend'] += "Ichimoku: Bearish; "

    # 11. Keltner Channels
    if current_price > df['KC_High'].iloc[-1]:
        trend_score += 0
        details['Trend'] += "Keltner Channel: Overbought (Bearish); "
    elif current_price < df['KC_Low'].iloc[-1]:
        trend_score += 1
        details['Trend'] += "Keltner Channel: Oversold (Bullish); "
    else:
        trend_score += 0.5
        details['Trend'] += "Keltner Channel: Neutral; "

    # 12. LSMA (Least Squares Moving Average)
    if current_price > df['LSMA'].iloc[-1]:
        trend_score += 1
        details['Trend'] += "LSMA: Bullish; "
    else:
        trend_score += 0
        details['Trend'] += "LSMA: Bearish; "

    # 13. Moving Average Channel (MAC)
    if current_price > df['MAC_Upper'].iloc[-1]:
        trend_score += 0
        details['Trend'] += "Moving Average Channel: Overbought (Bearish); "
    elif current_price < df['MAC_Lower'].iloc[-1]:
        trend_score += 1
        details['Trend'] += "Moving Average Channel: Oversold (Bullish); "
    else:
        trend_score += 0.5
        details['Trend'] += "Moving Average Channel: Neutral; "

    # 14. MACD Histogram
    macd_hist = df['MACD_hist'].iloc[-1]
    if macd_hist > 0 and macd_hist > df['MACD_hist'].iloc[-2]:
        trend_score += 1
        details['Trend'] += "MACD: Strong Bullish; "
    elif macd_hist > 0:
        trend_score += 0.5
        details['Trend'] += "MACD: Moderate Bullish; "
    elif macd_hist < 0 and macd_hist < df['MACD_hist'].iloc[-2]:
        trend_score += 0
        details['Trend'] += "MACD: Strong Bearish; "

    # 15. Parabolic SAR
    psar = df['Parabolic_SAR'].iloc[-1]
    previous_psar = df['Parabolic_SAR'].iloc[-2]

    if current_price > psar and current_price > previous_psar:
        trend_score += 1
        details['Trend'] += "Parabolic SAR: Strong Bullish; "
    elif current_price > psar:
        trend_score += 0.75
        details['Trend'] += "Parabolic SAR: Moderate Bullish; "
    elif current_price < psar and current_price < previous_psar:
        trend_score += 0
        details['Trend'] += "Parabolic SAR: Strong Bearish; "
    else:
        trend_score += 0.25
        details['Trend'] += "Parabolic SAR: Weak Bearish; "

    # 16. Price Channel
    if current_price > df['Price_Channel_Upper'].iloc[-1]:
        trend_score += 1
        details['Trend'] += "Price Channel: Breakout Bullish; "
    elif current_price < df['Price_Channel_Lower'].iloc[-1]:
        trend_score += 0
        details['Trend'] += "Price Channel: Breakout Bearish; "
    else:
        trend_score += 0.5
        details['Trend'] += "Price Channel: Within Channel; "

    # 17. SuperTrend
    if df['SuperTrend'].iloc[-1] < current_price:
        trend_score += 1
        details['Trend'] += "SuperTrend: Strong Bullish; "
    else:
        trend_score += 0
        details['Trend'] += "SuperTrend: Bearish; "

    # 18. Triple EMA (TEMA)
    if df['TEMA_20'].iloc[-1] < current_price:
        trend_score += 1
        details['Trend'] += "TEMA: Bullish; "
    else:
        trend_score += 0
        details['Trend'] += "TEMA: Bearish; "

    # 19. Advance/Decline Line
    if df['Advance_Decline'].iloc[-1] > 0:
        trend_score += 1
        details['Trend'] += "Advance/Decline: Advancing; "
    else:
        trend_score += 0
        details['Trend'] += "Advance/Decline: Declining; "

    # 20. Chande Kroll Stop
    if current_price > df['Chande_Kroll_Stop_Long'].iloc[-1]:
        trend_score += 0
        details['Trend'] += "Chande Kroll Stop: Overbought (Bearish); "
    elif current_price < df['Chande_Kroll_Stop_Short'].iloc[-1]:
        trend_score += 1
        details['Trend'] += "Chande Kroll Stop: Oversold (Bullish); "
    else:
        trend_score += 0.5
        details['Trend'] += "Chande Kroll Stop: Neutral; "

    # 21. Williams Alligator
    if df['Williams_Alligator_Lips'].iloc[-1] > df['Williams_Alligator_Teeth'].iloc[-1] > df['Williams_Alligator_Jaw'].iloc[-1]:
        trend_score += 1
        details['Trend'] += "Williams Alligator: Bullish; "
    elif df['Williams_Alligator_Jaw'].iloc[-1] > df['Williams_Alligator_Teeth'].iloc[-1] > df['Williams_Alligator_Lips'].iloc[-1]:
        trend_score += 0
        details['Trend'] += "Williams Alligator: Bearish; "
    else:
        trend_score += 0.5
        details['Trend'] += "Williams Alligator: Neutral; "

    # 22. Donchian Channels
    if current_price > df['Donchian_High'].iloc[-1]:
        trend_score += 0
        details['Trend'] += "Donchian Channels: Overbought (Bearish); "
    elif current_price < df['Donchian_Low'].iloc[-1]:
        trend_score += 1
        details['Trend'] += "Donchian Channels: Oversold (Bullish); "
    else:
        trend_score += 0.5
        details['Trend'] += "Donchian Channels: Neutral; "

    # Normalize Trend Score
    scores['Trend'] = trend_score / 22

    ## Momentum Indicators----------------------------------------
    momentum_score = 0

    # 1. Awesome Oscillator (AO)
    if df['AO'].iloc[-1] > 0:
        momentum_score += 1
        details['Momentum'] += "AO: Bullish; "
    else:
        momentum_score += 0
        details['Momentum'] += "AO: Bearish; "

    # 2. Accelerator Oscillator (AC)
    if df['AC'].iloc[-1] > 0:
        momentum_score += 1
        details['Momentum'] += "AC: Bullish; "
    else:
        momentum_score += 0
        details['Momentum'] += "AC: Bearish; "

    # 3. Chande Momentum Oscillator (CMO)
    if df['CMO'].iloc[-1] > 50:
        momentum_score += 0
        details['Momentum'] += "CMO: Overbought (Bearish); "
    elif df['CMO'].iloc[-1] < -50:
        momentum_score += 1
        details['Momentum'] += "CMO: Oversold (Bullish); "
    else:
        momentum_score += 0.5
        details['Momentum'] += "CMO: Neutral; "

    # 4. Commodity Channel Index (CCI)
    if df['CCI'].iloc[-1] > 100:
        momentum_score += 0
        details['Momentum'] += "CCI: Overbought (Bearish); "
    elif df['CCI'].iloc[-1] < -100:
        momentum_score += 1
        details['Momentum'] += "CCI: Oversold (Bullish); "
    else:
        momentum_score += 0.5
        details['Momentum'] += "CCI: Neutral; "

    # 5. Connors RSI (CRSI)
    if df['CRSI'].iloc[-1] > 70:
        momentum_score += 0
        details['Momentum'] += "Connors RSI: Overbought (Bearish); "
    elif df['CRSI'].iloc[-1] < 30:
        momentum_score += 1
        details['Momentum'] += "Connors RSI: Oversold (Bullish); "
    else:
        momentum_score += 0.5
        details['Momentum'] += "Connors RSI: Neutral; "

    # 6. Coppock Curve
    if df['Coppock'].iloc[-1] > 0:
        momentum_score += 1
        details['Momentum'] += "Coppock Curve: Bullish; "
    else:
        momentum_score += 0
        details['Momentum'] += "Coppock Curve: Bearish; "

    # 7. Detrended Price Oscillator (DPO)
    if df['DPO'].iloc[-1] > 0:
        momentum_score += 1
        details['Momentum'] += "DPO: Bullish; "
    else:
        momentum_score += 0
        details['Momentum'] += "DPO: Bearish; "

    # 8. Directional Movement Index (DMI)
    plus_di = df['Plus_DI'].iloc[-1]
    minus_di = df['Minus_DI'].iloc[-1]
    adx = df['ADX'].iloc[-1]

    if plus_di > minus_di and adx > 25:
        momentum_score += 1
        details['Momentum'] += "DMI: Bullish; "
    elif plus_di < minus_di and adx > 25:
        momentum_score += 0
        details['Momentum'] += "DMI: Bearish; "
    else:
        momentum_score += 0.5
        details['Momentum'] += "DMI: Neutral; "

    # 9. Know Sure Thing (KST)
    if df['KST'].iloc[-1] > df['KST_Signal'].iloc[-1]:
        momentum_score += 1
        details['Momentum'] += "KST: Bullish; "
    else:
        momentum_score += 0
        details['Momentum'] += "KST: Bearish; "

    # 10. Momentum
    if df['Momentum'].iloc[-1] > 0:
        momentum_score += 1
        details['Momentum'] += "Momentum: Bullish; "
    else:
        momentum_score += 0
        details['Momentum'] += "Momentum: Bearish; "

    # 11. RSI (Relative Strength Index)
    rsi = df['RSI'].iloc[-1]
    if rsi > 70:
        momentum_score += 0
        details['Momentum'] += "RSI: Overbought (Bearish); "
    elif rsi > 60:
        momentum_score += 0.25
        details['Momentum'] += "RSI: Mildly Overbought (Bearish); "
    elif rsi > 50:
        momentum_score += 0.75
        details['Momentum'] += "RSI: Mild Bullish; "
    elif rsi > 40:
        momentum_score += 0.5
        details['Momentum'] += "RSI: Neutral; "
    elif rsi > 30:
        momentum_score += 0.25
        details['Momentum'] += "RSI: Mild Bearish; "
    else:
        momentum_score += 1
        details['Momentum'] += "RSI: Oversold (Bullish); "

    # 12. Rate of Change (ROC)
    if df['ROC'].iloc[-1] > 0:
        momentum_score += 1
        details['Momentum'] += "ROC: Bullish; "
    else:
        momentum_score += 0
        details['Momentum'] += "ROC: Bearish; "

    # 13. Stochastic Oscillator
    stoch_k = df['Stochastic_%K'].iloc[-1]
    stoch_d = df['Stochastic_%D'].iloc[-1]

    if stoch_k > 80:
        momentum_score += 0
        details['Momentum'] += "Stochastic Oscillator: Overbought (Bearish); "
    elif stoch_k < 20:
        momentum_score += 1
        details['Momentum'] += "Stochastic Oscillator: Oversold (Bullish); "
    elif stoch_k > stoch_d:
        momentum_score += 0.75
        details['Momentum'] += "Stochastic Oscillator: Bullish; "
    else:
        momentum_score += 0.25
        details['Momentum'] += "Stochastic Oscillator: Bearish; "

    # 14. Stochastic RSI
    if df['Stochastic_RSI'].iloc[-1] > 0.8:
        momentum_score += 0
        details['Momentum'] += "Stochastic RSI: Overbought (Bearish); "
    elif df['Stochastic_RSI'].iloc[-1] < 0.2:
        momentum_score += 1
        details['Momentum'] += "Stochastic RSI: Oversold (Bullish); "
    else:
        momentum_score += 0.5
        details['Momentum'] += "Stochastic RSI: Neutral; "

    # 15. TRIX
    if df['TRIX'].iloc[-1] > df['TRIX_Signal'].iloc[-1]:
        momentum_score += 1
        details['Momentum'] += "TRIX: Bullish; "
    else:
        momentum_score += 0
        details['Momentum'] += "TRIX: Bearish; "

    # 16. True Strength Index (TSI)
    if df['TSI'].iloc[-1] > df['TSI_Signal'].iloc[-1]:
        momentum_score += 1
        details['Momentum'] += "TSI: Bullish; "
    else:
        momentum_score += 0
        details['Momentum'] += "TSI: Bearish; "

    # 17. Ultimate Oscillator
    if df['Ultimate_Oscillator'].iloc[-1] > 70:
        momentum_score += 0
        details['Momentum'] += "Ultimate Oscillator: Overbought (Bearish); "
    elif df['Ultimate_Oscillator'].iloc[-1] < 30:
        momentum_score += 1
        details['Momentum'] += "Ultimate Oscillator: Oversold (Bullish); "
    else:
        momentum_score += 0.5
        details['Momentum'] += "Ultimate Oscillator: Neutral; "

    # 18. Relative Vigor Index (RVI)
    if df['Relative_Vigor_Index'].iloc[-1] > df['RVI_Signal'].iloc[-1]:
        momentum_score += 1
        details['Momentum'] += "RVI: Bullish; "
    else:
        momentum_score += 0
        details['Momentum'] += "RVI: Bearish; "

    # 19. SMI Ergodic Indicator/Oscillator
    if df['SMI_Ergodic'].iloc[-1] > df['SMI_Ergodic_Signal'].iloc[-1]:
        momentum_score += 1
        details['Momentum'] += "SMI Ergodic: Bullish; "
    else:
        momentum_score += 0
        details['Momentum'] += "SMI Ergodic: Bearish; "

    # 20. Fisher Transform
    if df['Fisher_Transform'].iloc[-1] > df['Fisher_Transform_Signal'].iloc[-1]:
        momentum_score += 1
        details['Momentum'] += "Fisher Transform: Bullish; "
    else:
        momentum_score += 0
        details['Momentum'] += "Fisher Transform: Bearish; "

    # 21. Williams %R
    williams_r = df['Williams_%R'].iloc[-1]
    if williams_r > -20:
        momentum_score += 0
        details['Momentum'] += "Williams %R: Overbought (Bearish); "
    elif williams_r < -80:
        momentum_score += 1
        details['Momentum'] += "Williams %R: Oversold (Bullish); "
    else:
        momentum_score += 0.5
        details['Momentum'] += "Williams %R: Neutral; "

    # Normalize Momentum Score
    scores['Momentum'] = momentum_score / 21

    # Volatility Indicators--------------------------------------------------------
    volatility_score = 0

    # 1. Average True Range (ATR)
    atr = df['ATR'].iloc[-1]
    atr_mean = df['ATR'].rolling(window=14).mean().iloc[-1]
    if atr > 1.5 * atr_mean:
        volatility_score += 1
        details['Volatility'] += "ATR: High Volatility; "
    elif atr < 0.5 * atr_mean:
        volatility_score += 0
        details['Volatility'] += "ATR: Low Volatility; "
    else:
        volatility_score += 0.5
        details['Volatility'] += "ATR: Moderate Volatility; "

    # 2. Bollinger Bands %B
    if df['BB_%B'].iloc[-1] > 1:
        volatility_score += 0
        details['Volatility'] += "Bollinger Bands %B: Overbought (Bearish); "
    elif df['BB_%B'].iloc[-1] < 0:
        volatility_score += 1
        details['Volatility'] += "Bollinger Bands %B: Oversold (Bullish); "
    else:
        volatility_score += 0.5
        details['Volatility'] += "Bollinger Bands %B: Neutral; "

    # 3. Bollinger Bands Width
    bb_width = df['BB_Width'].iloc[-1]
    bb_width_mean = df['BB_Width'].rolling(window=14).mean().iloc[-1]
    if bb_width > 1.5 * bb_width_mean:
        volatility_score += 1
        details['Volatility'] += "Bollinger Bands Width: Expanding (High Volatility); "
    elif bb_width < 0.5 * bb_width_mean:
        volatility_score += 0
        details['Volatility'] += "Bollinger Bands Width: Contracting (Low Volatility); "
    else:
        volatility_score += 0.5
        details['Volatility'] += "Bollinger Bands Width: Moderate Volatility; "

    # 4. Chaikin Volatility
    chaikin_volatility = df['Chaikin_Volatility'].iloc[-1]
    chaikin_volatility_mean = df['Chaikin_Volatility'].rolling(window=14).mean().iloc[-1]
    if chaikin_volatility > 1.5 * chaikin_volatility_mean:
        volatility_score += 1
        details['Volatility'] += "Chaikin Volatility: High Volatility; "
    elif chaikin_volatility < 0.5 * chaikin_volatility_mean:
        volatility_score += 0
        details['Volatility'] += "Chaikin Volatility: Low Volatility; "
    else:
        volatility_score += 0.5
        details['Volatility'] += "Chaikin Volatility: Moderate Volatility; "

    # 5. Choppiness Index
    choppiness_index = df['Choppiness_Index'].iloc[-1]
    if choppiness_index > 61.8:
        volatility_score += 1
        details['Volatility'] += "Choppiness Index: High Volatility (Choppy Market); "
    elif choppiness_index < 38.2:
        volatility_score += 0
        details['Volatility'] += "Choppiness Index: Low Volatility (Trending Market); "
    else:
        volatility_score += 0.5
        details['Volatility'] += "Choppiness Index: Neutral; "

    # 6. Historical Volatility
    hist_vol = df['Hist_Vol_Annualized'].iloc[-1]
    hist_vol_mean = df['Hist_Vol_Annualized'].rolling(window=252).mean().iloc[-1]
    if hist_vol > 1.5 * hist_vol_mean:
        volatility_score += 1
        details['Volatility'] += "Historical Volatility: High Volatility; "
    elif hist_vol < 0.5 * hist_vol_mean:
        volatility_score += 0
        details['Volatility'] += "Historical Volatility: Low Volatility; "
    else:
        volatility_score += 0.5
        details['Volatility'] += "Historical Volatility: Moderate Volatility; "

    # 7. Mass Index
    if df['Mass_Index'].iloc[-1] > 27:
        volatility_score += 1
        details['Volatility'] += "Mass Index: High Volatility (Potential Reversal); "
    elif df['Mass_Index'].iloc[-1] < 26.5:
        volatility_score += 0
        details['Volatility'] += "Mass Index: Low Volatility; "
    else:
        volatility_score += 0.5
        details['Volatility'] += "Mass Index: Neutral; "

    # 8. Relative Volatility Index (RVI)
    if df['RVI'].iloc[-1] > 60:
        volatility_score += 1
        details['Volatility'] += "Relative Volatility Index: High Volatility; "
    elif df['RVI'].iloc[-1] < 40:
        volatility_score += 0
        details['Volatility'] += "Relative Volatility Index: Low Volatility; "
    else:
        volatility_score += 0.5
        details['Volatility'] += "Relative Volatility Index: Neutral; "

    # 9. Standard Deviation
    std_dev = df['Standard_Deviation'].iloc[-1]
    std_dev_mean = df['Standard_Deviation'].rolling(window=20).mean().iloc[-1]
    if std_dev > 1.5 * std_dev_mean:
        volatility_score += 1
        details['Volatility'] += "Standard Deviation: High Volatility; "
    elif std_dev < 0.5 * std_dev_mean:
        volatility_score += 0
        details['Volatility'] += "Standard Deviation: Low Volatility; "
    else:
        volatility_score += 0.5
        details['Volatility'] += "Standard Deviation: Moderate Volatility; "

    # 10. Volatility Close-to-Close
    vol_ctc = df['Vol_CtC'].iloc[-1]
    vol_ctc_mean = df['Vol_CtC'].rolling(window=20).mean().iloc[-1]
    if vol_ctc > 1.5 * vol_ctc_mean:
        volatility_score += 1
        details['Volatility'] += "Volatility Close-to-Close: High Volatility; "
    elif vol_ctc < 0.5 * vol_ctc_mean:
        volatility_score += 0
        details['Volatility'] += "Volatility Close-to-Close: Low Volatility; "
    else:
        volatility_score += 0.5
        details['Volatility'] += "Volatility Close-to-Close: Moderate Volatility; "

    # 11. Volatility Zero Trend Close-to-Close
    vol_ztc = df['Vol_ZtC'].iloc[-1]
    vol_ztc_mean = df['Vol_ZtC'].rolling(window=20).mean().iloc[-1]
    if vol_ztc > 1.5 * vol_ztc_mean:
        volatility_score += 1
        details['Volatility'] += "Volatility Zero Trend Close-to-Close: High Volatility; "
    elif vol_ztc < 0.5 * vol_ztc_mean:
        volatility_score += 0
        details['Volatility'] += "Volatility Zero Trend Close-to-Close: Low Volatility; "
    else:
        volatility_score += 0.5
        details['Volatility'] += "Volatility Zero Trend Close-to-Close: Moderate Volatility; "

    # 12. Volatility O-H-L-C
    vol_ohlc = df['Vol_OHLC'].iloc[-1]
    vol_ohlc_mean = df['Vol_OHLC'].rolling(window=20).mean().iloc[-1]
    if vol_ohlc > 1.5 * vol_ohlc_mean:
        volatility_score += 1
        details['Volatility'] += "Volatility O-H-L-C: High Volatility; "
    elif vol_ohlc < 0.5 * vol_ohlc_mean:
        volatility_score += 0
        details['Volatility'] += "Volatility O-H-L-C: Low Volatility; "
    else:
        volatility_score += 0.5
        details['Volatility'] += "Volatility O-H-L-C: Moderate Volatility; "

    # 13. Volatility Index
    vol_index = df['Vol_Index'].iloc[-1]
    vol_index_mean = df['Vol_Index'].rolling(window=20).mean().iloc[-1]
    if vol_index > 1.5 * vol_index_mean:
        volatility_score += 1
        details['Volatility'] += "Volatility Index: High Volatility; "
    elif vol_index < 0.5 * vol_index_mean:
        volatility_score += 0
        details['Volatility'] += "Volatility Index: Low Volatility; "
    else:
        volatility_score += 0.5
        details['Volatility'] += "Volatility Index: Moderate Volatility; "

    # 14. Chop Zone
    chop_zone = df['Chop_Zone'].iloc[-1]
    if chop_zone > 61.8:
        volatility_score += 1
        details['Volatility'] += "Chop Zone: High Volatility (Choppy Market); "
    elif chop_zone < 38.2:
        volatility_score += 0
        details['Volatility'] += "Chop Zone: Low Volatility (Trending Market); "
    else:
        volatility_score += 0.5
        details['Volatility'] += "Chop Zone: Neutral; "

    # 15. ZigZag
    zigzag_change = df['ZigZag'].diff().abs().max()
    if zigzag_change > 0.1:
        volatility_score += 1
        details['Volatility'] += "ZigZag: High Volatility; "
    else:
        volatility_score += 0
        details['Volatility'] += "ZigZag: Low Volatility; "

    # 16. Keltner Channels
    if current_price > df['Keltner_High'].iloc[-1]:
        volatility_score += 0
        details['Volatility'] += "Keltner Channels: Overbought (Bearish); "
    elif current_price < df['Keltner_Low'].iloc[-1]:
        volatility_score += 1
        details['Volatility'] += "Keltner Channels: Oversold (Bullish); "
    else:
        volatility_score += 0.5
        details['Volatility'] += "Keltner Channels: Neutral; "

    # Normalize Volatility Score
    scores['Volatility'] = volatility_score / 16

    ## Volume Indicators-------------------------------
    volume_score = 0

    # 1. Accumulation/Distribution Line (A/D)
    ad_line = df['AD'].iloc[-1]
    if ad_line > df['AD'].shift(1).iloc[-1]:
        volume_score += 1
        details['Volume'] += "A/D Line: Increasing; "
    elif ad_line < df['AD'].shift(1).iloc[-1]:
        volume_score += 0
        details['Volume'] += "A/D Line: Decreasing; "
    else:
        volume_score += 0.5
        details['Volume'] += "A/D Line: Flat; "

    # 2. Balance of Power (BOP)
    if df['BoP'].iloc[-1] > 0:
        volume_score += 1
        details['Volume'] += "BOP: Bullish; "
    else:
        volume_score += 0.5
        details['Volume'] += "BOP: Bearish; "

    # 3. Chaikin Money Flow (CMF)
    cmf = df['CMF'].iloc[-1]
    if cmf > 0 and cmf > df['CMF'].shift(1).iloc[-1]:
        volume_score += 1
        details['Volume'] += "CMF: Increasing; "
    elif cmf > 0:
        volume_score += 0.5
        details['Volume'] += "CMF: Positive but decreasing; "
    else:
        volume_score += 0
        details['Volume'] += "CMF: Decreasing; "

    # 4. Chaikin Oscillator
    chaikin_oscillator = df['CO'].iloc[-1]
    previous_chaikin_oscillator = df['CO'].shift(1).iloc[-1]
    if chaikin_oscillator > previous_chaikin_oscillator and chaikin_oscillator > 0:
        volume_score += 1
        details['Volume'] += "Chaikin Oscillator: Increasing and positive; "
    elif chaikin_oscillator > previous_chaikin_oscillator and chaikin_oscillator < 0:
        volume_score += 0.75
        details['Volume'] += "Chaikin Oscillator: Increasing but negative; "
    elif chaikin_oscillator < previous_chaikin_oscillator and chaikin_oscillator > 0:
        volume_score += 0.5
        details['Volume'] += "Chaikin Oscillator: Decreasing but positive; "
    else:
        volume_score += 0
        details['Volume'] += "Chaikin Oscillator: Decreasing and negative; "

    # 5. Ease of Movement (EMV)
    emv = df['EMV'].iloc[-1]
    previous_emv = df['EMV'].shift(1).iloc[-1]
    if emv > previous_emv and emv > 0:
        volume_score += 1
        details['Volume'] += "EMV: Increasing and positive; "
    elif emv < previous_emv and emv < 0:
        volume_score += 0
        details['Volume'] += "EMV: Decreasing and negative; "
    else:
        volume_score += 0.5
        details['Volume'] += "EMV: Flat; "

    # 6. Elder's Force Index (EFI)
    efi = df['EFI'].iloc[-1]
    if efi > df['EFI'].shift(1).iloc[-1]:
        volume_score += 1
        details['Volume'] += "EFI: Increasing; "
    elif efi < df['EFI'].shift(1).iloc[-1]:
        volume_score += 0
        details['Volume'] += "EFI: Decreasing; "
    else:
        volume_score += 0.5
        details['Volume'] += "EFI: Flat; "

    # 7. Klinger Oscillator
    kvo = df['KVO'].iloc[-1]
    kvo_signal = df['KVO_Signal'].iloc[-1]
    if kvo > kvo_signal and kvo > 0:
        volume_score += 1
        details['Volume'] += "KVO: Bullish crossover with positive money flow; "
    elif kvo > kvo_signal and kvo < 0:
        volume_score += 0.75
        details['Volume'] += "KVO: Bullish crossover but negative money flow; "
    elif kvo < kvo_signal and kvo > 0:
        volume_score += 0.5
        details['Volume'] += "KVO: Bearish crossover but positive money flow; "
    else:
        volume_score += 0
        details['Volume'] += "KVO: Bearish crossover with negative money flow; "

    # 8. Money Flow Index (MFI)
    mfi = df['MFI'].iloc[-1]
    if mfi > 80:
        volume_score += 0
        details['Volume'] += "MFI: Overbought; "
    elif mfi < 20:
        volume_score += 1
        details['Volume'] += "MFI: Oversold; "
    else:
        volume_score += 0.5
        details['Volume'] += "MFI: Neutral; "

    # 9. Net Volume
    net_volume = df['Net_Volume'].iloc[-1]
    previous_net_volume = df['Net_Volume'].shift(1).iloc[-1]
    if net_volume > previous_net_volume:
        volume_score += 1
        details['Volume'] += "Net Volume: Increasing sharply; "
    elif net_volume < previous_net_volume:
        volume_score += 0
        details['Volume'] += "Net Volume: Decreasing; "
    else:
        volume_score += 0.5
        details['Volume'] += "Net Volume: Flat; "

    # 10. On Balance Volume (OBV)
    obv = df['OBV'].iloc[-1]
    previous_obv = df['OBV'].shift(1).iloc[-1]
    if obv > previous_obv:
        volume_score += 1
        details['Volume'] += "OBV: Increasing sharply; "
    elif obv < previous_obv:
        volume_score += 0
        details['Volume'] += "OBV: Decreasing; "
    else:
        volume_score += 0.5
        details['Volume'] += "OBV: Flat; "

    # 11. Price Volume Trend (PVT)
    pvt = df['PVT'].iloc[-1]
    previous_pvt = df['PVT'].shift(1).iloc[-1]
    if pvt > previous_pvt:
        volume_score += 1
        details['Volume'] += "PVT: Increasing sharply; "
    elif pvt < previous_pvt:
        volume_score += 0
        details['Volume'] += "PVT: Decreasing; "
    else:
        volume_score += 0.5
        details['Volume'] += "PVT: Flat; "

    # 12. VWAP (Volume Weighted Average Price)
    if current_price > df['VWAP'].iloc[-1]:
        volume_score += 1
        details['Volume'] += "VWAP: Above VWAP (Bullish); "
    else:
        volume_score += 0
        details['Volume'] += "VWAP: Below VWAP (Bearish); "

    # 13. VWMA (Volume Weighted Moving Average)
    if current_price > df['VWMA'].iloc[-1]:
        volume_score += 1
        details['Volume'] += "VWMA: Above VWMA (Bullish); "
    else:
        volume_score += 0
        details['Volume'] += "VWMA: Below VWMA (Bearish); "

    # 14. Volume Oscillator
    volume_osc = df['VO'].iloc[-1]
    previous_volume_osc = df['VO'].shift(1).iloc[-1]
    if volume_osc > previous_volume_osc and volume_osc > 0:
        volume_score += 1
        details['Volume'] += "Volume Oscillator: Increasing sharply; "
    elif volume_osc < previous_volume_osc:
        volume_score += 0
        details['Volume'] += "Volume Oscillator: Decreasing; "
    else:
        volume_score += 0.5
        details['Volume'] += "Volume Oscillator: Flat; "

    # 15. Volume Profile Fixed Range (VPFR)
    if df['VPFR'].iloc[-1] > df['VPFR'].shift(1).iloc[-1]:
        volume_score += 1
        details['Volume'] += "VPFR: Increasing sharply; "
    else:
        volume_score += 0
        details['Volume'] += "VPFR: Decreasing; "

    # 16. Volume Profile Visible Range (VPVR)
    if df['VPVR'].iloc[-1] > df['VPVR'].shift(1).iloc[-1]:
        volume_score += 1
        details['Volume'] += "VPVR: Increasing sharply; "
    else:
        volume_score += 0
        details['Volume'] += "VPVR: Decreasing; "

    # 17. Vortex Indicator
    if df['Vortex_Pos'].iloc[-1] > df['Vortex_Neg'].iloc[-1]:
        volume_score += 1
        details['Volume'] += "Vortex: Bullish trend; "
    else:
        volume_score += 0
        details['Volume'] += "Vortex: Bearish trend; "

    # 18. Williams Accumulation/Distribution (WAD)
    if df['Williams_AD'].iloc[-1] > df['Williams_AD'].shift(1).iloc[-1]:
        volume_score += 1
        details['Volume'] += "WAD: Increasing sharply; "
    else:
        volume_score += 0
        details['Volume'] += "WAD: Decreasing; "

    # Normalize Volume Score
    scores['Volume'] = volume_score / 18

    ## Support/Resistance Indicators-------------------------------
    support_resistance_score = 0

    # 1. Williams Fractal
    if df['Fractal_Up'].iloc[-1] > 0:
        support_resistance_score += 0.25
        details['Support_Resistance'] += "Fractal: Potential Resistance detected; "
    elif df['Fractal_Down'].iloc[-1] > 0:
        support_resistance_score += 0.75
        details['Support_Resistance'] += "Fractal: Potential Support detected; "
    else:
        support_resistance_score += 0.5
        details['Support_Resistance'] += "Fractal: No significant support/resistance detected; "

    # 2. Pivot Points
    price = df['Close'].iloc[-1]
    if price > df['Resistance_3'].iloc[-1]:
        support_resistance_score += 0
        details['Support_Resistance'] += "Pivot Points: Price significantly above Resistance 3 (Overbought); "
    elif price > df['Resistance_2'].iloc[-1]:
        support_resistance_score += 0.25
        details['Support_Resistance'] += "Pivot Points: Price above Resistance 2 (Mildly Overbought); "
    elif price > df['Resistance_1'].iloc[-1]:
        support_resistance_score += 0.5
        details['Support_Resistance'] += "Pivot Points: Price above Resistance 1 (Moderately Bullish); "
    elif price < df['Support_3'].iloc[-1]:
        support_resistance_score += 1
        details['Support_Resistance'] += "Pivot Points: Price significantly below Support 3 (Oversold); "
    elif price < df['Support_2'].iloc[-1]:
        support_resistance_score += 0.75
        details['Support_Resistance'] += "Pivot Points: Price below Support 2 (Mildly Oversold); "
    elif price < df['Support_1'].iloc[-1]:
        support_resistance_score += 0.5
        details['Support_Resistance'] += "Pivot Points: Price below Support 1 (Mildly Bullish); "
    else:
        support_resistance_score += 0.25
        details['Support_Resistance'] += "Pivot Points: Price within range of Pivot (Neutral to Slightly Bullish); "

    # 3. Typical Price
    if price < df['Typical_Price'].iloc[-1]:
        support_resistance_score += 0.75
        details['Support_Resistance'] += "Typical Price: Price below typical price (Support); "
    elif price > df['Typical_Price'].iloc[-1]:
        support_resistance_score += 0.25
        details['Support_Resistance'] += "Typical Price: Price above typical price (Resistance); "
    else:
        support_resistance_score += 0.5
        details['Support_Resistance'] += "Typical Price: Price at typical price (Neutral); "

    # 4. Darvas Box Theory
    if price > df['Darvas_High'].iloc[-1]:
        support_resistance_score += 0.75
        details['Support_Resistance'] += "Darvas Box: Price breaking out above Darvas high (Bullish); "
    elif price < df['Darvas_Low'].iloc[-1]:
        support_resistance_score += 0.25
        details['Support_Resistance'] += "Darvas Box: Price breaking down below Darvas low (Bearish); "
    else:
        support_resistance_score += 0.5
        details['Support_Resistance'] += "Darvas Box: Price within Darvas range (Neutral); "

    # 5. Fibonacci Levels
    if price > df['Fib_0.618'].iloc[-1]:
        support_resistance_score += 1
        details['Support_Resistance'] += "Fibonacci: Price above 61.8% retracement (Strong Support); "
    elif price > df['Fib_0.382'].iloc[-1]:
        support_resistance_score += 0.75
        details['Support_Resistance'] += "Fibonacci: Price between 38.2% and 61.8% retracement (Moderate Support); "
    elif price > df['Fib_0.236'].iloc[-1]:
        support_resistance_score += 0.5
        details['Support_Resistance'] += "Fibonacci: Price between 23.6% and 38.2% retracement (Weak Support); "
    else:
        support_resistance_score += 0
        details['Support_Resistance'] += "Fibonacci: Price below 23.6% retracement (Weak Resistance); "

    # Normalize Support/Resistance Score
    scores['Support_Resistance'] = support_resistance_score / 5

    return scores, details



def get_recommendation(overall_score):
    if overall_score >= 0.8:
        return "Strong Buy"
    elif 0.6 <= overall_score < 0.8:
        return "Buy"
    elif 0.4 <= overall_score < 0.6:
        return "Hold"
    elif 0.2 <= overall_score < 0.4:
        return "Sell"
    else:
        return "Strong Sell"

def create_gauge(value, title, width=300, height=300):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': title, 'font': {'size': 24}},
        delta={'reference': 0.5, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.2], 'color': "red"},
                {'range': [0.2, 0.4], 'color': "orange"},
                {'range': [0.4, 0.6], 'color': "yellow"},
                {'range': [0.6, 0.8], 'color': "lightgreen"},
                {'range': [0.8, 1], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(width=width, height=height)
    return fig

def create_indicator_table(details, key):
    split_details = [item.split(": ") for item in details[key].split("; ")[:-1]]
    detail_table = pd.DataFrame(split_details, columns=['Indicator', 'Status'])
    return detail_table

def create_combined_chart(data, group_name, indicators, ticker, use_candlestick):
    selected_indicators = st.multiselect(f'Select {group_name} Indicators', indicators, default=[indicators[0]] if indicators else [])

    if selected_indicators:
        # Use Plotly's built-in color scale for better visualization
        color_scale = px.colors.qualitative.Plotly

        fig = make_subplots(
            rows=1, cols=1, shared_xaxes=True, 
            specs=[[{"secondary_y": True}]],
            subplot_titles=(f"{ticker} - {group_name} Indicators",)
        )

        # Display either candlestick or line chart based on user selection
        if use_candlestick:
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Candlestick',
                increasing_line_color='green', decreasing_line_color='red'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['Close'], mode='lines', 
                name='Close Price', hoverinfo='x+y', line=dict(color='blue', width=2)
            ))

        for i, indicator in enumerate(selected_indicators):
            if indicator in data.columns:
                color = color_scale[i % len(color_scale)]
                if indicator == 'MACD_hist':
                    colors = get_macd_hist_colors(data[indicator])
                    fig.add_trace(go.Bar(
                        x=data.index[1:], y=data[indicator][1:], 
                        name=indicator, marker_color=colors, hoverinfo='x+y'
                    ), secondary_y=True)
                else:
                    secondary_y = True if group_name == "Momentum" or "Volume" or "Volatility" else False
                    fig.add_trace(go.Scatter(
                        x=data.index, y=data[indicator], mode='lines', 
                        name=indicator, hoverinfo='x+y', line=dict(width=2, color=color)
                    ), secondary_y=secondary_y)

        # Add Bollinger Bands
        if 'BB_High' in indicators and 'BB_Low' in indicators:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['BB_High'], mode='lines', 
                name='BB High', line=dict(color='rgba(255, 0, 255, 0.1)', width=1)
            ))
            fig.add_trace(go.Scatter(
                x=data.index, y=data['BB_Low'], mode='lines', 
                name='BB Low', line=dict(color='rgba(255, 0, 255, 0.1)', width=1), fill='tonexty'
            ))

        # Add annotations for key indicators
        fig.add_annotation(
            x=data.index[-1], y=data['Close'].iloc[-1],
            text="Current Price", showarrow=True, arrowhead=1
        )

        # Enhance layout and interaction features
        fig.update_layout(
            autosize=True,
            height=500,
            margin=dict(t=80, b=80, l=60, r=60),
            yaxis=dict(title='Price', side='left', fixedrange=False, automargin=True),
            yaxis2=dict(title=f'{group_name} Indicator', side='right', overlaying='y', showgrid=False, fixedrange=False, automargin=True),
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
                type='date',
                automargin=True,
            ),
            legend=dict(x=0.5, y=-0.4, orientation='h', xanchor='center', yanchor='top'),
            hovermode='x unified',
            hoverlabel=dict(bgcolor="sky blue", font_size=12, font_family="Rockwell"),
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write(f"No indicators selected for {group_name}.")



def get_macd_hist_colors(macd_hist):
    colors = []
    for i in range(1, len(macd_hist)):
        if macd_hist.iloc[i] > 0:
            color = 'green' if macd_hist.iloc[i] > macd_hist.iloc[i - 1] else 'lightgreen'
        else:
            color = 'red' if macd_hist.iloc[i] < macd_hist.iloc[i - 1] else 'lightcoral'
        colors.append(color)
    return colors




def get_available_sources():
    try:
        sources = newsapi.get_sources(language='en')
        available_sources = [source['id'] for source in sources['sources']]
        return available_sources
    except Exception as e:
        st.error(f"An error occurred while fetching sources: {e}")
        return []

def fetch_news(company_name, start_date, end_date, selected_sources=None):
    try:
        if selected_sources:
            all_articles = newsapi.get_everything(
                q=company_name,
                language='en',
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                sort_by='publishedAt',
                page_size=50,
                sources=','.join(selected_sources)
            )
        else:
            all_articles = newsapi.get_everything(
                q=company_name,
                language='en',
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                sort_by='publishedAt',
                page_size=50
            )
        
        articles = []
        if 'articles' in all_articles:
            for article in all_articles['articles']:
                articles.append({
                    'title': article['title'],
                    'description': article.get('description', ''),
                    'url': article['url'],
                    'publishedAt': article['publishedAt'],
                    'source': article['source']['name']
                })
        return articles
    except Exception as e:
        st.error(f"An error occurred while fetching news: {e}")
        return []

def perform_sentiment_analysis(articles):
    sentiments = []
    for article in articles:
        if article['description']:
            description_score = analyzer.polarity_scores(article['description'])
            title_score = analyzer.polarity_scores(article['title'])
            
            # Weighted average of title and description sentiments
            combined_score = {
                'compound': (description_score['compound'] * 0.7 + title_score['compound'] * 0.3),
                'neg': (description_score['neg'] * 0.7 + title_score['neg'] * 0.3),
                'neu': (description_score['neu'] * 0.7 + title_score['neu'] * 0.3),
                'pos': (description_score['pos'] * 0.7 + title_score['pos'] * 0.3)
            }
            
            article['sentiment'] = combined_score
            article['length'] = len(article['description']) + len(article['title'])  # Add article length
            sentiments.append(article)
        else:
            article['sentiment'] = {'compound': 0, 'neg': 0, 'neu': 0, 'pos': 0}
            article['length'] = len(article['title'])
            sentiments.append(article)
    return sentiments

def make_recommendation(sentiments):
    current_date = datetime.now().date()
    
    # Calculate weighted average sentiment
    total_weighted_score = 0
    total_weight = 0
    
    for article in sentiments:
        days_old = (current_date - pd.to_datetime(article['publishedAt']).date()).days + 1
        weight = (1 / days_old) * article['length']  # Weight by recency and length
        total_weighted_score += article['sentiment']['compound'] * weight
        total_weight += weight
    
    weighted_avg_sentiment = total_weighted_score / total_weight if total_weight > 0 else 0

    if weighted_avg_sentiment > 0.15:
        return f"Based on the weighted sentiment analysis, it is recommended to BUY the stock. (Score: {weighted_avg_sentiment:.2f})"
    elif weighted_avg_sentiment < -0.15:
        return f"Based on the weighted sentiment analysis, it is recommended to SELL the stock. (Score: {weighted_avg_sentiment:.2f})"
    else:
        return f"Based on the weighted sentiment analysis, it is recommended to HOLD the stock. (Score: {weighted_avg_sentiment:.2f})"

def count_sentiments(sentiments):
    positive = sum(1 for s in sentiments if s['sentiment']['compound'] > 0.15)
    negative = sum(1 for s in sentiments if s['sentiment']['compound'] < -0.15)
    neutral = len(sentiments) - positive - negative
    return positive, negative, neutral

def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    st.pyplot(plt)

def plot_sentiment_histogram(df):
    fig_hist = px.histogram(df, x=df['sentiment'].apply(lambda x: x['compound']),
                            nbins=20, labels={'x': 'Sentiment Score'},
                            title='Sentiment Score Distribution')
    st.plotly_chart(fig_hist)

def plot_sentiment_over_time(df):
    df['publishedAt'] = pd.to_datetime(df['publishedAt']).dt.tz_localize(None)
    df = df.sort_values(by='publishedAt')

    fig_time = px.scatter(df, x='publishedAt', y=df['sentiment'].apply(lambda x: x['compound']),
                          color=df['sentiment'].apply(lambda x: 'Positive' if x['compound'] > 0 else ('Negative' if x['compound'] < 0 else 'Neutral')),
                          labels={'x': 'Date', 'y': 'Sentiment Score'},
                          title='Sentiment Score Over Time')
    st.plotly_chart(fig_time)

def sentiment_analysis_section(ticker, start_date, end_date):
    st.subheader(f"Sentiment Analysis for {ticker}")

    # Fetch company name using yfinance
    try:
        company_info = yf.Ticker(ticker)
        company_name = company_info.info['shortName']
    except KeyError:
        company_name = ticker

    st.write(f"Fetching news for: {company_name}")

    if company_name:
        # Fetch available news sources
        available_sources = get_available_sources()
        selected_sources = st.multiselect('Select News Sources', available_sources)

        with st.spinner("Fetching news..."):
            articles = fetch_news(company_name, end_date - timedelta(days=29), end_date, selected_sources)

        if articles:
            with st.spinner("Performing sentiment analysis..."):
                sentiments = perform_sentiment_analysis(articles)

            df = pd.DataFrame(sentiments)

            # Display Articles or Summary
            view_option = st.sidebar.radio("View Options", ["Articles", "Summary"])

            if view_option == "Articles":
                st.write("Recent News Articles:")
                for i, row in df.iterrows():
                    st.write(f"**Article {i+1}:** {row['title']}")
                    st.write(f"**Published At:** {row['publishedAt']}")
                    st.write(f"**Source:** {row['source']}")
                    st.write(f"**Description:** {row['description']}")
                    st.write(f"**URL:** [Read more]({row['url']})")
                    st.write(f"**Sentiment Score:** {row['sentiment']['compound']:.2f}")
                    st.write("---")

            elif view_option == "Summary":
                st.subheader("Sentiment Analysis Summary:")
                avg_sentiment = sum(df['sentiment'].apply(lambda x: x['compound'])) / len(df)
                st.write(f"Average Sentiment Score: {avg_sentiment:.2f}")

                positive, negative, neutral = count_sentiments(sentiments)
                st.write(f"Positive Articles: {positive}")
                st.write(f"Negative Articles: {negative}")
                st.write(f"Neutral Articles: {neutral}")

                # Sentiment distribution histogram
                plot_sentiment_histogram(df)

                # Sentiment over time graph
                plot_sentiment_over_time(df)

                # Sentiment distribution pie chart
                sentiment_counts = pd.Series([positive, negative, neutral], index=['Positive', 'Negative', 'Neutral'])
                fig_pie = px.pie(sentiment_counts, values=sentiment_counts, names=sentiment_counts.index, title='Sentiment Distribution')
                st.plotly_chart(fig_pie)

                # Generate and display word clouds for positive and negative articles
                positive_text = ' '.join(df[df['sentiment'].apply(lambda x: x['compound']) > 0.15]['description'].dropna())
                negative_text = ' '.join(df[df['sentiment'].apply(lambda x: x['compound']) < -0.15]['description'].dropna())
                if positive_text:
                    st.write("Positive Articles Word Cloud:")
                    generate_wordcloud(positive_text, 'Positive Articles')
                if negative_text:
                    st.write("Negative Articles Word Cloud:")
                    generate_wordcloud(negative_text, 'Negative Articles')

                recommendation = make_recommendation(sentiments)
                st.subheader("Investment Recommendation:")
                st.write(recommendation)

        else:
            st.write("No recent news headlines found.")
    else:
        st.write("Invalid ticker symbol.")



# Main Streamlit App
def stock_analysis_app():
    st.sidebar.subheader("Stock Analysis")

    # Replace text input with a selectbox for company name auto-suggestion
    selected_company = st.sidebar.selectbox('Select or Enter Company Name:', company_names)

    # Retrieve the corresponding ticker for the selected company
    ticker = [ticker for ticker, company in ticker_to_company.items() if company == selected_company][0]

    submenu = st.sidebar.selectbox("Select Analysis Type", ["Technical Analysis", "Sentiment Analysis", "Price Forecast"])
    start_date, end_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=365)), st.sidebar.date_input("End Date", value=datetime.now() + timedelta(days=1))

    if submenu == "Technical Analysis":
        st.title('Stock Technical Analysis')

        if ticker:
            data = download_data(ticker, start_date, end_date)
            if data.empty:
                st.write("No data available for the selected ticker.")
                return

            data = calculate_technical_indicators(data)
            
            # Calculate scores and details
            scores, details = calculate_scores(data)

            # Define columns for each category
            trend_columns = ["MACD_hist","5_day_EMA", "10_day_EMA", "20_day_EMA", "ALMA", "Aroon_Up", "Aroon_Down", "ADX", "Plus_DI", "Minus_DI", "BB_Middle", "BB_Std", "BB_High", "BB_Low", "DEMA", "Envelope_High", "Envelope_Low", "GMMA_Short", "GMMA_Long", "HMA", "Ichimoku_Tenkan", "Ichimoku_Kijun", "Ichimoku_Senkou_Span_A", "Ichimoku_Senkou_Span_B", "KC_Middle", "ATR_10", "KC_High", "KC_Low", "LSMA", "MAC_Upper", "MAC_Lower", "EMA_12", "EMA_26", "MACD", "MACD_signal",  "Parabolic_SAR", "SuperTrend", "Price_Channel_Upper", "Price_Channel_Lower", "TEMA_20", "Advance_Decline", "Chande_Kroll_Stop_Long", "Chande_Kroll_Stop_Short", "Williams_Alligator_Jaw", "Williams_Alligator_Teeth", "Williams_Alligator_Lips", "Donchian_High", "Donchian_Low"]
            momentum_columns = ["RSI","AO", "AC", "CMO", "CCI", "CRSI", "Coppock", "DPO", "KST", "KST_Signal", "Momentum",  "ROC",  "Stochastic_%K", "Stochastic_%D", "Stochastic_RSI", "TRIX", "TRIX_Signal", "TSI", "TSI_Signal", "Ultimate_Oscillator", "Relative_Vigor_Index", "RVI_Signal", "SMI_Ergodic", "SMI_Ergodic_Signal", "Fisher_Transform", "Fisher_Transform_Signal", "Williams_%R", "Plus_DI", "Minus_DI"]
            volatility_columns = ["ATR", "BB_%B", "BB_Width", "Chaikin_Volatility",  "Choppiness_Index", "Hist_Vol_Annualized", "Mass_Index", "RVI", "Standard_Deviation", "Vol_CtC", "Vol_ZtC", "Vol_OHLC", "Vol_Index", "Chop_Zone", "ZigZag", "Keltner_High", "Keltner_Low"]
            volume_columns = ["AD", "BoP", "CMF", "CO", "EMV", "EFI", "KVO", "KVO_Signal", "MFI", "Net_Volume", "OBV", "PVT", "VWAP", "VO", "Vortex_Pos", "Vortex_Neg", "Volume", "VWMA", "VPFR", "VPVR", "Spread", "Elder_Ray_Bull", "Elder_Ray_Bear", "Volume_Profile", "Price_to_Volume", "McClellan_Oscillator", "TRIN", "Williams_AD", "Ease_of_Movement"]
            support_resistance_columns = ["Pivot_Point","Fractal_Up", "Fractal_Down",  "Resistance_1", "Support_1", "Resistance_2", "Support_2", "Resistance_3", "Support_3", "Typical_Price", "Darvas_High", "Darvas_Low", "Fib_0.0", "Fib_0.236", "Fib_0.382", "Fib_0.5", "Fib_0.618", "Fib_1.0"]
            #statistical_columns = ["Correlation_Coefficient", "Log_Correlation", "Linear_Regression_Curve", "Linear_Regression_Slope", "Standard_Error", "Standard_Error_Band_Upper",  "Standard_Error_Band_Lower", "Median_Price"]

            indicator_groups = {
                "Trend": trend_columns,
                "Momentum": momentum_columns,
                "Volatility": volatility_columns,
                "Volume": volume_columns,
                "Support_Resistance": support_resistance_columns,
                #"Statistics":statistical_columns
            }

            # Checkbox to toggle candlestick chart
            use_candlestick = st.sidebar.checkbox("Use Candlestick Chart", value=False)

            # Process each section
            for group_name, indicators in indicator_groups.items():
                st.subheader(f'{group_name} Indicators')

                col1, col2 = st.columns([1, 2])

                with col1:
                    # Display the gauge for the group
                    st.plotly_chart(create_gauge(scores[group_name], f'{group_name} Score'))

                with col2:
                    # Set table height using CSS
                    table_css = """
                    <style>
                    .custom-table {
                        height: 250px;
                        overflow-y: auto;
                        display: block;
                        padding-left: 10px;
                    }
                    </style>
                    """
                    st.markdown(table_css, unsafe_allow_html=True)

                    # Display the indicator table with custom height
                    indicator_table = create_indicator_table(details, group_name)
                    st.markdown(f'<div class="custom-table">{indicator_table.to_html(index=False)}</div>', unsafe_allow_html=True)

                # Display the combined chart with selected indicators
                create_combined_chart(data, group_name, indicators, ticker, use_candlestick)

                st.divider()

            # Calculate overall weightage score
            overall_score = (
                scores['Trend'] * 0.30 + 
                scores['Momentum'] * 0.25 + 
                scores['Volume'] * 0.25 + 
                scores['Volatility'] * 0.10 + 
                scores['Support_Resistance'] * 0.10
            )
            recommendation = get_recommendation(overall_score)

            st.subheader("Overall Analysis")

            col1, col2 = st.columns([1, 2])  # Layout for overall score

            with col1:
                st.plotly_chart(create_gauge(overall_score, 'Overall Score'))

            with col2:
                st.markdown(f"### Overall Score: {overall_score:.2f}")
                st.markdown(f"<p style='font-size:20px;'>Recommendation: {recommendation}</p>", unsafe_allow_html=True)

    elif submenu == "Sentiment Analysis":
        st.title('Stock Sentiment Analysis')
        if ticker:
            sentiment_analysis_section(ticker, start_date, end_date)
        else:
            st.write("Please enter a valid stock ticker to perform sentiment analysis.")

    elif submenu == "Price Forecast":
        pass

if __name__ == '__main__':
    stock_analysis_app()


