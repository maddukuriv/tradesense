import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import linregress
import pandas_ta as ta

from utils.constants import bse_largecap, bse_smallcap, bse_midcap, sp500_tickers, ftse100_tickers

# Helper functions for calculating technical indicators and scoring
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




# Add all the technical indicator functions here...

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
    df['KST_Signal'] = df['KST'].rolling(window=9).mean()
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    df['RSI'] = rsi(df['Close'])
    df['ROC'] = df['Close'].pct_change(12)
    df['Stochastic_%K'] = (df['Close'] - df['Low'].rolling(window=14).min()) / (df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min()) * 100
    df['Stochastic_%D'] = df['Stochastic_%K'].rolling(window=3).mean()
    df['Stochastic_RSI'] = (rsi(df['Close'], window=14) - rsi(df['Close'], window=14).rolling(window=14).min()) / (rsi(df['Close'], window=14).rolling(window=14).max() - rsi(df['Close'], window=14).rolling(window=14).min())
    df['TRIX'] = df['Close'].ewm(span=15, adjust=False).mean().pct_change(1)
    df['TSI'] = df['Close'].diff(1).ewm(span=25, adjust=False).mean() / df['Close'].diff(1).abs().ewm(span=13, adjust=False).mean()
    df['TSI_Signal'] = df['TSI'].ewm(span=9, adjust=False).mean()
    df['Ultimate_Oscillator'] = (4 * (df['Close'] - df['Low']).rolling(window=7).sum() + 2 * (df['Close'] - df['Low']).rolling(window=14).sum() + (df['Close'] - df['Low']).rolling(window=28).sum()) / ((df['High'] - df['Low']).rolling(window=7).sum() + (df['High'] - df['Low']).rolling(window=14).sum() + (df['High'] - df['Low']).rolling(window=28).sum()) * 100

    # Volume Indicators
    df['AD'] = (df['Close'] - df['Low'] - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
    df['BoP'] = (df['Close'] - df['Open']) / (df['High'] - df['Low'])
    df['CMF'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
    df['CO'] = df['Close'].diff(3).ewm(span=10, adjust=False).mean()
    df['EMV'] = (df['High'] - df['Low']) / df['Volume']
    df['EFI'] = df['Close'].diff(1) * df['Volume']
    df['KVO'] = (df['High'] - df['Low']).ewm(span=34, adjust=False).mean() - (df['High'] - df['Low']).ewm(span=55, adjust=False).mean()
    df['KVO_Signal'] = df['KVO'].ewm(span=13, adjust=False).mean()
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
    df['Hist_Vol_Annualized'] = historical_volatility(df)
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
    
    # Statistical indicators
    df['Correlation_Coefficient'] = correlation_coefficient(df['Close'], df['Close'].shift(1))
    df['Log_Correlation'] = log_correlation(df['Close'], df['Close'].shift(1))
    df['Linear_Regression_Curve'] = linear_regression_curve(df['Close'])
    df['Linear_Regression_Slope'] = linear_regression_slope(df['Close'])
    df['Standard_Error'] = standard_error(df['Close'])
    df['Standard_Error_Band_Upper'], df['Standard_Error_Band_Lower'] = standard_error_bands(df['Close'])


    # Calculate Advance/Decline
    df['Advance_Decline'] = df['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)).cumsum()
    df['Chop_Zone'] = choppiness_index(df['High'], df['Low'], df['Close'])
    df['Chande_Kroll_Stop_Long'], df['Chande_Kroll_Stop_Short'] = chande_kroll_stop(df['High'], df['Low'], df['Close'])
    df['Fisher_Transform'], df['Fisher_Transform_Signal'] = fisher_transform(df['Close'])
    # Calculate Median Price
    df['Median_Price'] = (df['High'] + df['Low']) / 2
    df['Relative_Vigor_Index'] = relative_vigor_index(df['Open'], df['High'], df['Low'], df['Close'])
    df['SMI_Ergodic'], df['SMI_Ergodic_Signal'] = smi_ergodic(df['Close'])
    # Calculate Spread
    df['Spread'] = df['High'] - df['Low']

    # Calculate Typical Price
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Williams_%R'] = williams_r(df['High'], df['Low'], df['Close'])
    df['Williams_Alligator_Jaw'], df['Williams_Alligator_Teeth'], df['Williams_Alligator_Lips'] = alligator(df['High'], df['Low'], df['Close'])
    df['ZigZag'] = zigzag(df['Close'])

    return df

def add_scoring(df):
    # Customize scoring for each indicator
    volume_score_conditions = {
        # When the A/D line rises, indicating accumulation over multiple periods
        'AD': (df['AD'] > df['AD'].shift(1)) & (df['AD'].rolling(window=3).mean() > df['AD'].rolling(window=3).mean().shift(1)),

        # When BOP turns positive, confirming over the last few periods
        'BoP': (df['BoP'] > 0) & (df['BoP'].rolling(window=3).mean() > 0),

        # When CMF crosses above the zero line with confirmation
        'CMF': (df['CMF'] > 0) & (df['CMF'].rolling(window=3).mean() > 0),

        # When the Chaikin Oscillator crosses above the zero line with trend confirmation
        'CO': (df['CO'] > 0) & (df['CO'].rolling(window=3).mean() > 0),

        # When EMV crosses above the zero line with confirmation
        'EMV': (df['EMV'] > 0) & (df['EMV'].rolling(window=3).mean() > 0),

        # When EFI turns positive and is trending upward over the last few periods
        'EFI': (df['EFI'] > 0) & (df['EFI'] > df['EFI'].shift(1)) & (df['EFI'].rolling(window=3).mean() > df['EFI'].rolling(window=3).mean().shift(1)),

        # When the Klinger Oscillator crosses above its signal line with trend confirmation
        'KVO': (df['KVO'] > df['KVO_Signal']) & (df['KVO'].rolling(window=3).mean() > df['KVO_Signal'].rolling(window=3).mean()),

        # When MFI moves from oversold territory (below 20) upward, confirmed by trend
        'MFI': (df['MFI'] > 20) & (df['MFI'].shift(1) <= 20) & (df['MFI'].rolling(window=3).mean() > 20),

        # When net volume is consistently positive over multiple periods
        'Net_Volume': (df['Net_Volume'] > 0) & (df['Net_Volume'].rolling(window=3).mean() > 0),

        # When OBV trends upward, confirmed over multiple periods
        'OBV': (df['OBV'] > df['OBV'].shift(1)) & (df['OBV'].rolling(window=3).mean() > df['OBV'].rolling(window=3).mean().shift(1)),

        # When PVT trends upward, confirmed over multiple periods
        'PVT': (df['PVT'] > df['PVT'].shift(1)) & (df['PVT'].rolling(window=3).mean() > df['PVT'].rolling(window=3).mean().shift(1)),

        # When the price is below VWAP, indicating a potential buy opportunity, with trend confirmation
        'VWAP_Below_Price': (df['Close'] < df['VWAP']) & (df['Close'].rolling(window=3).mean() < df['VWAP'].rolling(window=3).mean()),

        # When the price crosses above the VWMA, confirmed by trend
        'VWMA_Cross': (df['Close'] > df['VWMA']) & (df['Close'].rolling(window=3).mean() > df['VWMA'].rolling(window=3).mean()),

        # When the volume oscillator crosses above the zero line, confirmed by trend
        'VO': (df['VO'] > 0) & (df['VO'].rolling(window=3).mean() > 0),

        # When the price is near a high-volume node in the VPFR, confirmed by close proximity
        'VPFR_Near_High_Volume_Node': (df['VPFR'] == df['VPFR'].max()) & (df['Close'] > df['VPFR'].shift(1)),

        # When the price is near a high-volume node in the VPVR, confirmed by close proximity
        'VPVR_Near_High_Volume_Node': (df['VPVR'] == df['VPVR'].max()) & (df['Close'] > df['VPVR'].shift(1)),

        # When the positive VI crosses above the negative VI, confirmed by trend
        'Vortex_Positive_Cross': (df['Vortex_Pos'] > df['Vortex_Neg']) & (df['Vortex_Pos'].rolling(window=3).mean() > df['Vortex_Neg'].rolling(window=3).mean()),

        # When a price rise is accompanied by higher-than-average volume, confirmed over multiple periods
        'Price_Volume_Rise': (df['Close'] > df['Close'].shift(1)) & (df['Volume'] > df['Volume'].rolling(window=20).mean()) & (df['Volume'].rolling(window=3).mean() > df['Volume'].rolling(window=20).mean()),

        # Narrow spreads can indicate liquidity and good entry points
        'Spread_Narrow': df['Spread'] < df['Spread'].rolling(window=20).mean(),
    }


    # Customize scoring for each trend indicator
    trend_score_conditions = {
        # When the stock price crosses above the ALMA and ALMA is sloping upwards (for trend confirmation)
        'ALMA': (df['Close'] > df['ALMA']) & (df['ALMA'].diff() > 0),

        # When the Aroon Up crosses above the Aroon Down and both indicate a strong trend (up > 70, down < 30)
        'Aroon': (df['Aroon_Up'] > df['Aroon_Down']) & (df['Aroon_Up'] > 70) & (df['Aroon_Down'] < 30),

        # ADX confirms a strong trend; ensure ADX is rising for continued momentum
        'ADX': (df['ADX'] > 25) & (df['ADX'].diff() > 0),

        # Price touches the lower Bollinger Band and starts moving up; also check for price bounce (close > open)
        'BB_Low': (df['Close'] > df['BB_Low']) & (df['Close'] > df['Open']),

        # When the price crosses above the DEMA and DEMA is trending upwards
        'DEMA': (df['Close'] > df['DEMA']) & (df['DEMA'].diff() > 0),

        # When the price moves above the lower envelope, with an additional check for a recent upward price momentum
        'Envelope_Low': (df['Close'] > df['Envelope_Low']) & (df['Close'].diff() > 0),

        # Short-term moving averages cross above long-term moving averages, and the long-term average is also trending upwards
        'GMMA': (df['GMMA_Short'] > df['GMMA_Long']) & (df['GMMA_Long'].diff() > 0),

        # When the price crosses above the HMA, and the HMA itself is sloping upwards
        'HMA': (df['Close'] > df['HMA']) & (df['HMA'].diff() > 0),

        # When the price crosses above the Ichimoku cloud, and the cloud is showing a bullish sentiment (Senkou Span A > Senkou Span B)
        'Ichimoku_Cloud': (df['Close'] > df[['Ichimoku_Senkou_Span_A', 'Ichimoku_Senkou_Span_B']].max(axis=1)) & 
                        (df['Ichimoku_Senkou_Span_A'] > df['Ichimoku_Senkou_Span_B']),

        # Price bounces off the lower Keltner Channel band, confirmed with a bullish candlestick (close > open)
        'KC_Low': (df['Close'] > df['KC_Low']) & (df['Close'] > df['Open']),

        # When the price crosses above the LSMA, with LSMA trending upwards
        'LSMA': (df['Close'] > df['LSMA']) & (df['LSMA'].diff() > 0),

        # When the price rebounds off the lower channel line, ensure the price is also above the open
        'Price_Channel_Lower': (df['Close'] > df['Price_Channel_Lower']) & (df['Close'] > df['Open']),

        # MACD line crosses above the signal line, with MACD histogram also positive and increasing
        'MACD': (df['MACD'] > df['MACD_signal']) & (df['MACD_hist'] > 0) & (df['MACD_hist'].diff() > 0),

        # When the price crosses above the Parabolic SAR, confirming with an upward price momentum
        'Parabolic_SAR': (df['Close'] > df['Parabolic_SAR']) & (df['Close'].diff() > 0),

        # When the price breaks above the upper channel, confirmed by a bullish candlestick
        'Price_Channel_Upper': (df['Close'] > df['Price_Channel_Upper']) & (df['Close'] > df['Open']),

        # When the price crosses above the SuperTrend line, with SuperTrend indicating a bullish trend
        'SuperTrend': (df['Close'] > df['SuperTrend']) & (df['SuperTrend'].diff() < 0),

        # When the price crosses above the TEMA, with TEMA trending upwards
        'TEMA_20': (df['Close'] > df['TEMA_20']) & (df['TEMA_20'].diff() > 0),

        # A rising A/D line indicates broad market strength
        'Advance_Decline': df['Advance_Decline'] > df['Advance_Decline'].shift(1),

        # When the price crosses above the Chande Kroll Stop (using the Long Stop as reference)
        'Chande_Kroll_Stop_Cross_Long': df['Median_Price'] > df['Chande_Kroll_Stop_Long'],

        # When the Williams Alligator Lips cross above the Teeth and Jaw, indicating a new trend
        'Williams_Alligator_Cross': (df['Williams_Alligator_Lips'] > df['Williams_Alligator_Teeth']) & (df['Williams_Alligator_Teeth'] > df['Williams_Alligator_Jaw']),
    }

    # Customize scoring for each momentum indicator
    momentum_score_conditions = {
        # When the AC turns positive, indicating momentum is building
        'AC': df['AC'] > 0,

        # When AO crosses above the zero line, indicating bullish momentum
        'AO': df['AO'] > 0,

        # When CMO crosses above the zero line from negative territory
        'CMO': df['CMO'] > 0,

        # When CCI moves from negative to positive territory (crosses zero)
        'CCI': df['CCI'] > 0,

        # When the Connors RSI drops to oversold levels (below 20) and turns upward
        'CRSI': (df['CRSI'] < 20) & (df['CRSI'] > df['CRSI'].shift(1)),

        # When the Coppock Curve turns upward from below zero
        'Coppock': (df['Coppock'] < 0) & (df['Coppock'].diff() > 0),

        # When DPO crosses above the zero line
        'DPO': df['DPO'] > 0,

        # When the +DI line crosses above the -DI line (Directional Movement)
        #'DI_Plus_Minus': df['DI_Plus'] > df['DI_Minus'],

        # When KST crosses above its signal line
        'KST': df['KST'] > df['KST_Signal'],

        # When the momentum indicator turns positive
        'Momentum': df['Momentum'] > 0,

        # When RSI moves from oversold territory (below 30) upward
        'RSI': (df['RSI'] < 30) & (df['RSI'] > df['RSI'].shift(1)),

        # When ROC turns positive from negative territory
        'ROC': df['ROC'] > 0,

        # When the %K line crosses above the %D line from oversold levels
        'Stochastic_%K_D_Cross': df['Stochastic_%K'] > df['Stochastic_%D'],

        # When Stochastic RSI crosses above the 20 level from oversold conditions
        'Stochastic_RSI': (df['Stochastic_RSI'] < 20) & (df['Stochastic_RSI'] > df['Stochastic_RSI'].shift(1)),

        # When TRIX crosses above the zero line
        'TRIX': df['TRIX'] > 0,

        # When TSI crosses above its signal line
        'TSI': df['TSI'] > df['TSI_Signal'],

        # When the Ultimate Oscillator moves from below 30 to above 50
        'Ultimate_Oscillator': (df['Ultimate_Oscillator'] > 30) & (df['Ultimate_Oscillator'] > df['Ultimate_Oscillator'].shift(1)) & (df['Ultimate_Oscillator'] > 50),

        # When Williams %R moves from oversold territory (below -80) to above -80
        'Williams_%R_Oversold': (df['Williams_%R'] > -80) & (df['Williams_%R'].shift(1) <= -80),

        # When the Fisher Transform turns positive from negative territory
        'Fisher_Transform': (df['Fisher_Transform'] > 0) & (df['Fisher_Transform'].shift(1) <= 0),

        # When RVI crosses above its signal line (SMI Ergodic is used here, which is similar)
        'SMI_Ergodic_Cross': df['SMI_Ergodic'] > df['SMI_Ergodic_Signal'],
    }


    volatility_score_conditions = {
        # When %B is near 0, indicating the price is near the lower band (potential for reversal)
        'BB_%B': df['BB_%B'] < 0.2,

        # Narrowing Bollinger Bands may indicate a period of consolidation before a breakout
        'BB_Width': df['BB_Width'] < df['BB_Width'].rolling(window=20).mean(),

        # A spike in Chaikin Volatility following a decline may indicate a reversal
        'Chaikin_Volatility': (df['Chaikin_Volatility'] > df['Chaikin_Volatility'].rolling(window=5).mean()) & (df['Chaikin_Volatility'].diff() > 0),

        # Lower Choppiness Index values indicate a trending market, which may precede a strong move
        'Choppiness_Index': df['Choppiness_Index'] < 50,

        # Historical Volatility below average might indicate consolidation before a breakout
        'Hist_Vol_Annualized': df['Hist_Vol_Annualized'] < df['Hist_Vol_Annualized'].rolling(window=20).mean(),

        # When the Mass Index rises above 27 and then falls below 26.5, signaling a potential reversal
        'Mass_Index': (df['Mass_Index'] > 27) & (df['Mass_Index'] < 26.5),

        # When RVI crosses above 50, indicating increasing volatility in a bullish direction
        'RVI': df['RVI'] > 50,

        # Low standard deviation can indicate consolidation before a breakout
        'Standard_Deviation': df['Standard_Deviation'] < df['Standard_Deviation'].rolling(window=20).mean(),

        # Decreasing Close-to-Close volatility can indicate a potential breakout
        'Vol_CtC': df['Vol_CtC'] < df['Vol_CtC'].rolling(window=20).mean(),

        # Decreasing Zero-to-Close volatility can indicate a potential breakout
        'Vol_ZtC': df['Vol_ZtC'] < df['Vol_ZtC'].rolling(window=20).mean(),

        # Decreasing OHLC (Open, High, Low, Close) volatility can indicate a potential breakout
        'Vol_OHLC': df['Vol_OHLC'] < df['Vol_OHLC'].rolling(window=20).mean(),

        # Lower VIX values often correlate with rising markets, indicating potential bullish conditions
        'Vol_Index': df['Vol_Index'] < df['Vol_Index'].rolling(window=20).mean(),

        # Indicates low chop when a strong trend might be forming
        'Chop_Zone': df['Chop_Zone'] < 50,
        
        # When the ZigZag indicator turns upward, indicating a potential reversal
        'ZigZag_Upturn': df['ZigZag'] > df['ZigZag'].shift(1)
    }



    support_resistance_score_conditions = {
        # When the price is above the pivot point, suggesting a bullish bias
        'Pivot_Point': df['Close'] > df['Pivot_Point'],

        # When the price is above Resistance 1, indicating further bullish momentum
        'Resistance_1': df['Close'] > df['Resistance_1'],

        # When the price is above Support 1, indicating a potential bounce from support
        'Support_1': df['Close'] > df['Support_1'],

        # When the price is above Resistance 2, indicating strong bullish momentum
        'Resistance_2': df['Close'] > df['Resistance_2'],

        # When the price is above Support 2, indicating another potential bounce from a deeper support
        'Support_2': df['Close'] > df['Support_2'],

        # When the price is above Resistance 3, indicating very strong bullish momentum
        'Resistance_3': df['Close'] > df['Resistance_3'],

        # When the price is above Support 3, indicating a strong bounce from a significant support level
        'Support_3': df['Close'] > df['Support_3'],

        # When a down fractal is formed, it could indicate a bottoming out (buy signal)
        'Fractal_Down': df['Fractal_Down'] > 0
    }

    statistical_score_conditions = {
        # Positive correlation with a leading indicator may suggest buying opportunities
        'Correlation_Coefficient': df['Correlation_Coefficient'] > 0,

        # Logarithmic correlation is positive, indicating a buying opportunity
        'Log_Correlation': df['Log_Correlation'] > 0,

        # When the price is below the linear regression curve and starts to turn upward
        'Price_Below_Regression_Curve': (df['Close'] < df['Linear_Regression_Curve']) & (df['Close'].shift(1) >= df['Close'].shift(2)),

        # A positive slope suggests an upward trend
        'Linear_Regression_Slope': df['Linear_Regression_Slope'] > 0,

        # Lower standard error indicates stronger confidence in the trend
        'Standard_Error': df['Standard_Error'] < df['Standard_Error'].rolling(window=20).mean(),

        # When the price touches the lower standard error band and starts moving up
        'Standard_Error_Band_Lower_Touch': (df['Close'] <= df['Standard_Error_Band_Lower']) & (df['Close'].shift(1) < df['Close'].shift(2))
    }



    # Define weightages for each category
    weightage = {
        'Volume_Score': 0.20,             # 15% weight
        'Trend_Score': 0.25,              # 25% weight
        'Momentum_Score': 0.25,           # 25% weight
        'Volatility_Score': 0.15,         # 15% weight
        'Support_Resistance_Score': 0.10, # 10% weight
        'Statistical_Score': 0.05,        # 5% weight
        
    }

    # Calculate scores based on conditions
    df['Volume_Score'] = sum([cond.astype(int) for cond in volume_score_conditions.values()])
    df['Trend_Score'] = sum([cond.astype(int) for cond in trend_score_conditions.values()])
    df['Momentum_Score'] = sum([cond.astype(int) for cond in momentum_score_conditions.values()])
    df['Volatility_Score'] = sum([cond.astype(int) for cond in volatility_score_conditions.values()])
    df['Support_Resistance_Score'] = sum([cond.astype(int) for cond in support_resistance_score_conditions.values()])
    df['Statistical_Score'] = sum([cond.astype(int) for cond in statistical_score_conditions.values()])
    

    # Normalize scores to be between -1 and 1
    df['Volume_Score'] = df['Volume_Score'] / df['Volume_Score'].abs().max()
    df['Trend_Score'] = df['Trend_Score'] / df['Trend_Score'].abs().max()
    df['Momentum_Score'] = df['Momentum_Score'] / df['Momentum_Score'].abs().max()
    df['Volatility_Score'] = df['Volatility_Score'] / df['Volatility_Score'].abs().max()
    df['Support_Resistance_Score'] = df['Support_Resistance_Score'] / df['Support_Resistance_Score'].abs().max()
    df['Statistical_Score'] = df['Statistical_Score'] / df['Statistical_Score'].abs().max()


    # Calculate the overall score as a weighted sum of the scores
    df['Overall_Score'] = (
        df['Volume_Score'] * weightage['Volume_Score'] +
        df['Trend_Score'] * weightage['Trend_Score'] +
        df['Momentum_Score'] * weightage['Momentum_Score'] +
        df['Volatility_Score'] * weightage['Volatility_Score'] +
        df['Support_Resistance_Score'] * weightage['Support_Resistance_Score'] +
        df['Statistical_Score'] * weightage['Statistical_Score'] 
    )
    return df

def fetch_latest_data(tickers_with_dates):
    technical_data = []
    for ticker, occurrence_date in tickers_with_dates:
        try:
            data = yf.download(ticker, start="2023-01-01", end="2023-12-31")
            if data.empty:
                continue
            data = calculate_indicators(data)
            latest_data = data.iloc[-1].copy()
            latest_data['Ticker'] = ticker  # Ensure 'Ticker' is added to the row
            technical_data.append(latest_data)
        except Exception as e:
            st.error(f"Error fetching data for ticker '{ticker}': {e}")
    return pd.DataFrame(technical_data)

# Main Streamlit app to display the comparison and analysis
def display_stock_comparison():
    st.title("Enhanced Stock Comparison Tool")

    tickers = st.text_input("Enter Stock Ticker(s):").split(',')

    if st.button("Analyze"):
        tickers_with_signals = [(ticker, None) for ticker in tickers]

        df_signals = fetch_latest_data(tickers_with_signals)

        if not df_signals.empty:
            df_signals = add_scoring(df_signals)

            
            trend_columns = ['Ticker', 'Close','ALMA','Aroon_Up', 'Aroon_Down','ADX', 'BB_High', 'SMA_20','BB_Low', 'DEMA', 'Envelope_High', 'Envelope_Low', 'GMMA_Short', 'GMMA_Long', 'HMA', 'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'Ichimoku_Senkou_Span_A', 'Ichimoku_Senkou_Span_B', 'KC_High', 'KC_Low', 'LSMA','EMA_20','MACD', 'MACD_signal', 'MACD_hist', 'Parabolic_SAR', 'SuperTrend','MAC_Upper','MAC_Lower','Price_Channel_Upper','Price_Channel_Lower','TEMA_20','Advance_Decline','Chande_Kroll_Stop_Long', 'Chande_Kroll_Stop_Short','Williams_Alligator_Jaw', 'Williams_Alligator_Teeth', 'Williams_Alligator_Lips']
            momentum_columns = ['Ticker','AO', 'AC', 'CMO', 'CCI', 'CRSI', 'Coppock', 'DPO', 'KST','KST_Signal', 'Momentum', 'RSI','ROC', 'Stochastic_%K', 'Stochastic_%D', 'Stochastic_RSI', 'TRIX', 'TSI','TSI_Signal', 'Ultimate_Oscillator','Relative_Vigor_Index', 'SMI_Ergodic', 'SMI_Ergodic_Signal','Fisher_Transform', 'Fisher_Transform_Signal', 'Williams_%R']
            volume_columns = ['Ticker','AD', 'BoP', 'CMF', 'CO', 'EMV', 'EFI', 'KVO','KVO_Signal', 'MFI', 'Net_Volume','OBV', 'PVT', 'VWAP','VWMA', 'VO','VPFR','VPVR', 'Vortex_Pos', 'Vortex_Neg', 'Volume', 'Spread']
            volatility_columns = ['Ticker','ATR','BB_%B', 'BB_Width', 'Chaikin_Volatility', 'Choppiness_Index', 'Hist_Vol_Annualized', 'Mass_Index', 'RVI', 'Standard_Deviation','Vol_CtC','Vol_ZtC','Vol_OHLC','Vol_Index','Chop_Zone' , 'ZigZag']
            support_resistance_columns = ['Ticker', 'Close','Pivot_Point', 'Resistance_1', 'Support_1', 'Resistance_2', 'Support_2', 'Resistance_3', 'Support_3','Fractal_Up','Fractal_Down','Typical_Price']
            statitical_columns=['Ticker','Correlation_Coefficient','Log_Correlation','Linear_Regression_Curve','Linear_Regression_Slope','Standard_Error','Standard_Error_Band_Upper','Standard_Error_Band_Lower', 'Median_Price']
        

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

            st.subheader("Statisical Indicators")
            st.dataframe(df_signals[statitical_columns])

            st.subheader("Stock Comparision")
            st.dataframe(df_signals[['Ticker','Overall_Score','Trend_Score', 'Momentum_Score', 'Volume_Score','Volatility_Score', 'Support_Resistance_Score', 'Statistical_Score']])
        else:
            st.write("No valid data found for the selected tickers.")

# Call the main function
display_stock_comparison()

