import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pandas_ta as ta
import plotly.graph_objects as go
import streamlit as st

# Set the date range
end_date = datetime.now()
start_date = end_date - timedelta(days=90)  # for 3 months of data

# List of stock tickers
tickers = ['AAPL', 'META', 'NVDA']

# Function to download data
def download_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    return df

# Define a function to calculate William Arbitrage
def calculate_williams_alligator(df):
    jaw_length = 13
    teeth_length = 8
    lips_length = 5

    df['Jaw'] = df['Close'].shift(jaw_length).rolling(window=jaw_length).mean()
    df['Teeth'] = df['Close'].shift(teeth_length).rolling(window=teeth_length).mean()
    df['Lips'] = df['Close'].shift(lips_length).rolling(window=lips_length).mean()

    return df

# Define a function to calculate technical indicators
def calculate_indicators(df):
    df['CMO'] = ta.cmo(df['Close'], length=14)
    
    keltner = ta.kc(df['High'], df['Low'], df['Close'])
    df['Keltner_High'] = keltner['KCUe_20_2']
    df['Keltner_Low'] = keltner['KCLe_20_2']
    
    df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)
    df['Ultimate_Oscillator'] = ta.uo(df['High'], df['Low'], df['Close'])
    
    
    
    donchian = ta.donchian(df['High'], df['Low'])
    df['Donchian_High'] = donchian['DCU_20_20']
    df['Donchian_Low'] = donchian['DCL_20_20']
    
    df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume']).astype(float)
    
    distance_moved = ((df['High'] + df['Low']) / 2) - ((df['High'].shift(1) + df['Low'].shift(1)) / 2)
    box_ratio = (df['Volume'] / 1e8) / (df['High'] - df['Low'])
    emv = distance_moved / box_ratio
    df['Ease_of_Movement'] = emv.rolling(window=14).mean()
    
    df['Chaikin_MF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'])
    
    df['Williams_R'] = ta.willr(df['High'], df['Low'], df['Close'])
    
    trix = ta.trix(df['Close'])
    df['Trix'] = trix['TRIX_30_9']
    df['Trix_Signal'] = trix['TRIXs_30_9']
    
    vortex = ta.vortex(df['High'], df['Low'], df['Close'])
    df['Vortex_Pos'] = vortex['VTXP_14']
    df['Vortex_Neg'] = vortex['VTXM_14']
    
    supertrend = ta.supertrend(df['High'], df['Low'], df['Close'], length=7, multiplier=3.0)
    df['SuperTrend'] = supertrend['SUPERT_7_3.0']
    
    df['RVI'] = ta.rvi(df['High'], df['Low'], df['Close'])
    df['RVI_Signal'] = ta.ema(df['RVI'], length=14)
    
    bull_power = df['High'] - ta.ema(df['Close'], length=13)
    bear_power = df['Low'] - ta.ema(df['Close'], length=13)
    df['Elder_Ray_Bull'] = bull_power
    df['Elder_Ray_Bear'] = bear_power
    
    wad = ta.ad(df['High'], df['Low'], df['Close'], df['Volume'])
    df['Williams_AD'] = wad
    
    # Darvas Box Theory
    df['Darvas_High'] = df['High'].rolling(window=20).max()
    df['Darvas_Low'] = df['Low'].rolling(window=20).min()
    
    # Volume Profile calculation
    df['Volume_Profile'] = df.groupby(pd.cut(df['Close'], bins=20))['Volume'].transform('sum')

    # Additional technical indicators
    df['5_day_EMA'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['10_day_EMA'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['20_day_EMA'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['12_day_EMA'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['26_day_EMA'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['12_day_EMA'] - df['26_day_EMA']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stochastic_%K'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
    df['Stochastic_%D'] = df['Stochastic_%K'].rolling(window=3).mean()
    
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    df['A/D_line'] = (clv * df['Volume']).fillna(0).cumsum()
    
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    df['5_day_Volume_MA'] = df['Volume'].rolling(window=5).mean()
    df['10_day_Volume_MA'] = df['Volume'].rolling(window=10).mean()
    df['20_day_Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['20_day_SMA'] = df['Close'].rolling(window=20).mean()
    df['Std_Dev'] = df['Close'].rolling(window=20).std()
    df['BB_High'] = df['20_day_SMA'] + (df['Std_Dev'] * 2)
    df['BB_Low'] = df['20_day_SMA'] - (df['Std_Dev'] * 2)
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = high_low.combine(high_close, np.maximum).combine(low_close, np.maximum)
    df['ATR'] = tr.rolling(window=14).mean()
    
    # Parabolic SAR calculation
    df['Parabolic_SAR'] = calculate_parabolic_sar(df)
    
    # ADX calculation
    df['ADX'] = calculate_adx(df)
    
    # Ichimoku Cloud calculation
    df['Ichimoku_conv'], df['Ichimoku_base'], df['Ichimoku_A'], df['Ichimoku_B'] = calculate_ichimoku(df)
    
    # Other indicators
    df['ROC'] = ((df['Close'] - df['Close'].shift(12)) / df['Close'].shift(12)) * 100
    df['DPO'] = df['Close'] - df['Close'].shift(21).rolling(window=21).mean()
    df['Williams_%R'] = (high_14 - df['Close']) / (high_14 - low_14) * -100
    df['McClellan_Oscillator'] = df['Close'].ewm(span=19, adjust=False).mean() - df['Close'].ewm(span=39, adjust=False).mean()
    
    advances = (df['Close'] > df['Open']).astype(int)
    declines = (df['Close'] < df['Open']).astype(int)
    df['TRIN'] = (advances.rolling(window=14).sum() / declines.rolling(window=14).sum()) / (df['Volume'].rolling(window=14).mean() / df['Volume'].rolling(window=14).mean())
    df['Price_to_Volume'] = df['Close'] / df['Volume']
    df['Trend_Line'] = df['Close'].rolling(window=30).mean()
    
    # Pivot Points calculation
    df['Pivot_Point'], df['Resistance_1'], df['Support_1'], df['Resistance_2'], df['Support_2'], df['Resistance_3'], df['Support_3'] = calculate_pivot_points(df)
    
    # Fibonacci Levels calculation
    df = calculate_fibonacci_levels(df)
    
    # Gann Levels calculation
    df = calculate_gann_levels(df)
    
    # Advance Decline Line calculation
    df['Advance_Decline_Line'] = advances.cumsum() - declines.cumsum()
    
    # William Arbitrage calculation
    df = calculate_williams_alligator(df)
    
    return df

def calculate_parabolic_sar(df):
    af = 0.02
    uptrend = True
    df['Parabolic_SAR'] = np.nan
    ep = df['Low'][0] if uptrend else df['High'][0]
    df['Parabolic_SAR'].iloc[0] = df['Close'][0]
    for i in range(1, len(df)):
        if uptrend:
            df['Parabolic_SAR'].iloc[i] = df['Parabolic_SAR'].iloc[i - 1] + af * (ep - df['Parabolic_SAR'].iloc[i - 1])
            if df['Low'].iloc[i] < df['Parabolic_SAR'].iloc[i]:
                uptrend = False
                df['Parabolic_SAR'].iloc[i] = ep
                af = 0.02
                ep = df['High'].iloc[i]
        else:
            df['Parabolic_SAR'].iloc[i] = df['Parabolic_SAR'].iloc[i - 1] + af * (ep - df['Parabolic_SAR'].iloc[i - 1])
            if df['High'].iloc[i] > df['Parabolic_SAR'].iloc[i]:
                uptrend = True
                df['Parabolic_SAR'].iloc[i] = ep
                af = 0.02
                ep = df['Low'].iloc[i]
    return df['Parabolic_SAR']

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

def calculate_ichimoku(df):
    high_9 = df['High'].rolling(window=9).max()
    low_9 = df['Low'].rolling(window=9).min()
    high_26 = df['High'].rolling(window=26).max()
    low_26 = df['Low'].rolling(window=26).min()
    high_52 = df['High'].rolling(window=52).max()
    low_52 = df['Low'].rolling(window=52).min()
    ichimoku_conv = (high_9 + low_9) / 2
    ichimoku_base = (high_26 + low_26) / 2
    ichimoku_a = ((ichimoku_conv + ichimoku_base) / 2).shift(26)
    ichimoku_b = ((high_52 + low_52) / 2).shift(26)
    return ichimoku_conv, ichimoku_base, ichimoku_a, ichimoku_b

def calculate_pivot_points(df):
    pivot = (df['High'] + df['Low'] + df['Close']) / 3
    resistance_1 = (2 * pivot) - df['Low']
    support_1 = (2 * pivot) - df['High']
    resistance_2 = pivot + (df['High'] - df['Low'])
    support_2 = pivot - (df['High'] - df['Low'])
    resistance_3 = df['High'] + 2 * (pivot - df['Low'])
    support_3 = df['Low'] - 2 * (df['High'] - pivot)
    return pivot, resistance_1, support_1, resistance_2, support_2, resistance_3, support_3

def calculate_fibonacci_levels(df):
    high = df['High'].max()
    low = df['Low'].min()
    diff = high - low
    df['Fib_0.0'] = high
    df['Fib_0.236'] = high - 0.236 * diff
    df['Fib_0.382'] = high - 0.382 * diff
    df['Fib_0.5'] = high - 0.5 * diff
    df['Fib_0.618'] = high - 0.618 * diff
    df['Fib_1.0'] = low
    return df

def calculate_gann_levels(df):
    high = df['High'].max()
    low = df['Low'].min()
    diff = high - low
    df['Gann_0.25'] = low + 0.25 * diff
    df['Gann_0.5'] = low + 0.5 * diff
    df['Gann_0.75'] = low + 0.75 * diff
    return df

# Function to calculate the scores based on the provided criteria
def calculate_scores(data):
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

    # Trend Indicators
    trend_score = 0

    if data['5_day_EMA'].iloc[-1] > data['10_day_EMA'].iloc[-1] > data['20_day_EMA'].iloc[-1]:
        trend_score += 2
        details['Trend'] += "EMA: Strong Bullish; "
    elif data['5_day_EMA'].iloc[-1] > data['10_day_EMA'].iloc[-1] > data['20_day_EMA'].iloc[-1] and abs(data['5_day_EMA'].iloc[-1] - data['20_day_EMA'].iloc[-1]) < 0.5:
        trend_score += 1.75
        details['Trend'] += "EMA: Moderate Bullish; "
    elif data['5_day_EMA'].iloc[-1] > data['10_day_EMA'].iloc[-1] and data['10_day_EMA'].iloc[-1] < data['20_day_EMA'].iloc[-1]:
        trend_score += 1.5
        details['Trend'] += "EMA: Moderate Bullish; "
    elif data['5_day_EMA'].iloc[-1] < data['10_day_EMA'].iloc[-1] > data['20_day_EMA'].iloc[-1]:
        trend_score += 1
        details['Trend'] += "EMA: Neutral; "
    elif data['5_day_EMA'].iloc[-1] < data['10_day_EMA'].iloc[-1] < data['20_day_EMA'].iloc[-1]:
        trend_score += 0
        details['Trend'] += "EMA: Bearish; "

    # MACD Histogram
    macd_hist = data['MACD_hist'].iloc[-1]
    if macd_hist > 0 and macd_hist > data['MACD_hist'].iloc[-2]:
        trend_score += 2
        details['Trend'] += "MACD: Strong Bullish; "
    elif macd_hist > 0:
        trend_score += 1.75
        details['Trend'] += "MACD: Moderate Bullish; "
    elif macd_hist > 0 and macd_hist < data['MACD_hist'].iloc[-2]:
        trend_score += 1.5
        details['Trend'] += "MACD: Mild Bullish; "
    elif macd_hist < 0 and macd_hist > data['MACD_hist'].iloc[-2]:
        trend_score += 1
        details['Trend'] += "MACD: Mild Bearish; "
    elif macd_hist < 0:
        trend_score += 0.75
        details['Trend'] += "MACD: Neutral; "
    elif macd_hist < 0 and macd_hist < data['MACD_hist'].iloc[-2]:
        trend_score += 0
        details['Trend'] += "MACD: Strong Bearish; "

    # Ichimoku
    ichimoku_conv = data['Ichimoku_conv'].iloc[-1]
    ichimoku_base = data['Ichimoku_base'].iloc[-1]
    ichimoku_a = data['Ichimoku_A'].iloc[-1]
    ichimoku_b = data['Ichimoku_B'].iloc[-1]
    if ichimoku_conv > ichimoku_base and ichimoku_conv > ichimoku_a and ichimoku_conv > ichimoku_b:
        trend_score += 1
        details['Trend'] += "Ichimoku: Strong Bullish; "
    elif ichimoku_conv > ichimoku_base and (ichimoku_conv > ichimoku_a or ichimoku_conv > ichimoku_b):
        trend_score += 0.75
        details['Trend'] += "Ichimoku: Moderate Bullish; "
    elif ichimoku_conv > ichimoku_base:
        trend_score += 0.5
        details['Trend'] += "Ichimoku: Mild Bullish; "
    elif ichimoku_conv < ichimoku_base and (ichimoku_conv > ichimoku_a or ichimoku_conv > ichimoku_b):
        trend_score += 0.5
        details['Trend'] += "Ichimoku: Neutral; "
    elif ichimoku_conv < ichimoku_base:
        trend_score += 0.25
        details['Trend'] += "Ichimoku: Mild Bearish; "
    else:
        trend_score += 0
        details['Trend'] += "Ichimoku: Bearish; "

    # Parabolic SAR
    psar = data['Parabolic_SAR'].iloc[-1]
    price = data['Close'].iloc[-1]
    if psar < price and psar > data['Parabolic_SAR'].iloc[-2]:
        trend_score += 1
        details['Trend'] += "Parabolic SAR: Strong Bullish; "
    elif psar < price:
        trend_score += 0.75
        details['Trend'] += "Parabolic SAR: Moderate Bullish; "
    elif psar < price:
        trend_score += 0.5
        details['Trend'] += "Parabolic SAR: Mild Bullish; "
    elif psar < price:
        trend_score += 0.5
        details['Trend'] += "Parabolic SAR: Neutral; "
    else:
        trend_score += 0
        details['Trend'] += "Parabolic SAR: Strong Bearish; "

    # SuperTrend
    supertrend = data['SuperTrend'].iloc[-1]
    if supertrend < price:
        trend_score += 1
        details['Trend'] += "SuperTrend: Strong Bullish; "
    elif supertrend < price:
        trend_score += 0.75
        details['Trend'] += "SuperTrend: Moderate Bullish; "
    else:
        trend_score += 0
        details['Trend'] += "SuperTrend: Bearish; "

    # Donchian Channels
    donchian_high = data['Donchian_High'].iloc[-1]
    donchian_low = data['Donchian_Low'].iloc[-1]
    if price > donchian_high:
        trend_score += 1
        details['Trend'] += "Donchian Channels: Strong Bullish; "
    elif price > donchian_high:
        trend_score += 0.75
        details['Trend'] += "Donchian Channels: Moderate Bullish; "
    elif price > donchian_low:
        trend_score += 0.5
        details['Trend'] += "Donchian Channels: Mild Bullish; "
    elif price > donchian_low:
        trend_score += 0.5
        details['Trend'] += "Donchian Channels: Neutral; "
    else:
        trend_score += 0.25
        details['Trend'] += "Donchian Channels: Mild Bearish; "

    # Vortex Indicator
    vortex_pos = data['Vortex_Pos'].iloc[-1]
    vortex_neg = data['Vortex_Neg'].iloc[-1]
    if vortex_pos > vortex_neg and vortex_pos > data['Vortex_Pos'].iloc[-2]:
        trend_score += 1
        details['Trend'] += "Vortex: Strong Bullish; "
    elif vortex_pos > vortex_neg:
        trend_score += 0.75
        details['Trend'] += "Vortex: Moderate Bullish; "
    elif vortex_pos > vortex_neg:
        trend_score += 0.5
        details['Trend'] += "Vortex: Mild Bullish; "
    elif vortex_pos < vortex_neg and vortex_pos > data['Vortex_Pos'].iloc[-2]:
        trend_score += 0.5
        details['Trend'] += "Vortex: Neutral; "
    elif vortex_pos < vortex_neg:
        trend_score += 0.25
        details['Trend'] += "Vortex: Mild Bearish; "
    else:
        trend_score += 0
        details['Trend'] += "Vortex: Bearish; "

    # ADX
    adx = data['ADX'].iloc[-1]
    if adx > 25 and adx > data['ADX'].iloc[-2]:
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

    scores['Trend'] = trend_score / 11  # Normalize to 1

    # Momentum Indicators
    momentum_score = 0
    rsi = data['RSI'].iloc[-1]
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

    # Stochastic %K and %D
    stoch_k = data['Stochastic_%K'].iloc[-1]
    stoch_d = data['Stochastic_%D'].iloc[-1]
    if stoch_k > stoch_d and stoch_k > 80:
        momentum_score += 0
        details['Momentum'] += "Stochastic: Overbought (Bearish); "
    elif stoch_k > stoch_d and stoch_k > 50:
        momentum_score += 0.75
        details['Momentum'] += "Stochastic: Bullish; "
    elif stoch_k > stoch_d:
        momentum_score += 0.5
        details['Momentum'] += "Stochastic: Neutral Bullish; "
    elif stoch_k < stoch_d and stoch_k < 20:
        momentum_score += 1
        details['Momentum'] += "Stochastic: Oversold (Bullish); "
    elif stoch_k < stoch_d:
        momentum_score += 0.5
        details['Momentum'] += "Stochastic: Neutral Bearish; "

    # Rate of Change (ROC)
    roc = data['ROC'].iloc[-1]
    if roc > 10:
        momentum_score += 1
        details['Momentum'] += "ROC: Strong Bullish; "
    elif roc > 0:
        momentum_score += 0.75
        details['Momentum'] += "ROC: Mild Bullish; "
    elif roc > -10:
        momentum_score += 0.25
        details['Momentum'] += "ROC: Mild Bearish; "
    else:
        momentum_score += 0
        details['Momentum'] += "ROC: Strong Bearish; "

    # Detrended Price Oscillator (DPO)
    dpo = data['DPO'].iloc[-1]
    if dpo > 1:
        momentum_score += 1
        details['Momentum'] += "DPO: Strong Bullish; "
    elif dpo > 0:
        momentum_score += 0.75
        details['Momentum'] += "DPO: Mild Bullish; "
    elif dpo > -1:
        momentum_score += 0.25
        details['Momentum'] += "DPO: Mild Bearish; "
    else:
        momentum_score += 0
        details['Momentum'] += "DPO: Strong Bearish; "

    # Williams %R
    williams_r = data['Williams_%R'].iloc[-1]
    if williams_r > -20:
        momentum_score += 0
        details['Momentum'] += "Williams %R: Overbought (Bearish); "
    elif williams_r > -50:
        momentum_score += 0.25
        details['Momentum'] += "Williams %R: Neutral Bearish; "
    elif williams_r > -80:
        momentum_score += 0.5
        details['Momentum'] += "Williams %R: Neutral Bullish; "
    else:
        momentum_score += 1
        details['Momentum'] += "Williams %R: Oversold (Bullish); "

    # Chande Momentum Oscillator (CMO)
    cmo = data['CMO'].iloc[-1]
    if cmo > 50:
        momentum_score += 0
        details['Momentum'] += "CMO: Overbought (Bearish); "
    elif cmo > 0:
        momentum_score += 0.75
        details['Momentum'] += "CMO: Bullish; "
    elif cmo > -50:
        momentum_score += 0.5
        details['Momentum'] += "CMO: Neutral; "
    else:
        momentum_score += 1
        details['Momentum'] += "CMO: Oversold (Bullish); "

    # Commodity Channel Index (CCI)
    cci = data['CCI'].iloc[-1]
    if cci > 200:
        momentum_score += 1
        details['Momentum'] += "CCI: Strong Bullish; "
    elif cci > 100:
        momentum_score += 0.75
        details['Momentum'] += "CCI: Mild Bullish; "
    elif cci > -100:
        momentum_score += 0.5
        details['Momentum'] += "CCI: Neutral; "
    elif cci > -200:
        momentum_score += 0.25
        details['Momentum'] += "CCI: Mild Bearish; "
    else:
        momentum_score += 0
        details['Momentum'] += "CCI: Strong Bearish; "

    # Relative Vigor Index (RVI)
    rvi = data['RVI'].iloc[-1]
    rvi_signal = data['RVI_Signal'].iloc[-1]
    if rvi > rvi_signal:
        momentum_score += 1
        details['Momentum'] += "RVI: Strong Bullish; "
    elif rvi > rvi_signal:
        momentum_score += 0.75
        details['Momentum'] += "RVI: Mild Bullish; "
    elif rvi < rvi_signal:
        momentum_score += 0.25
        details['Momentum'] += "RVI: Mild Bearish; "
    else:
        momentum_score += 0
        details['Momentum'] += "RVI: Strong Bearish; "

    # Ultimate Oscillator
    uo = data['Ultimate_Oscillator'].iloc[-1]
    if uo > 70:
        momentum_score += 0
        details['Momentum'] += "Ultimate Oscillator: Overbought (Bearish); "
    elif uo > 60:
        momentum_score += 0.25
        details['Momentum'] += "Ultimate Oscillator: Mild Bearish; "
    elif uo > 50:
        momentum_score += 0.75
        details['Momentum'] += "Ultimate Oscillator: Neutral Bullish; "
    elif uo > 40:
        momentum_score += 0.25
        details['Momentum'] += "Ultimate Oscillator: Neutral Bearish; "
    elif uo > 30:
        momentum_score += 0.75
        details['Momentum'] += "Ultimate Oscillator: Mild Bullish; "
    else:
        momentum_score += 1
        details['Momentum'] += "Ultimate Oscillator: Oversold (Bullish); "

    # Trix and Trix Signal
    trix = data['Trix'].iloc[-1]
    trix_signal = data['Trix_Signal'].iloc[-1]
    if trix > trix_signal:
        momentum_score += 1
        details['Momentum'] += "Trix: Strong Bullish; "
    elif trix > trix_signal:
        momentum_score += 0.75
        details['Momentum'] += "Trix: Mild Bullish; "
    elif trix < trix_signal:
        momentum_score += 0.25
        details['Momentum'] += "Trix: Mild Bearish; "
    else:
        momentum_score += 0
        details['Momentum'] += "Trix: Strong Bearish; "



    scores['Momentum'] = momentum_score / 12  # Normalize to 1

    # Volatility Indicators
    volatility_score = 0
    atr = data['ATR'].iloc[-1]
    std_dev = data['Std_Dev'].iloc[-1]
    bb_high = data['BB_High'].iloc[-1]
    bb_low = data['BB_Low'].iloc[-1]
    sma_20 = data['20_day_SMA'].iloc[-1]
    keltner_high = data['Keltner_High'].iloc[-1]
    keltner_low = data['Keltner_Low'].iloc[-1]

    # ATR
    if atr > 1.5 * data['ATR'].rolling(window=50).mean().iloc[-1]:
        volatility_score += 0.5
        details['Volatility'] += "ATR: High volatility; "
    elif atr > 1.2 * data['ATR'].rolling(window=50).mean().iloc[-1]:
        volatility_score += 0.4
        details['Volatility'] += "ATR: Moderate volatility; "
    else:
        volatility_score += 0.3
        details['Volatility'] += "ATR: Low volatility; "

    # Standard Deviation
    if std_dev > 1.5 * data['Std_Dev'].rolling(window=50).mean().iloc[-1]:
        volatility_score += 0.5
        details['Volatility'] += "Std Dev: High volatility; "
    elif std_dev > 1.2 * data['Std_Dev'].rolling(window=50).mean().iloc[-1]:
        volatility_score += 0.4
        details['Volatility'] += "Std Dev: Moderate volatility; "
    else:
        volatility_score += 0.3
        details['Volatility'] += "Std Dev: Low volatility; "

    # Bollinger Bands
    if price > bb_high:
        volatility_score += 0
        details['Volatility'] += "BB: Strong Overbought (Bearish); "
    elif price > bb_high:
        volatility_score += 0.25
        details['Volatility'] += "BB: Mild Overbought (Bearish); "
    elif price < bb_low:
        volatility_score += 1
        details['Volatility'] += "BB: Strong Oversold (Bullish); "
    elif price < bb_low:
        volatility_score += 0.75
        details['Volatility'] += "BB: Mild Oversold (Bullish); "
    else:
        volatility_score += 0.5
        details['Volatility'] += "BB: Neutral; "

    # 20-day SMA
    if price > sma_20:
        volatility_score += 1
        details['Volatility'] += "20-day SMA: Strong Bullish; "
    elif price > sma_20:
        volatility_score += 0.75
        details['Volatility'] += "20-day SMA: Mild Bullish; "
    elif price < sma_20:
        volatility_score += 0.25
        details['Volatility'] += "20-day SMA: Mild Bearish; "
    else:
        volatility_score += 0
        details['Volatility'] += "20-day SMA: Strong Bearish; "

    # Keltner Channels
    if price > keltner_high:
        volatility_score += 0
        details['Volatility'] += "Keltner: Strong Overbought (Bearish); "
    elif price > keltner_high:
        volatility_score += 0.25
        details['Volatility'] += "Keltner: Mild Overbought (Bearish); "
    elif price < keltner_low:
        volatility_score += 1
        details['Volatility'] += "Keltner: Strong Oversold (Bullish); "
    elif price < keltner_low:
        volatility_score += 0.75
        details['Volatility'] += "Keltner: Mild Oversold (Bullish); "
    else:
        volatility_score += 0.5
        details['Volatility'] += "Keltner: Neutral; "

    scores['Volatility'] = volatility_score / 5  # Normalize to 1

    # Volume Indicators
    volume_score = 0
    obv = data['OBV'].iloc[-1]
    ad_line = data['A/D_line'].iloc[-1]
    price_to_volume = data['Price_to_Volume'].iloc[-1]
    trin = data['TRIN'].iloc[-1]
    advance_decline_line = data['Advance_Decline_Line'].iloc[-1]
    mcclellan_oscillator = data['McClellan_Oscillator'].iloc[-1]
    volume_profile = data['Volume_Profile'].iloc[-1]
    cmf = data['Chaikin_MF'].iloc[-1]
    williams_ad = data['Williams_AD'].iloc[-1]
    ease_of_movement = data['Ease_of_Movement'].iloc[-1]
    mfi = data['MFI'].iloc[-1]
    elder_ray_bull = data['Elder_Ray_Bull'].iloc[-1]
    elder_ray_bear = data['Elder_Ray_Bear'].iloc[-1]
    vwap = data['VWAP'].iloc[-1]

    if obv > data['OBV'].iloc[-2]:
        volume_score += 1
        details['Volume'] += "OBV: Increasing sharply; "
    elif obv < data['OBV'].iloc[-2]:
        volume_score += 0
        details['Volume'] += "OBV: Decreasing; "

    if ad_line > data['A/D_line'].iloc[-2]:
        volume_score += 1
        details['Volume'] += "A/D Line: Increasing sharply; "
    elif ad_line < data['A/D_line'].iloc[-2]:
        volume_score += 0
        details['Volume'] += "A/D Line: Decreasing; "

    # Add more volume indicators scoring
    # Price to Volume
    if price_to_volume > data['Price_to_Volume'].mean():
        volume_score += 0.75
        details['Volume'] += "Price to Volume: Increasing; "
    elif price_to_volume < data['Price_to_Volume'].mean():
        volume_score += 0.25
        details['Volume'] += "Price to Volume: Decreasing; "
    else:
        volume_score += 0.5
        details['Volume'] += "Price to Volume: Neutral; "

    # TRIN (Arms Index)
    if trin < 0.8:
        volume_score += 1
        details['Volume'] += "TRIN: Strong Bullish; "
    elif trin < 1:
        volume_score += 0.75
        details['Volume'] += "TRIN: Mild Bullish; "
    elif trin < 1.2:
        volume_score += 0.5
        details['Volume'] += "TRIN: Neutral; "
    elif trin < 1.5:
        volume_score += 0.25
        details['Volume'] += "TRIN: Mild Bearish; "
    else:
        volume_score += 0
        details['Volume'] += "TRIN: Strong Bearish; "

    # Advance/Decline Line
    if advance_decline_line > data['Advance_Decline_Line'].iloc[-2]:
        volume_score += 1
        details['Volume'] += "Advance/Decline Line: Increasing sharply; "
    elif advance_decline_line > data['Advance_Decline_Line'].iloc[-2]:
        volume_score += 0.75
        details['Volume'] += "Advance/Decline Line: Increasing slowly; "
    elif advance_decline_line < data['Advance_Decline_Line'].iloc[-2]:
        volume_score += 0.25
        details['Volume'] += "Advance/Decline Line: Decreasing slowly; "
    else:
        volume_score += 0
        details['Volume'] += "Advance/Decline Line: Decreasing sharply; "

    # McClellan Oscillator
    if mcclellan_oscillator > 50:
        volume_score += 1
        details['Volume'] += "McClellan Oscillator: Strong Bullish; "
    elif mcclellan_oscillator > 0:
        volume_score += 0.75
        details['Volume'] += "McClellan Oscillator: Mild Bullish; "
    elif mcclellan_oscillator < -50:
        volume_score += 0.25
        details['Volume'] += "McClellan Oscillator: Mild Bearish; "
    else:
        volume_score += 0
        details['Volume'] += "McClellan Oscillator: Strong Bearish; "

    # Volume Profile
    if volume_profile > data['Volume_Profile'].mean():
        volume_score += 0.75
        details['Volume'] += "Volume Profile: High volume nodes well above support; "
    else:
        volume_score += 0
        details['Volume'] += "Volume Profile: High volume nodes at resistance; "

    # Chaikin Money Flow (CMF)
    if cmf > 0.2:
        volume_score += 1
        details['Volume'] += "CMF: Strong Bullish; "
    elif cmf > 0:
        volume_score += 0.75
        details['Volume'] += "CMF: Mild Bullish; "
    elif cmf < -0.2:
        volume_score += 0.25
        details['Volume'] += "CMF: Mild Bearish; "
    else:
        volume_score += 0
        details['Volume'] += "CMF: Strong Bearish; "

    # Williams Accumulation/Distribution
    if williams_ad > data['Williams_AD'].iloc[-2]:
        volume_score += 1
        details['Volume'] += "Williams AD: Increasing sharply; "
    elif williams_ad < data['Williams_AD'].iloc[-2]:
        volume_score += 0.25
        details['Volume'] += "Williams AD: Decreasing slowly; "
    else:
        volume_score += 0
        details['Volume'] += "Williams AD: Decreasing sharply; "

    # Ease of Movement
    if ease_of_movement > data['Ease_of_Movement'].mean():
        volume_score += 1
        details['Volume'] += "Ease of Movement: Positive and increasing; "
    elif ease_of_movement > 0:
        volume_score += 0.75
        details['Volume'] += "Ease of Movement: Positive but flat; "
    else:
        volume_score += 0.25
        details['Volume'] += "Ease of Movement: Negative but flat; "

    # MFI (Money Flow Index)
    if mfi > 80:
        volume_score += 0
        details['Volume'] += "MFI: Overbought (Bearish); "
    elif mfi > 70:
        volume_score += 0.25
        details['Volume'] += "MFI: Mild Overbought (Bearish); "
    elif mfi > 50:
        volume_score += 0.75
        details['Volume'] += "MFI: Neutral Bullish; "
    elif mfi > 30:
        volume_score += 0.5
        details['Volume'] += "MFI: Neutral; "
    else:
        volume_score += 1
        details['Volume'] += "MFI: Oversold (Bullish); "

    # Elder-Ray Bull Power and Bear Power
    if elder_ray_bull > 0 and elder_ray_bull > data['Elder_Ray_Bull'].mean():
        volume_score += 1
        details['Volume'] += "Elder Ray Bull Power: Strong Bullish; "
    elif elder_ray_bull > 0:
        volume_score += 0.75
        details['Volume'] += "Elder Ray Bull Power: Mild Bullish; "
    elif elder_ray_bear < 0 and elder_ray_bear < data['Elder_Ray_Bear'].mean():
        volume_score += 0
        details['Volume'] += "Elder Ray Bear Power: Strong Bearish; "
    else:
        volume_score += 0.25
        details['Volume'] += "Elder Ray Bear Power: Mild Bearish; "

    # VWAP (Volume Weighted Average Price)
    if price > vwap:
        volume_score += 1
        details['Volume'] += "VWAP: Strong Bullish; "
    elif price > vwap:
        volume_score += 0.75
        details['Volume'] += "VWAP: Mild Bullish; "
    elif price < vwap:
        volume_score += 0.25
        details['Volume'] += "VWAP: Mild Bearish; "
    else:
        volume_score += 0
        details['Volume'] += "VWAP: Strong Bearish; "

    scores['Volume'] = volume_score / 13  # Normalize to 1

    # Support and Resistance Indicators
    support_resistance_score = 0
    price = data['Close'].iloc[-1]
    pivot_point = data['Pivot_Point'].iloc[-1]
    resistance_1 = data['Resistance_1'].iloc[-1]
    support_1 = data['Support_1'].iloc[-1]
    fib_0 = data['Fib_0.0'].iloc[-1]
    fib_0_236 = data['Fib_0.236'].iloc[-1]
    fib_0_382 = data['Fib_0.382'].iloc[-1]
    fib_0_5 = data['Fib_0.5'].iloc[-1]
    fib_0_618 = data['Fib_0.618'].iloc[-1]
    fib_1 = data['Fib_1.0'].iloc[-1]
    darvas_high = data['Darvas_High'].iloc[-1]
    darvas_low = data['Darvas_Low'].iloc[-1]

    if price > pivot_point:
        support_resistance_score += 1
        details['Support_Resistance'] += "Pivot Point: Above Pivot Point; "
    elif price < pivot_point:
        support_resistance_score += 0
        details['Support_Resistance'] += "Pivot Point: Below Pivot Point; "

    if price > resistance_1:
        support_resistance_score += 0
        details['Support_Resistance'] += "Support/Resistance: Near Resistance; "
    elif price < support_1:
        support_resistance_score += 1
        details['Support_Resistance'] += "Support/Resistance: Near Support; "

    # Add Fibonacci Levels scoring
    if price > fib_0_618:
        support_resistance_score += 1
        details['Support_Resistance'] += "Fibonacci: Strong support/resistance; "
    elif price > fib_0_5:
        support_resistance_score += 0.75
        details['Support_Resistance'] += "Fibonacci: Moderate support/resistance; "
    elif price > fib_0_382:
        support_resistance_score += 0.5
        details['Support_Resistance'] += "Fibonacci: Mild support/resistance; "
    elif price > fib_0_236:
        support_resistance_score += 0.25
        details['Support_Resistance'] += "Fibonacci: Weak support/resistance; "
    else:
        support_resistance_score += 0.5
        details['Support_Resistance'] += "Fibonacci: Potential reversal; "

    # Darvas Box Theory
    if price > darvas_high:
        support_resistance_score += 1
        details['Support_Resistance'] += "Darvas: Strong Bullish; "
    elif price > darvas_high:
        support_resistance_score += 0.75
        details['Support_Resistance'] += "Darvas: Mild Bullish; "
    elif price < darvas_low:
        support_resistance_score += 0.25
        details['Support_Resistance'] += "Darvas: Mild Bearish; "
    else:
        support_resistance_score += 0.5
        details['Support_Resistance'] += "Darvas: Neutral; "

    scores['Support_Resistance'] = support_resistance_score / 4  # Normalize to 1

    return scores, details

# Function to create gauge charts
def create_gauge(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={'axis': {'range': [0, 1]}},
    ))
    return fig

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

# Initialize DataFrame to store scores
scores_df = pd.DataFrame()

# Streamlit app
st.title("Stock Analysis Scores")

# Loop through each ticker to calculate scores
for ticker in tickers:
    st.header(f"Ticker: {ticker}")
    
    # Download stock data
    df = download_data(ticker, start_date, end_date)
    
    # Calculate technical indicators
    df = calculate_indicators(df)
    
    # Initialize list to store scores for each day
    daily_scores = []
    
    # Calculate scores for each day
    for i in range(len(df)):
        scores, details = calculate_scores(df.iloc[:i+1])
        scores['Date'] = df.index[i]
        scores['Ticker'] = ticker
        daily_scores.append(scores)
    
    # Append scores to the DataFrame
    ticker_scores_df = pd.DataFrame(daily_scores)
    scores_df = pd.concat([scores_df, ticker_scores_df], ignore_index=True)
    
    # Display scores in Streamlit
    st.dataframe(ticker_scores_df)

# Save scores to a CSV file
scores_df.to_csv('daily_stock_scores.csv', index=False)
st.write("Scores saved to 'daily_stock_scores.csv'")
