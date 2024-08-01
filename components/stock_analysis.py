import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pmdarima import auto_arima
from scipy.signal import hilbert, cwt, ricker

def stock_analysis_app():
    st.sidebar.subheader("Stock Analysis")

    # User input for the stock ticker
    ticker = st.sidebar.text_input('Enter Stock Ticker (e.g., BAJAJFINSV.NS): ', 'BAJAJFINSV.NS')
    submenu = st.sidebar.selectbox("Select Analysis Type", [ "Technical Analysis", "Sentiment Analysis", "Price Forecast"])

    # Date inputs limited to the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=395)

    

    if submenu == "Technical Analysis":
        # Define a function to download data
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
            
            kvo = ta.kvo(df['High'], df['Low'], df['Close'], df['Volume'])
            df['Klinger'] = kvo['KVO_34_55_13']
            
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

            # Klinger Oscillator
            klinger = data['Klinger'].iloc[-1]
            if klinger > 50:
                momentum_score += 1
                details['Momentum'] += "Klinger: Strong Bullish; "
            elif klinger > 0:
                momentum_score += 0.75
                details['Momentum'] += "Klinger: Mild Bullish; "
            elif klinger > -50:
                momentum_score += 0.25
                details['Momentum'] += "Klinger: Mild Bearish; "
            else:
                momentum_score += 0
                details['Momentum'] += "Klinger: Strong Bearish; "

            scores['Momentum'] = momentum_score / 13  # Normalize to 1

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
            
        # Streamlit App
        st.title('Stock Technical Analysis')
        
        

        # Date inputs
        start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=365))
        end_date = st.sidebar.date_input("End Date", value=datetime.now() + timedelta(days=1))

        if ticker:
            # Download data from Yahoo Finance
            data = download_data(ticker, start_date, end_date)
            index_ticker = '^NSEI'  # Nifty 50 index ticker
            index_data = yf.download(index_ticker, period='1y', interval='1d')

            # Ensure the index is in datetime format
            data.index = pd.to_datetime(data.index)
            index_data.index = pd.to_datetime(index_data.index)

            # Calculate technical indicators
            data = calculate_indicators(data)

            # Option to select table or visualization
            view_option = st.sidebar.radio("Select View", ('Visualization', 'Analysis'))

            if view_option == 'Visualization':
                # Define indicator groups
                indicator_groups = {
                    "Trend Indicators": ["5_day_EMA", "10_day_EMA", "20_day_EMA", "MACD", "MACD_signal", "MACD_hist", "Trend_Line", "Ichimoku_conv", "Ichimoku_base", "Ichimoku_A", "Ichimoku_B", "Parabolic_SAR", "SuperTrend", "Donchian_High", "Donchian_Low", "Vortex_Pos", "Vortex_Neg", "ADX", "Jaw", "Teeth", "Lips"],
                    "Momentum Indicators": ["RSI", "Stochastic_%K", "Stochastic_%D", "ROC", "DPO", "Williams_%R", "CMO", "CCI", "RVI", "RVI_Signal", "Ultimate_Oscillator", "Trix", "Trix_Signal", "Klinger"],
                    "Volatility": ["ATR", "Std_Dev", "BB_High", "BB_Low","20_day_SMA", "Keltner_High", "Keltner_Low"],
                    "Volume Indicators": ["OBV", "A/D_line", "Price_to_Volume", "TRIN", "Advance_Decline_Line", "McClellan_Oscillator", "Volume_Profile", "Chaikin_MF", "Williams_AD", "Ease_of_Movement", "MFI", "Elder_Ray_Bull", "Elder_Ray_Bear", "VWAP"],
                    "Support and Resistance Levels": ["Pivot_Point", "Resistance_1", "Support_1", "Resistance_2", "Support_2", "Resistance_3", "Support_3", "Fib_0.0", "Fib_0.236", "Fib_0.382", "Fib_0.5", "Fib_0.618", "Fib_1.0", "Darvas_High", "Darvas_Low"],
                    "Other Indicators": ["Relative_Strength", "Performance_vs_Index"]
                }

                # Create multiselect options for each indicator group
                selected_indicators = []
                for group_name, indicators in indicator_groups.items():
                    with st.expander(group_name):
                        selected_indicators.extend(st.sidebar.multiselect(f'Select {group_name}', indicators))
                
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
                    y_axis_name = f'y{i+2}'

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
                        'text': f'{ticker} Price and Technical Indicators',
                        'y': 0.97,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    },
                    height=900,
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
                    legend=dict(x=0.5, y=-0.02, orientation='h', xanchor='center', yanchor='top')
                )

                fig.update_layout(
                    hovermode='x unified',
                    hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell")
                )

                st.plotly_chart(fig)
            else:
                # Calculate scores
                scores, details = calculate_scores(data)
                
                # Display data table
                st.write(data)

                # Create and display gauges and details in two columns
                for key, value in scores.items():
                    col1, col2 = st.columns([1, 2])
                    with col2:
                        st.plotly_chart(create_gauge(value, key))
                    with col1:
                        st.markdown(f"### {key} Indicators")
                        details_list = details[key].split("; ")
                        for detail in details_list:
                            if detail:  # Check if detail is not an empty string
                                st.markdown(f"- {detail}")

                # Calculate overall weightage score
                overall_score = np.mean(list(scores.values()))
                overall_description = f"{overall_score*100:.1f}%"
                recommendation = get_recommendation(overall_score)
            
                col1, col2 = st.columns([1, 2])
                with col2:
                    st.plotly_chart(create_gauge(overall_score, 'Overall Score'))
                with col1:
                    st.markdown(f"### Overall Score: {overall_description}")
                    st.markdown(f"<p style='font-size:20px;'>Recommendation: {recommendation}</p>", unsafe_allow_html=True)

    elif submenu == "Sentiment Analysis":

        # Initialize NewsApiClient with your API key
            newsapi = NewsApiClient(api_key='252b2075083945dfbed8945ddc240a2b')
            analyzer = SentimentIntensityAnalyzer()

            def fetch_news(company_name, start_date, end_date):
                # Fetch news articles related to the company name
                all_articles = newsapi.get_everything(q=company_name,
                                                    language='en',
                                                    from_param=start_date.strftime('%Y-%m-%d'),
                                                    to=end_date.strftime('%Y-%m-%d'),
                                                    sort_by='publishedAt',
                                                    page_size=50,
                                                    sources='the-times-of-india, financial-express, the-hindu, bloomberg, cnbc')
                articles = []
                for article in all_articles['articles']:
                    articles.append({
                        'title': article['title'],
                        'description': article['description'],
                        'url': article['url'],
                        'publishedAt': article['publishedAt'],
                        'source': article['source']['name']
                    })
                return articles

            def perform_sentiment_analysis(articles):
                sentiments = []
                for article in articles:
                    if article['description']:
                        score = analyzer.polarity_scores(article['description'])
                        article['sentiment'] = score
                        sentiments.append(article)
                    else:
                        article['sentiment'] = {'compound': 0, 'neg': 0, 'neu': 0, 'pos': 0}
                        sentiments.append(article)
                return sentiments

            def make_recommendation(sentiments):
                avg_sentiment = sum([s['sentiment']['compound'] for s in sentiments]) / len(sentiments)
                if avg_sentiment > 0.1:
                    return "Based on the sentiment analysis, it is recommended to BUY the stock."
                elif avg_sentiment < -0.1:
                    return "Based on the sentiment analysis, it is recommended to NOT BUY the stock."
                else:
                    return "Based on the sentiment analysis, it is recommended to HOLD OFF on any action."

            def count_sentiments(sentiments):
                positive = sum(1 for s in sentiments if s['sentiment']['compound'] > 0.1)
                negative = sum(1 for s in sentiments if s['sentiment']['compound'] < -0.1)
                neutral = len(sentiments) - positive - negative
                return positive, negative, neutral

            def generate_wordcloud(text, title):
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.title(title)
                plt.axis('off')
                st.pyplot(plt)

            st.title("Stock News Sentiment Analysis")

            if ticker:
                # Fetch company name using yfinance
                try:
                    company_info = yf.Ticker(ticker)
                    company_name = company_info.info['shortName']
                except KeyError:
                    company_name = ticker

                st.write(f"Fetching news for: {company_name}")

                if company_name:
                    with st.spinner("Fetching news..."):
                        articles = fetch_news(company_name, start_date, end_date)

                    if articles:
                        with st.spinner("Performing sentiment analysis..."):
                            sentiments = perform_sentiment_analysis(articles)

                        df = pd.DataFrame(sentiments)

                        # Radio button options for Articles and Summary
                        view_option = st.sidebar.radio("View Options", ["Articles", "Summary"])

                        if view_option == "Articles":
                            st.write("Recent News Articles:")
                            for i, row in df.iterrows():
                                st.write(f"**Article {i+1}:** {row['title']}")
                                st.write(f"**Published At:** {row['publishedAt']}")
                                st.write(f"**Description:** {row['description']}")
                                st.write(f"**URL:** {row['url']}")
                                st.write(f"**Source:** {row['source']}")
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

                            # Sentiment score bar chart
                            fig_bar = px.bar(df, x=df.index, y=df['sentiment'].apply(lambda x: x['compound']),
                                            labels={'x': 'Article', 'y': 'Sentiment Score'},
                                            title='Sentiment Score per Article')
                            st.plotly_chart(fig_bar)

                            # Sentiment trend over time
                            df['publishedAt'] = pd.to_datetime(df['publishedAt']).dt.tz_localize(None)
                            df.set_index('publishedAt', inplace=True)
                            df = df.sort_index()

                            # Aggregate sentiment by day
                            daily_sentiment = df['sentiment'].apply(lambda x: x['compound']).resample('D').mean()

                            fig_line = px.line(daily_sentiment.rolling(window=5).mean(),
                                            labels={'value': 'Sentiment Score', 'index': 'Date'},
                                            title='Sentiment Trend Over Time')
                            st.plotly_chart(fig_line)

                            # Correlate sentiment with stock price
                            
                            stock_data = yf.download(ticker, start=start_date, end=end_date)
                            if not stock_data.empty:
                                stock_data.index = stock_data.index.tz_localize(None)
                                stock_data = stock_data[['Close']]
                                stock_data['Sentiment'] = daily_sentiment
                                combined_data = pd.concat([stock_data, daily_sentiment], axis=1).dropna()

                                fig_combined = go.Figure()
                                fig_combined.add_trace(go.Scatter(x=combined_data.index, y=combined_data['Close'],
                                                                mode='lines', name='Stock Price'))
                                fig_combined.add_trace(go.Scatter(x=combined_data.index, y=combined_data['Sentiment'],
                                                                mode='lines', name='Sentiment Score', yaxis='y2'))

                                fig_combined.update_layout(
                                    title='Stock Price and Sentiment Over Time',
                                    xaxis_title='Date',
                                    yaxis_title='Stock Price',
                                    yaxis2=dict(title='Sentiment Score', overlaying='y', side='right'),
                                    legend=dict(x=0, y=1, traceorder='normal')
                                )
                                st.plotly_chart(fig_combined)

                            # Sentiment distribution pie chart
                            
                            sentiment_counts = pd.Series([positive, negative, neutral], index=['Positive', 'Negative', 'Neutral'])
                            fig_pie = px.pie(sentiment_counts, values=sentiment_counts, names=sentiment_counts.index, title='Sentiment Distribution')
                            st.plotly_chart(fig_pie)

                            # Generate and display word clouds for positive and negative articles
                            positive_text = ' '.join(df[df['sentiment'].apply(lambda x: x['compound']) > 0.1]['description'].dropna())
                            negative_text = ' '.join(df[df['sentiment'].apply(lambda x: x['compound']) < -0.1]['description'].dropna())
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
                        st.write("No news articles found for this company.")
                else:
                    st.write("Invalid ticker symbol.")

    elif submenu == "Price Forecast":

        # Function to fetch stock data from Yahoo Finance
        def get_stock_data(ticker, start_date, end_date):
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            stock_data.reset_index(inplace=True)
            return stock_data

        # Function to calculate technical indicators
        def calculate_technical_indicators(df):
            # Moving Average Convergence Divergence (MACD)
            df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
                        
            # Relative Strength Index (RSI)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
                        
            # Bollinger Bands
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['Std_Dev'] = df['Close'].rolling(window=20).std()
            df['Upper_Band'] = df['SMA_20'] + (df['Std_Dev'] * 2)
            df['Lower_Band'] = df['SMA_20'] - (df['Std_Dev'] * 2)
            df['BBW'] = (df['Upper_Band'] - df['Lower_Band']) / df['SMA_20']  # Bollinger Band Width
                        
            # Exponential Moving Average (EMA)
            df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
                        
            # Average True Range (ATR)
            df['High-Low'] = df['High'] - df['Low']
            df['High-PrevClose'] = np.abs(df['High'] - df['Close'].shift(1))
            df['Low-PrevClose'] = np.abs(df['Low'] - df['Close'].shift(1))
            df['True_Range'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
            df['ATR'] = df['True_Range'].rolling(window=14).mean()
                        
            # Rate of Change (ROC)
            df['ROC'] = df['Close'].pct_change(periods=12) * 100
                        
            # On-Balance Volume (OBV)
            df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
                        
            return df

        # Functions for Fourier, Wavelet, and Hilbert transforms
        def calculate_fourier(df, n=5):
            close_fft = np.fft.fft(df['Close'].values)
            fft_df = pd.DataFrame({'fft': close_fft})
            fft_df['absolute'] = np.abs(fft_df['fft'])
            fft_df['angle'] = np.angle(fft_df['fft'])
            fft_df = fft_df.sort_values(by='absolute', ascending=False).head(n)
            fft_df = fft_df.reindex(range(len(df)), fill_value=0)  # Ensure it matches the length of the stock data
            return fft_df['absolute']

        def calculate_wavelet(df):
            widths = np.arange(1, 31)
            cwt_matrix = cwt(df['Close'], ricker, widths)
            max_wavelet = np.max(cwt_matrix, axis=0)
            return pd.Series(max_wavelet).reindex(range(len(df)), fill_value=0)  # Ensure it matches the length of the stock data

        def calculate_hilbert(df):
            analytic_signal = hilbert(df['Close'])
            amplitude_envelope = np.abs(analytic_signal)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            return pd.Series(amplitude_envelope), pd.Series(instantaneous_phase)

        # Streamlit UI
        st.title("Stock Price Prediction with SARIMA Model")

        start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=1500))
        end_date = st.sidebar.date_input("End Date", value=datetime.now() + timedelta(days=1))
    
        forecast_period = st.sidebar.number_input("Forecast Period (days)", value=10, min_value=1, max_value=30)

        # Fetch the data
        stock_data = get_stock_data(ticker, start_date, end_date)

        # Calculate technical indicators
        stock_data = calculate_technical_indicators(stock_data)

        # Calculate Fourier, Wavelet, and Hilbert transforms
        stock_data['Fourier'] = calculate_fourier(stock_data)
        stock_data['Wavelet'] = calculate_wavelet(stock_data)
        stock_data['Hilbert_Amplitude'], stock_data['Hilbert_Phase'] = calculate_hilbert(stock_data)

        # Drop rows with NaN values
        stock_data.dropna(inplace=True)

        # Extract the closing prices and technical indicators
        close_prices = stock_data['Close']
        technical_indicators = stock_data[['MACD', 'Signal_Line', 'MACD_Hist', 'RSI', 'SMA_20', 'Upper_Band', 'Lower_Band', 'BBW', 'EMA_50', 'ATR', 'ROC', 'OBV', 'Fourier', 'Wavelet', 'Hilbert_Amplitude', 'Hilbert_Phase']]

        # Check correlation with close prices
        correlations = technical_indicators.corrwith(close_prices).sort_values()

        # Display correlation as a bar chart
        fig_corr = go.Figure(go.Bar(
            x=correlations.values,
            y=correlations.index,
            orientation='h'
        ))
        fig_corr.update_layout(
            title="Correlation with Close Prices",
            xaxis_title="Correlation",
            yaxis_title="Indicators",
            yaxis=dict(tickmode='linear')
        )
        st.plotly_chart(fig_corr)

        # Train SARIMA model with exogenous variables (technical indicators)
        sarima_model = auto_arima(
            close_prices,
            exogenous=technical_indicators,
            start_p=1,
            start_q=1,
            max_p=3,
            max_q=3,
            m=7,  # Weekly seasonality
            start_P=0,
            seasonal=True,
            d=1,
            D=1,
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )

        # Forecasting the next n business days (excluding weekends)
        future_technical_indicators = technical_indicators.tail(forecast_period).values
        forecast = sarima_model.predict(n_periods=forecast_period, exogenous=future_technical_indicators)

        # Generate the forecasted dates excluding weekends
        forecasted_dates = pd.bdate_range(start=stock_data['Date'].iloc[-1], periods=forecast_period + 1)[1:]

        forecasted_df = pd.DataFrame({'Date': forecasted_dates, 'Forecasted_Close': forecast})

        # Plotly interactive figure with time slider
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=stock_data['Date'],
            y=stock_data['Close'],
            mode='lines',
            name='Close Price'
        ))

        # Add forecasted prices
        fig.add_trace(go.Scatter(
            x=forecasted_df['Date'],
            y=forecasted_df['Forecasted_Close'],
            mode='lines',
            name='Forecasted Close Price',
            line=dict(color='orange')
        ))

        fig.update_layout(
            title=f'Stock Price Forecast for {ticker}',
            xaxis_title='Date',
            yaxis_title='Close Price',
            legend=dict(x=0.01, y=0.99),
        )

        st.plotly_chart(fig)

        # Display forecasted prices
        st.write("Forecasted Prices for the next {} days:".format(forecast_period))
        st.dataframe(forecasted_df)

        # Display model summary
        st.write("Model Summary:")
        st.text(sarima_model.summary())