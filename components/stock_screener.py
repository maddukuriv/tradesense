import streamlit as st
import pandas as pd
import numpy as np
from supabase import create_client
from datetime import datetime, timedelta
from utils.constants import SUPABASE_URL,SUPABASE_KEY,sp500_tickers, ftse100_tickers, Largecap,Midcap,Smallcap,crypto_largecap,crypto_midcap,Currencies,Commodities,Indices
import pandas_ta as ta
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from scipy.stats import linregress


# Supabase client setup 
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def stock_screener_app():
    st.sidebar.subheader("Stock Screener")
    st.title('Stock Screener')

    # Dropdown for selecting ticker category
    ticker_category = st.sidebar.selectbox("Select Index/Crypto", [ "Indices","Commodities","Currencies","Cryptocurrencies","Stocks-Largecap","Stocks-Midcap","Stocks-Smallcap","Stocks-Largemidcap","Stocks-Midsmall","Stocks-Multicap","Stocks-S&P 500", "Stocks-FTSE 100"])

    # Dropdown for Strategies
    #submenu = st.sidebar.selectbox("Select Strategy", ["Momentum", "Mean Reversion", "volume Driven", "Trend Following","Breakout","Volatility Based","Reversal","Trend Conformation","Volatility Reversion","volume & Momentum"])




    # Define strategy options with icons
    strategy_map = {
        "Momentum": ":chart_with_upwards_trend:",
        "Mean Reversion": ":repeat:",
        "Volume Driven": ":bar_chart:",
        "Trend Following": ":chart:",
        "Breakout": ":fire:",
        "Volatility Based": ":cyclone:",
        "Reversal": ":arrows_clockwise:",
        "Trend Confirmation": ":white_check_mark:",
        "Volatility Reversion": ":arrows_counterclockwise:",
        "Volume & Momentum": ":rocket:",
    }

    st.sidebar.write("## Select Strategy")

    # Initialize selected strategy with "Momentum" as default
    selected_strategy = "Momentum"

    # Display buttons in sidebar (single column)
    for strategy, icon in strategy_map.items():
        if st.sidebar.button(f"{icon} {strategy}", key=strategy):
            selected_strategy = strategy

    # Assign selected strategy to submenu
    submenu = selected_strategy

    # Display the selected strategy in the sidebar
    st.sidebar.write(f"### Selected Strategy: {strategy_map[selected_strategy]} {selected_strategy}")

    # Example of using submenu elsewhere
    if submenu:
        st.write(f"Running analysis for strategy: {submenu}")

 


    # Set tickers based on selected category
    tickers = {
        "Stocks-Largecap":Largecap,
        "Stocks-Midcap": Midcap,
        "Stocks-Smallcap": Smallcap,
        "Stocks-Largemidcap":Largecap + Midcap,
        "Stocks-Midsmallcap":Midcap +Smallcap,
        "Stocks-Multicap": Largecap + Midcap +Smallcap,
        "Stocks-S&P 500": sp500_tickers,
        "Stocks-FTSE 100": ftse100_tickers,
        "Cryptocurrencies": crypto_largecap + crypto_midcap,
        "Indices":Indices,
        "Commodities":Commodities,
        "Currencies":Currencies

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
                plus_dm = df['high'].diff()
                minus_dm = df['low'].diff()
                plus_dm[plus_dm < 0] = 0
                minus_dm[minus_dm > 0] = 0
                tr = pd.concat([df['high'] - df['low'], 
                                (df['high'] - df['close'].shift()).abs(), 
                                (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
                atr = tr.rolling(window=14).mean()
                plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
                minus_di = abs(100 * (minus_dm.ewm(alpha=1/14).mean() / atr))
                adx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).ewm(alpha=1/14).mean()
                return adx
    
    def volume_profile_fixed_range(df, start_idx, end_idx, bins=50):
        data = df.iloc[start_idx:end_idx]
        price_min, price_max = data['close'].min(), data['close'].max()
        price_range = np.linspace(price_min, price_max, bins)
        volume_profile = np.zeros(len(price_range))

        for i in range(len(price_range) - 1):
            mask = (data['close'] >= price_range[i]) & (data['close'] < price_range[i + 1])
            volume_profile[i] = data['volume'][mask].sum()

        df['VPFR'] = 0
        df.iloc[start_idx:end_idx, df.columns.get_loc('VPFR')] = volume_profile[np.searchsorted(price_range, data['close']) - 1]

        return df['VPFR']

    def volume_profile_visible_range(df, visible_range=100, bins=50):
        end_idx = len(df)
        start_idx = max(0, end_idx - visible_range)
        return volume_profile_fixed_range(df, start_idx, end_idx, bins)
    
    def calculate_accelerator_oscillator(df):
        # Awesome Oscillator
        df['AO'] = df['high'].rolling(window=5).mean() - df['low'].rolling(window=34).mean()
        # 5-period SMA of AO
        df['AO_SMA_5'] = df['AO'].rolling(window=5).mean()
        # Accelerator Oscillator
        df['AC'] = df['AO'] - df['AO_SMA_5']
        return df['AC']

    def calculate_awesome_oscillator(df):
        midpoint = (df['high'] + df['low']) / 2
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
        # close-to-close Volatility
        df['Vol_CtC'] = df['close'].pct_change().rolling(window=window).std() * np.sqrt(window)
        return df['Vol_CtC']

    def volatility_zero_trend_close_to_close(df, window):
        # Zero Trend close-to-close Volatility
        returns = df['close'].pct_change()
        zero_returns = returns[returns == 0]
        df['Vol_ZtC'] = zero_returns.rolling(window=window).std() * np.sqrt(window)
        df['Vol_ZtC'].fillna(0, inplace=True)  # Handle NaNs
        return df['Vol_ZtC']

    def volatility_ohlc(df, window):
        # OHLC Volatility
        df['HL'] = df['high'] - df['low']
        df['OC'] = np.abs(df['close'] - df['open'])
        df['Vol_OHLC'] = df[['HL', 'OC']].max(axis=1).rolling(window=window).mean()
        return df['Vol_OHLC']

    def volatility_index(df, window):
        # Volatility Index (standard deviation of returns)
        df['Vol_Index'] = df['close'].pct_change().rolling(window=window).std() * np.sqrt(window)
        return df['Vol_Index']
    
    def historical_volatility(df, window=252):
        # Calculate the daily returns
        df['Returns'] = df['close'].pct_change()
        
        # Calculate the rolling standard deviation of returns over the specified window
        df['Hist_Vol'] = df['Returns'].rolling(window=window).std()
        
        # Annualize the historical volatility (assuming 252 trading days in a year)
        df['Hist_Vol_Annualized'] = df['Hist_Vol'] * np.sqrt(window)
        
        return df['Hist_Vol_Annualized']    
    

    def williams_fractal(df):
        def fractal_high(df, n):
            return df['high'][(df['high'] == df['high'].rolling(window=n, center=True).max()) &
                            (df['high'] > df['high'].shift(1)) &
                            (df['high'] > df['high'].shift(2)) &
                            (df['high'] > df['high'].shift(-1)) &
                            (df['high'] > df['high'].shift(-2))]

        def fractal_low(df, n):
            return df['low'][(df['low'] == df['low'].rolling(window=n, center=True).min()) &
                            (df['low'] < df['low'].shift(1)) &
                            (df['low'] < df['low'].shift(2)) &
                            (df['low'] < df['low'].shift(-1)) &
                            (df['low'] < df['low'].shift(-2))]

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

    
    def get_stock_data(ticker):


        try:
            response = (
            supabase.table("stock_data")
            .select("*")
            .filter("ticker", "eq", ticker)
            .order("date", desc=False)  # Order by latest date
            .execute()
        )
            if response.data:
                data = pd.DataFrame(response.data)
                if 'date' in data.columns:
                    data['date'] = pd.to_datetime(data['date'])
                    data.drop_duplicates(subset=['date'], keep='first', inplace=True)  # Fix here
                    data.set_index('date', inplace=True)
                    
                return data
            else:
                return pd.DataFrame()
        except Exception as e:
            return pd.DataFrame()


    @st.cache_data
    def fetch_and_calculate_data(ticker):
        try:
            response = supabase.table("stock_data").select("*").eq("ticker", ticker).execute()
            
            if response.data:
                data = pd.DataFrame(response.data)
                
                if not data.empty:
                    data = calculate_indicators(data)
                
                return data
            else:
                return pd.DataFrame()  # Return an empty DataFrame if no data is found

        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return pd.DataFrame()  # Return an empty DataFrame in case of an error

    

    def calculate_indicators(df):

        df['ALMA'] = ta.alma(df['close'])

        # Simple Moving Average (SMA)
        df['SMA_20'] = df['close'].rolling(window=20).mean()

        # Exponential Moving Average (EMA)
        df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()

        # Bollinger Bands
        df['BB_Middle'] = df['close'].rolling(window=20).mean()
        df['BB_Std'] = df['close'].rolling(window=20).std()
        df['BB_high'] = df['BB_Middle'] + (df['BB_Std'] * 2)
        df['BB_low'] = df['BB_Middle'] - (df['BB_Std'] * 2)

        # MACD
        df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        # ADX calculation
        df['ADX'] = calculate_adx(df)
     
        
        # On-Balance volume (OBV)
        df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()

        # Average True Range (ATR)
        df['ATR'] = atr(df['high'], df['low'], df['close'])

        # Aroon Indicator
        df['Aroon_Up'], df['Aroon_Down'] = aroon_up_down(df['high'], df['low'])

        # Double Exponential Moving Average (DEMA)
        df['DEMA'] = 2 * df['EMA_20'] - df['EMA_20'].ewm(span=20, adjust=False).mean()

        # Envelopes
        df['Envelope_high'] = df['close'].rolling(window=20).mean() * 1.02
        df['Envelope_low'] = df['close'].rolling(window=20).mean() * 0.98

        # Guppy Multiple Moving Average (GMMA)
        df['GMMA_Short'] = df['close'].ewm(span=3, adjust=False).mean()
        df['GMMA_Long'] = df['close'].ewm(span=30, adjust=False).mean()

        # Hull Moving Average (HMA)
        df['HMA'] = hull_moving_average(df['close'])

        # Ichimoku Cloud
        df['Ichimoku_Tenkan'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
        df['Ichimoku_Kijun'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
        df['Ichimoku_Senkou_Span_A'] = ((df['Ichimoku_Tenkan'] + df['Ichimoku_Kijun']) / 2).shift(26)
        df['Ichimoku_Senkou_Span_B'] = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)

        # Keltner Channels
        df['KC_Middle'] = df['close'].rolling(window=20).mean()
        df['ATR_10'] = atr(df['high'], df['low'], df['close'], window=10)
        df['KC_high'] = df['KC_Middle'] + (df['ATR_10'] * 2)
        df['KC_low'] = df['KC_Middle'] - (df['ATR_10'] * 2)

        # Least Squares Moving Average (LSMA)
        df['LSMA'] = lsma(df['close'])

        # Parabolic SAR
        df['Parabolic_SAR'] = parabolic_sar(df['high'], df['low'], df['close'])
        
        # SuperTrend
        supertrend = ta.supertrend(df['high'], df['low'], df['close'], length=7, multiplier=3.0)
        df['SuperTrend'] = supertrend['SUPERT_7_3.0']

        # Moving Average Channel (MAC)
        df['MAC_Upper'], df['MAC_lower'] = moving_average_channel(df['close'], window=20, offset=2)

        # Price Channel
        df['Price_Channel_Upper'], df['Price_Channel_lower'] = price_channel(df['high'], df['low'], window=20)

        # Triple EMA (TEMA)
        df['TEMA_20'] = triple_ema(df['close'], window=20)

        # Momentum Indicators
        df['AO'] = calculate_awesome_oscillator(df)
        df['AC'] = calculate_accelerator_oscillator(df)
        df['CMO'] = rsi(df['close'], window=14) - 50
        df['CCI'] = (df['close'] - df['close'].rolling(window=20).mean()) / (0.015 * df['close'].rolling(window=20).std())
        df['CRSI'] = (rsi(df['close'], window=3) + rsi(df['close'], window=2) + rsi(df['close'], window=5)) / 3
        df['Coppock'] = df['close'].diff(14).ewm(span=10, adjust=False).mean() + df['close'].diff(11).ewm(span=10, adjust=False).mean()
        df['DPO'] = df['close'].shift(int(20 / 2 + 1)) - df['close'].rolling(window=20).mean()
        df['KST'] = df['close'].rolling(window=10).mean() + df['close'].rolling(window=15).mean() + df['close'].rolling(window=20).mean() + df['close'].rolling(window=30).mean()
        df['KST_Signal'] = df['KST'].rolling(window=9).mean()
        df['Momentum'] = df['close'] - df['close'].shift(10)
        df['RSI'] = rsi(df['close'])
        df['ROC'] = df['close'].pct_change(12)
        df['Stochastic_%K'] = (df['close'] - df['low'].rolling(window=14).min()) / (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()) * 100
        df['Stochastic_%D'] = df['Stochastic_%K'].rolling(window=3).mean()
        df['Stochastic_RSI'] = (rsi(df['close'], window=14) - rsi(df['close'], window=14).rolling(window=14).min()) / (rsi(df['close'], window=14).rolling(window=14).max() - rsi(df['close'], window=14).rolling(window=14).min())
        df['TRIX'] = df['close'].ewm(span=15, adjust=False).mean().pct_change(1)
        df['TSI'] = df['close'].diff(1).ewm(span=25, adjust=False).mean() / df['close'].diff(1).abs().ewm(span=13, adjust=False).mean()
        df['TSI_Signal'] = df['TSI'].ewm(span=9, adjust=False).mean()
        df['Ultimate_Oscillator'] = (4 * (df['close'] - df['low']).rolling(window=7).sum() + 2 * (df['close'] - df['low']).rolling(window=14).sum() + (df['close'] - df['low']).rolling(window=28).sum()) / ((df['high'] - df['low']).rolling(window=7).sum() + (df['high'] - df['low']).rolling(window=14).sum() + (df['high'] - df['low']).rolling(window=28).sum()) * 100

        # volume Indicators
        df['10_volume_MA'] = df['volume'].rolling(window=10).mean()
        df['30_volume_MA'] = df['volume'].rolling(window=30).mean()
        df['AD'] = (df['close'] - df['low'] - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
        df['BoP'] = (df['close'] - df['open']) / (df['high'] - df['low'])
        df['CMF'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
        df['CO'] = df['close'].diff(3).ewm(span=10, adjust=False).mean()
        df['EMV'] = (df['high'] - df['low']) / df['volume']
        df['EFI'] = df['close'].diff(1) * df['volume']
        df['KVO'] = (df['high'] - df['low']).ewm(span=34, adjust=False).mean() - (df['high'] - df['low']).ewm(span=55, adjust=False).mean()
        df['KVO_Signal'] = df['KVO'].ewm(span=13, adjust=False).mean()
        df['MFI'] = (df['close'].diff(1) / df['close'].shift(1) * df['volume']).rolling(window=14).mean()
        df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        df['PVT'] = (df['close'].pct_change(1) * df['volume']).cumsum()
        df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['VO'] = df['volume'].pct_change(12)
        df['Vortex_Pos'] = df['high'].diff(1).abs().rolling(window=14).sum() / atr(df['high'], df['low'], df['close'])
        df['Vortex_Neg'] = df['low'].diff(1).abs().rolling(window=14).sum() / atr(df['high'], df['low'], df['close'])
        df['volume'] = df['volume']

        # volume Weighted Moving Average (VWMA)
        df['VWMA'] = ta.vwma(df['close'], df['volume'], length=20)

        # Net volume
        df['Net_volume'] = df['volume'] * (df['close'].diff() / df['close'].shift(1))

        # volume Profile Fixed Range (VPFR)
        df['VPFR'] = volume_profile_fixed_range(df, start_idx=0, end_idx=len(df)-1)

        # volume Profile Visible Range (VPVR)
        df['VPVR'] = volume_profile_visible_range(df, visible_range=100)

        # Volatility Indicators
        df['BB_%B'] = (df['close'] - df['BB_low']) / (df['BB_high'] - df['BB_low'])
        df['BB_Width'] = (df['BB_high'] - df['BB_low']) / df['close']
        df['Chaikin_Volatility'] = (df['high'] - df['low']).ewm(span=10, adjust=False).mean()
        df['Choppiness_Index'] = np.log10((df['high'] - df['low']).rolling(window=14).sum() / (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min())) * 100
        df['Hist_Vol_Annualized'] = historical_volatility(df)
        df['Mass_Index'] = (df['high'] - df['low']).rolling(window=25).sum() / (df['high'] - df['low']).rolling(window=9).sum()
        df['RVI'] = df['close'].rolling(window=10).mean() / df['close'].rolling(window=10).std()
        df['Standard_Deviation'] = df['close'].rolling(window=20).std()

        # Volatility close-to-close
        df['Vol_CtC'] = volatility_close_to_close(df, window=20)

        # Volatility Zero Trend close-to-close
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

        df['Pivot_Point'], df['Resistance_1'], df['Support_1'], df['Resistance_2'], df['Support_2'], df['Resistance_3'], df['Support_3'] = pivot_points(df['high'], df['low'], df['close'])
        
        # Statistical indicators
        df['Correlation_Coefficient'] = correlation_coefficient(df['close'], df['close'].shift(1))
        df['Log_Correlation'] = log_correlation(df['close'], df['close'].shift(1))
        df['Linear_Regression_Curve'] = linear_regression_curve(df['close'])
        df['Linear_Regression_Slope'] = linear_regression_slope(df['close'])
        df['Standard_Error'] = standard_error(df['close'])
        df['Standard_Error_Band_Upper'], df['Standard_Error_Band_lower'] = standard_error_bands(df['close'])


        # Calculate Advance/Decline
        df['Advance_Decline'] = df['close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)).cumsum()
        df['Chop_Zone'] = choppiness_index(df['high'], df['low'], df['close'])
        df['Chande_Kroll_Stop_Long'], df['Chande_Kroll_Stop_Short'] = chande_kroll_stop(df['high'], df['low'], df['close'])
        df['Fisher_Transform'], df['Fisher_Transform_Signal'] = fisher_transform(df['close'])
        # Calculate Median Price
        df['Median_Price'] = (df['high'] + df['low']) / 2
        df['Relative_Vigor_Index'] = relative_vigor_index(df['open'], df['high'], df['low'], df['close'])
        df['SMI_Ergodic'], df['SMI_Ergodic_Signal'] = smi_ergodic(df['close'])
        # Calculate Spread
        df['Spread'] = df['high'] - df['low']

        # Calculate Typical Price
        df['Typical_Price'] = (df['high'] + df['low'] + df['close']) / 3
        df['Williams_%R'] = williams_r(df['high'], df['low'], df['close'])
        df['Williams_Alligator_Jaw'], df['Williams_Alligator_Teeth'], df['Williams_Alligator_Lips'] = alligator(df['high'], df['low'], df['close'])
        df['ZigZag'] = zigzag(df['close'])


        return df

        
        

    def fetch_company_info(ticker):
        try:
            # Fetch data from the Supabase database
            response = supabase.table("stock_info").select("*").execute()

            # Check if data exists and convert it to a DataFrame
            if response.data:
                df = pd.DataFrame(response.data)
                
                # Filter the data based on the ticker symbol
                company_data = df[df['ticker'] == ticker]
                
                if not company_data.empty:
                    long_name = company_data['longname'].iloc[0]  # Get the long name
                    sector = company_data['sector'].iloc[0]  # Get the sector
                    industry = company_data['industry'].iloc[0]  # Get the industry
                    return long_name, sector, industry
                else:
                    st.error(f"No data found for ticker '{ticker}' in the database.")
                    return None, None, None
            else:
                st.error(f"No data found in the database.")
                return None, None, None

        except Exception as e:
            st.error(f"Error fetching company info for ticker '{ticker}': {e}")
            return None, None, None

    def fetch_latest_data(tickers_with_dates):
        technical_data = []
        for ticker, occurrence_date in tickers_with_dates:
            try:
                data = get_stock_data(ticker)
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
                    'close': latest_data['close'],
                    'open':latest_data['open'],
                    'volume': latest_data['volume'],
                    'SMA_20': latest_data['SMA_20'],
                    'EMA_20': latest_data['EMA_20'],
                    'BB_high': latest_data['BB_high'],
                    'BB_low': latest_data['BB_low'],
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
                    'Envelope_high': latest_data['Envelope_high'],
                    'Envelope_low': latest_data['Envelope_low'],
                    'GMMA_Short': latest_data['GMMA_Short'],
                    'GMMA_Long': latest_data['GMMA_Long'],
                    'HMA': latest_data['HMA'],
                    'Ichimoku_Tenkan': latest_data['Ichimoku_Tenkan'],
                    'Ichimoku_Kijun': latest_data['Ichimoku_Kijun'],
                    'Ichimoku_Senkou_Span_A': latest_data['Ichimoku_Senkou_Span_A'],
                    'Ichimoku_Senkou_Span_B': latest_data['Ichimoku_Senkou_Span_B'],
                    'KC_high': latest_data['KC_high'],
                    'KC_low': latest_data['KC_low'],
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
                    'KST_Signal': latest_data['KST_Signal'],
                    'Momentum': latest_data['Momentum'],
                    'ROC': latest_data['ROC'],
                    'Stochastic_%K': latest_data['Stochastic_%K'],
                    'Stochastic_%D': latest_data['Stochastic_%D'],
                    'Stochastic_RSI': latest_data['Stochastic_RSI'],
                    'TRIX': latest_data['TRIX'],
                    'TSI': latest_data['TSI'],
                    'TSI_Signal': latest_data['TSI_Signal'],
                    'Ultimate_Oscillator': latest_data['Ultimate_Oscillator'],
                    'AD': latest_data['AD'],
                    'BoP': latest_data['BoP'],
                    'CMF': latest_data['CMF'],
                    'CO': latest_data['CO'],
                    'EMV': latest_data['EMV'],
                    'EFI': latest_data['EFI'],
                    'KVO': latest_data['KVO'],
                    'KVO_Signal': latest_data['KVO_Signal'],
                    'MFI': latest_data['MFI'],
                    'PVT': latest_data['PVT'],
                    'VWAP': latest_data['VWAP'],
                    'VO': latest_data['VO'],
                    'Vortex_Pos': latest_data['Vortex_Pos'],
                    'Vortex_Neg': latest_data['Vortex_Neg'],
                    'volume': latest_data['volume'],
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
                    'Net_volume':latest_data['Net_volume'],
                    'VPFR':latest_data['VPFR'],
                    'VPVR':latest_data['VPVR'],
                    'MAC_Upper':latest_data['MAC_Upper'],
                    'MAC_lower':latest_data['MAC_lower'],
                    'Price_Channel_Upper':latest_data['Price_Channel_Upper'],
                    'Price_Channel_lower':latest_data['Price_Channel_lower'],
                    'TEMA_20':latest_data['TEMA_20'],
                    'Vol_CtC':latest_data['Vol_CtC'],
                    'Vol_ZtC':latest_data['Vol_ZtC'],
                    'Vol_OHLC':latest_data['Vol_OHLC'],
                    'Vol_Index':latest_data['Vol_Index'],
                    'Fractal_Up':latest_data['Fractal_Up'],
                    'Fractal_Down':latest_data['Fractal_Down'],
                    'Correlation_Coefficient':latest_data['Correlation_Coefficient'],
                    'Log_Correlation':latest_data['Log_Correlation'],
                    'Linear_Regression_Curve':latest_data['Linear_Regression_Curve'],
                    'Linear_Regression_Slope':latest_data['Linear_Regression_Slope'],
                    'Standard_Error':latest_data['Standard_Error'],
                    'Standard_Error_Band_Upper':latest_data['Standard_Error_Band_Upper'],
                    'Standard_Error_Band_lower':latest_data['Standard_Error_Band_lower'],
                    'Advance_Decline':latest_data['Advance_Decline'],
                    'Chop_Zone':latest_data['Chop_Zone'],
                    'Chande_Kroll_Stop_Long':latest_data['Chande_Kroll_Stop_Long'],
                    'Chande_Kroll_Stop_Short':latest_data['Chande_Kroll_Stop_Short'],
                    'Fisher_Transform':latest_data['Fisher_Transform'],
                    'Fisher_Transform_Signal':latest_data['Fisher_Transform_Signal'],
                    'Median_Price':latest_data['Median_Price'],
                    'Relative_Vigor_Index':latest_data['Relative_Vigor_Index'],
                    'SMI_Ergodic':latest_data['SMI_Ergodic'],
                    'SMI_Ergodic_Signal':latest_data['SMI_Ergodic_Signal'],
                    'Spread':latest_data['Spread'],
                    'Typical_Price':latest_data['Typical_Price'],
                    'Williams_%R':latest_data['Williams_%R'],
                    'Williams_Alligator_Jaw':latest_data['Williams_Alligator_Jaw'],
                    'Williams_Alligator_Teeth':latest_data['Williams_Alligator_Teeth'],
                    'Williams_Alligator_Lips':latest_data['Williams_Alligator_Lips'],
                    'ZigZag':latest_data['ZigZag'],
                  

                })
            except Exception as e:
                st.error(f"Error fetching latest data for ticker '{ticker}': {e}")

        return pd.DataFrame(technical_data)

    def check_signal(data, strategy):
        recent_data = data[-5:]
        if strategy == "Momentum":
            # Calculate MACD AND SIGNAL
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data['MACD_hist'] = data['MACD'] - data['MACD_signal']
            # Query stocks where macd is above the macd signal in the last 5 days
            for i in range(1, len(recent_data)):
                if (recent_data['MACD'].iloc[i] > recent_data['MACD_signal'].iloc[i] and
                    recent_data['MACD'].iloc[i-1] < recent_data['MACD_signal'].iloc[i-1] and
                    recent_data['MACD_hist'].iloc[i] > 0):
                    return recent_data.index[i]

        elif strategy == "Mean Reversion":
            # Query stocks where price is above the bollinger low in the last 5 days
            for i in range(1, len(recent_data)):
                if (recent_data['close'].iloc[i] > recent_data['BB_low'].iloc[i] and
                    recent_data['close'].iloc[i-1] <= recent_data['BB_low'].iloc[i-1]):
                    return recent_data.index[i]

        elif strategy == "volume Driven":
            data['10_volume_MA'] = data['volume'].rolling(window=10).mean()
            data['30_volume_MA'] = data['volume'].rolling(window=30).mean()

            for i in range(1, len(recent_data)):
                if (recent_data['10_volume_MA'].iloc[i] > recent_data['30_volume_MA'].iloc[i] and
                    recent_data['10_volume_MA'].iloc[i-1] <= recent_data['30_volume_MA'].iloc[i-1]):
                    return recent_data.index[i]
                
        elif strategy == "Trend Following":
            # Calculate Ichimoku Cloud components
            data['Ichimoku_Tenkan'] = (data['high'].rolling(window=9).max() + data['low'].rolling(window=9).min()) / 2
            data['Ichimoku_Kijun'] = (data['high'].rolling(window=26).max() + data['low'].rolling(window=26).min()) / 2
            data['Ichimoku_Senkou_Span_A'] = ((data['Ichimoku_Tenkan'] + data['Ichimoku_Kijun']) / 2).shift(26)
            data['Ichimoku_Senkou_Span_B'] = ((data['high'].rolling(window=52).max() + data['low'].rolling(window=52).min()) / 2).shift(26)

            # Query stocks where price is above the Ichimoku Cloud in the last 5 days
            for i in range(len(recent_data) - 5, len(recent_data)):
                if (recent_data['close'].iloc[i] > recent_data['Ichimoku_Senkou_Span_A'].iloc[i] and
                    recent_data['close'].iloc[i] > recent_data['Ichimoku_Senkou_Span_B'].iloc[i]):
                    return recent_data.index[i]
                
        elif strategy == "Breakout":
            # Calculate Keltner Channel components
            data['KC_Middle'] = data['close'].rolling(window=20).mean()
            data['ATR_10'] = atr(data['high'], data['low'], data['close'], window=10)
            data['KC_high'] = data['KC_Middle'] + (data['ATR_10'] * 2)
            data['KC_low'] = data['KC_Middle'] - (data['ATR_10'] * 2)

            # Query stocks where price breaks out of the Keltner Channel
            for i in range(1, len(recent_data)):
                if (recent_data['close'].iloc[i] > recent_data['KC_high'].iloc[i] and  
                    recent_data['close'].iloc[i-1] <= recent_data['KC_high'].iloc[i-1]): # Price breakout above the channel
                    return recent_data.index[i]
        
        elif strategy == "Reversal":
            # Calculate Williams %R and CMO
            data['Williams_%R'] = williams_r(data['high'], data['low'], data['close'])
            data['CMO'] = rsi(data['close'], window=14) - 50

            # Query stocks where Williams %R is below -80 and CMO is below -50
            for i in range(1, len(recent_data)):
                if (recent_data['Williams_%R'].iloc[i] < -80 and  # Williams %R below -80
                    recent_data['CMO'].iloc[i] < -50):  # CMO below -50
                    return recent_data.index[i]

        elif strategy == "Trend Confirmation ":
            # Calculate GMMA components
            data['GMMA_Short'] = data['close'].ewm(span=3, adjust=False).mean()
            data['GMMA_Long'] = data['close'].ewm(span=30, adjust=False).mean()

            # Query stocks where GMMA Short is greater than GMMA Long (indicating a potential breakout)
            for i in range(1, len(recent_data)):
                if (recent_data['GMMA_Short'].iloc[i] > recent_data['GMMA_Long'].iloc[i] and 
                    recent_data['GMMA_Short'].iloc[i-1] <= recent_data['GMMA_Long'].iloc[i-1]):  # GMMA Short above GMMA Long
                    return recent_data.index[i]
                
        elif strategy == "Volatility Reversion":
            # Calculate Choppiness Index and ATR
            data['Choppiness_Index'] = np.log10((data['high'] - data['low']).rolling(window=14).sum() / \
                                                (data['high'].rolling(window=14).max() - data['low'].rolling(window=14).min())) * 100
            data['ATR'] = atr(data['high'], data['low'], data['close'], window=14)

            # Query stocks where market is choppy (Choppiness Index high) and ATR is low
            for i in range(1, len(recent_data)):
                if (recent_data['Choppiness_Index'].iloc[i] > 60 and  # high Choppiness Index indicates choppy market
                    recent_data['ATR'].iloc[i] < recent_data['ATR'].rolling(window=14).mean().iloc[i]):  # low ATR
                    return recent_data.index[i]

            # Prepare for breakout when ATR increases as Choppiness Index decreases
            for i in range(2, len(recent_data)):
                if (recent_data['Choppiness_Index'].iloc[i] < 50 and  # Choppiness Index decreases
                    recent_data['ATR'].iloc[i] > recent_data['ATR'].iloc[i-1]):  # ATR increases
                    return recent_data.index[i]


        elif strategy == "volume & Momentum":

            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
         
            for i in range(1, len(recent_data)):
                if (recent_data['MACD'].iloc[i] > recent_data['MACD_signal'].iloc[i] and
                    recent_data['MACD'].iloc[i-1] < recent_data['MACD_signal'].iloc[i-1] and
                    recent_data['MACD'].iloc[i] > 0):
                    return recent_data.index[i]
                
        elif strategy == "Volatility Based":
            # Calculate Bollinger Bands Width and ATR
            data['BB_high'] = data['close'].rolling(window=20).mean() + (2 * data['close'].rolling(window=20).std())
            data['BB_low'] = data['close'].rolling(window=20).mean() - (2 * data['close'].rolling(window=20).std())
            data['BB_Width'] = (data['BB_high'] - data['BB_low']) / data['close']
            data['ATR'] = atr(data['high'], data['low'], data['close'], window=14)

            # Query stocks with low ATR and narrow Bollinger Bands Width
            for i in range(1, len(recent_data)):
                if (recent_data['ATR'].iloc[i] < recent_data['ATR'].rolling(window=14).mean().iloc[i] and  # low ATR
                    recent_data['BB_Width'].iloc[i] < recent_data['BB_Width'].rolling(window=14).mean().iloc[i]):  # Narrow BB Width
                    return recent_data.index[i]


        return None
    
    def count_scanned_tickers(tickers):
            valid_tickers = []
            for ticker in tickers:
                try:
                    data = get_stock_data(ticker)
                    if not data.empty:
                        valid_tickers.append(ticker)
                except Exception:
                    pass
            return len(valid_tickers)
    
    tickers_with_signals = []
    progress_bar = st.progress(0)
    progress_step = 1 / len(tickers)

    total_tickers = len(tickers)
    scanned_tickers = 0

    for i, ticker in enumerate(tickers):
            try:
                data = get_stock_data(ticker)
                if data.empty:
                    continue
                data = calculate_indicators(data)
                occurrence_date = check_signal(data, submenu)
                if occurrence_date:
                    tickers_with_signals.append((ticker, occurrence_date))
                scanned_tickers += 1
            except Exception as e:
                st.error(f"Error processing data for ticker '{ticker}': {str(e)}")

            progress_bar.progress((i + 1) * progress_step)

    st.write(f"Successfully scanned {scanned_tickers} out of {total_tickers} tickers.")

    df_signals = fetch_latest_data(tickers_with_signals)



    if not df_signals.empty:
        
        table1_columns = ['Ticker', 'Date of Occurrence', 'Company Name', 'Sector', 'Industry', 'close', 'volume']
        trend_columns = ['Ticker', 'close','ALMA','Aroon_Up', 'Aroon_Down','ADX', 'BB_high', 'SMA_20','BB_low', 'DEMA', 'Envelope_high', 'Envelope_low', 'GMMA_Short', 'GMMA_Long', 'HMA', 'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'Ichimoku_Senkou_Span_A', 'Ichimoku_Senkou_Span_B', 'KC_high', 'KC_low', 'LSMA','EMA_20','MACD', 'MACD_signal', 'MACD_hist', 'Parabolic_SAR', 'SuperTrend','MAC_Upper','MAC_lower','Price_Channel_Upper','Price_Channel_lower','TEMA_20','Advance_Decline','Chande_Kroll_Stop_Long', 'Chande_Kroll_Stop_Short','Williams_Alligator_Jaw', 'Williams_Alligator_Teeth', 'Williams_Alligator_Lips']
        momentum_columns = ['Ticker','AO', 'AC', 'CMO', 'CCI', 'CRSI', 'Coppock', 'DPO', 'KST','KST_Signal', 'Momentum', 'RSI','ROC', 'Stochastic_%K', 'Stochastic_%D', 'Stochastic_RSI', 'TRIX', 'TSI','TSI_Signal', 'Ultimate_Oscillator','Relative_Vigor_Index', 'SMI_Ergodic', 'SMI_Ergodic_Signal','Fisher_Transform', 'Fisher_Transform_Signal', 'Williams_%R']
        volume_columns = ['Ticker','AD', 'BoP', 'CMF', 'CO', 'EMV', 'EFI', 'KVO','KVO_Signal', 'MFI', 'Net_volume','OBV', 'PVT', 'VWAP','VWMA', 'VO','VPFR','VPVR', 'Vortex_Pos', 'Vortex_Neg', 'volume', 'Spread']
        volatility_columns = ['Ticker','ATR','BB_%B', 'BB_Width', 'Chaikin_Volatility', 'Choppiness_Index', 'Hist_Vol_Annualized', 'Mass_Index', 'RVI', 'Standard_Deviation','Vol_CtC','Vol_ZtC','Vol_OHLC','Vol_Index','Chop_Zone' , 'ZigZag']
        support_resistance_columns = ['Ticker', 'close','Pivot_Point', 'Resistance_1', 'Support_1', 'Resistance_2', 'Support_2', 'Resistance_3', 'Support_3','Fractal_Up','Fractal_Down','Typical_Price']
        statitical_columns=['Ticker','Correlation_Coefficient','Log_Correlation','Linear_Regression_Curve','Linear_Regression_Slope','Standard_Error','Standard_Error_Band_Upper','Standard_Error_Band_lower', 'Median_Price']
        

        st.title("Stocks Based on Selected Strategy")
        st.write(f"Stocks with {submenu} signal in the last 5 days:")

        
        st.dataframe(df_signals[table1_columns])

        st.subheader("volume Indicators")
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



if __name__ == "__main__":
    stock_screener_app()
