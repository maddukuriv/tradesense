import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import pandas_ta as pta
import plotly.graph_objects as go
import numpy as np
from scipy.stats import linregress

def display():
    st.title("Technical Analysis")

    # Sidebar setup
    st.sidebar.subheader("Interactive Charts")

    # Sidebar for user input
    ticker = st.sidebar.text_input("Enter Stock Symbol", value='RVNL.NS')
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2023-01-01'))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

    # Load stock data
    @st.cache_data
    def load_data(ticker, start, end):
        data = yf.download(ticker, start=start, end=end)
        data.reset_index(inplace=True)
        return data

    # Load index data
    @st.cache_data
    def load_index_data(ticker, start, end):
        data = yf.download(ticker, start=start, end=end)
        data.reset_index(inplace=True)
        return data

    st.title('Stock Technical Analysis')

    index_ticker = "^NSEI"  # NIFTY 50 index ticker

    # Load data
    data_load_state = st.text('Loading data...')
    data = load_data(ticker, start_date, end_date).copy()
    index_data = load_index_data(index_ticker, start_date, end_date).copy()
    data_load_state.text('Loading data...done!')

    # Calculate technical indicators
    def calculate_technical_indicators(df, index_df):
        # Moving averages
        df['10_SMA'] = ta.trend.sma_indicator(df['Close'], window=10)
        df['20_SMA'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['10_EMA'] = ta.trend.ema_indicator(df['Close'], window=10)
        df['20_EMA'] = ta.trend.ema_indicator(df['Close'], window=20)
        df['10_WMA'] = ta.trend.wma_indicator(df['Close'], window=10)
        df['20_WMA'] = ta.trend.wma_indicator(df['Close'], window=20)

        # Volume Moving Averages
        df['5_VMA'] = df['Volume'].rolling(window=5).mean()
        df['10_VMA'] = df['Volume'].rolling(window=10).mean()
        df['20_VMA'] = df['Volume'].rolling(window=20).mean()

        # Momentum Indicators
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['%K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
        df['%D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
        df['MACD'] = ta.trend.macd(df['Close'])
        df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # Volume Indicators
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'], window=14)
        df['A/D Line'] = ta.volume.acc_dist_index(df['High'], df['Low'], df['Close'], df['Volume'])

        # Volatility Indicators
        df['BB_High'], df['BB_Middle'], df['BB_Low'] = ta.volatility.bollinger_hband(df['Close']), ta.volatility.bollinger_mavg(df['Close']), ta.volatility.bollinger_lband(df['Close'])
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        df['Std Dev'] = ta.volatility.bollinger_wband(df['Close'])

        # Trend Indicators
        df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
        df['+DI'] = ta.trend.adx_pos(df['High'], df['Low'], df['Close'])
        df['-DI'] = ta.trend.adx_neg(df['High'], df['Low'], df['Close'])
        psar = pta.psar(df['High'], df['Low'], df['Close'])
        df['Parabolic_SAR'] = psar['PSARl_0.02_0.2']
        df['Ichimoku_a'] = ta.trend.ichimoku_a(df['High'], df['Low'])
        df['Ichimoku_b'] = ta.trend.ichimoku_b(df['High'], df['Low'])
        df['Ichimoku_base'] = ta.trend.ichimoku_base_line(df['High'], df['Low'])
        df['Ichimoku_conv'] = ta.trend.ichimoku_conversion_line(df['High'], df['Low'])

        # Support and Resistance Levels
        df = find_support_resistance(df)
        df['Support_Trendline'] = calculate_trendline(df, kind='support')
        df['Resistance_Trendline'] = calculate_trendline(df, kind='resistance')
        df = pivot_points(df)

        # Price Oscillators
        df['ROC'] = ta.momentum.roc(df['Close'], window=12)
        df['DPO'] = ta.trend.dpo(df['Close'], window=20)
        df['Williams %R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14)

        # Market Breadth Indicators
        df['Advances'] = df['Close'].diff().apply(lambda x: 1 if x > 0 else 0)
        df['Declines'] = df['Close'].diff().apply(lambda x: 1 if x < 0 else 0)
        df['McClellan Oscillator'] = (df['Advances'] - df['Declines']).rolling(window=19).mean() - (df['Advances'] - df['Declines']).rolling(window=39).mean()
        df['TRIN'] = (df['Advances'] / df['Declines']) / (df['Volume'][df['Advances'] > 0].sum() / df['Volume'][df['Declines'] > 0].sum())
        df['Advance-Decline Line'] = df['Advances'].cumsum() - df['Declines'].cumsum()

        # Relative Performance Indicators
        df['Price-to-Volume Ratio'] = df['Close'] / df['Volume']
        df['Relative Strength Comparison'] = df['Close'] / index_df['Close']
        df['Performance Relative to an Index'] = df['Close'].pct_change().cumsum() - index_df['Close'].pct_change().cumsum()

        return df

    # Identify Horizontal Support and Resistance
    def find_support_resistance(df, window=20):
        df['Support'] = df['Low'].rolling(window, center=True).min()
        df['Resistance'] = df['High'].rolling(window, center=True).max()
        return df

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

    # Calculate Pivot Points
    def pivot_points(df):
        df['Pivot'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
        df['R1'] = 2 * df['Pivot'] - df['Low'].shift(1)
        df['S1'] = 2 * df['Pivot'] - df['High'].shift(1)
        df['R2'] = df['Pivot'] + (df['High'].shift(1) - df['Low'].shift(1))
        df['S2'] = df['Pivot'] - (df['High'].shift(1) - df['Low'].shift(1))
        return df

    data = calculate_technical_indicators(data, index_data)

    # Function to add range buttons to the plot
    def add_range_buttons(fig):
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="7d", step="day", stepmode="backward"),
                        dict(count=14, label="14d", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True)
            )
        )

    # Plotly visualization functions
    def plot_indicator(df, indicator, title, yaxis_title='Price', secondary_y=False):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df[indicator], mode='lines', name=indicator, yaxis="y2" if secondary_y else "y1"))
        
        if secondary_y:
            fig.update_layout(
                yaxis2=dict(
                    title=indicator,
                    overlaying='y',
                    side='right'
                )
            )
        
        fig.update_layout(title=title, xaxis_title='Date', yaxis_title=yaxis_title)
        add_range_buttons(fig)
        st.plotly_chart(fig)

    def plot_moving_average(df, ma_type):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))
        fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color='blue', opacity=0.5, yaxis='y2'))
        if ma_type == 'SMA':
            fig.add_trace(go.Scatter(x=df['Date'], y=df['10_SMA'], mode='lines', name='10_SMA'))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['20_SMA'], mode='lines', name='20_SMA'))
        elif ma_type == 'EMA':
            fig.add_trace(go.Scatter(x=df['Date'], y=df['10_EMA'], mode='lines', name='10_EMA'))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['20_EMA'], mode='lines', name='20_EMA'))
        elif ma_type == 'WMA':
            fig.add_trace(go.Scatter(x=df['Date'], y=df['10_WMA'], mode='lines', name='10_WMA'))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['20_WMA'], mode='lines', name='20_WMA'))
        fig.update_layout(title=f'{ma_type} Moving Averages', xaxis_title='Date', yaxis_title='Price', yaxis2=dict(title='Volume', overlaying='y', side='right'))
        add_range_buttons(fig)
        st.plotly_chart(fig)

    def plot_macd(df):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], mode='lines', name='MACD'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_Signal'], mode='lines', name='MACD Signal'))

        # Plot MACD Histogram with different colors
        macd_hist_colors = []
        for i in range(1, len(df)):
            if df['MACD_Hist'].iloc[i] > 0:
                color = 'green' if df['MACD_Hist'].iloc[i] > df['MACD_Hist'].iloc[i - 1] else 'lightgreen'
            else:
                color = 'red' if df['MACD_Hist'].iloc[i] < df['MACD_Hist'].iloc[i - 1] else 'lightcoral'
            macd_hist_colors.append(color)

        fig.add_trace(go.Bar(x=df['Date'][1:], y=df['MACD_Hist'][1:], name='MACD Histogram', marker_color=macd_hist_colors, yaxis='y2'))

        fig.update_layout(
            title='MACD',
            xaxis_title='Date',
            yaxis_title='Price',
            yaxis2=dict(
                title='MACD Histogram',
                overlaying='y',
                side='right'
            )
        )
        add_range_buttons(fig)
        st.plotly_chart(fig)

    def plot_trendlines(df):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))

        fig.add_trace(go.Scatter(x=df['Date'], y=df['Support_Trendline'], mode='lines', name='Support Trendline', line=dict(color='green', dash='dash')))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Resistance_Trendline'], mode='lines', name='Resistance Trendline', line=dict(color='red', dash='dash')))

        fig.update_layout(title='Trendlines', xaxis_title='Date', yaxis_title='Price')
        add_range_buttons(fig)
        st.plotly_chart(fig)

    def plot_fibonacci_retracement(df):
        high = df['High'].max()
        low = df['Low'].min()

        diff = high - low
        levels = [high, high - 0.236 * diff, high - 0.382 * diff, high - 0.5 * diff, high - 0.618 * diff, low]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))

        for level in levels:
            fig.add_trace(go.Scatter(x=[df['Date'].iloc[0], df['Date'].iloc[-1]],
                                    y=[level, level],
                                    mode='lines', name=f'Level {level}', line=dict(dash='dash')))

        fig.update_layout(title='Fibonacci Retracement Levels', xaxis_title='Date', yaxis_title='Price')
        add_range_buttons(fig)
        st.plotly_chart(fig)

    def plot_gann_fan_lines(df):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))

        # Adding Gann fan lines (simple example, for more advanced lines use a proper method)
        for i in range(1, 5):
            fig.add_trace(go.Scatter(x=[df['Date'].iloc[0], df['Date'].iloc[-1]],
                                    y=[df['Close'].iloc[0], df['Close'].iloc[0] + i * (df['Close'].iloc[-1] - df['Close'].iloc[0]) / 4],
                                    mode='lines', name=f'Gann Fan {i}', line=dict(dash='dash')))

        fig.update_layout(title='Gann Fan Lines', xaxis_title='Date', yaxis_title='Price')
        add_range_buttons(fig)
        st.plotly_chart(fig)

    def plot_chart_patterns(df, pattern):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))

        # Adding example chart patterns (simple example, for more advanced patterns use a proper method)
        pattern_data = detect_chart_patterns(df, pattern)
        if pattern_data:
            for pattern_info in pattern_data:
                fig.add_trace(go.Scatter(x=pattern_info['x'], y=pattern_info['y'], mode='lines+markers', name=pattern_info['name'], line=dict(color=pattern_info['color'])))

        fig.update_layout(title=f'{pattern}', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig)

    def detect_chart_patterns(df, pattern):
        patterns = []
        if pattern == 'Head and Shoulders':
            patterns = detect_head_and_shoulders(df)
        elif pattern == 'Double Tops and Bottoms':
            patterns = detect_double_tops_and_bottoms(df)
        elif pattern == 'Flags and Pennants':
            patterns = detect_flags_and_pennants(df)
        elif pattern == 'Triangles':
            patterns = detect_triangles(df)
        elif pattern == 'Cup and Handle':
            patterns = detect_cup_and_handle(df)
        return patterns

    def detect_head_and_shoulders(df):
        patterns = []
        window = 20  # Sliding window size
        for i in range(window, len(df) - window):
            window_df = df.iloc[i - window:i + window]
            if len(window_df) < 3:  # Ensure there are enough data points
                continue
            max_high = window_df['High'].max()
            min_low = window_df['Low'].min()
            middle_idx = window_df['High'].idxmax()
            left_idx = window_df.iloc[:middle_idx]['High'].idxmax()
            right_idx = window_df.iloc[middle_idx + 1:]['High'].idxmax()

            if window_df['High'].iloc[left_idx] < max_high and window_df['High'].iloc[right_idx] < max_high and \
                    window_df['Low'].iloc[middle_idx] > min_low:
                patterns.append({
                    "x": [window_df['Date'].iloc[left_idx], window_df['Date'].iloc[middle_idx], window_df['Date'].iloc[right_idx]],
                    "y": [window_df['High'].iloc[left_idx], window_df['High'].iloc[middle_idx], window_df['High'].iloc[right_idx]],
                    "name": "Head and Shoulders",
                    "color": "orange"
                })
        return patterns

    def detect_double_tops_and_bottoms(df):
        patterns = []
        window = 20  # Sliding window size
        for i in range(window, len(df) - window):
            window_df = df.iloc[i - window:i + window]
            if len(window_df) < 3:  # Ensure there are enough data points
                continue
            max_high = window_df['High'].max()
            min_low = window_df['Low'].min()
            double_top = window_df['High'].value_counts().get(max_high, 0) > 1
            double_bottom = window_df['Low'].value_counts().get(min_low, 0) > 1

            if double_top:
                patterns.append({
                    "x": [window_df['Date'].iloc[0], window_df['Date'].iloc[-1]],
                    "y": [max_high, max_high],
                    "name": "Double Top",
                    "color": "red"
                })
            elif double_bottom:
                patterns.append({
                    "x": [window_df['Date'].iloc[0], window_df['Date'].iloc[-1]],
                    "y": [min_low, min_low],
                    "name": "Double Bottom",
                    "color": "green"
                })
        return patterns

    def detect_flags_and_pennants(df):
        patterns = []
        window = 20  # Sliding window size
        for i in range(window, len(df) - window):
            window_df = df.iloc[i - window:i + window]
            if len(window_df) < 3:  # Ensure there are enough data points
                continue
            min_low = window_df['Low'].min()
            max_high = window_df['High'].max()
            flag_pattern = ((window_df['High'] - window_df['Low']) / window_df['Low']).mean() < 0.05

            if flag_pattern:
                patterns.append({
                    "x": [window_df['Date'].iloc[0], window_df['Date'].iloc[-1]],
                    "y": [min_low, max_high],
                    "name": "Flag",
                    "color": "purple"
                })
        return patterns

    def detect_triangles(df):
        patterns = []
        window = 20  # Sliding window size
        for i in range(window, len(df) - window):
            window_df = df.iloc[i - window:i + window]
            if len(window_df) < 3:  # Ensure there are enough data points
                continue
            max_high = window_df['High'].max()
            min_low = window_df['Low'].min()
            triangle_pattern = np.all(np.diff(window_df['High']) < 0) and np.all(np.diff(window_df['Low']) > 0)

            if triangle_pattern:
                patterns.append({
                    "x": [window_df['Date'].iloc[0], window_df['Date'].iloc[-1]],
                    "y": [min_low, max_high],
                    "name": "Triangle",
                    "color": "blue"
                })
        return patterns

    def detect_cup_and_handle(df):
        patterns = []
        window = 50  # Sliding window size
        for i in range(window, len(df) - window):
            window_df = df.iloc[i - window:i + window]
            if len(window_df) < 3:  # Ensure there are enough data points
                continue
            max_high = window_df['High'].max()
            min_low = window_df['Low'].min()
            cup_shape = ((window_df['High'] - window_df['Low']) / window_df['Low']).mean() < 0.1

            if cup_shape:
                patterns.append({
                    "x": [window_df['Date'].iloc[0], window_df['Date'].iloc[len(window_df) // 2], window_df['Date'].iloc[-1]],
                    "y": [max_high, min_low, max_high],
                    "name": "Cup and Handle",
                    "color": "brown"
                })
        return patterns

    def plot_mcclellan_oscillator(df):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['McClellan Oscillator'], mode='lines', name='McClellan Oscillator'))
        fig.update_layout(title='McClellan Oscillator', xaxis_title='Date', yaxis_title='Value')
        add_range_buttons(fig)
        st.plotly_chart(fig)

    def plot_trin(df):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['TRIN'], mode='lines', name='TRIN'))
        fig.update_layout(title='Arms Index (TRIN)', xaxis_title='Date', yaxis_title='Value')
        add_range_buttons(fig)
        st.plotly_chart(fig)

    def get_signals(df):
        signals = []

        # Example logic for signals (these can be customized)
        if df['Close'].iloc[-1] > df['20_SMA'].iloc[-1]:
            signals.append(("Simple Moving Average (20_SMA)", "Hold", "Price is above the SMA."))
        else:
            signals.append(("Simple Moving Average (20_SMA)", "Sell", "Price crossed below the SMA."))

        if df['Close'].iloc[-1] > df['20_EMA'].iloc[-1]:
            signals.append(("Exponential Moving Average (20_EMA)", "Hold", "Price is above the EMA."))
        else:
            signals.append(("Exponential Moving Average (20_EMA)", "Sell", "Price crossed below the EMA."))

        if df['Close'].iloc[-1] > df['20_WMA'].iloc[-1]:
            signals.append(("Weighted Moving Average (20_WMA)", "Hold", "Price is above the WMA."))
        else:
            signals.append(("Weighted Moving Average (20_WMA)", "Sell", "Price crossed below the WMA."))

        if df['RSI'].iloc[-1] < 30:
            signals.append(("Relative Strength Index (RSI)", "Buy", "RSI crosses below 30 (oversold)."))
        elif df['RSI'].iloc[-1] > 70:
            signals.append(("Relative Strength Index (RSI)", "Sell", "RSI crosses above 70 (overbought)."))
        else:
            signals.append(("Relative Strength Index (RSI)", "Hold", "RSI is between 30 and 70."))

        if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
            signals.append(("Moving Average Convergence Divergence (MACD)", "Buy", "MACD line crosses above the signal line."))
        else:
            signals.append(("Moving Average Convergence Divergence (MACD)", "Sell", "MACD line crosses below the signal line."))

        if df['%K'].iloc[-1] < 20 and df['%D'].iloc[-1] < 20 and df['%K'].iloc[-1] > df['%D'].iloc[-1]:
            signals.append(("Stochastic Oscillator", "Buy", "%K line crosses above %D line and both are below 20."))
        elif df['%K'].iloc[-1] > 80 and df['%D'].iloc[-1] > 80 and df['%K'].iloc[-1] < df['%D'].iloc[-1]:
            signals.append(("Stochastic Oscillator", "Sell", "%K line crosses below %D line and both are above 80."))
        else:
            signals.append(("Stochastic Oscillator", "Hold", "No clear buy or sell signal."))

        if df['OBV'].diff().iloc[-1] > 0:
            signals.append(("On-Balance Volume (OBV)", "Buy", "OBV is increasing."))
        else:
            signals.append(("On-Balance Volume (OBV)", "Sell", "OBV is decreasing."))

        if df['Close'].iloc[-1] > df['VWAP'].iloc[-1]:
            signals.append(("Volume Weighted Average Price (VWAP)", "Buy", "Price crosses above the VWAP."))
        else:
            signals.append(("Volume Weighted Average Price (VWAP)", "Sell", "Price crosses below the VWAP."))

        if df['A/D Line'].diff().iloc[-1] > 0:
            signals.append(("Accumulation/Distribution Line (A/D Line)", "Buy", "A/D Line is increasing."))
        else:
            signals.append(("Accumulation/Distribution Line (A/D Line)", "Sell", "A/D Line is decreasing."))

        if df['Close'].iloc[-1] < df['BB_Low'].iloc[-1]:
            signals.append(("Bollinger Bands", "Buy", "Price crosses below the lower band."))
        elif df['Close'].iloc[-1] > df['BB_High'].iloc[-1]:
            signals.append(("Bollinger Bands", "Sell", "Price crosses above the upper band."))
        else:
            signals.append(("Bollinger Bands", "Hold", "Price is within Bollinger Bands."))

        if df['ATR'].iloc[-1] > df['ATR'].rolling(window=14).mean().iloc[-1]:
            signals.append(("Average True Range (ATR)", "Buy", "ATR is increasing, indicating higher volatility."))
        else:
            signals.append(("Average True Range (ATR)", "Sell", "ATR is decreasing, indicating lower volatility."))

        if df['Std Dev'].iloc[-1] > df['Close'].rolling(window=20).std().iloc[-1]:
            signals.append(("Standard Deviation", "Buy", "Price is below the mean minus 2 standard deviations."))
        else:
            signals.append(("Standard Deviation", "Sell", "Price is above the mean plus 2 standard deviations."))

        if df['Parabolic_SAR'].iloc[-1] < df['Close'].iloc[-1]:
            signals.append(("Parabolic SAR (Stop and Reverse)", "Buy", "Price crosses above the SAR."))
        else:
            signals.append(("Parabolic SAR (Stop and Reverse)", "Sell", "Price crosses below the SAR."))

        if df['ROC'].iloc[-1] > 0:
            signals.append(("Price Rate of Change (ROC)", "Buy", "ROC crosses above zero."))
        else:
            signals.append(("Price Rate of Change (ROC)", "Sell", "ROC crosses below zero."))

        if df['DPO'].iloc[-1] > 0:
            signals.append(("Detrended Price Oscillator (DPO)", "Buy", "DPO crosses above zero."))
        else:
            signals.append(("Detrended Price Oscillator (DPO)", "Sell", "DPO crosses below zero."))

        if df['Williams %R'].iloc[-1] < -80:
            signals.append(("Williams %R", "Buy", "Williams %R crosses above -80 (indicating oversold)."))
        elif df['Williams %R'].iloc[-1] > -20:
            signals.append(("Williams %R", "Sell", "Williams %R crosses below -20 (indicating overbought)."))
        else:
            signals.append(("Williams %R", "Hold", "Williams %R is between -80 and -20."))

        if df['Close'].iloc[-1] > df['Pivot'].iloc[-1]:
            signals.append(("Pivot Points", "Buy", "Price crosses above the pivot point."))
        else:
            signals.append(("Pivot Points", "Sell", "Price crosses below the pivot point."))

        high = df['High'].max()
        low = df['Low'].min()
        diff = high - low
        fib_levels = [high, high - 0.236 * diff, high - 0.382 * diff, high - 0.5 * diff, high - 0.618 * diff, low]
        for level in fib_levels:
            if df['Close'].iloc[-1] > level:
                signals.append(("Fibonacci Retracement Levels", "Buy", "Price crosses above a Fibonacci retracement level."))
                break
            elif df['Close'].iloc[-1] < level:
                signals.append(("Fibonacci Retracement Levels", "Sell", "Price crosses below a Fibonacci retracement level."))
                break

        gann_fan_line = [df['Close'].iloc[0] + i * (df['Close'].iloc[-1] - df['Close'].iloc[0]) / 4 for i in range(1, 5)]
        for line in gann_fan_line:
            if df['Close'].iloc[-1] > line:
                signals.append(("Gann Fan Lines", "Buy", "Price crosses above a Gann fan line."))
                break
            elif df['Close'].iloc[-1] < line:
                signals.append(("Gann Fan Lines", "Sell", "Price crosses below a Gann fan line."))
                break

        if df['McClellan Oscillator'].iloc[-1] > 0:
            signals.append(("McClellan Oscillator", "Buy", "Oscillator crosses above zero."))
        else:
            signals.append(("McClellan Oscillator", "Sell", "Oscillator crosses below zero."))

        if df['TRIN'].iloc[-1] < 1:
            signals.append(("Arms Index (TRIN)", "Buy", "TRIN below 1.0 (more advancing volume)."))
        else:
            signals.append(("Arms Index (TRIN)", "Sell", "TRIN above 1.0 (more declining volume)."))

        # Chart Patterns
        patterns = detect_chart_patterns(df, 'Summary')
        signals.extend(patterns)

        # Additional Indicators
        if df['Ichimoku_a'].iloc[-1] > df['Ichimoku_b'].iloc[-1]:
            signals.append(("Ichimoku Cloud", "Buy", "Ichimoku conversion line above baseline."))
        else:
            signals.append(("Ichimoku Cloud", "Sell", "Ichimoku conversion line below baseline."))

        if df['Relative Strength Comparison'].iloc[-1] > 1:
            signals.append(("Relative Strength Comparison", "Buy", "Stock outperforms index."))
        else:
            signals.append(("Relative Strength Comparison", "Sell", "Stock underperforms index."))

        if df['Performance Relative to an Index'].iloc[-1] > 0:
            signals.append(("Performance Relative to an Index", "Buy", "Stock outperforms index over time."))
        else:
            signals.append(("Performance Relative to an Index", "Sell", "Stock underperforms index over time."))

        if df['Advance-Decline Line'].diff().iloc[-1] > 0:
            signals.append(("Advance-Decline Line", "Buy", "Advances exceed declines."))
        else:
            signals.append(("Advance-Decline Line", "Sell", "Declines exceed advances."))

        if df['Price-to-Volume Ratio'].iloc[-1] > df['Price-to-Volume Ratio'].rolling(window=14).mean().iloc[-1]:
            signals.append(("Price-to-Volume Ratio", "Buy", "Price-to-Volume ratio increasing."))
        else:
            signals.append(("Price-to-Volume Ratio", "Sell", "Price-to-Volume ratio decreasing."))

        return signals

    signals = get_signals(data)

    # Sidebar for technical indicators
    st.sidebar.header('Technical Indicators')
    indicator_category = st.sidebar.radio('Select Indicator Category', [
        'Moving Averages', 'Momentum Indicators', 'Volume Indicators', 'Volatility Indicators', 'Trend Indicators',
        'Support and Resistance Levels', 'Price Oscillators', 'Market Breadth Indicators', 'Chart Patterns', 'Relative Performance Indicators', 'Summary'
    ])

    # Display technical indicators
    st.subheader('Technical Indicators')
    if indicator_category != 'Summary':
        if indicator_category == 'Moving Averages':
            indicators = st.selectbox("Select Moving Average", ['SMA', 'EMA', 'WMA'])
            plot_moving_average(data, indicators)
        elif indicator_category == 'Momentum Indicators':
            indicators = st.selectbox("Select Momentum Indicator", ['RSI', 'Stochastic Oscillator', 'MACD'])
            if indicators == 'RSI':
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
                fig.add_trace(go.Scatter(x=data['Date'], y=data['RSI'], mode='lines', name='RSI'))
                fig.add_trace(go.Scatter(x=data['Date'], y=[70] * len(data), mode='lines', name='RSI 70', line=dict(color='red', dash='dash')))
                fig.add_trace(go.Scatter(x=data['Date'], y=[30] * len(data), mode='lines', name='RSI 30', line=dict(color='green', dash='dash')))
                fig.update_layout(title='Relative Strength Index (RSI)', xaxis_title='Date', yaxis_title='Value')
                add_range_buttons(fig)
                st.plotly_chart(fig)
            elif indicators == 'Stochastic Oscillator':
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
                fig.add_trace(go.Scatter(x=data['Date'], y=data['%K'], mode='lines', name='%K'))
                fig.add_trace(go.Scatter(x=data['Date'], y=data['%D'], mode='lines', name='%D'))
                fig.add_trace(go.Scatter(x=data['Date'], y=[80] * len(data), mode='lines', name='%K 80', line=dict(color='red', dash='dash')))
                fig.add_trace(go.Scatter(x=data['Date'], y=[20] * len(data), mode='lines', name='%K 20', line=dict(color='green', dash='dash')))
                fig.update_layout(title='Stochastic Oscillator', xaxis_title='Date', yaxis_title='Value')
                add_range_buttons(fig)
                st.plotly_chart(fig)
            elif indicators == 'MACD':
                plot_macd(data)
        elif indicator_category == 'Volume Indicators':
            indicators = st.selectbox("Select Volume Indicator", ['OBV', 'VWAP', 'A/D Line', 'Volume Moving Averages'])
            if indicators == 'OBV':
                plot_indicator(data, 'OBV', 'On-Balance Volume (OBV)')
            elif indicators == 'VWAP':
                plot_indicator(data, 'VWAP', 'Volume Weighted Average Price (VWAP)')
            elif indicators == 'A/D Line':
                plot_indicator(data, 'A/D Line', 'Accumulation/Distribution Line')
            elif indicators == 'Volume Moving Averages':
                fig = go.Figure()
                fig.add_trace(go.Bar(x=data['Date'], y=data['Volume'], name='Volume', marker_color='blue', opacity=0.5))
                fig.add_trace(go.Scatter(x=data['Date'], y=data['5_VMA'], mode='lines', name='5_VMA'))
                fig.add_trace(go.Scatter(x=data['Date'], y=data['10_VMA'], mode='lines', name='10_VMA'))
                fig.add_trace(go.Scatter(x=data['Date'], y=data['20_VMA'], mode='lines', name='20_VMA'))
                fig.update_layout(title='Volume Moving Averages', xaxis_title='Date', yaxis_title='Volume')
                add_range_buttons(fig)
                st.plotly_chart(fig)
        elif indicator_category == 'Volatility Indicators':
            indicators = st.selectbox("Select Volatility Indicator", ['Bollinger Bands', 'ATR', 'Standard Deviation'])
            if indicators == 'Bollinger Bands':
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
                fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_High'], mode='lines', name='BB High'))
                fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_Middle'], mode='lines', name='BB Middle'))
                fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_Low'], mode='lines', name='BB Low'))
                fig.update_layout(title='Bollinger Bands', xaxis_title='Date', yaxis_title='Price')
                add_range_buttons(fig)
                st.plotly_chart(fig)
            elif indicators == 'ATR':
                plot_indicator(data, 'ATR', 'Average True Range (ATR)')
            elif indicators == 'Standard Deviation':
                plot_indicator(data, 'Std Dev', 'Standard Deviation')
        elif indicator_category == 'Trend Indicators':
            indicators = st.selectbox("Select Trend Indicator", ['Trendlines', 'Parabolic SAR', 'Ichimoku Cloud', 'ADX'])
            if indicators == 'Trendlines':
                plot_trendlines(data)
            elif indicators == 'Parabolic SAR':
                plot_indicator(data, 'Parabolic_SAR', 'Parabolic SAR')
            elif indicators == 'Ichimoku Cloud':
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
                fig.add_trace(go.Scatter(x=data['Date'], y=data['Ichimoku_a'], mode='lines', name='Ichimoku A'))
                fig.add_trace(go.Scatter(x=data['Date'], y=data['Ichimoku_b'], mode='lines', name='Ichimoku B'))
                fig.add_trace(go.Scatter(x=data['Date'], y=data['Ichimoku_base'], mode='lines', name='Ichimoku Base Line'))
                fig.add_trace(go.Scatter(x=data['Date'], y=data['Ichimoku_conv'], mode='lines', name='Ichimoku Conversion Line'))
                fig.update_layout(title='Ichimoku Cloud', xaxis_title='Date', yaxis_title='Value')
                add_range_buttons(fig)
                st.plotly_chart(fig)
            elif indicators == 'ADX':
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
                fig.add_trace(go.Scatter(x=data['Date'], y=data['ADX'], mode='lines', name='ADX'))
                fig.add_trace(go.Scatter(x=data['Date'], y=data['+DI'], mode='lines', name='+DI'))
                fig.add_trace(go.Scatter(x=data['Date'], y=data['-DI'], mode='lines', name='-DI'))
                fig.update_layout(title='Average Directional Index (ADX)', xaxis_title='Date', yaxis_title='Value')
                add_range_buttons(fig)
                st.plotly_chart(fig)
        elif indicator_category == 'Support and Resistance Levels':
            indicators = st.selectbox("Select Support and Resistance Level", ['Pivot Points', 'Fibonacci Retracement Levels', 'Gann Fan Lines'])
            if indicators == 'Pivot Points':
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
                fig.add_trace(go.Scatter(x=data['Date'], y=data['Pivot'], mode='lines', name='Pivot'))
                fig.add_trace(go.Scatter(x=data['Date'], y=data['R1'], mode='lines', name='R1'))
                fig.add_trace(go.Scatter(x=data['Date'], y=data['S1'], mode='lines', name='S1'))
                fig.add_trace(go.Scatter(x=data['Date'], y=data['R2'], mode='lines', name='R2'))
                fig.add_trace(go.Scatter(x=data['Date'], y=data['S2'], mode='lines', name='S2'))
                fig.update_layout(title='Pivot Points', xaxis_title='Date', yaxis_title='Price')
                add_range_buttons(fig)
                st.plotly_chart(fig)
            elif indicators == 'Fibonacci Retracement Levels':
                plot_fibonacci_retracement(data)
            elif indicators == 'Gann Fan Lines':
                plot_gann_fan_lines(data)
        elif indicator_category == 'Price Oscillators':
            indicators = st.selectbox("Select Price Oscillator", ['ROC', 'DPO', 'Williams %R'])
            if indicators == 'ROC':
                plot_indicator(data, 'ROC', 'Rate of Change (ROC)')
            elif indicators == 'DPO':
                plot_indicator(data, 'DPO', 'Detrended Price Oscillator (DPO)')
            elif indicators == 'Williams %R':
                plot_indicator(data, 'Williams %R', 'Williams %R')
        elif indicator_category == 'Market Breadth Indicators':
            indicators = st.selectbox("Select Market Breadth Indicator", ['Advance-Decline Line', 'McClellan Oscillator', 'TRIN'])
            if indicators == 'Advance-Decline Line':
                plot_indicator(data, 'Advance-Decline Line', 'Advance-Decline Line')
            elif indicators == 'McClellan Oscillator':
                plot_mcclellan_oscillator(data)
            elif indicators == 'TRIN':
                plot_trin(data)
        elif indicator_category == 'Chart Patterns':
            indicators = st.selectbox("Select Chart Pattern", ['Head and Shoulders', 'Double Tops and Bottoms', 'Flags and Pennants', 'Triangles', 'Cup and Handle'])
            if indicators == 'Head and Shoulders':
                plot_chart_patterns(data, 'Head and Shoulders')
            elif indicators == 'Double Tops and Bottoms':
                plot_chart_patterns(data, 'Double Tops and Bottoms')
            elif indicators == 'Flags and Pennants':
                plot_chart_patterns(data, 'Flags and Pennants')
            elif indicators == 'Triangles':
                plot_chart_patterns(data, 'Triangles')
            elif indicators == 'Cup and Handle':
                plot_chart_patterns(data, 'Cup and Handle')
        elif indicator_category == 'Relative Performance Indicators':
            indicators = st.selectbox("Select Relative Performance Indicator", ['Price-to-Volume Ratio', 'Relative Strength Comparison', 'Performance Relative to an Index'])
            if indicators == 'Price-to-Volume Ratio':
                plot_indicator(data, 'Price-to-Volume Ratio', 'Price-to-Volume Ratio', secondary_y=True)
            elif indicators == 'Relative Strength Comparison':
                plot_indicator(data, 'Relative Strength Comparison', 'Relative Strength Comparison')
            elif indicators == 'Performance Relative to an Index':
                plot_indicator(data, 'Performance Relative to an Index', 'Performance Relative to an Index')
    else:
        # Display signals in a dataframe with improved visualization
        st.subheader('Technical Indicator Signals')
        signals_df = pd.DataFrame(signals, columns=['Technical Indicator', 'Signal', 'Reason'])
        st.dataframe(signals_df.style.applymap(lambda x: 'background-color: lightgreen' if 'Buy' in x else 'background-color: lightcoral' if 'Sell' in x else '', subset=['Signal']))


# Call the display function
if __name__ == "__main__":
    display()