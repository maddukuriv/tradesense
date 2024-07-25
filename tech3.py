import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta

# Define a function to download data
def download_data(ticker):
    df = yf.download(ticker,start_date, end_date, interval="1d")
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

# Streamlit App
st.title('Technical Indicators Dashboard')

# User input for the stock ticker
ticker = st.sidebar.text_input('Enter Stock Ticker (e.g., BAJAJFINSV.NS): ', 'BAJAJFINSV.NS')

# Date inputs
start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", value=datetime.now() + timedelta(days=1))

if ticker:
    # Download data from Yahoo Finance
    data = download_data(ticker)
    index_ticker = '^NSEI'  # Nifty 50 index ticker
    index_data = yf.download(index_ticker, period='1y', interval='1d')

    # Ensure the index is in datetime format
    data.index = pd.to_datetime(data.index)
    index_data.index = pd.to_datetime(index_data.index)

    # Calculate technical indicators
    data = calculate_indicators(data)

    # Option to select table or visualization
    view_option = st.sidebar.radio("Select View", ('Visualization', 'Table'))

if view_option == 'Visualization':
    # Define indicator groups
    indicator_groups = {
        "Trend Indicators": ["5_day_EMA", "10_day_EMA", "20_day_EMA", "MACD", "MACD_signal", "MACD_hist", "Trend_Line", "Ichimoku_conv", "Ichimoku_base", "Ichimoku_A", "Ichimoku_B", "Parabolic_SAR", "SuperTrend", "Donchian_High", "Donchian_Low", "Vortex_Pos", "Vortex_Neg", "ADX"],
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

    show_candlestick = st.sidebar.checkbox('Show Candlestick Chart')
   

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
    st.write(data)
