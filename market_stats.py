import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# List of stock tickers (example tickers; replace with your full list)

tickers = ["ITC.NS", "JBCHEPHARM.BO", "JWL.BO", "JYOTHYLAB.BO", "KPRMILL.NS", "KAJARIACER.NS", "KEI.BO",
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
                "CHAMBLFERT.BO", "COALINDIA.NS", "COFORGE.BO", "COLPAL.NS", "CONCORDBIO.BO", "COROMANDEL.BO", "CREDITACC.BO" ] # Add all 100 tickers here


# Fetch historical data for all stocks
data = {}
for ticker in tickers:
    try:
        data[ticker] = yf.download(ticker, period='1y')
    except Exception as e:
        st.write(f"Error fetching data for {ticker}: {e}")

def calculate_atr(df, n=14):
    df['High_Low'] = df['High'] - df['Low']
    df['High_Close'] = abs(df['High'] - df['Close'].shift(1))
    df['Low_Close'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=n).mean()
    return df

def calculate_std(df, n=14):
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(window=n).std() * np.sqrt(n)  # Annualized volatility
    return df

def calculate_liquidity(df, n=14):
    df['Average_Volume'] = df['Volume'].rolling(window=n).mean()
    return df

# Apply the calculations to all dataframes
for ticker in tickers:
    if ticker in data:
        data[ticker] = calculate_atr(data[ticker], n=14)
        data[ticker] = calculate_std(data[ticker], n=14)
        data[ticker] = calculate_liquidity(data[ticker], n=14)

# Extract and rank metrics
metrics_list = []

for ticker in tickers:
    if ticker in data and not data[ticker].empty:
        atr = data[ticker]['ATR'].iloc[-1]
        volatility = data[ticker]['Volatility'].iloc[-1]
        avg_volume = data[ticker]['Average_Volume'].iloc[-1]
        metrics_list.append({'Ticker': ticker, 'ATR': atr, 'Volatility': volatility, 'Average_Volume': avg_volume})

metrics_summary = pd.DataFrame(metrics_list)

# Rank the stocks by ATR, Volatility, and Average Volume
metrics_summary['ATR_Rank'] = metrics_summary['ATR'].rank(ascending=False)
metrics_summary['Volatility_Rank'] = metrics_summary['Volatility'].rank(ascending=False)
metrics_summary['Volume_Rank'] = metrics_summary['Average_Volume'].rank(ascending=False)

# Composite rank: lower is better (average of the ranks)
metrics_summary['Composite_Rank'] = (metrics_summary['ATR_Rank'] + metrics_summary['Volatility_Rank'] + metrics_summary['Volume_Rank']) / 3
metrics_summary = metrics_summary.sort_values(by='Composite_Rank')

# Streamlit UI
st.title("Stock Volatility and Liquidity Analysis")
st.dataframe(metrics_summary)

# Plot metrics using Plotly
fig = go.Figure()

fig.add_trace(go.Bar(
    x=metrics_summary['Ticker'],
    y=metrics_summary['ATR'],
    name='ATR',
    marker_color='indianred'
))

fig.add_trace(go.Bar(
    x=metrics_summary['Ticker'],
    y=metrics_summary['Volatility'],
    name='Volatility',
    marker_color='lightsalmon'
))

fig.add_trace(go.Bar(
    x=metrics_summary['Ticker'],
    y=metrics_summary['Average_Volume'],
    name='Average Volume',
    marker_color='lightblue'
))

fig.update_layout(
    title='Stock Metrics: ATR, Volatility, and Average Volume',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='Value',
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1
)

st.plotly_chart(fig)
