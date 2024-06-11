import yfinance as yf
import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# List of tickers
tickers = ["ABBOTINDIA.NS", "ADANIPOWER.NS", "AFFLE.BO", "AIAENG.BO", "AJANTPHARM.BO", "APLLTD.BO", "ALKEM.BO",
           "ARE&M.NS", "ANANDRATHI.BO", "APARINDS.BO", "ASIANPAINT.NS", "ASTRAL.NS", "ASTRAZEN.NS", "BAJFINANCE.NS",
           "BASF.NS", "BAYERCROP.BO", "BERGEPAINT.BO", "BDL.NS", "BEL.NS", "BSOFT.BO", "CDSL.NS", "CAMS.NS",
           "CARBORUNIV.BO", "CASTROLIND.NS", "CHAMBLFERT.BO", "COALINDIA.NS", "COFORGE.BO", "COLPAL.NS",
           "CONCORDBIO.BO", "COROMANDEL.BO", "CREDITACC.BO", "CUMMINSIND.NS", "CYIENT.NS", "DATAPATTNS.NS",
           "DEEPAKNTR.NS", "DIVISLAB.NS", "LALPATHLAB.NS", "RDY", "ELGIEQUIP.NS", "EMAMILTD.NS", "FIVESTAR.BO",
           "GRINFRA.NS", "GILLETTE.NS", "GLAXO.NS", "GODFRYPHLP.NS", "GRINDWELL.NS", "HAVELLS.NS", "HCLTECH.NS",
           "HAL.BO", "HONAUT.BO", "IRCTC.NS", "ISEC.BO", "INFY.NS", "IPCALAB.BO", "ITC.NS", "JBCHEPHARM.BO", "JWL.BO",
           "JYOTHYLAB.BO", "KPRMILL.NS", "KAJARIACER.NS", "KEI.BO", "LTTS.NS", "LTIM.NS", "MANKIND.NS", "MARICO.NS",
           "METROBRAND.BO", "MOTILALOFS.NS", "MPHASIS.NS", "MUTHOOTFIN.NS", "NH.NS", "NAVINFLUOR.NS", "NAM-INDIA.BO",
           "NMDC.NS", "OFSS.NS", "PGHH.NS", "PIIND.NS", "PAGEIND.NS", "PERSISTENT.NS", "PETRONET.NS", "PFIZER.NS",
           "PIDILITIND.NS", "POLYMED.NS", "POLYCAB.NS", "RRKABEL.NS", "RVNL.NS", "RATNAMANI.NS", "RITES.NS",
           "SANOFI.NS", "SCHAEFFLER.NS", "SKFINDIA.NS", "SOLARINDS.NS", "SONACOMS.NS", "SUMICHEM.NS", "SUNTV.NS",
           "SUNDRMFAST.NS", "SUPREMEIND.BO", "TATAELXSI.NS", "TATATECH.NS", "TCS.NS", "TECHM.NS", "TIMKEN.NS",
           "TITAN.NS", "TRITURBINE.NS", "TIINDIA.NS", "UNITDSPR.BO", "VGUARD.NS", "MANYAVAR.NS", "VINATIORGA.NS",
           "WIPRO.NS", "ZYDUSLIFE.NS"]

# Function to fetch data
def fetch_data(tickers, period='1d', interval='1m'):
    data = yf.download(tickers, period=period, interval=interval)
    return data['Close']

# Fetch the data for daily, weekly, and monthly periods
data_daily = fetch_data(tickers, period='1d', interval='1m')
data_weekly = fetch_data(tickers, period='5d', interval='1d')
data_monthly = fetch_data(tickers, period='1mo', interval='1d')

# Drop columns with all NaN values
data_daily.dropna(axis=1, how='all', inplace=True)
data_weekly.dropna(axis=1, how='all', inplace=True)
data_monthly.dropna(axis=1, how='all', inplace=True)

# Fill missing values with forward fill
data_daily.fillna(method='ffill', inplace=True)
data_weekly.fillna(method='ffill', inplace=True)
data_monthly.fillna(method='ffill', inplace=True)

# Fill any remaining NaNs with backward fill (in case the first row is NaN)
data_daily.fillna(method='bfill', inplace=True)
data_weekly.fillna(method='bfill', inplace=True)
data_monthly.fillna(method='bfill', inplace=True)

# Calculate daily, weekly, and monthly changes
daily_change = data_daily.iloc[-1] - data_daily.iloc[0]
percent_change_daily = (daily_change / data_daily.iloc[0]) * 100

weekly_change = data_weekly.iloc[-1] - data_weekly.iloc[0]
percent_change_weekly = (weekly_change / data_weekly.iloc[0]) * 100

monthly_change = data_monthly.iloc[-1] - data_monthly.iloc[0]
percent_change_monthly = (monthly_change / data_monthly.iloc[0]) * 100

# Create DataFrames
df_daily = pd.DataFrame({'Ticker': data_daily.columns, 'Last Traded Price': data_daily.iloc[-1].values, '% Change': percent_change_daily.values})
df_weekly = pd.DataFrame({'Ticker': data_weekly.columns, 'Last Traded Price': data_weekly.iloc[-1].values, '% Change': percent_change_weekly.values})
df_monthly = pd.DataFrame({'Ticker': data_monthly.columns, 'Last Traded Price': data_monthly.iloc[-1].values, '% Change': percent_change_monthly.values})

# Remove rows with NaN values
df_daily.dropna(inplace=True)
df_weekly.dropna(inplace=True)
df_monthly.dropna(inplace=True)

# Round off the % Change values
df_daily['% Change'] = df_daily['% Change'].round(2)
df_weekly['% Change'] = df_weekly['% Change'].round(2)
df_monthly['% Change'] = df_monthly['% Change'].round(2)

# Sort the DataFrames by '% Change'
df_daily_sorted = df_daily.sort_values(by='% Change', ascending=True)
df_weekly_sorted = df_weekly.sort_values(by='% Change', ascending=True)
df_monthly_sorted = df_monthly.sort_values(by='% Change', ascending=True)

# Function to reshape data for heatmap
def reshape_for_heatmap(df, num_columns=10):
    num_rows = int(np.ceil(len(df) / num_columns))
    reshaped_data = np.zeros((num_rows, num_columns))
    reshaped_tickers = np.empty((num_rows, num_columns), dtype=object)
    reshaped_data[:] = np.nan
    reshaped_tickers[:] = ''

    index = 0
    for y in range(num_rows):
        for x in range(num_columns):
            if index < len(df):
                reshaped_data[y, x] = df['% Change'].values[index]
                reshaped_tickers[y, x] = df['Ticker'].values[index]
                index += 1

    return reshaped_data, reshaped_tickers

# Create annotated heatmaps using Plotly
def create_horizontal_annotated_heatmap(df, title, num_columns=10):
    reshaped_data, tickers = reshape_for_heatmap(df, num_columns)
    annotations = []
    for y in range(reshaped_data.shape[0]):
        for x in range(reshaped_data.shape[1]):
            text = f'<b>{tickers[y, x]}</b><br>{reshaped_data[y, x]}%'
            annotations.append(
                go.layout.Annotation(
                    text=text,
                    x=x,
                    y=y,
                    xref='x',
                    yref='y',
                    showarrow=False,
                    font=dict(size=10, color="black", family="Arial, sans-serif"),
                    align="left"
                )
            )
    fig = go.Figure(data=go.Heatmap(
        z=reshaped_data,
        x=list(range(reshaped_data.shape[1])),
        y=list(range(reshaped_data.shape[0])),
        hoverinfo='text',
        colorscale='Blues',
        showscale=False,
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        annotations=annotations,
        autosize=False,
        width=1800,
        height=200 + 120 * len(reshaped_data),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

# Streamlit app
st.title('Stock Dashboard')

# Dropdown menu to select the period
heatmap_option = st.selectbox('Select period to view heatmap:', ['Daily Gainers/Losers', 'Weekly Gainers/Losers', 'Monthly Gainers/Losers'])

# Display the selected heatmap
if heatmap_option == 'Daily Gainers/Losers':

    fig = create_horizontal_annotated_heatmap(df_daily_sorted, 'Daily Gainers/Losers')
    st.plotly_chart(fig)
elif heatmap_option == 'Weekly Gainers/Losers':

    fig = create_horizontal_annotated_heatmap(df_weekly_sorted, 'Weekly Gainers/Losers')
    st.plotly_chart(fig)
elif heatmap_option == 'Monthly Gainers/Losers':

    fig = create_horizontal_annotated_heatmap(df_monthly_sorted, 'Monthly Gainers/Losers')
    st.plotly_chart(fig)
