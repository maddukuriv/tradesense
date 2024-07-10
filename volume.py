import yfinance as yf
import pandas as pd
import streamlit as st

# List of large-cap tickers
largecap_tickers = [
    "ITC.NS", "JBCHEPHARM.BO", "JWL.BO", "JYOTHYLAB.BO", "KPRMILL.NS", "KAJARIACER.NS", "KEI.BO",
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
    "CHAMBLFERT.BO", "COALINDIA.NS", "COFORGE.BO", "COLPAL.NS", "CONCORDBIO.BO", "COROMANDEL.BO", "CREDITACC.BO"
]

def fetch_data(ticker):
    data = yf.download(ticker, period="1mo", interval="1d")
    data['20D_MA_Volume'] = data['Volume'].rolling(window=20).mean()
    data['Volume_Change'] = data['Volume'].pct_change()
    data['Price_Change'] = data['Close'].pct_change()
    
    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    return data

# Streamlit app
st.title("Volume and Price Analysis of Large-Cap Stocks with Momentum Indicators")
st.write("This app analyzes the volume and price changes of large-cap stocks, displaying the 20-day moving average volume, today's volume, percentage difference, RSI, MACD, and closing price.")

# Create a list to store results
results = []

for ticker in largecap_tickers:
    try:
        ticker_data = fetch_data(ticker)
        latest_data = ticker_data.iloc[-1]
        ma_volume = latest_data['20D_MA_Volume']
        today_volume = latest_data['Volume']
        price_change = latest_data['Price_Change'] * 100  # Convert to percentage
        rsi = latest_data['RSI']
        macd = latest_data['MACD']
        signal = latest_data['Signal']
        close_price = latest_data['Close']
        if pd.notna(ma_volume) and pd.notna(today_volume):
            pct_diff = ((today_volume - ma_volume) / ma_volume) * 100  # Convert to percentage
            results.append({
                'Ticker': ticker,
                '20D_MA_Volume': ma_volume,
                'Volume_Today': today_volume,
                '%_Difference': pct_diff,
                'Price_Change (%)': price_change,
                'RSI': rsi,
                'MACD': macd,
                'Signal': signal,
                'Close': close_price
            })
    except Exception as e:
        st.write(f"Could not fetch data for {ticker}: {e}")

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Query stocks with RSI < 65, Volume % change > 0, MACD < Signal
query_df = results_df[(results_df['RSI'] < 65) & (results_df['%_Difference'] > 0) & (results_df['MACD'] < results_df['Signal'])]

# Display the main DataFrame
st.write("### Volume and Price Analysis Data with Momentum Indicators")
st.write(results_df)

# Display the queried DataFrame
st.write("### Queried Stocks (RSI < 65, Volume % Change > 0, MACD < Signal)")
st.write(query_df)

# Download link for the main DataFrame
@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

csv = convert_df(results_df)

st.download_button(
    label="Download main data as CSV",
    data=csv,
    file_name='volume_price_analysis_with_momentum.csv',
    mime='text/csv',
)

# Download link for the queried DataFrame
query_csv = convert_df(query_df)

st.download_button(
    label="Download queried data as CSV",
    data=query_csv,
    file_name='queried_stocks.csv',
    mime='text/csv',
)

if st.button('Refresh Data'):
    st.experimental_rerun()
