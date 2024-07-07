import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import numpy as np
import plotly.graph_objects as go
from scipy.stats import linregress


def display():
    st.title("Stock Screener")

    # Sidebar for stock selection
    st.sidebar.subheader("Indices")
    submenu = st.sidebar.radio("Select Option", ["India-LargeCap", "India-MidCap", "FTSE100", "S&P500"])

    # List of stock tickers
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

    midcap_tickers = [
        "PNCINFRA.NS", "INDIASHLTR.NS", "RAYMOND.NS", "KAMAHOLD.BO", "BENGALASM.BO", "CHOICEIN.NS",
        "GRAVITA.NS", "HGINFRA.NS", "JKPAPER.NS", "MTARTECH.NS", "HAPPSTMNDS.NS", "SARDAEN.NS",
        "WELENT.NS", "LTFOODS.NS", "GESHIP.NS", "SHRIPISTON.NS", "SHAREINDIA.NS", "CYIENTDLM.NS", "VTL.NS",
        "EASEMYTRIP.NS", "LLOYDSME.NS", "ROUTE.NS", "VAIBHAVGBL.NS", "GOKEX.NS", "USHAMART.NS", "EIDPARRY.NS",
        "KIRLOSBROS.NS", "MANINFRA.NS", "CMSINFO.NS", "RALLIS.NS", "GHCL.NS", "NEULANDLAB.NS", "SPLPETRO.NS",
        "MARKSANS.NS", "NAVINFLUOR.NS", "ELECON.NS", "TANLA.NS", "KFINTECH.NS", "TIPSINDLTD.NS", "ACI.NS",
        "SURYAROSNI.NS", "GPIL.NS", "GMDCLTD.NS", "MAHSEAMLES.NS", "TDPOWERSYS.NS", "TECHNOE.NS", "JLHL.NS"
    ]

    ftse100_tickers = [
        "III.L", "ADM.L", "AAF.L", "AAL.L", "ANTO.L", "AHT.L", "ABF.L", "AZN.L",
        "AUTO.L", "AV.L", "BME.L", "BA.L", "BARC.L", "BDEV.L", "BEZ.L", "BKGH.L",
        "BP.L", "BATS.L", "BT.A.L", "BNZL.L", "BRBY.L", "CNA.L", "CCH.L", "CPG.L",
        "CTEC.L", "CRDA.L", "DARK.L", "DCC.L", "DGE.L", "DPLM.L", "EZJ.L", "ENT.L",
        "EXPN.L", "FCIT.L", "FRAS.L", "FRES.L", "GLEN.L", "GSK.L", "HLN.L", "HLMA.L",
        "HL.L", "HIK.L", "HWDN.L", "HSBA.L", "IMI.L", "IMB.L", "INF.L", "IHG.L",
        "ICP.L", "IAG.L", "ITRK.L", "JD.L", "KGF.L", "LAND.L", "LGEN.L", "LLOY.L",
        "LSEG.L", "LMP.L", "MNG.L", "MKS.L", "MRO.L", "MNDI.L", "NG.L", "NWG.L",
        "NXT.L", "PSON.L", "PSH.L", "PSN.L", "PHNX.L", "PRU.L", "RKT.L", "REL.L",
        "RTO.L", "RMV.L", "RIO.L", "RR.L", "SGE.L", "SBRY.L", "SDR.L", "SMT.L",
        "SGRO.L", "SVT.L", "SHEL.L", "SN.L", "SMDS.L", "SMIN.L", "SKG.L", "SPX.L",
        "SSE.L", "STAN.L", "TW.L", "TSCO.L", "ULVR.L", "UTG.L", "UU.L", "VTY.L",
        "VOD.L", "WEIR.L", "WTB.L", "WPP.L"
    ]

    sp500_tickers = [
        "MMM", "AOS", "ABT", "ABBV", "ACN", "ADBE", "AMD", "AES", "AFL", "A", "APD", "ABNB",
        "AKAM", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", "GOOGL", "GOOG", "MO", "AMZN",
        "AMCR", "AEE", "AAL", "AEP", "AXP", "AIG", "AMT", "AWK", "AMP", "AME", "AMGN", "APH",
        "ADI", "ANSS", "AON", "APA", "AAPL", "AMAT", "APTV", "ACGL", "ADM", "ANET", "AJG",
        "AIZ", "T", "ATO", "ADSK", "ADP", "AZO", "AVB", "AVY", "AXON", "BKR", "BALL", "BAC",
        "BK", "BBWI", "BAX", "BDX", "BRK.B", "BBY", "BIO", "TECH", "BIIB", "BLK", "BX", "BA",
        "BKNG", "BWA", "BXP", "BSX", "BMY", "AVGO", "BR", "BRO", "BF.B", "BLDR", "BG", "CDNS",
        "CZR", "CPT", "CPB", "COF", "CAH", "KMX", "CCL", "CARR", "CTLT", "CAT", "CBOE", "CBRE",
        "CDW", "CE", "COR", "CNC", "CNP", "CF", "CHRW", "CRL", "SCHW", "CHTR", "CVX", "CMG",
        "CB", "CHD", "CI", "CINF", "CTAS", "CSCO", "C", "CFG", "CLX", "CME", "CMS", "KO",
        "CTSH", "CL", "CMCSA", "CAG", "COP", "ED", "STZ", "CEG", "COO", "CPRT", "GLW", "CPAY",
        "CTVA", "CSGP", "COST", "CTRA", "CRWD", "CCI", "CSX", "CMI", "CVS", "DHR", "DRI",
        "DVA", "DAY", "DECK", "DE", "DAL", "DVN", "DXCM", "FANG", "DLR", "DFS", "DG", "DLTR",
        "D", "DPZ", "DOV", "DOW", "DHI", "DTE", "DUK", "DD", "EMN", "ETN", "EBAY", "ECL",
        "EIX", "EW", "EA", "ELV", "LLY", "EMR", "ENPH", "ETR", "EOG", "EPAM", "EQT", "EFX",
        "EQIX", "EQR", "ESS", "EL", "ETSY", "EG", "EVRG", "ES", "EXC", "EXPE", "EXPD", "EXR",
        "XOM", "FFIV", "FDS", "FICO", "FAST", "FRT", "FDX", "FIS", "FITB", "FSLR", "FE",
        "FI", "FMC", "F", "FTNT", "FTV", "FOXA", "FOX", "BEN", "FCX", "GRMN", "IT", "GE",
        "GEHC", "GEV", "GEN", "GNRC", "GD", "GIS", "GM", "GPC", "GILD", "GPN", "GL", "GDDY",
        "GS", "HAL", "HIG", "HAS", "HCA", "DOC", "HSIC", "HSY", "HES", "HPE", "HLT", "HOLX",
        "HD", "HON", "HRL", "HST", "HWM", "HPQ", "HUBB", "HUM", "HBAN", "HII", "IBM", "IEX",
        "IDXX", "ITW", "INCY", "IR", "PODD", "INTC", "ICE", "IFF", "IP", "IPG", "INTU", "ISRG",
        "IVZ", "INVH", "IQV", "IRM", "JBHT", "JBL", "JKHY", "J", "JNJ", "JCI", "JPM", "JNPR",
        "K", "KVUE", "KDP", "KEY", "KEYS", "KMB", "KIM", "KMI", "KKR", "KLAC", "KHC", "KR",
        "LHX", "LH", "LRCX", "LW", "LVS", "LDOS", "LEN", "LIN", "LYV", "LKQ", "LMT", "L",
        "LOW", "LULU", "LYB", "MTB", "MRO", "MPC", "MKTX", "MAR", "MMC", "MLM", "MAS", "MA",
        "MTCH", "MKC", "MCD", "MCK", "MDT", "MRK", "META", "MET", "MTD", "MGM", "MCHP", "MU",
        "MSFT", "MAA", "MRNA", "MHK", "MOH", "TAP", "MDLZ", "MPWR", "MNST", "MCO", "MS",
        "MOS", "MSI", "MSCI", "NDAQ", "NTAP", "NFLX", "NEM", "NWSA", "NWS", "NEE", "NKE",
        "NI", "NDSN", "NSC", "NTRS", "NOC", "NCLH", "NRG", "NUE", "NVDA", "NVR", "NXPI",
        "ORLY", "OXY", "ODFL", "OMC", "ON", "OKE", "ORCL", "OTIS", "PCAR", "PKG", "PANW",
        "PARA", "PH", "PAYX", "PAYC", "PYPL", "PNR", "PEP", "PFE", "PCG", "PM", "PSX", "PNW",
        "PNC", "POOL", "PPG", "PPL", "PFG", "PG", "PGR", "PLD", "PRU", "PEG", "PTC", "PSA",
        "PHM", "QRVO", "PWR", "QCOM", "DGX", "RL", "RJF", "RTX", "O", "REG", "REGN", "RF",
        "RSG", "RMD", "RVTY", "ROK", "ROL", "ROP", "ROST", "RCL", "SPGI", "CRM", "SBAC",
        "SLB", "STX", "SRE", "NOW", "SHW", "SPG", "SWKS", "SJM", "SNA", "SOLV", "SO", "LUV",
        "SWK", "SBUX", "STT", "STLD", "STE", "SYK", "SMCI", "SYF", "SNPS", "SYY", "TMUS",
        "TROW", "TTWO", "TPR", "TRGP", "TGT", "TEL", "TDY", "TFX", "TER", "TSLA", "TXN",
        "TXT", "TMO", "TJX", "TSCO", "TT", "TDG", "TRV", "TRMB", "TFC", "TYL", "TSN", "USB",
        "UBER", "UDR", "ULTA", "UNP", "UAL", "UPS", "URI", "UNH", "UHS", "VLO", "VTR", "VLTO",
        "VRSN", "VRSK", "VZ", "VRTX", "VTRS", "VICI", "V", "VST", "VMC", "WRB", "GWW",
        "WAB", "WBA", "WMT", "DIS", "WBD", "WM", "WAT", "WEC", "WFC", "WELL", "WST", "WDC",
        "WRK", "WY", "WMB", "WTW", "WYNN", "XEL", "XYL", "YUM", "ZBRA", "ZBH", "ZTS"
    ]

    # Function to create Plotly figure
    def create_figure(data, indicators, title):
        fig = go.Figure()
        fig.update_layout(
            title=title, 
            xaxis_title='Date', 
            yaxis_title='Price',
            xaxis_rangeslider_visible=True,
            plot_bgcolor='darkgrey',
            paper_bgcolor='white',
            font=dict(color='black'),
            hovermode='x',
            xaxis=dict(
                rangeselector=dict(),
                rangeslider=dict(visible=True),
                type='date'
            ),
            yaxis=dict(fixedrange=False),
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Reset Zoom",
                              method="relayout",
                              args=[{"xaxis.range": [None, None],
                                     "yaxis.range": [None, None]}])]
            )]
        )
        return fig

    # Function to fetch and process stock data
    @st.cache_data(ttl=3600)
    def get_stock_data(ticker_symbols, period):
        try:
            stock_data = {}
            for ticker_symbol in ticker_symbols:
                df = yf.download(ticker_symbol, period=period)
                if not df.empty:
                    df.interpolate(method='linear', inplace=True)
                    df = calculate_indicators(df)
                    df.dropna(inplace=True)
                    stock_data[ticker_symbol] = df
            return stock_data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return {}

    # Function to calculate technical indicators
    @st.cache_data(ttl=3600)
    def calculate_indicators(df):
        # Calculate Moving Averages
        df['5_MA'] = ta.trend.WMAIndicator(close=df['Close'], window=5).wma()
        df['20_MA'] = ta.trend.WMAIndicator(close=df['Close'], window=20).wma()
        df['50_MA'] = ta.trend.WMAIndicator(close=df['Close'], window=50).wma()

        # Calculate MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Histogram'] = macd.macd_diff()

        # Calculate ADX
        adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'])
        df['ADX'] = adx.adx()

        # Calculate Parabolic SAR
        psar = ta.trend.PSARIndicator(df['High'], df['Low'], df['Close'])
        df['Parabolic_SAR'] = psar.psar()

        # Calculate RSI
        rsi = ta.momentum.RSIIndicator(df['Close'])
        df['RSI'] = rsi.rsi()

        # Calculate Volume Moving Averages
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()

        # Calculate Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['Bollinger_High'] = bollinger.bollinger_hband()
        df['Bollinger_Low'] = bollinger.bollinger_lband()
        df['Bollinger_Middle'] = bollinger.bollinger_mavg()

        # Calculate Detrended Price Oscillator (DPO)
        df['DPO'] = ta.trend.DPOIndicator(close=df['Close']).dpo()

        # Calculate On-Balance Volume (OBV)
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()

        # Calculate Volume Weighted Average Price (VWAP)
        df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).volume_weighted_average_price()

        # Calculate Accumulation/Distribution Line (A/D Line)
        df['A/D Line'] = ta.volume.AccDistIndexIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).acc_dist_index()

        # Calculate Average True Range (ATR)
        df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()

        return df

    # Function to query the stocks
    @st.cache_data(ttl=3600)
    def query_stocks(stock_data, conditions):
        results = []
        for ticker, df in stock_data.items():
            if df.empty or len(df) < 1:
                continue
            condition_met = True
            for condition in conditions:
                col1, op, col2 = condition
                if col1 not in df.columns or col2 not in df.columns:
                    condition_met = False
                    break
                if op == '>':
                    if not (df[col1] > df[col2]).iloc[-1]:
                        condition_met = False
                        break
                elif op == '<':
                    if not (df[col1] < df[col2]).iloc[-1]:
                        condition_met = False
                        break
                elif op == '>=':
                    if not (df[col1] >= df[col2]).iloc[-1]:
                        condition_met = False
                        break
                elif op == '<=':
                    if not (df[col1] <= df[col2]).iloc[-1]:
                        condition_met = False
                        break
            if condition_met:
                row = {
                    'Ticker': ticker,
                    'MACD': df['MACD'].iloc[-1],
                    'MACD_Signal': df['MACD_Signal'].iloc[-1],
                    'MACD_Hist': df['MACD_Histogram'].iloc[-1],
                    'RSI': df['RSI'].iloc[-1],
                    'ADX': df['ADX'].iloc[-1],
                    'Close': df['Close'].iloc[-1],
                    '5_MA': df['5_MA'].iloc[-1],
                    '20_MA': df['20_MA'].iloc[-1],
                    'Bollinger_High': df['Bollinger_High'].iloc[-1],
                    'Bollinger_Low': df['Bollinger_Low'].iloc[-1],
                    'Bollinger_Middle': df['Bollinger_Middle'].iloc[-1],
                    'Parabolic_SAR': df['Parabolic_SAR'].iloc[-1],
                    'Volume': df['Volume'].iloc[-1],
                    'Volume_MA_10': df['Volume_MA_10'].iloc[-1],
                    'Volume_MA_20': df['Volume_MA_20'].iloc[-1],
                    'DPO': df['DPO'].iloc[-1]
                }
                results.append(row)
        return pd.DataFrame(results)

    # Determine tickers based on submenu selection
    if submenu == "India-LargeCap":
        st.subheader("India-LargeCap")
        tickers = largecap_tickers
    elif submenu == "India-MidCap":
        st.subheader("India-MidCap")
        tickers = midcap_tickers
    elif submenu == "FTSE100":
        st.subheader("FTSE100")
        tickers = ftse100_tickers
    elif submenu == "S&P500":
        st.subheader("S&P500")
        tickers = sp500_tickers

    # Fetch data and calculate indicators for each stock
    stock_data = get_stock_data(tickers, period='3mo')

    # Define first set of conditions
    first_conditions = [
        ('MACD', '>', 'MACD_Signal'),
        ('Parabolic_SAR', '<', 'Close')
    ]

    # Query stocks based on the first set of conditions
    first_query_df = query_stocks(stock_data, first_conditions)

    # Filter stocks in an uptrend with high volume and positive DPO
    second_query_df = first_query_df[
        (first_query_df['RSI'] < 65) & (first_query_df['RSI'] > 45) &
        (first_query_df['ADX'] > 25) & (first_query_df['MACD'] > 0)
    ]

    st.write("Stocks in an uptrend with high volume and positive DPO:")
    st.dataframe(second_query_df)

    # Dropdown for analysis type
    col1, col2 = st.columns(2)

    # Set up the start and end date inputs
    with col1:
        selected_stock = st.selectbox("Select Stock", second_query_df['Ticker'].tolist())

    with col2:
        analysis_type = st.selectbox("Select Analysis Type", ["Trend Analysis", "Volume Analysis", "Support & Resistance Levels"])

    # Create two columns
    col1, col2 = st.columns(2)

    # Set up the start and end date inputs
    with col1:
        START = st.date_input('Start Date', pd.to_datetime("2023-06-01"))

    with col2:
        END = st.date_input('End Date', pd.to_datetime("today"))

    # If a stock is selected, plot its data with the selected indicators
    if selected_stock:
        @st.cache_data(ttl=3600)
        def load_data(ticker, start, end):
            df = yf.download(ticker, start=start, end=end)
            df.reset_index(inplace=True)
            return df

        df = load_data(selected_stock, START, END)

        if df.empty:
            st.write("No data available for the provided ticker.")
        else:
            df.interpolate(method='linear', inplace=True)
            df = calculate_indicators(df)

            # Identify Horizontal Support and Resistance
            def find_support_resistance(df, window=20):
                df['Support'] = df['Low'].rolling(window, center=True).min()
                df['Resistance'] = df['High'].rolling(window, center=True).max()
                return df

            df = find_support_resistance(df)

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

            df['Support_Trendline'] = calculate_trendline(df, kind='support')
            df['Resistance_Trendline'] = calculate_trendline(df, kind='resistance')

            # Calculate Fibonacci Retracement Levels
            def fibonacci_retracement_levels(high, low):
                diff = high - low
                levels = {
                    'Level_0': high,
                    'Level_0.236': high - 0.236 * diff,
                    'Level_0.382': high - 0.382 * diff,
                    'Level_0.5': high - 0.5 * diff,
                    'Level_0.618': high - 0.618 * diff,
                    'Level_1': low
                }
                return levels

            recent_high = df['High'].max()
            recent_low = df['Low'].min()
            fib_levels = fibonacci_retracement_levels(recent_high, recent_low)

            # Calculate Pivot Points
            def pivot_points(df):
                df['Pivot'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
                df['R1'] = 2 * df['Pivot'] - df['Low'].shift(1)
                df['S1'] = 2 * df['Pivot'] - df['High'].shift(1)
                df['R2'] = df['Pivot'] + (df['High'].shift(1) - df['Low'].shift(1))
                df['S2'] = df['Pivot'] - (df['High'].shift(1) - df['Low'].shift(1))
                return df

            df = pivot_points(df)

            # Function to generate buy/sell signals
            def generate_signals(macd, signal, rsi, close):
                buy_signals = [0] * len(macd)
                sell_signals = [0] * len(macd)
                for i in range(1, len(macd)):
                    if macd[i] > signal[i] and macd[i-1] <= signal[i-1]:
                        buy_signals[i] = 1
                    elif macd[i] < signal[i] and macd[i-1] >= signal[i-1]:
                        sell_signals[i] = 1
                return buy_signals, sell_signals

            df['Buy_Signal'], df['Sell_Signal'] = generate_signals(df['MACD'], df['MACD_Signal'], df['RSI'], df['Close'])

            if analysis_type == "Trend Analysis":
                st.subheader("Trend Analysis")

                indicators = st.multiselect(
                    "Select Indicators",
                    ['Close', '20_MA', '50_MA', '200_MA', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'RSI', 'Buy_Signal', 'Sell_Signal', 'ADX',
                     'Parabolic_SAR', 'Bollinger_High', 'Bollinger_Low', 'Bollinger_Middle', 'ATR'],
                    default=['Close', 'Buy_Signal', 'Sell_Signal']
                )
                timeframe = st.radio(
                    "Select Timeframe",
                    ['15 days', '30 days', '90 days', '180 days', '1 year', 'All'],
                    index=2,
                    horizontal=True
                )

                if timeframe == '15 days':
                    df = df[-15:]
                elif timeframe == '30 days':
                    df = df[-30:]
                elif timeframe == '90 days':
                    df = df[-90:]
                elif timeframe == '180 days':
                    df = df[-180:]
                elif timeframe == '1 year':
                    df = df[-365:]

                fig = create_figure(df.set_index('Date'), indicators, f"Trend Analysis for {selected_stock}")

                colors = {'Close': 'blue', '20_MA': 'orange', '50_MA': 'green', '200_MA': 'red', 'MACD': 'purple',
                          'MACD_Signal': 'brown', 'RSI': 'pink', 'Buy_Signal': 'green', 'Sell_Signal': 'red', 'ADX': 'magenta',
                          'Parabolic_SAR': 'yellow', 'Bollinger_High': 'black', 'Bollinger_Low': 'cyan',
                          'Bollinger_Middle': 'grey', 'ATR': 'darkblue'}

                for indicator in indicators:
                    if indicator == 'Buy_Signal':
                        fig.add_trace(
                            go.Scatter(x=df[df[indicator] == 1]['Date'],
                                       y=df[df[indicator] == 1]['Close'], mode='markers', name='Buy Signal',
                                       marker=dict(color='green', symbol='triangle-up')))
                    elif indicator == 'Sell_Signal':
                        fig.add_trace(
                            go.Scatter(x=df[df[indicator] == 1]['Date'],
                                       y=df[df[indicator] == 1]['Close'], mode='markers', name='Sell Signal',
                                       marker=dict(color='red', symbol='triangle-down')))
                    elif indicator == 'MACD_Histogram':
                        fig.add_trace(
                            go.Bar(x=df['Date'], y=df[indicator], name=indicator, marker_color='gray'))
                    else:
                        fig.add_trace(
                            go.Scatter(x=df['Date'], y=df[indicator], mode='lines', name=indicator,
                                       line=dict(color=colors.get(indicator, 'black'))))

                st.plotly_chart(fig)

            elif analysis_type == "Volume Analysis":
                st.subheader("Volume Analysis")
                volume_indicators = st.multiselect(
                    "Select Volume Indicators",
                    ['Close', 'Volume', 'Volume_MA_20', 'Volume_MA_10', 'Volume_MA_5', 'OBV', 'VWAP', 'A/D Line'],
                    default=['Close', 'VWAP']
                )
                volume_timeframe = st.radio(
                    "Select Timeframe",
                    ['15 days', '30 days', '90 days', '180 days', '1 year', 'All'],
                    index=2,
                    horizontal=True
                )

                if volume_timeframe == '15 days':
                    df = df[-15:]
                elif volume_timeframe == '30 days':
                    df = df[-30:]
                elif volume_timeframe == '90 days':
                    df = df[-90:]
                elif volume_timeframe == '180 days':
                    df = df[-180:]
                elif volume_timeframe == '1 year':
                    df = df[-365:]

                fig = create_figure(df.set_index('Date'), volume_indicators, f"Volume Analysis for {selected_stock}")

                for indicator in volume_indicators:
                    fig.add_trace(go.Scatter(x=df['Date'], y=df[indicator], mode='lines', name=indicator))

                st.plotly_chart(fig)

            elif analysis_type == "Support & Resistance Levels":
                st.subheader("Support & Resistance Levels")
                sr_indicators = st.multiselect(
                    "Select Indicators",
                    ['Close', '20_MA', '50_MA', '200_MA', 'Support', 'Resistance', 'Support_Trendline',
                     'Resistance_Trendline', 'Pivot', 'R1', 'S1', 'R2', 'S2'],
                    default=['Close', 'Support', 'Resistance']
                )
                sr_timeframe = st.radio(
                    "Select Timeframe",
                    ['15 days', '30 days', '90 days', '180 days', '1 year', 'All'],
                    index=2,
                    horizontal=True
                )

                if sr_timeframe == '15 days':
                    df = df[-15:]
                elif sr_timeframe == '30 days':
                    df = df[-30:]
                elif sr_timeframe == '90 days':
                    df = df[-90:]
                elif sr_timeframe == '180 days':
                    df = df[-180:]
                elif sr_timeframe == '1 year':
                    df = df[-365:]

                fig = create_figure(df.set_index('Date'), sr_indicators, f"Support & Resistance Levels for {selected_stock}")

                for indicator in sr_indicators:
                    fig.add_trace(go.Scatter(x=df['Date'], y=df[indicator], mode='lines', name=indicator))

                st.plotly_chart(fig)



# Call the display function
if __name__ == "__main__":
    display()
