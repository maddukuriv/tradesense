elif choice == "Stock Prediction":
            # Your existing 'Stock Price Forecasting' code-----------------------------------------------------------------------------------------------------------------------

            # Sidebar for user input
            st.sidebar.subheader("Prediction")

            submenu = st.sidebar.selectbox("Select Option", ["Trend", "Price"])  

            

            if submenu == "Trend":
                tickers = st.sidebar.multiselect("Enter Stock Symbols", options=bse_largecap+bse_midcap+bse_smallcap)
                time_period = st.sidebar.selectbox("Select Time Period", options=["6mo", "1y", "5y"], index=0)
                @st.cache_data
                def load_data(ticker, period):
                    return yf.download(ticker, period=period)

                def remove_outliers(df, column='Close', z_thresh=3):
                    df['zscore'] = zscore(df[column])
                    df = df[df['zscore'].abs() <= z_thresh]
                    df.drop(columns='zscore', inplace=True)
                    return df

                def calculate_indicators(df):
                    df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
                    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
                    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
                    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
                    df['MACD'] = ta.trend.macd(df['Close'])
                    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
                    df['MACD_Hist'] = ta.trend.macd_diff(df['Close'])
                    df['Bollinger_High'] = ta.volatility.bollinger_hband(df['Close'])
                    df['Bollinger_Low'] = ta.volatility.bollinger_lband(df['Close'])
                    df['Stochastic_Oscillator'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
                    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
                    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
                    df['VPT'] = ta.volume.volume_price_trend(df['Close'], df['Volume'])
                    df['CMF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'])
                    df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
                    return df

                def calculate_peaks_troughs(df):
                    peaks, _ = find_peaks(df['Close'])
                    troughs, _ = find_peaks(-df['Close'])
                    df['Peaks'] = np.nan
                    df['Troughs'] = np.nan
                    df.loc[df.index[peaks], 'Peaks'] = df['Close'].iloc[peaks]
                    df.loc[df.index[troughs], 'Troughs'] = df['Close'].iloc[troughs]
                    return df

                def calculate_fourier(df, n=5):
                    close_fft = np.fft.fft(df['Close'].values)
                    fft_df = pd.DataFrame({'fft': close_fft})
                    fft_df['absolute'] = np.abs(fft_df['fft'])
                    fft_df['angle'] = np.angle(fft_df['fft'])
                    fft_df = fft_df.sort_values(by='absolute', ascending=False).head(n)
                    
                    # Calculate dominant frequency in days
                    freqs = np.fft.fftfreq(len(df))
                    dominant_freqs = freqs[np.argsort(np.abs(close_fft))[-n:]]
                    # Ensure frequencies are positive
                    positive_freqs = np.abs(dominant_freqs)
                    cycles_in_days = 1 / positive_freqs
                    fft_df['cycles_in_days'] = cycles_in_days
                    return fft_df

                def calculate_wavelet(df):
                    widths = np.arange(1, 31)
                    cwt_matrix = cwt(df['Close'], ricker, widths)
                    return cwt_matrix

                def calculate_hilbert(df):
                    analytic_signal = hilbert(df['Close'])
                    amplitude_envelope = np.abs(analytic_signal)
                    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
                    return amplitude_envelope, instantaneous_phase

                # Streamlit UI
                st.title("Multi-Stock Cycle Detection and Analysis")

                results = []

                for ticker in tickers:
                    st.subheader(f"Analysis for {ticker}")
                    data = load_data(ticker, time_period)
                    
                    if not data.empty:
                        df = data.copy()
                        df = remove_outliers(df)
                        df = calculate_indicators(df)
                        df = calculate_peaks_troughs(df)

                        st.subheader(f"{ticker} Data and Indicators ({time_period} Period)")
                        st.dataframe(df.tail())

                        # Fourier Transform results
                        st.subheader("Fourier Transform - Dominant Cycles")
                        fft_df = calculate_fourier(df)
                        st.write("Dominant Cycles from Fourier Transform (Top 5):")
                        st.dataframe(fft_df)

                        # Determine current position in the cycle
                        dominant_cycle = fft_df.iloc[1] if fft_df.iloc[1]['absolute'] > fft_df.iloc[2]['absolute'] else fft_df.iloc[2]
                        current_phase_angle = dominant_cycle['angle']
                        current_position = 'upward' if current_phase_angle > 0 else 'downward'

                        st.subheader("Current Cycle Position")
                        st.write(f"The stock is currently in an '{current_position}' phase of the dominant cycle with a phase angle of {current_phase_angle:.2f} radians.")

                        # Explanation of what this means
                        if current_position == 'upward':
                            st.write("This means the stock price is currently in an upward trend within its dominant cycle, and it is expected to continue rising.")
                        else:
                            st.write("This means the stock price is currently in a downward trend within its dominant cycle, and it is expected to continue falling.")

                        # Calculate and display Wavelet Transform results
                        st.subheader("Wavelet Transform")
                        cwt_matrix = calculate_wavelet(df)

                        # Insights from Wavelet Transform
                        max_wavelet_amplitude = np.max(np.abs(cwt_matrix))
                        st.write("**Wavelet Transform Insights:**")
                        st.write(f"The maximum amplitude in the wavelet transform is {max_wavelet_amplitude:.2f}.")

                        # Calculate and display Hilbert Transform results
                        st.subheader("Hilbert Transform")
                        amplitude_envelope, instantaneous_phase = calculate_hilbert(df)

                        # Insights from Hilbert Transform
                        st.write("**Hilbert Transform Insights:**")
                        st.write(f"The current amplitude envelope is {amplitude_envelope[-1]:.2f}.")
                        st.write(f"The current instantaneous phase is {instantaneous_phase[-1]:.2f} radians.")

                        # Collecting results for comparison
                        results.append({
                            'Ticker': ticker,
                            'Fourier_Dominant_Cycle_Days': fft_df['cycles_in_days'].iloc[0],
                            'Fourier_Angle': fft_df['angle'].iloc[1],  # Correctly selecting the dominant cycle angle
                            'Fourier_Trend': current_position,
                            'Wavelet_Max_Amplitude': max_wavelet_amplitude,
                            'Hilbert_Amplitude': amplitude_envelope[-1],
                            'Hilbert_Instantaneous_Phase': instantaneous_phase[-1]
                        })

                # Comparison and Final Recommendation
                st.subheader("Comparison and Final Recommendation")
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)

                # Final Recommendation based on collected results
                buy_recommendation = results_df[
                    (results_df['Wavelet_Max_Amplitude'] > 1000) &  # Adjust threshold as necessary
                    (results_df['Hilbert_Amplitude'] > results_df['Hilbert_Amplitude'].mean()) &  # Strong current price movement
                    (results_df['Fourier_Trend'] == 'upward')  # Current price trend is upward
                ]

                if not buy_recommendation.empty:
                    st.write("**Recommendation: Buy**")
                    st.dataframe(buy_recommendation[['Ticker', 'Fourier_Dominant_Cycle_Days', 'Fourier_Angle', 'Fourier_Trend', 'Wavelet_Max_Amplitude', 'Hilbert_Amplitude']])
                else:
                    st.write("No strong buy recommendations based on the current analysis.")

            else:

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
                    freqs = np.fft.fftfreq(len(df))
                    dominant_freqs = freqs[np.argsort(np.abs(close_fft))[-n:]]
                    positive_freqs = np.abs(dominant_freqs)
                    cycles_in_days = 1 / positive_freqs
                    fft_df['cycles_in_days'] = cycles_in_days
                    return fft_df

                def calculate_wavelet(df):
                    widths = np.arange(1, 31)
                    cwt_matrix = cwt(df['Close'], ricker, widths)
                    return cwt_matrix

                def calculate_hilbert(df):
                    analytic_signal = hilbert(df['Close'])
                    amplitude_envelope = np.abs(analytic_signal)
                    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
                    return amplitude_envelope, instantaneous_phase

                # Streamlit UI
                st.title("Stock Price Prediction with SARIMA Model")

                # Sidebar for user input
                st.sidebar.subheader("Settings")
                ticker_symbol = st.sidebar.text_input("Enter Stock Symbol", value="CUMMINSIND.NS")
                start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2021-06-01"))
                end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))
                forecast_period = st.sidebar.number_input("Forecast Period (days)", value=10, min_value=1, max_value=30)

                # Fetch the data
                stock_data = get_stock_data(ticker_symbol, start_date, end_date)

                # Calculate technical indicators
                stock_data = calculate_technical_indicators(stock_data)

                # Calculate Fourier, Wavelet, and Hilbert transforms
                fft_df = calculate_fourier(stock_data)
                cwt_matrix = calculate_wavelet(stock_data)
                amplitude_envelope, instantaneous_phase = calculate_hilbert(stock_data)

                # Drop rows with NaN values
                stock_data.dropna(inplace=True)

                # Extract the closing prices and technical indicators
                close_prices = stock_data['Close']
                technical_indicators = stock_data[['MACD', 'Signal_Line', 'MACD_Hist', 'RSI', 'SMA_20', 'Upper_Band', 'Lower_Band', 'BBW', 'EMA_50', 'ATR', 'ROC', 'OBV']]

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
                    title=f'Stock Price Forecast for {ticker_symbol}',
                    xaxis_title='Date',
                    yaxis_title='Close Price',
                    legend=dict(x=0.01, y=0.99),
                )

                st.plotly_chart(fig)

                # Display forecasted prices
                st.write("Forecasted Prices for the next {} days:".format(forecast_period))
                st.dataframe(forecasted_df)

