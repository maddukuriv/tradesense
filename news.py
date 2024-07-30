import streamlit as st
from newsapi.newsapi_client import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.sidebar.subheader("Stock Analysis")
submenu = st.sidebar.selectbox("Select Analysis Type", ["Financial Analysis", "Technical Analysis", "Sentiment Analysis", "Price Forecast"])

# User input for the stock ticker
ticker = st.sidebar.text_input('Enter Stock Ticker (e.g., BAJAJFINSV.NS): ', 'BAJAJFINSV.NS')

# Date inputs limited to the last 30 days
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

if submenu == "Financial Analysis":
    pass

elif submenu == "Technical Analysis":
    pass

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
                    st.write("Sentiment Analysis Summary:")
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
                    st.write("Sentiment vs Stock Price:")
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
                    st.write("Sentiment Distribution:")
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
                    st.write("Investment Recommendation:")
                    st.write(recommendation)

                    # Downloadable report
                    st.write("Download Report:")
                    csv = df.to_csv(index=False)
                    st.download_button('Download CSV', data=csv, file_name='sentiment_analysis_report.csv', mime='text/csv')
            else:
                st.write("No news articles found for this company.")
        else:
            st.write("Invalid ticker symbol.")

elif submenu == "Price Forecast":
    pass
