#Your API key is: 252b2075083945dfbed8945ddc240a2b



import streamlit as st
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# Initialize NewsApiClient with your API key
newsapi = NewsApiClient(api_key='252b2075083945dfbed8945ddc240a2b')
analyzer = SentimentIntensityAnalyzer()

def fetch_news(ticker):
    # Fetch news articles related to the ticker
    all_articles = newsapi.get_everything(q=ticker,
                                          language='en',
                                          sort_by='publishedAt',
                                          page_size=10,
                                          sources='the-times-of-india, financial-express, the-hindu, bloomberg, cnbc')
    articles = []
    for article in all_articles['articles']:
        articles.append({
            'title': article['title'],
            'description': article['description'],
            'url': article['url'],
            'publishedAt': article['publishedAt']
        })
    return articles

def perform_sentiment_analysis(articles):
    sentiments = []
    for article in articles:
        if article['description']:
            score = analyzer.polarity_scores(article['description'])
            sentiment = score['compound']
            sentiments.append(sentiment)
        else:
            sentiments.append(0)
    return sentiments

def make_recommendation(sentiments):
    avg_sentiment = sum(sentiments) / len(sentiments)
    if avg_sentiment > 0.1:
        return "Based on the sentiment analysis, it is recommended to BUY the stock."
    elif avg_sentiment < -0.1:
        return "Based on the sentiment analysis, it is recommended to NOT BUY the stock."
    else:
        return "Based on the sentiment analysis, it is recommended to HOLD OFF on any action."

st.title("Stock News Sentiment Analysis")
ticker = st.text_input("Enter the Company Name:")

if ticker:
    with st.spinner("Fetching news..."):
        articles = fetch_news(ticker)
    
    if articles:
        with st.spinner("Performing sentiment analysis..."):
            sentiments = perform_sentiment_analysis(articles)
        
        df = pd.DataFrame({
            'Title': [article['title'] for article in articles],
            'Description': [article['description'] for article in articles],
            'URL': [article['url'] for article in articles],
            'Published At': [article['publishedAt'] for article in articles],
            'Sentiment': sentiments
        })
        
        st.write("Recent News Articles:")
        for i, row in df.iterrows():
            st.write(f"**Article {i+1}:** {row['Title']}")
            st.write(f"**Published At:** {row['Published At']}")
            st.write(f"**Description:** {row['Description']}")
            st.write(f"**URL:** {row['URL']}")
            st.write(f"**Sentiment Score:** {row['Sentiment']:.2f}")
            st.write("---")
        
        st.write("Sentiment Analysis Summary:")
        st.bar_chart(df['Sentiment'])

        recommendation = make_recommendation(sentiments)
        st.write("Investment Recommendation:")
        st.write(recommendation)
    else:
        st.write("No news articles found for this ticker.")
