import streamlit as st
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from newspaper import Article
from transformers import pipeline
import time
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

def news_app():


    # Function to summarize articles
    def summarize_article(url):
        try:
            article = Article(url)
            article.download()
            article.parse()
            article_text = article.text

            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            summary = summarizer(article_text, max_length=150, min_length=50, do_sample=False)

            return {"url": url, "title": article.title, "summary": summary[0]["summary_text"]}
        except Exception as e:
            return {"url": url, "title": "Failed to fetch", "summary": str(e)}

    # Function to click 'Load More' button
    def load_all_articles(driver, load_more_button_css, max_clicks):
        clicks = 0
        last_article_count = 0

        while clicks < max_clicks:
            try:
                load_more_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, load_more_button_css))
                )
                load_more_button.click()
                time.sleep(1)

                current_article_count = len(driver.find_elements(By.XPATH, "//h2[contains(text(),'')]"))
                if current_article_count == last_article_count:
                    st.warning("No more new articles found. Stopping.")
                    break

                last_article_count = current_article_count
                clicks += 1
            except Exception as e:
                st.error(f"No more articles to load or button not found: {e}")
                break

        if clicks >= max_clicks:
            st.info("Reached the maximum number of clicks.")

    # Function to fetch article links
    def get_article_links_selenium(main_url, max_clicks):
        driver = webdriver.Chrome()  # Ensure ChromeDriver is installed and in PATH
        driver.get(main_url)
        time.sleep(3)

        load_more_button_css = "button[class='list-with-sidebar-m__view-all-large__Lx51z button-m__btn__Hlh8p button-m__btn-medium__AQNFd button-m__btn-outline__zjMCO button-m__btn-rounded-full__cKE9X ']"
        load_all_articles(driver, load_more_button_css, max_clicks)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        driver.quit()

        elements = soup.find_all("h2")
        article_links = [element.find_parent("a")["href"] for element in elements if element.find_parent("a")]
        return list(set(article_links))

    # Main function to summarize all articles
    def summarize_all_articles(main_url, max_clicks):
        article_links = get_article_links_selenium(main_url, max_clicks)
        st.write(f"Found {len(article_links)} articles.")
        summaries = []

        # Use threading to speed up summarization
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(summarize_article, article_url if article_url.startswith("http") else "https://www.ndtvprofit.com" + article_url) for article_url in article_links]
            for future in futures:
                summaries.append(future.result())

        return summaries

    # Streamlit App
    st.title("News Summarizer")

    menu_options = {
        "Business": "https://www.ndtvprofit.com/business?query=read-more",
        "Markets": "https://www.ndtvprofit.com/markets?query=read-more",
        "Economy & Finance": "https://www.ndtvprofit.com/economy-finance?query=read-more",
        "Earnings": "https://www.ndtvprofit.com/quarterly-earnings?query=read-more"
    }

    selected_option = st.sidebar.selectbox("Select a category:", list(menu_options.keys()))
    main_url = menu_options[selected_option]

    max_clicks = st.sidebar.number_input("Set the maximum number of 'Load More' clicks:", min_value=1, max_value=20, value=1, step=1)

    if st.sidebar.button("Load and Summarize Articles"):
        with st.spinner("Fetching and summarizing articles..."):
            results = summarize_all_articles(main_url, max_clicks)

        st.success("Summarization complete!")

        for result in results:
            st.subheader(result['title'])
            st.write(f"**Summary:** {result['summary']}")
            st.write(f"[Read full article]({result['url']})")


if __name__ == "__main__":
    news_app_app()