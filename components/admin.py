import streamlit as st
import pandas as pd
from utils.mongodb import users_collection, watchlists_collection, trades_collection

# Display tables function
def display_tables():
    st.title('Database Admin Page')

    st.header("Users Collection")
    users = list(users_collection.find())
    st.dataframe(pd.DataFrame(users))

    st.header("Watchlists Collection")
    watchlists = list(watchlists_collection.find())
    st.dataframe(pd.DataFrame(watchlists))

    st.header("Trades Collection")
    trades = list(trades_collection.find())
    st.dataframe(pd.DataFrame(trades))
