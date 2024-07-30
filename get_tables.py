import streamlit as st
import sqlite3
import pandas as pd

# Path to your SQLite database file
db_path = 'new_etrade.db'

# Function to load data from a table
def load_data(table_name):
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name};"
    data = pd.read_sql_query(query, conn)
    conn.close()
    return data

def display_tables():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # List tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    st.title('Database Admin Page')

    # Display tables
    for table in tables:
        table_name = table[0]
        st.header(f"Table: {table_name}")

        # Load data
        data = load_data(table_name)

        # Display data
        st.dataframe(data)

    conn.close()
