import os
import pymongo
from dotenv import load_dotenv
from urllib.parse import quote_plus
import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import yfinance as yf

# Load environment variables from .env file
load_dotenv()

# MongoDB connection
def get_mongo_client():
    try:
        username = os.getenv('MONGO_USERNAME')
        password = os.getenv('MONGO_PASSWORD')
        uri = f'mongodb+srv://{quote_plus(username)}:{quote_plus(password)}@tradesense.uq6adbz.mongodb.net/?retryWrites=true&w=majority&appName=tradesense'

        client = pymongo.MongoClient(uri)
        return client
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

# MongoDB Database and Collections
client = get_mongo_client()
if client:
    try:
        db = client['etrade']

        # Existing collections
        users_collection = db['users']
        watchlists_collection = db['watchlists']

        # Collection for trades
        trades_collection = db['trades']

        print("Database connected successfully")
    except pymongo.errors.ServerSelectionTimeoutError as err:
        print(f"Error selecting database or collections: {err}")

# Function to initialize MongoDB collections
def init_db():
    try:
        if client:
            # Ensure indexes for unique fields
            users_collection.create_index('email', unique=True)
            watchlists_collection.create_index('user_id')

            print("Database initialized with indexes")
    except pymongo.errors.PyMongoError as e:
        print(f"Error initializing database: {e}")

# Initialize the database
init_db()
