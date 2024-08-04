import os
import pymongo
from dotenv import load_dotenv
from urllib.parse import quote_plus

# Load environment variables from .env file
load_dotenv()

# MongoDB connection
def get_mongo_client():
    try:
        username = os.getenv('MONGO_USERNAME')
        password = os.getenv('MONGO_PASSWORD')
        uri = f'mongodb+srv://{username}:{password}@tradesense.uq6adbz.mongodb.net/?retryWrites=true&w=majority&appName=tradesense'

      
        # print(f"MongoDB URI: {uri}, {username} :: {password}")  # Print URI for debugging
        client = pymongo.MongoClient(uri)
        # print("MongoDB Client:", client)
        return client
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

# MongoDB Database and Collections
client = get_mongo_client()
if client:
    try:
        db = client['etrade']
        users_collection = db['users']
        watchlists_collection = db['watchlists']
        portfolios_collection = db['portfolios']
        print("database connected===>")
    except pymongo.errors.ServerSelectionTimeoutError as err:
        print(f"Error selecting database or collections: {err}")

# Function to initialize MongoDB collections``
def init_db():
    try:
        if client:
            # Ensure indexes for unique fields
            users_collection.create_index('email', unique=True)
            watchlists_collection.create_index('user_id')
            portfolios_collection.create_index('user_id')
    except pymongo.errors.PyMongoError as e:
        print(f"Error initializing database: {e}")

# # Initialize the database
# init_db()
