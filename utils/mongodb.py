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
        portfolios_collection = db['portfolios']

        # New collections for buy and sell trades
        buy_trades_collection = db['buy_trades']  # Collection for buy trades
        sell_trades_collection = db['sell_trades']  # Collection for sell trades

        print("database connected===>")
    except pymongo.errors.ServerSelectionTimeoutError as err:
        print(f"Error selecting database or collections: {err}")

# Function to initialize MongoDB collections
def init_db():
    try:
        if client:
            # Ensure indexes for unique fields
            users_collection.create_index('email', unique=True)
            watchlists_collection.create_index('user_id')
            portfolios_collection.create_index('user_id')

            # Creating indexes for buy_trades and sell_trades
            buy_trades_collection.create_index([('user_id', pymongo.ASCENDING), ('ticker', pymongo.ASCENDING)])
            sell_trades_collection.create_index([('user_id', pymongo.ASCENDING), ('ticker', pymongo.ASCENDING)])

            # Ensure unique trade IDs (if you're generating unique IDs for each trade)
            buy_trades_collection.create_index('trade_id', unique=True, sparse=True)
            sell_trades_collection.create_index('trade_id', unique=True, sparse=True)

            print("Database initialized with indexes")
    except pymongo.errors.PyMongoError as e:
        print(f"Error initializing database: {e}")

# Initialize the database
init_db()
