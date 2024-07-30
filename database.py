import sqlite3

# Replace 'new_etrade.db' with the path to your database file
db_path = 'new_etrade.db'

# Connect to the database
conn = sqlite3.connect(db_path)

# Create a cursor object
cursor = conn.cursor()

# List tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

# Fetch and print rows from each table
for table in tables:
    table_name = table[0]
    print(f"Table: {table_name}")
    
    # Get the column names
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    column_names = [col[1] for col in columns]
    
    # Print the column names
    print("Columns:", ", ".join(column_names))
    
    # Fetch and print all rows from the table
    cursor.execute(f"SELECT * FROM {table_name};")
    rows = cursor.fetchall()
    
    for row in rows:
        print(row)
    
    print("\n" + "-"*50 + "\n")

# Close the connection
conn.close()
