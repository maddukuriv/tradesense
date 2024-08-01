
import bcrypt  # for password hashing
from datetime import datetime
# Function to verify password using bcrypt
def verify_password(hashed_password, plain_password):
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except ValueError as e:
        print(f"Error verifying password: {e}")
        return False


# Function to hash password using bcrypt
def hash_password(password):
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password.decode('utf-8')



# Function to convert string date to datetime.date
def string_to_date(date_string):
    return datetime.strptime(date_string, '%Y-%m-%d').date()
