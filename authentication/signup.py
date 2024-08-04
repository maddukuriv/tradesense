import streamlit as st
from utils.mongodb import users_collection
from utils.hash_utils import hash_password
import re

# Email validation function
def is_valid_email(email):
    regex = r'^\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.match(regex, email)

# Password validation function
def is_valid_password(password):
    if len(password) < 8:
        return False
    if not re.search(r'[A-Z]', password):
        return False
    if not re.search(r'[a-z]', password):
        return False
    if not re.search(r'[0-9]', password):
        return False
    if not re.search(r'[@$!%*?&]', password):
        return False
    return True

# Signup function
def signup():
    st.subheader("Sign Up")
    name = st.text_input("Enter your name", key='signup_name')
    email = st.text_input("Enter your email", key='signup_email')
    dob = st.date_input("Enter your date of birth", key='signup_dob')
    pob = st.text_input("Enter your place of birth", key='signup_pob')
    password = st.text_input("Enter a new password", type="password", key='signup_password')
    confirm_password = st.text_input("Confirm your password", type="password", key='signup_confirm_password')

    if st.button("Sign Up"):
        if not is_valid_email(email):
            st.error("Invalid email format.")
        elif users_collection.find_one({"email": email}):
            st.error("Email already exists. Try a different email.")
        elif not is_valid_password(password):
            st.error("Password must be at least 8 characters long and include an uppercase letter, a lowercase letter, a number, and a special character.")
        elif password != confirm_password:
            st.error("Passwords do not match.")
        else:
            hashed_password = hash_password(password)
            new_user = {
                "name": name,
                "email": email,
                "password": hashed_password,
                "dob": dob.strftime('%Y-%m-%d'), 
                "pob": pob
            }
            users_collection.insert_one(new_user)
            st.success("User registered successfully!")
