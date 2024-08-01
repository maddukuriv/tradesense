import streamlit as st
from utils.mongodb import users_collection
from utils.hash_utils import hash_password

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
        existing_user = users_collection.find_one({"email": email})
        if existing_user:
            st.error("Email already exists. Try a different email.")
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
