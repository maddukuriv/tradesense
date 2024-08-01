import streamlit as st
from utils.mongodb import users_collection
from utils.hash_utils import verify_password

# Login function
def login():
    st.subheader("Login")
    email = st.text_input("Enter your email", key='login_email')
    password = st.text_input("Enter your password", type="password", key='login_password')

    if st.button("Login"):
        user = users_collection.find_one({"email": email})
        if user and verify_password(user['password'], password):
            st.success("Login successful!")
            st.session_state.logged_in = True
            st.session_state.username = user['name']
            st.session_state.email = user['email']
            st.session_state.user_id = user['_id']
        else:
            st.error("Invalid email or password.")
