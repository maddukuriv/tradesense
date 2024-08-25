import streamlit as st
from utils.mongodb import users_collection
from utils.hash_utils import hash_password
from bson.objectid import ObjectId
import re
from datetime import datetime
import time  # Import time module for delay

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

# Forgot password function
def forgot_password():
    st.subheader("Forgot Password")
    email = st.text_input("Enter your email", key='forgot_email')
    dob = st.date_input("Enter your date of birth", min_value=datetime(1950, 1, 1), max_value=datetime.now(), key='forgot_dob')
    pob = st.text_input("Enter your place of birth", key='forgot_pob')

    if 'identity_verified' not in st.session_state:
        st.session_state.identity_verified = False

    if st.button("Submit"):
        # Convert DOB to string format to match database format
        dob_str = dob.strftime('%Y-%m-%d')
        user = users_collection.find_one({"email": email, "dob": dob_str, "pob": pob})
        
        if user:
            st.session_state.email = email
            st.session_state.user_id = str(user['_id'])  # Ensure user_id is a string
            st.session_state.identity_verified = True
            st.success("Identity verified. Please reset your password.")
        else:
            st.error("Invalid details provided.")

    if st.session_state.identity_verified:
        new_password = st.text_input("Enter a new password", type="password", key='reset_new_password')
        confirm_new_password = st.text_input("Confirm your new password", type="password", key='reset_confirm_new_password')

        if st.button("Reset Password"):
            if new_password != confirm_new_password:
                st.error("Passwords do not match.")
            elif not is_valid_password(new_password):
                st.error("Password must be at least 8 characters long and include an uppercase letter, a lowercase letter, a number, and a special character.")
            else:
                hashed_password = hash_password(new_password)
                users_collection.update_one({"_id": ObjectId(st.session_state.user_id)}, {"$set": {"password": hashed_password}})
                st.success("Password reset successfully. You can now log in with the new password.")
                
                # Delay before rerun to allow message to be seen
                time.sleep(2)  # 2-second delay
                st.session_state.identity_verified = False
                st.experimental_rerun()
