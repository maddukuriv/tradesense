
import streamlit as st
from utils.mongodb import users_collection
from utils.hash_utils import hash_password
from bson.objectid import ObjectId

# Forgot password function
def forgot_password():
    st.subheader("Forgot Password")
    email = st.text_input("Enter your email", key='forgot_email')
    dob = st.date_input("Enter your date of birth", key='forgot_dob')
    pob = st.text_input("Enter your place of birth", key='forgot_pob')

    if st.button("Submit"):
        user = users_collection.find_one({"email": email, "dob": dob, "pob": pob})
        if user:
            st.session_state.email = email
            st.session_state.user_id = user['_id']
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
            else:
                hashed_password = hash_password(new_password)
                users_collection.update_one({"_id": ObjectId(st.session_state.user_id)}, {"$set": {"password": hashed_password}})
                st.success("Password reset successfully. You can now log in with the new password.")
                st.session_state.identity_verified = False
