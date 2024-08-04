import streamlit as st
from utils.mongodb import users_collection
from utils.hash_utils import hash_password
from bson.objectid import ObjectId

# Forgot password function
def forgot_password():
    st.subheader("Forgot Password")
    email = st.text_input("Enter your email", key='forgot_email')
    pob = st.text_input("Enter your place of birth", key='forgot_pob')

    if 'identity_verified' not in st.session_state:
        st.session_state.identity_verified = False

    if st.button("Submit"):
        user = users_collection.find_one({"email": email, "pob": pob})
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
                st.session_state.identity_verified = False
                st.experimental_rerun()
