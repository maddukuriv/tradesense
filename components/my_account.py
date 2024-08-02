import streamlit as st
from utils.mongodb import users_collection
from bson.objectid import ObjectId
from utils.hash_utils import string_to_date, verify_password, hash_password
from datetime import datetime

# My Account function
def my_account():
    st.subheader("My Account")

    if st.session_state.logged_in:
        user = users_collection.find_one({"_id": ObjectId(st.session_state.user_id)})

        new_name = st.text_input("Update your name", value=user['name'], key='account_name')
        new_dob = st.date_input("Update your date of birth", value=string_to_date(user['dob']), key='account_dob')
        new_pob = st.text_input("Update your place of birth", value=user['pob'], key='account_pob')

        if st.button("Update Details"):
            new_dob_str = new_dob.strftime('%Y-%m-%d')  # Convert the date to string
            users_collection.update_one({"_id": ObjectId(user['_id'])}, {"$set": {"name": new_name, "dob": new_dob_str, "pob": new_pob}})
            st.success("Details updated successfully!")

        st.subheader("Change Password")
        current_password = st.text_input("Enter your current password", type="password", key='account_current_password')
        new_password = st.text_input("Enter a new password", type="password", key='account_new_password')
        confirm_new_password = st.text_input("Confirm your new password", type="password", key='account_confirm_new_password')

        if st.button("Change Password"):
            if verify_password(user['password'], current_password):
                if new_password != confirm_new_password:
                    st.error("Passwords do not match.")
                else:
                    hashed_password = hash_password(new_password)
                    users_collection.update_one({"_id": ObjectId(user['_id'])}, {"$set": {"password": hashed_password}})
                    st.success("Password changed successfully!")
            else:
                st.error("Current password is incorrect.")

# To call the function in your main Streamlit app
if __name__ == "__main__":
    my_account()
