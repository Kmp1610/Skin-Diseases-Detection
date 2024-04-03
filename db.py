import streamlit as st
import mysql.connector
from mysql.connector import Error
from passlib.hash import sha256_crypt  # For password hashing
import subprocess

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def create_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
        print("Connection to MySQL DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection

# Function to register a new user
def register_user(username, password):
    hashed_password = sha256_crypt.hash(password)  # Hashing the password
    try:
        cursor = connection.cursor()
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
        connection.commit()
        st.success("Registration successful")
    except Error as e:
        st.error(f"Error occurred: {e}")

# Function to authenticate user during signin
def authenticate_user(username, password):
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        if user:  # If user exists
            if sha256_crypt.verify(password, user[2]):  # Verify hashed password
                st.success("Login successful")
                subprocess.Popen(["streamlit", "run", "predict.py"])
        # Exit the current script
                raise SystemExit
            
            else:
                st.error("Invalid password")
        else:
            st.error("User not found")
    except Error as e:
        st.error(f"Error occurred: {e}")

# Main function
def main():
    st.title("User Authentication")

    # Streamlit pages for signup and signin
    page = st.sidebar.radio("Navigation", ["Signup", "Signin"])

    if page == "Signup":
        st.subheader("Signup")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")

        if st.button("Register"):
            if password == confirm_password:
                register_user(username, password)
            else:
                st.error("Passwords do not match")

    elif page == "Signin":
        st.subheader("Signin")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
             if authenticate_user(username, password):  # Check if login is successful
                st.success("You are logged in!")
    
# Database connection details
HOST = "localhost"
USER = "root"
PASSWORD = "Kunjan@123"
DATABASE = "Fsd"

# Creating a connection to the MySQL database
connection = create_connection(HOST, USER, PASSWORD, DATABASE)

if connection is not None:
    main()
    connection.close()  # Closing the database connection when done
else:
    st.error("Unable to connect to the database")

