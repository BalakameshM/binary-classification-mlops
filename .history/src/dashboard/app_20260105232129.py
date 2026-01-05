# src/dashboard/app.py
import streamlit as st

st.set_page_config(page_title="Cereal MLOps Dashboard", layout="wide")
st.sidebar.title("Navigation")
st.sidebar.page_link("pages/overview.py", label="Overview", icon="📊")

st.title("Cereal MLOps Dashboard")
st.write("Use the sidebar to open pages.")
