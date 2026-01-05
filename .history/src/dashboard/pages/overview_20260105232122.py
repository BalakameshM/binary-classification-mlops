# src/dashboard/pages/overview.py
import streamlit as st
from src.dashboard.utils import predict_single

st.title("Cereal Classifier Dashboard")
st.caption("Demo UI for inference + quick sanity checks")

with st.form("predict_form"):
    st.subheader("Enter Features")
    protein = st.number_input("protein", value=3.0)
    fat = st.number_input("fat", value=1.0)
    sodium = st.number_input("sodium", value=200.0)
    carbo = st.number_input("carbo", value=15.0)
    potass = st.number_input("potass", value=100.0)
    vitamins = st.number_input("vitamins", value=25.0)
    weight = st.number_input("weight", value=1.0)
    cups = st.number_input("cups", value=1.0)

    mfr = st.text_input("mfr", value="K")
    type_ = st.text_input("type", value="C")
    shelf = st.number_input("shelf", value=2)

    submitted = st.form_submit_button("Predict")

if submitted:
    features = {
        "protein": protein, "fat": fat, "sodium": sodium,
        "carbo": carbo, "potass": potass, "vitamins": vitamins,
        "weight": weight, "cups": cups,
        "mfr": mfr, "type": type_, "shelf": shelf
    }
    pred, run_id = predict_single(features)
    label = {0: "Unhealthy", 1: "Moderately healthy", 2: "Healthy"}.get(pred, str(pred))

    st.success(f"Prediction: {label} (class={pred})")
    st.info(f"Model run id: {run_id}")
