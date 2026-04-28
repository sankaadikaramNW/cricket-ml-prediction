import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("simple_model.pkl")

st.set_page_config(page_title="Cricket Match Predictor", layout="wide")

st.title("🏏 T20 Match Outcome Predictor")

st.write("Enter match situation to predict win probability")

# =========================
# USER INPUT
# =========================

runs = st.number_input("Current Runs", 0, 300, 50)
wickets = st.number_input("Wickets Lost", 0, 10, 2)
overs = st.number_input("Overs Completed", 0.0, 20.0, 5.0)
target = st.number_input("Target Score", 0, 300, 150)

# =========================
# FEATURE ENGINEERING
# =========================

balls = int(overs * 6)
balls_remaining = 120 - balls

current_rr = runs / overs if overs > 0 else 0
runs_to_get = target - runs
required_rr = runs_to_get / (balls_remaining / 6) if balls_remaining > 0 else 0

pressure = required_rr - current_rr

# Create input dataframe
input_data = pd.DataFrame({
    'Innings': [2],
    'Ball_Number': [balls],
    'Balls_Remaining': [balls_remaining],
    'Innings Runs': [runs],
    'Wickets_Lost': [wickets],
    'Current_RunRate': [current_rr],
    'Required_RunRate': [required_rr],
    'Pressure_Index': [pressure]
})

# =========================
# PREDICTION
# =========================

if st.button("Predict Outcome"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"🏆 Batting Team Likely to WIN ({probability:.2f})")
    else:
        st.error(f"❌ Batting Team Likely to LOSE ({probability:.2f})")