import streamlit as st
import pickle
import numpy as np

# Load saved model and scaler
ridge = pickle.load(open('ridge.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Streamlit UI
st.title("🔥 FWI Prediction")

st.markdown("### Enter the following inputs:")

# Input fields
Temperature = st.number_input("Temperature", value=25.0)
RH = st.number_input("Relative Humidity (RH)", value=40.0)
Ws = st.number_input("Wind Speed (Ws)", value=10.0)
Rain = st.number_input("Rain (mm)", value=0.0)
FFMC = st.number_input("FFMC", value=85.0)
DMC = st.number_input("DMC", value=50.0)
ISI = st.number_input("ISI", value=5.0)
Classes = st.number_input("Classes (0 or 1)", value=0)
Region = st.number_input("Region (1 or 0)", value=1)

# Predict button
if st.button("Predict"):
    # Create input array
    input_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])

    # Scale input
    scaled_data = scaler.transform(input_data)

    # Make prediction
    prediction = ridge.predict(scaled_data)[0]

    st.markdown(f"### 🌲 THE FWI prediction is **{prediction:.6f}**")
