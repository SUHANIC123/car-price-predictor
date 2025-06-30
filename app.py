import streamlit as st
import pickle
import numpy as np

# Load model and encoders
model = pickle.load(open('car_price_model.pkl', 'rb'))
le_fuel = pickle.load(open('fuel_encoder.pkl', 'rb'))
le_trans = pickle.load(open('transmission_encoder.pkl', 'rb'))

st.title("🚗 Car Price Predictor")

kms = st.number_input("KMs Driven", 0)
fuel = st.selectbox("Fuel Type", le_fuel.classes_)
trans = st.selectbox("Transmission", le_trans.classes_)

if st.button("Predict Price"):
    input_data = np.array([[kms, le_fuel.transform([fuel])[0], le_trans.transform([trans])[0]]])
    price = model.predict(input_data)[0]
    st.success(f"Estimated Selling Price: ₹ {round(price, 2)}")
