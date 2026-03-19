import streamlit as st
import pickle
import pandas as pd

# Load model
with open("insurance_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Insurance Cost Predictor")

age = st.number_input("Age")
bmi = st.number_input("BMI")
children = st.number_input("Number of Children")
sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northwest", "southeast", "southwest"])

if st.button("Predict"):
    input_data = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        "sex_male": [1 if sex == "male" else 0],
        "smoker_yes": [1 if smoker == "yes" else 0],
        "region_northwest": [1 if region == "northwest" else 0],
        "region_southeast": [1 if region == "southeast" else 0],
        "region_southwest": [1 if region == "southwest" else 0]
    })

    prediction = model.predict(input_data)

    st.success(f"Predicted Insurance Cost: {prediction[0]:.2f}")