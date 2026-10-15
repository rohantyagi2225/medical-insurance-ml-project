import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# Load the model artifact tracked in this repository.
MODEL_PATH = Path(__file__).resolve().parent / "insurance_model.pkl"

@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)

st.set_page_config(page_title="Medical Insurance Cost Predictor", page_icon="🩺")
st.title("Medical Insurance Cost Predictor")
st.caption(
    "Educational estimate only. This output is not an insurance quote, "
    "medical advice, or a basis for real underwriting decisions."
)

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load model artifact at {MODEL_PATH.name}. Check the file and package compatibility.")
    st.exception(exc)
    st.stop()

age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=25.0, step=0.1)
children = st.number_input("Number of children", min_value=0, max_value=10, value=0, step=1)
sex = st.selectbox("Sex", ["female", "male"])
smoker = st.selectbox("Smoker", ["no", "yes"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

if st.button("Estimate insurance charges", type="primary"):
    input_data = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        "sex_male": [1 if sex == "male" else 0],
        "smoker_yes": [1 if smoker == "yes" else 0],
        "region_northwest": [1 if region == "northwest" else 0],
        "region_southeast": [1 if region == "southeast" else 0],
        "region_southwest": [1 if region == "southwest" else 0],
    })

    try:
        prediction = model.predict(input_data)
        st.metric("Estimated insurance charges", f"{float(prediction[0]):,.2f}")
        st.caption("Estimate from the saved educational model; actual premiums depend on insurer-specific underwriting.")
    except Exception as exc:
        st.error("Prediction failed. The saved model may expect a different feature schema than this app supplies.")
        st.exception(exc)
