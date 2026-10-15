import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "insurance_model.pkl"

@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)

st.set_page_config(page_title="Medical Insurance Cost Predictor", page_icon="🩺", layout="centered")
st.title("Medical Insurance Cost Predictor")
st.caption(
    "Educational estimate only. This is not a real insurer quote, medical advice, "
    "or a basis for insurance underwriting."
)

try:
    model = load_model()
except Exception:
    st.error(
        "The saved model could not be loaded. Run train_model.py to generate a compatible "
        "insurance_model.pkl, then replace the repository's model artifact."
    )
    st.stop()

st.subheader("Enter example details")
age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=25.0, step=0.1)
children = st.number_input("Number of children", min_value=0, max_value=10, value=0, step=1)
sex = st.selectbox("Sex", ["female", "male"])
smoker = st.selectbox("Smoker", ["no", "yes"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

if st.button("Estimate insurance charges", type="primary"):
    row = pd.DataFrame([{
        "age": age, "bmi": bmi, "children": children,
        "sex": sex, "smoker": smoker, "region": region,
    }])
    row = pd.get_dummies(row, columns=["sex", "smoker", "region"], drop_first=True, dtype=int)

    # Reindex to the exact training feature order stored by the fitted pipeline.
    expected = getattr(model, "feature_names_in_", None)
    if expected is None and hasattr(model, "named_steps"):
        expected = getattr(model.named_steps.get("scaler", model), "feature_names_in_", None)
    if expected is not None:
        row = row.reindex(columns=list(expected), fill_value=0)

    try:
        estimate = float(model.predict(row)[0])
        st.metric("Estimated insurance charges", f"{estimate:,.2f}")
        st.caption("Estimate from a historical educational dataset; actual premiums vary by insurer and policy.")
    except Exception as exc:
        st.error("Prediction failed because the saved model's feature schema does not match the app.")
        st.exception(exc)
