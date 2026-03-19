# Medical Insurance Cost Prediction

## 📌 Problem Statement
Predict medical insurance charges based on demographic and health-related features.

---

## 📊 Dataset
- 1338 records
- Features: age, sex, bmi, children, smoker, region
- Target: charges

---

## 🧠 Approach
1. Data Cleaning  
2. Exploratory Data Analysis (EDA)  
3. Feature Encoding  
4. Train-Test Split  
5. Feature Scaling  
6. Model Training  
   - Linear Regression  
   - Random Forest  
7. Model Evaluation (R², RMSE)  
8. Feature Importance Analysis  

---

## 📈 Model Performance

| Model | R² Score | RMSE |
|------|--------|------|
| Linear Regression | 0.78 | ~5796 |
| Random Forest | 0.86 | ~4567 |

Random Forest performed better and was selected as final model.

---

## 🔥 Key Insights
- Smoking status is the most important factor affecting insurance cost  
- Charges are right-skewed  
- Non-linear models perform better  

---

## 🚀 Features
- End-to-end ML pipeline  
- Model comparison  
- Feature importance visualization  
- Prediction system  

---

## ▶️ Run the App (Streamlit)

```bash
pip install streamlit
streamlit run app.py
