
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

df = load_data()

st.title("Heart Failure Prediction App")
st.write("This app predicts the risk of heart failure based on patient data.")

# User inputs
st.sidebar.header("Patient Information")
age = st.sidebar.slider("Age", 18, 100, 50)
anaemia = st.sidebar.selectbox("Anaemia", [0, 1])
creatinine_phosphokinase = st.sidebar.slider("Creatinine Phosphokinase", 20, 8000, 250)
diabetes = st.sidebar.selectbox("Diabetes", [0, 1])
ejection_fraction = st.sidebar.slider("Ejection Fraction (%)", 10, 80, 38)
high_blood_pressure = st.sidebar.selectbox("High Blood Pressure", [0, 1])
platelets = st.sidebar.slider("Platelets", 25000, 900000, 265000)
serum_creatinine = st.sidebar.slider("Serum Creatinine", 0.5, 10.0, 1.1)
serum_sodium = st.sidebar.slider("Serum Sodium", 110, 150, 137)
sex = st.sidebar.selectbox("Sex", [0, 1])
smoking = st.sidebar.selectbox("Smoking", [0, 1])
time = st.sidebar.slider("Follow-up Period (days)", 0, 300, 100)

# Prepare input
input_data = pd.DataFrame({
    'age': [age],
    'anaemia': [anaemia],
    'creatinine_phosphokinase': [creatinine_phosphokinase],
    'diabetes': [diabetes],
    'ejection_fraction': [ejection_fraction],
    'high_blood_pressure': [high_blood_pressure],
    'platelets': [platelets],
    'serum_creatinine': [serum_creatinine],
    'serum_sodium': [serum_sodium],
    'sex': [sex],
    'smoking': [smoking],
    'time': [time]
})

# Train model
X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier()
model.fit(X_scaled, y)

# Predict
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]
proba = model.predict_proba(input_scaled)[0]

# Output
st.subheader("Prediction")
st.write("Risk of death:", "Yes" if prediction == 1 else "No")
st.write("Confidence: {:.2f}%".format(np.max(proba) * 100))

st.subheader("Patient Data")
st.dataframe(input_data)
