import streamlit as st
import numpy as np
import pickle
from sklearn.datasets import load_breast_cancer

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Load feature names
data = load_breast_cancer()
features = data.feature_names

st.title("🔍 Breast Cancer Prediction App")
st.write("Input the values and click **Predict** to see if the tumor is likely malignant or benign.")

# User input
input_data = []
for feature in features[:10]:  # Only first 10 features for simplicity
    value = st.slider(f"{feature}", float(X_train[:, features.tolist().index(feature)].min()), 
                      float(X_train[:, features.tolist().index(feature)].max()), 
                      float(X_train[:, features.tolist().index(feature)].mean()))
    input_data.append(value)

# Predict button
if st.button("Predict"):
    prediction = model.predict([input_data])
    result = "Benign 😊" if prediction[0] == 1 else "Malignant ⚠️"
    st.success(f"Prediction: {result}")
