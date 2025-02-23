
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
from fpdf import FPDF
from PIL import Image
import io

# Load Model & Scaler
def load_model():
    model = joblib.load("water_purity_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# Function to Predict Water Purity
def predict_purity(features):
    features_scaled = scaler.transform([features])  # Apply same scaling as training
    prediction = model.predict(features_scaled)
    return "Pure" if prediction[0] == 1 else "Impure"

# Function to Store Prediction in Database
def store_prediction(features, result):
    conn = sqlite3.connect("water_purity.db")
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO predictions (pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, OrganicCarbon, Trihalomethanes, Turbidity, Prediction)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (*features, result))
    
    conn.commit()
    conn.close()

# Generate PDF Report
def generate_report(result, features):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Water Purity Analysis Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Prediction: {result}", ln=True)
    pdf.ln(10)
    
    params = ["pH", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity", "Organic Carbon", "Trihalomethanes", "Turbidity"]
    for param, value in zip(params, features):
        pdf.cell(200, 10, txt=f"{param}: {value}", ln=True)
    
    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output

# Streamlit UI
st.set_page_config(page_title="Water Purity Assessment", layout="wide")
st.title("💧 Water Purity Assessment")
st.markdown("### Check the purity of water using Machine Learning")

# Sidebar Inputs
st.sidebar.header("Input Water Parameters")
pH = st.sidebar.slider("pH Level", 0.0, 14.0, 7.0)
Hardness = st.sidebar.slider("Hardness", 0, 300, 150)
Solids = st.sidebar.slider("Solids (ppm)", 0, 50000, 20000)
Chloramines = st.sidebar.slider("Chloramines", 0.0, 15.0, 7.0)
Sulfate = st.sidebar.slider("Sulfate", 0.0, 500.0, 250.0)
Conductivity = st.sidebar.slider("Conductivity", 0.0, 1000.0, 500.0)
Organic_carbon = st.sidebar.slider("Organic Carbon", 0.0, 30.0, 15.0)
Trihalomethanes = st.sidebar.slider("Trihalomethanes", 0.0, 120.0, 60.0)
Turbidity = st.sidebar.slider("Turbidity", 0.0, 10.0, 5.0)

if st.sidebar.button("Check Purity"):
    features = [pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]
    result = predict_purity(features)
    
    # Store in database
    store_prediction(features, result)

    st.subheader(f"Water is **{result}**")
    pdf = generate_report(result, features)
    st.download_button(label="Download Report", data=pdf, file_name="Water_Purity_Report.pdf", mime="application/pdf")

