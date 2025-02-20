import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

# Load the model and scaler
model = tf.keras.models.load_model("water_quality_model.h5")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Water Quality Prediction App")

# User input for features
feature_names = ["pH", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity", "Organic Carbon", "Trihalomethanes", "Turbidity"]
user_input = []

for feature in feature_names:
    value = st.number_input(f"Enter {feature}", value=0.0)
    user_input.append(value)

# Convert input to numpy array and scale it
if st.button("Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)
    
    # Display result
    if prediction[0][0] > 0.5:
        st.success("The water is **safe** for drinking!")
    else:
        st.error("The water is **not safe** for drinking!")
