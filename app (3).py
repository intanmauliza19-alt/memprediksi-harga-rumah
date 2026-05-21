import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set page title
st.set_page_config(page_title="House Price Predictor")

st.title("🏠 House Price Prediction App")
st.markdown("Enter the details of the house below to estimate its market price.")

# Load the saved scaler and model
@st.cache_resource
def load_artifacts():
    with open('standard_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('linear_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return scaler, model

try:
    scaler, model = load_artifacts()

    # Create input fields
    col1, col2 = st.columns(2)
    
    with col1:
        sq_ft = st.number_input("Square Footage", min_value=100, max_value=10000, value=2500)
        bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
        bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
        year_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000)

    with col2:
        lot_size = st.number_input("Lot Size (Acres)", min_value=0.1, max_value=50.0, value=2.5)
        garage_size = st.number_input("Garage Size (Cars)", min_value=0, max_value=5, value=2)
        neighborhood = st.slider("Neighborhood Quality (1-10)", 1, 10, 7)

    if st.button("Predict Price"):
        # Prepare input dataframe
        input_data = pd.DataFrame({
            'Square_Footage': [sq_ft],
            'Num_Bedrooms': [bedrooms],
            'Num_Bathrooms': [bathrooms],
            'Year_Built': [year_built],
            'Lot_Size': [lot_size],
            'Garage_Size': [garage_size],
            'Neighborhood_Quality': [neighborhood]
        })

        # Preprocess and Predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        st.success(f"### Estimated Price: ${prediction[0]:,.2f}")

except FileNotFoundError:
    st.error("Error: Scaler or Model files not found. Please ensure 'standard_scaler.pkl' and 'linear_regression_model.pkl' are in the same directory.")
