import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Load your pre-trained model and transformers
@st.cache_data()
def load_resources():
    model = joblib.load('Prostate_cancer_rf_model.joblib')
    scaler = joblib.load('min_max_scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    return model, scaler, label_encoders

model, scaler, label_encoders = load_resources()

# Define the Streamlit app
def main():
    st.title("Prostate Cancer Diagnosis Prediction")

    st.markdown("#### PROJECT WORK BY: Ajala Tamilore Ayomitomiwa")
    
    st.subheader("**Input the required features for diagnosis prediction:**")

    # Input fields for features
    radius = st.number_input('Radius', min_value=9.0, max_value=25.0, value=16.85)
    texture = st.number_input('Texture', min_value=11.0, max_value=27.0, value=18.23)
    perimeter = st.number_input('Perimeter', min_value=52.0, max_value=172.0, value=95.78)
    area = st.number_input('Area', min_value=202.0, max_value=1878.0, value=702.88)
    smoothness = st.number_input('Smoothness', min_value=0.07, max_value=0.143, value=0.10273)
    compactness = st.number_input('Compactness', min_value=0.038, max_value=0.345, value=0.12170)
    symmetry = st.number_input('Symmetry', min_value=0.135, max_value=0.304, value=0.19317)
    fractal_dimension = st.number_input('Fractal Dimension', min_value=0.053, max_value=0.097, value=0.06469)

    # Button to make prediction
    if st.button('Predict Diagnosis'):
        input_df = pd.DataFrame([[radius, texture, perimeter, area, smoothness, compactness, symmetry, fractal_dimension]],
                                columns=['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'symmetry', 'fractal_dimension'])
        input_df_scaled = scaler.transform(input_df)
        prediction = model.predict(input_df_scaled)
        prediction_proba = model.predict_proba(input_df_scaled)
        max_proba = np.max(prediction_proba) * 100

        # Output prediction
        result = label_encoders['diagnosis_result'].inverse_transform(prediction)[0]
        st.success(f"The predicted diagnosis is **{result}** with a probability of **{max_proba:.2f}%**")

        # Display the confidence score as a gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = max_proba,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Confidence Score", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': 'red'},
                    {'range': [50, 75], 'color': 'yellow'},
                    {'range': [75, 100], 'color': 'green'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.90,
                    'value': max_proba}
            }))
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
