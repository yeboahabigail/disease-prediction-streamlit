"""
Disease Prediction System using Machine Learning
A Streamlit web application for predicting diseases based on symptoms and environmental factors
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 48px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .sub-header {
        font-size: 24px;
        color: #424242;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1565c0;
        color: white;
    }
    .disclaimer-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 20px 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
    .feature-card {
        background-color: #f8f9fa;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        height: 100%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .feature-icon {
        font-size: 48px;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Load the trained model and preprocessors
@st.cache_resource
def load_model():
    """
    Load the trained Random Forest model, scaler, and label encoder.
    
    Returns:
        tuple: (model, scaler, label_encoder)
    
    Raises:
        FileNotFoundError: If any required model file is missing
    """
    try:
        model = joblib.load(open('Random_Forest_model.pkl', 'rb'))
        scaler = joblib.load(open('scaler.pkl', 'rb'))
        label_encoder = joblib.load(open('label_encoder.pkl', 'rb'))
        return model, scaler, label_encoder
    except FileNotFoundError as e:
        st.error(f"❌ Error loading model files: {e}")
        st.info("Please ensure the following files are in the same directory as app.py:")
        st.code("- Random_Forest_model.pkl\n- scaler.pkl\n- label_encoder.pkl")
        st.stop()

# Load models
model, scaler, label_encoder = load_model()

# Sidebar Navigation
with st.sidebar:
    st.markdown("# ")
    st.title("Navigation")
    
    if st.button(" Home", use_container_width=True):
        st.session_state.page = 'Home'
    
    if st.button(" About", use_container_width=True):
        st.session_state.page = 'About'
    
    if st.button(" Disease Prediction", use_container_width=True):
        st.session_state.page = 'Prediction'
    
    if st.button(" Model Information", use_container_width=True):
        st.session_state.page = 'Model'
    
    

# HOME PAGE
if st.session_state.page == 'Home':
    st.markdown('<div class="main-header"> Disease Prediction System</div>', unsafe_allow_html=True)
    st.markdown("### *Powered by Machine Learning & Environmental Data Analysis*")
    
    st.markdown("""
    <div class="disclaimer-box">
        <h4>⚠️ Medical Disclaimer</h4>
        <p>This is a <strong>predictive tool for educational purposes only</strong>. 
        It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
        Always consult with qualified healthcare professionals for medical concerns.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## Welcome to the Disease Prediction System")
    
    st.markdown("""
    This advanced machine learning application helps predict potential diseases based on:
    - Patient symptoms
    - Environmental conditions
    - Demographic information
    
    Our Random Forest model achieves **98.29% accuracy** in predicting across 11 different disease classes.
    """)
    
    # Feature Cards
    st.markdown("### 🌟 Key Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon"></div>
            <h3>High Accuracy</h3>
            <p>98.29% prediction accuracy with advanced Random Forest algorithm</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon"></div>
            <h3>Environmental Analysis</h3>
            <p>Considers temperature, humidity, and wind speed for better predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon"></div>
            <h3>Comprehensive Results</h3>
            <p>Detailed probability distribution and visual analytics</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Disease Classes
    st.markdown("###  Supported Disease Classes")
    disease_cols = st.columns(4)
    diseases = [
        "Heart Attack", "Migraine", "Influenza", "Heat Stroke",
        "Malaria", "Stroke", "Eczema", "Dengue",
        "Common Cold", "Arthritis", "Sinusitis"
    ]
    
    for idx, disease in enumerate(diseases):
        with disease_cols[idx % 4]:
            st.info(f"✓ {disease}")
    
    st.markdown("---")
    
    # Call to Action
    st.markdown("###  Get Started")
    st.info("Click on **'Disease Prediction'** in the navigation menu to begin your analysis!")

# ABOUT PAGE
elif st.session_state.page == 'About':
    st.markdown('<div class="main-header"> About This System</div>', unsafe_allow_html=True)
    
    st.markdown("## Overview")
    st.write("""
    The Disease Prediction System is an advanced machine learning application designed to assist in 
    preliminary disease identification based on patient symptoms and environmental factors. This system 
    combines medical symptom analysis with environmental data to provide comprehensive disease predictions.
    """)
    
    st.markdown("##  Purpose")
    st.info("""
    This application serves as an **educational tool** to demonstrate the application of machine learning 
    in healthcare diagnostics. It is designed to:
    - Showcase the potential of AI in medical preliminary screening
    - Provide insights into disease-symptom relationships
    - Demonstrate the importance of environmental factors in disease prediction
    - Serve as a learning resource for machine learning applications in healthcare
    """)
    
    st.markdown("##  How It Works")
    st.write("""
    The system uses a **Random Forest** machine learning algorithm trained on a comprehensive dataset 
    that includes:
    
    1. **Patient Demographics**: Age and gender information
    2. **Environmental Data**: Temperature, humidity, and wind speed
    3. **Symptom Profiles**: 47 different symptoms across multiple categories
    4. **Disease Classes**: 11 distinct disease categories
    
    The model analyzes the combination of these factors to predict the most likely disease condition.
    """)
    
    st.markdown("##  Dataset Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Original Dataset", "5,200 rows", help="Total number of records collected")
    
    with col2:
        st.metric("After Cleaning", "4,981 rows", help="Final dataset after data cleaning and preprocessing")
    
    with col3:
        st.metric("Data Retention", "95.79%", help="Percentage of data retained after cleaning")
    
    st.info("""
    **Data Cleaning Process:**
    - Removed 219 duplicate entries
    - Handled missing values
    - Validated feature ranges
    - Ensured data consistency
    - Applied stratified sampling for balanced class distribution
    """)
    
    st.markdown("---")
    
    st.markdown("##  Disease Categories")
    disease_info = {
        "Heart Attack": "Cardiovascular emergency requiring immediate medical attention",
        "Migraine": "Severe headache disorder with neurological symptoms",
        "Influenza": "Viral respiratory infection affecting the respiratory system",
        "Heat Stroke": "Life-threatening condition caused by overheating",
        "Malaria": "Mosquito-borne infectious disease",
        "Stroke": "Medical emergency when blood flow to brain is interrupted",
        "Eczema": "Inflammatory skin condition causing irritation and itching",
        "Dengue": "Mosquito-borne viral infection",
        "Common Cold": "Viral infection of the upper respiratory tract",
        "Arthritis": "Inflammatory joint disorder causing pain and stiffness",
        "Sinusitis": "Inflammation of sinus cavities"
    }
    
    for disease, description in disease_info.items():
        with st.expander(f"**{disease}**"):
            st.write(description)
    
    st.markdown("---")
    
    st.markdown("## ⚠️ Important Disclaimer")
    st.warning("""
    **This system is NOT a replacement for professional medical advice, diagnosis, or treatment.**
    
    - Always consult with qualified healthcare professionals for any health concerns
    - This tool is for educational and informational purposes only
    - Do not use this system to self-diagnose or self-medicate
    - In case of medical emergency, contact emergency services immediately
    - The predictions provided are probabilistic and should not be considered definitive
    """)
    
    

# PREDICTION PAGE
elif st.session_state.page == 'Prediction':
    st.markdown('<div class="main-header"> Disease Prediction</div>', unsafe_allow_html=True)
    
    # Main Input Section
    st.markdown("##  Patient Information")
    
    # Create three columns for demographic information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("👤 Age (years)", min_value=1, max_value=100, value=30, step=1)
    
    with col2:
        gender = st.radio("⚥ Gender", options=["Male", "Female"], horizontal=True)
        gender_encoded = 0 if gender == "Female" else 1
    
    # Input validation
    if age < 1 or age > 100:
        st.error("⚠️ Age must be between 1 and 100 years")
        st.stop()
    
    # Environmental Conditions Section
    st.markdown("##  Environmental Conditions")
    env_col1, env_col2, env_col3 = st.columns(3)
    
    with env_col1:
        temperature = st.slider(" Temperature (°C)", min_value=-15.0, max_value=41.0, value=25.0, step=0.5)
    
    with env_col2:
        humidity = st.slider(" Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
        humidity_normalized = humidity / 100.0
    
    with env_col3:
        wind_speed = st.slider(" Wind Speed (km/h)", min_value=0.0, max_value=32.0, value=10.0, step=0.5)
    
    # Symptoms Input Section
    st.markdown("##  Symptom Selection")
    st.markdown("*Select all symptoms that apply*")
    
    # Define all 47 symptoms organized by category
    symptom_categories = {
        "General Symptoms": [
            "nausea", "fatigue", "dizziness", "weakness", "fever", "chills", "shivering", "high_fever"
        ],
        "Pain Symptoms": [
            "joint_pain", "abdominal_pain", "headache", "chest_pain", "pain_behind_the_eyes",
            "back_pain", "knee_ache", "severe_headache", "throbbing_headache", "facial_pain",
            "sinus_headache", "pain_behind_eyes"
        ],
        "Respiratory Symptoms": [
            "runny_nose", "cough", "sore_throat", "sneezing", "shortness_of_breath",
            "rapid_breathing", "reduced_smell_and_taste"
        ],
        "Cardiovascular Symptoms": [
            "rapid_heart_rate"
        ],
        "Gastrointestinal Symptoms": [
            "vomiting", "diarrhea"
        ],
        "Skin Symptoms": [
            "rashes", "skin_irritation", "itchiness"
        ],
        "Neurological Symptoms": [
            "trouble_seeing", "confusion"
        ],
        "Other Symptoms": [
            "body_aches", "swollen_glands"
        ],
        "Medical History": [
            "asthma_history", "high_cholesterol", "diabetes", "obesity", "hiv_aids",
            "nasal_polyps", "asthma", "high_blood_pressure"
        ]
    }
    
    # Create symptom selection with expandable sections
    selected_symptoms = {}
    
    for category, symptoms in symptom_categories.items():
        with st.expander(f"**{category}** ({len(symptoms)} items)", expanded=(category == "General Symptoms")):
            cols = st.columns(3)
            for idx, symptom in enumerate(symptoms):
                col_idx = idx % 3
                with cols[col_idx]:
                    symptom_display = symptom.replace("_", " ").title()
                    selected_symptoms[symptom] = st.checkbox(symptom_display, key=symptom)
    
    # Count selected symptoms
    total_selected = sum(selected_symptoms.values())
    st.info(f" **{total_selected} symptoms selected**")
    
    # Warning for too many symptoms
    if total_selected > 40:
        st.warning("⚠️ You have selected many symptoms. This may reduce prediction accuracy. Please ensure all selections are accurate.")
    
    # Prediction Button
    st.markdown("---")
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    
    with predict_col2:
        predict_button = st.button(" Predict Disease", use_container_width=True)
    
    # Prediction Logic
    if predict_button:
        if total_selected == 0:
            st.warning("⚠️ Please select at least one symptom before predicting.")
        else:
            with st.spinner(" Analyzing symptoms and environmental data..."):
                # Create feature array in the correct order
                feature_order = [
                    'Age', 'Gender', 'Temperature (C)', 'Humidity', 'Wind Speed (km/h)',
                    'nausea', 'joint_pain', 'abdominal_pain', 'high_fever', 'chills',
                    'fatigue', 'runny_nose', 'pain_behind_the_eyes', 'dizziness', 'headache',
                    'chest_pain', 'vomiting', 'cough', 'shivering', 'asthma_history',
                    'high_cholesterol', 'diabetes', 'obesity', 'hiv_aids', 'nasal_polyps',
                    'asthma', 'severe_headache', 'weakness', 'trouble_seeing',
                    'fever', 'body_aches', 'sore_throat', 'sneezing', 'diarrhea',
                    'rapid_breathing', 'rapid_heart_rate', 'pain_behind_eyes', 'swollen_glands',
                    'rashes', 'sinus_headache', 'facial_pain', 'shortness_of_breath',
                    'reduced_smell_and_taste', 'skin_irritation', 'itchiness',
                    'throbbing_headache', 'confusion', 'back_pain', 'knee_ache'
                ]
                
                # Build feature array
                features = []
                features.append(age)
                features.append(gender_encoded)
                features.append(temperature)
                features.append(humidity_normalized)
                features.append(wind_speed)
                
                # Add all symptoms
                for symptom in feature_order[5:]:
                    features.append(1 if selected_symptoms.get(symptom, False) else 0)
                
                # Convert to numpy array and reshape
                features_array = np.array(features).reshape(1, -1)
                
                # Scale the features
                features_scaled = scaler.transform(features_array)
                
                # Make prediction
                prediction = model.predict(features_scaled)
                prediction_proba = model.predict_proba(features_scaled)
                
                # Decode prediction
                predicted_disease = label_encoder.inverse_transform(prediction)[0]
                confidence = np.max(prediction_proba) * 100
                
                # Store results in session state
                st.session_state.prediction_results = {
                    'disease': predicted_disease,
                    'confidence': confidence,
                    'proba': prediction_proba,
                    'age': age,
                    'gender': gender,
                    'temperature': temperature,
                    'humidity': humidity,
                    'wind_speed': wind_speed,
                    'selected_symptoms': selected_symptoms
                }
                
                # Display Results
                st.markdown("---")
                st.markdown("##  Prediction Results")
                
                # Main prediction box
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>Predicted Disease</h2>
                    <h1 style="font-size: 48px; margin: 20px 0;">{predicted_disease}</h1>
                    <h3>Confidence: {confidence:.2f}%</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence interpretation
                if confidence >= 95:
                    st.success(" **Very High Confidence**: The model is very certain about this prediction.")
                elif confidence >= 80:
                    st.info(" **High Confidence**: Strong likelihood, but consider other possibilities.")
                elif confidence >= 60:
                    st.warning("⚠️ **Moderate Confidence**: Multiple diseases possible. Consult a professional.")
                else:
                    st.error("❌ **Low Confidence**: Uncertain prediction. Medical consultation strongly recommended.")
                
                # Show all probabilities
                st.markdown("###  Probability Distribution Across All Diseases")
                
                # Create DataFrame for probabilities
                disease_names = label_encoder.classes_
                probabilities = prediction_proba[0] * 100
                
                prob_df = pd.DataFrame({
                    'Disease': disease_names,
                    'Probability (%)': probabilities
                }).sort_values('Probability (%)', ascending=False)
                
                # Create interactive bar chart
                fig = px.bar(
                    prob_df,
                    x='Probability (%)',
                    y='Disease',
                    orientation='h',
                    color='Probability (%)',
                    color_continuous_scale='Blues',
                    title='Disease Prediction Probabilities'
                )
                fig.update_layout(
                    height=500,
                    showlegend=False,
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show top 3 predictions
                st.markdown("### 🏆 Top 3 Most Likely Diseases")
                top_3 = prob_df.head(3)
                
                cols = st.columns(3)
                for idx, (_, row) in enumerate(top_3.iterrows()):
                    with cols[idx]:
                        rank = ["🥇", "🥈", "🥉"][idx]
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{rank} Rank {idx+1}</h3>
                            <h4>{row['Disease']}</h4>
                            <h2 style="color: #1f77b4;">{row['Probability (%)']:.2f}%</h2>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Recommendations
                st.markdown("###  Recommendations")
                st.info(f"""
                Based on the prediction of **{predicted_disease}** with {confidence:.2f}% confidence:
                
                1.  **Consult a healthcare professional** immediately for proper diagnosis
                2.  Share your symptoms and this prediction with your doctor
                3.  Consider getting appropriate medical tests
                4.  Do NOT self-medicate based on this prediction
                5.  If symptoms are severe, seek emergency medical care
                """)
                
                # Show input summary
                with st.expander(" Input Summary"):
                    st.markdown("**Demographic Information:**")
                    st.write(f"- Age: {age} years")
                    st.write(f"- Gender: {gender}")
                    
                    st.markdown("**Environmental Conditions:**")
                    st.write(f"- Temperature: {temperature}°C")
                    st.write(f"- Humidity: {humidity}%")
                    st.write(f"- Wind Speed: {wind_speed} km/h")
                    
                    st.markdown("**Selected Symptoms:**")
                    selected_list = [s.replace("_", " ").title() for s, v in selected_symptoms.items() if v]
                    if selected_list:
                        for symptom in selected_list:
                            st.write(f"- {symptom}")
                    else:
                        st.write("No symptoms selected")

# MODEL INFORMATION PAGE
elif st.session_state.page == 'Model':
    st.markdown('<div class="main-header"> Model Information</div>', unsafe_allow_html=True)
    
    st.markdown("##  Machine Learning Model")
    st.write("""
    This system employs a **Random Forest Classifier**, an ensemble learning method that operates by 
    constructing multiple decision trees during training and outputting the mode of predictions.
    """)
    
    st.markdown("##  Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Testing Accuracy", "98.29%", help="Overall prediction accuracy on test data")
    
    with col2:
        st.metric("Precision", "0.9832", help="Proportion of correct positive predictions")
    
    with col3:
        st.metric("F1-Score", "0.9830", help="Harmonic mean of precision and recall")
    
    with col4:
        st.metric("Disease Classes", "11", help="Number of diseases the model can predict")
    
    st.markdown("---")
    
    st.markdown("##  Model Features")
    st.write("The model uses **49 input features** divided into the following categories:")
    
    feature_col1, feature_col2 = st.columns(2)
    
    with feature_col1:
        st.markdown("### Demographic Features (2)")
        st.write("- Age")
        st.write("- Gender")
        
        st.markdown("### Environmental Features (3)")
        st.write("- Temperature (°C)")
        st.write("- Humidity (%)")
        st.write("- Wind Speed (km/h)")
    
    with feature_col2:
        st.markdown("### Symptom Features (44)")
        st.write("- General symptoms (8)")
        st.write("- Pain symptoms (12)")
        st.write("- Respiratory symptoms (7)")
        st.write("- Cardiovascular symptoms (1)")
        st.write("- Gastrointestinal symptoms (2)")
        st.write("- Skin symptoms (3)")
        st.write("- Neurological symptoms (2)")
        st.write("- Medical history (9)")
    
    st.markdown("---")
    
    st.markdown("##  Technical Details")
    
    with st.expander("**Algorithm: Random Forest**"):
        st.write("""
        Random Forest is an ensemble learning algorithm that:
        - Constructs multiple decision trees during training
        - Outputs the class that is the mode of predictions from individual trees
        - Reduces overfitting by averaging multiple deep decision trees
        - Provides feature importance rankings
        - Handles both numerical and categorical data effectively
        """)
    
    with st.expander("**Data Preprocessing**"):
        st.write("""
        The model pipeline includes:
        - **Feature Scaling**: StandardScaler for normalizing numerical features
        - **Label Encoding**: Converting disease names to numerical labels
        - **Feature Engineering**: Combining demographic, environmental, and symptom data
        - **Data Validation**: Ensuring all inputs are within expected ranges
        """)
    
    with st.expander("**Model Training**"):
        st.write("""
        Training specifications:
        - Training dataset with balanced disease classes
        - Cross-validation for robust performance estimation
        - Hyperparameter tuning for optimal performance
        - Evaluation on separate test dataset
        """)
    
    st.markdown("---")
    
    st.markdown("##  Why Random Forest?")
    
    benefits = {
        "High Accuracy": "Achieves excellent performance through ensemble learning",
        "Robustness": "Resistant to overfitting and handles noisy data well",
        "Feature Importance": "Provides insights into which features matter most",
        "Versatility": "Works well with both categorical and numerical data",
        "No Scaling Required": "Tree-based models don't require feature scaling (though we use it)",
        "Handles Missing Data": "Can work with incomplete data points"
    }
    
    for benefit, description in benefits.items():
        st.success(f"**{benefit}**: {description}")

# Footer (appears on all pages)
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><strong>Disease Prediction System v1.0</strong></p>
    <p>Machine Learning Individual Project | December 2025</p>
    <p>Powered by Random Forest Algorithm (Accuracy: 98.29%)</p>
    <p><em>For educational and research purposes only</em></p>
</div>
""", unsafe_allow_html=True)