import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page config
st.set_page_config(
    page_title="Multi-Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Multi-Disease Prediction System")
st.markdown("---")

# Load models
@st.cache_resource
def load_models():
    models = {}
    scalers = {}
    try:
        models['diabetes'] = joblib.load('models/diabetes_model.pkl')
        scalers['diabetes'] = joblib.load('models/scalers/diabetes_scaler.pkl')
        models['heart'] = joblib.load('models/heart_model.pkl')
        scalers['heart'] = joblib.load('models/scalers/heart_scaler.pkl')
        models['parkinsons'] = joblib.load('models/parkinsons_model.pkl')
        scalers['parkinsons'] = joblib.load('models/scalers/parkinsons_scaler.pkl')
        return models, scalers, True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, False

models, scalers, success = load_models()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Diabetes", "Heart Disease", "Parkinson's", "About"])

# Home page
if page == "Home":
    st.header("Welcome to AI-Powered Health Risk Assessment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("### 🩸 Diabetes")
        st.write("Predict diabetes risk using clinical parameters")
        if st.button("Go to Diabetes", key="home_dia"):
            st.session_state.page = "Diabetes"
            st.rerun()
    
    with col2:
        st.info("### ❤️ Heart Disease")
        st.write("Assess cardiovascular health and risk factors")
        if st.button("Go to Heart Disease", key="home_heart"):
            st.session_state.page = "Heart Disease"
            st.rerun()
    
    with col3:
        st.info("### 🧠 Parkinson's")
        st.write("Analyze voice patterns for Parkinson's detection")
        if st.button("Go to Parkinson's", key="home_park"):
            st.session_state.page = "Parkinson's"
            st.rerun()

# Diabetes page
elif page == "Diabetes":
    st.header("🩸 Diabetes Risk Assessment")
    
    with st.form("diabetes_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input("Pregnancies", 0, 20, 2)
            glucose = st.number_input("Glucose Level (mg/dL)", 0, 300, 120)
            bloodpressure = st.number_input("Blood Pressure (mm Hg)", 0, 200, 70)
            skinthickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
        
        with col2:
            insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 80)
            bmi = st.number_input("BMI", 10.0, 70.0, 32.0, 0.1)
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, 0.01)
            age = st.number_input("Age", 18, 120, 35)
        
        submitted = st.form_submit_button("Predict Risk", type="primary")
        
        if submitted:
            if not success:
                st.error("Models not loaded. Please train models first.")
            else:
                features = np.array([[pregnancies, glucose, bloodpressure, skinthickness,
                                    insulin, bmi, dpf, age]])
                features_scaled = scalers['diabetes'].transform(features)
                pred = models['diabetes'].predict(features_scaled)[0]
                prob = models['diabetes'].predict_proba(features_scaled)[0]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if pred == 1:
                        st.error(f"### 🔴 High Risk")
                    else:
                        st.success(f"### 🟢 Low Risk")
                
                with col2:
                    st.metric("Confidence", f"{prob[pred]*100:.1f}%")
                
                with col3:
                    st.metric("Risk Probability", f"{prob[1]*100:.1f}%")
                
                st.info(f"📊 Detailed Probabilities:")
                st.write(f"- Low Risk: {prob[0]*100:.1f}%")
                st.write(f"- High Risk: {prob[1]*100:.1f}%")

# Heart Disease page
elif page == "Heart Disease":
    st.header("❤️ Heart Disease Risk Assessment")
    
    with st.form("heart_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", 20, 100, 55)
            sex = st.selectbox("Sex", ["Female", "Male"])
            sex_val = 1 if sex == "Male" else 0
            cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
            cp_val = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp) + 1
            trestbps = st.number_input("Resting Blood Pressure", 80, 200, 130)
            chol = st.number_input("Cholesterol", 100, 600, 240)
            fbs = st.selectbox("Fasting Blood Sugar >120", ["No", "Yes"])
            fbs_val = 1 if fbs == "Yes" else 0
        
        with col2:
            restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "LV Hypertrophy"])
            restecg_val = ["Normal", "ST-T Abnormality", "LV Hypertrophy"].index(restecg)
            thalach = st.number_input("Max Heart Rate", 60, 220, 150)
            exang = st.selectbox("Exercise Angina", ["No", "Yes"])
            exang_val = 1 if exang == "Yes" else 0
            oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, 0.1)
            slope = st.selectbox("Slope", ["Upsloping", "Flat", "Downsloping"])
            slope_val = ["Upsloping", "Flat", "Downsloping"].index(slope) + 1
            ca = st.number_input("Major Vessels", 0, 3, 0)
            thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
            thal_val = [3, 6, 7][["Normal", "Fixed Defect", "Reversible Defect"].index(thal)]
        
        submitted = st.form_submit_button("Predict Risk", type="primary")
        
        if submitted and success:
            features = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val,
                                restecg_val, thalach, exang_val, oldpeak, slope_val, ca, thal_val]])
            features_scaled = scalers['heart'].transform(features)
            pred = models['heart'].predict(features_scaled)[0]
            prob = models['heart'].predict_proba(features_scaled)[0]
            
            if pred == 1:
                st.error(f"### 🔴 High Risk")
            else:
                st.success(f"### 🟢 Low Risk")
            
            st.metric("Confidence", f"{prob[pred]*100:.1f}%")
            st.info(f"Low Risk: {prob[0]*100:.1f}% | High Risk: {prob[1]*100:.1f}%")

# Parkinson's page
elif page == "Parkinson's":
    st.header("🧠 Parkinson's Risk Assessment")
    
    with st.form("parkinsons_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            mdvp_fo = st.number_input("MDVP:Fo(Hz)", 50.0, 300.0, 120.0, 1.0)
            mdvp_fhi = st.number_input("MDVP:Fhi(Hz)", 50.0, 600.0, 150.0, 1.0)
            mdvp_flo = st.number_input("MDVP:Flo(Hz)", 40.0, 250.0, 100.0, 1.0)
            jitter = st.number_input("Jitter(%)", 0.0, 0.1, 0.005, 0.001)
        
        with col2:
            shimmer = st.number_input("Shimmer", 0.0, 0.5, 0.03, 0.01)
            hnr = st.number_input("HNR", 0.0, 40.0, 20.0, 1.0)
            rpde = st.number_input("RPDE", 0.0, 1.0, 0.5, 0.01)
            dfa = st.number_input("DFA", 0.0, 1.0, 0.6, 0.01)
        
        submitted = st.form_submit_button("Predict Risk", type="primary")
        
        if submitted and success:
            features = np.array([[mdvp_fo, mdvp_fhi, mdvp_flo, jitter, shimmer, hnr, rpde, dfa] + [0]*14])
            features_scaled = scalers['parkinsons'].transform(features)
            pred = models['parkinsons'].predict(features_scaled)[0]
            prob = models['parkinsons'].predict_proba(features_scaled)[0]
            
            if pred == 1:
                st.error(f"### 🔴 High Risk")
            else:
                st.success(f"### 🟢 Low Risk")
            
            st.metric("Confidence", f"{prob[pred]*100:.1f}%")

# About page
elif page == "About":
    st.header("📚 About The Project")
    st.markdown("""
    ### Final Year Major Project
    **BSc Computer Science & Machine Learning**  
    **Loyola Academy**
    
    #### Model Performance
    - 🩸 **Diabetes:** Random Forest - 82% accuracy
    - ❤️ **Heart Disease:** SVM - 87% accuracy
    - 🧠 **Parkinson's:** Random Forest - 91% accuracy
    
    #### Datasets Used
    - Pima Indians Diabetes Dataset (UCI)
    - Cleveland Heart Disease Dataset (UCI)
    - UCI Parkinson's Dataset
    """)

# Debug info in sidebar
with st.sidebar.expander("Debug Info"):
    st.write("Models loaded:", success)
    if os.path.exists('models'):
        st.write("Models folder:", os.listdir('models'))
