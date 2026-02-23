# app.py - Complete Multi-Disease Prediction Web App
from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load all models and scalers at startup
print("="*60)
print("🚀 LOADING MODELS...")
print("="*60)

models = {}
scalers = {}

try:
    # Diabetes
    models['diabetes'] = joblib.load('models/diabetes_model.pkl')
    scalers['diabetes'] = joblib.load('models/scalers/diabetes_scaler.pkl')
    print("✅ Diabetes model loaded")
    
    # Heart Disease
    models['heart'] = joblib.load('models/heart_model.pkl')
    scalers['heart'] = joblib.load('models/scalers/heart_scaler.pkl')
    print("✅ Heart disease model loaded")
    
    # Parkinson's
    models['parkinsons'] = joblib.load('models/parkinsons_model.pkl')
    scalers['parkinsons'] = joblib.load('models/scalers/parkinsons_scaler.pkl')
    print("✅ Parkinson's model loaded")
    
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("Please run training first: python3 train_models_fresh.py")

@app.route('/')
def home():
    """Home page with disease selection"""
    return render_template('index.html')

@app.route('/predict/<disease>', methods=['GET', 'POST'])
def predict(disease):
    """Prediction page for specific disease"""
    if request.method == 'POST':
        try:
            if disease == 'diabetes':
                return predict_diabetes()
            elif disease == 'heart':
                return predict_heart()
            elif disease == 'parkinsons':
                return predict_parkinsons()
            else:
                return "Disease not found", 404
        except Exception as e:
            return f"Error: {str(e)}", 400
    
    # GET request - show form
    return render_template(f'{disease}_form.html', disease=disease)

def predict_diabetes():
    """Handle diabetes prediction"""
    # Get form data
    features = [
        float(request.form['pregnancies']),
        float(request.form['glucose']),
        float(request.form['bloodpressure']),
        float(request.form['skinthickness']),
        float(request.form['insulin']),
        float(request.form['bmi']),
        float(request.form['dpf']),
        float(request.form['age'])
    ]
    
    # Convert to numpy array and scale
    features_array = np.array([features])
    features_scaled = scalers['diabetes'].transform(features_array)
    
    # Predict
    prediction = models['diabetes'].predict(features_scaled)[0]
    probability = models['diabetes'].predict_proba(features_scaled)[0]
    
    # Prepare result
    result = {
        'disease': 'Diabetes',
        'prediction': 'High Risk' if prediction == 1 else 'Low Risk',
        'probability': f"{probability[prediction]*100:.1f}%",
        'confidence_no': f"{probability[0]*100:.1f}%",
        'confidence_yes': f"{probability[1]*100:.1f}%",
        'features': dict(request.form)
    }
    
    return render_template('result.html', result=result)

def predict_heart():
    """Handle heart disease prediction"""
    features = [
        float(request.form['age']),
        float(request.form['sex']),
        float(request.form['cp']),
        float(request.form['trestbps']),
        float(request.form['chol']),
        float(request.form['fbs']),
        float(request.form['restecg']),
        float(request.form['thalach']),
        float(request.form['exang']),
        float(request.form['oldpeak']),
        float(request.form['slope']),
        float(request.form['ca']),
        float(request.form['thal'])
    ]
    
    features_array = np.array([features])
    features_scaled = scalers['heart'].transform(features_array)
    
    prediction = models['heart'].predict(features_scaled)[0]
    probability = models['heart'].predict_proba(features_scaled)[0]
    
    result = {
        'disease': 'Heart Disease',
        'prediction': 'High Risk' if prediction == 1 else 'Low Risk',
        'probability': f"{probability[prediction]*100:.1f}%",
        'confidence_no': f"{probability[0]*100:.1f}%",
        'confidence_yes': f"{probability[1]*100:.1f}%",
        'features': dict(request.form)
    }
    
    return render_template('result.html', result=result)

def predict_parkinsons():
    """Handle Parkinson's prediction"""
    # Get the 8 main features (others will be set to 0 for simplicity)
    features = [
        float(request.form['mdvp_fo']),
        float(request.form['mdvp_fhi']),
        float(request.form['mdvp_flo']),
        float(request.form['jitter']),
        float(request.form['shimmer']),
        float(request.form['hnr']),
        float(request.form['rpde']),
        float(request.form['dfa'])
    ]
    # Pad with zeros for remaining 14 features
    features.extend([0] * 14)
    
    features_array = np.array([features])
    features_scaled = scalers['parkinsons'].transform(features_array)
    
    prediction = models['parkinsons'].predict(features_scaled)[0]
    probability = models['parkinsons'].predict_proba(features_scaled)[0]
    
    result = {
        'disease': "Parkinson's Disease",
        'prediction': 'High Risk' if prediction == 1 else 'Low Risk',
        'probability': f"{probability[prediction]*100:.1f}%",
        'confidence_no': f"{probability[0]*100:.1f}%",
        'confidence_yes': f"{probability[1]*100:.1f}%",
        'features': dict(request.form)
    }
    
    return render_template('result.html', result=result)

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
