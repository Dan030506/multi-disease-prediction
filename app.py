# app.py - FIXED VERSION with proper route handling
from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load models
models = {}
scalers = {}

try:
    models['diabetes'] = joblib.load('models/diabetes_model.pkl')
    scalers['diabetes'] = joblib.load('models/scalers/diabetes_scaler.pkl')
    print("✅ Diabetes model loaded")
except Exception as e:
    print(f"❌ Diabetes model failed: {e}")

try:
    models['heart'] = joblib.load('models/heart_model.pkl')
    scalers['heart'] = joblib.load('models/scalers/heart_scaler.pkl')
    print("✅ Heart model loaded")
except Exception as e:
    print(f"❌ Heart model failed: {e}")

try:
    models['parkinsons'] = joblib.load('models/parkinsons_model.pkl')
    scalers['parkinsons'] = joblib.load('models/scalers/parkinsons_scaler.pkl')
    print("✅ Parkinson's model loaded")
except Exception as e:
    print(f"❌ Parkinson's model failed: {e}")

@app.route('/')
def home():
    """Homepage with disease selection"""
    return render_template('index.html')

@app.route('/predict/<disease>', methods=['GET', 'POST'])
def predict(disease):
    """Handle both displaying form (GET) and prediction (POST)"""
    
    # If it's a GET request, show the form
    if request.method == 'GET':
        return render_template(f'{disease}_form.html', disease=disease)
    
    # If it's a POST request, make prediction
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
        return f"Error: {str(e)}", 500

def predict_diabetes():
    """Handle diabetes prediction"""
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
    
    features_array = np.array([features])
    features_scaled = scalers['diabetes'].transform(features_array)
    
    prediction = models['diabetes'].predict(features_scaled)[0]
    probability = models['diabetes'].predict_proba(features_scaled)[0]
    
    result = {
        'disease': 'Diabetes',
        'prediction': 'High Risk' if prediction == 1 else 'Low Risk',
        'probability': f"{probability[prediction]*100:.1f}%",
        'confidence_no': f"{probability[0]*100:.1f}%",
        'confidence_yes': f"{probability[1]*100:.1f}%",
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
    }
    
    return render_template('result.html', result=result)

def predict_parkinsons():
    """Handle Parkinson's prediction"""
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
    }
    
    return render_template('result.html', result=result)

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/debug')
def debug():
    """Debug endpoint"""
    import os
    return {
        'models_loaded': list(models.keys()),
        'models_folder': os.listdir('models') if os.path.exists('models') else [],
        'scalers_folder': os.listdir('models/scalers') if os.path.exists('models/scalers') else [],
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
