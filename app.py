# app.py - SIMPLIFIED VERSION
from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

def load_model(model_name, scaler_name):
    """Lazy load models when needed"""
    try:
        model = joblib.load(f'models/{model_name}.pkl')
        scaler = joblib.load(f'models/scalers/{scaler_name}.pkl')
        return model, scaler
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/debug')
def debug():
    import os
    files = os.listdir('models') if os.path.exists('models') else []
    scalers = os.listdir('models/scalers') if os.path.exists('models/scalers') else []
    return {
        'models_folder': files,
        'scalers_folder': scalers,
        'cwd': os.getcwd()
    }

@app.route('/predict/<disease>', methods=['GET', 'POST'])
def predict(disease):
    if request.method == 'POST':
        try:
            if disease == 'diabetes':
                return predict_diabetes()
            elif disease == 'heart':
                return predict_heart()
            elif disease == 'parkinsons':
                return predict_parkinsons()
        except Exception as e:
            return f"Error: {str(e)}", 500
    return render_template(f'{disease}_form.html')

def predict_diabetes():
    model, scaler = load_model('diabetes_model', 'diabetes_scaler')
    if model is None:
        return "Model not loaded"
    
    features = [float(request.form[f]) for f in 
                ['pregnancies', 'glucose', 'bloodpressure', 'skinthickness',
                 'insulin', 'bmi', 'dpf', 'age']]
    
    features_scaled = scaler.transform([features])
    pred = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0]
    
    return render_template('result.html', result={
        'disease': 'Diabetes',
        'prediction': 'High Risk' if pred == 1 else 'Low Risk',
        'probability': f"{prob[pred]*100:.1f}%"
    })

def predict_heart():
    model, scaler = load_model('heart_model', 'heart_scaler')
    if model is None:
        return "Model not loaded"
    
    features = [float(request.form[f]) for f in 
                ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
    
    features_scaled = scaler.transform([features])
    pred = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0]
    
    return render_template('result.html', result={
        'disease': 'Heart Disease',
        'prediction': 'High Risk' if pred == 1 else 'Low Risk',
        'probability': f"{prob[pred]*100:.1f}%"
    })

def predict_parkinsons():
    model, scaler = load_model('parkinsons_model', 'parkinsons_scaler')
    if model is None:
        return "Model not loaded"
    
    features = [float(request.form[f]) for f in 
                ['mdvp_fo', 'mdvp_fhi', 'mdvp_flo', 'jitter',
                 'shimmer', 'hnr', 'rpde', 'dfa']]
    features.extend([0] * 14)
    
    features_scaled = scaler.transform([features])
    pred = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0]
    
    return render_template('result.html', result={
        'disease': "Parkinson's Disease",
        'prediction': 'High Risk' if pred == 1 else 'Low Risk',
        'probability': f"{prob[pred]*100:.1f}%"
    })

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
