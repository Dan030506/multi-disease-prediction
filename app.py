# app.py - SIMPLEST POSSIBLE VERSION
from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load models ONCE at startup with error checking
print("="*60)
print("🚀 LOADING MODELS...")
print("="*60)

models = {}
scalers = {}

try:
    models['diabetes'] = joblib.load('models/diabetes_model.pkl')
    print("✅ Diabetes model loaded")
except Exception as e:
    print(f"❌ Diabetes model failed: {e}")

try:
    models['heart'] = joblib.load('models/heart_model.pkl')
    print("✅ Heart model loaded")
except Exception as e:
    print(f"❌ Heart model failed: {e}")

try:
    models['parkinsons'] = joblib.load('models/parkinsons_model.pkl')
    print("✅ Parkinson's model loaded")
except Exception as e:
    print(f"❌ Parkinson's model failed: {e}")

try:
    scalers['diabetes'] = joblib.load('models/scalers/diabetes_scaler.pkl')
    scalers['heart'] = joblib.load('models/scalers/heart_scaler.pkl')
    scalers['parkinsons'] = joblib.load('models/scalers/parkinsons_scaler.pkl')
    print("✅ All scalers loaded")
except Exception as e:
    print(f"❌ Scaler error: {e}")

print("="*60)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/debug')
def debug():
    import os
    return {
        'models_loaded': list(models.keys()),
        'models_failed': [k for k in ['diabetes','heart','parkinsons'] if k not in models],
        'models_folder': os.listdir('models') if os.path.exists('models') else [],
        'scalers_folder': os.listdir('models/scalers') if os.path.exists('models/scalers') else [],
    }

@app.route('/predict/<disease>', methods=['POST'])
def predict(disease):
    try:
        if disease == 'diabetes':
            return predict_diabetes()
        elif disease == 'heart':
            return predict_heart()
        elif disease == 'parkinsons':
            return predict_parkinsons()
    except Exception as e:
        return f"Error: {str(e)}"

def predict_diabetes():
    if 'diabetes' not in models:
        return "Model not loaded"
    
    features = [[
        float(request.form['pregnancies']),
        float(request.form['glucose']),
        float(request.form['bloodpressure']),
        float(request.form['skinthickness']),
        float(request.form['insulin']),
        float(request.form['bmi']),
        float(request.form['dpf']),
        float(request.form['age'])
    ]]
    
    features_scaled = scalers['diabetes'].transform(features)
    pred = models['diabetes'].predict(features_scaled)[0]
    
    return f"Prediction: {'High Risk' if pred==1 else 'Low Risk'}"

def predict_heart():
    return "Heart prediction endpoint"

def predict_parkinsons():
    return "Parkinson's prediction endpoint"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
