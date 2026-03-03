# app.py - FINAL FIXED VERSION with absolute paths
from flask import Flask, render_template, request
import numpy as np
import joblib
import os
import sys
import traceback

app = Flask(__name__)

# Get the absolute path to the project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("="*70)
print("🚀 STARTING APPLICATION")
print("="*70)
print(f"BASE_DIR: {BASE_DIR}")

# Print debug info
print(f"Current directory: {os.getcwd()}")
print(f"Files in current dir: {os.listdir('.')}")

# Check models folder with absolute path
models_path = os.path.join(BASE_DIR, 'models')
scalers_path = os.path.join(models_path, 'scalers')

print(f"\n📁 Models path: {models_path}")
print(f"Models exist: {os.path.exists(models_path)}")
if os.path.exists(models_path):
    print(f"Models folder contents: {os.listdir(models_path)}")

print(f"\n📁 Scalers path: {scalers_path}")
print(f"Scalers exist: {os.path.exists(scalers_path)}")
if os.path.exists(scalers_path):
    print(f"Scalers folder contents: {os.listdir(scalers_path)}")

# Load models with absolute paths
models = {}
scalers = {}

def safe_load(loader_func, name, file_path):
    """Safely load a model with error handling"""
    try:
        print(f"\n🔄 Attempting to load {name} from: {file_path}")
        if not os.path.exists(file_path):
            print(f"❌ File does not exist: {file_path}")
            return None
        
        obj = loader_func(file_path)
        print(f"✅ Successfully loaded {name}")
        return obj
    except Exception as e:
        print(f"❌ Error loading {name}: {str(e)}")
        traceback.print_exc()
        return None

# Load models using absolute paths
models['diabetes'] = safe_load(joblib.load, 'diabetes_model', 
                               os.path.join(models_path, 'diabetes_model.pkl'))
models['heart'] = safe_load(joblib.load, 'heart_model',
                           os.path.join(models_path, 'heart_model.pkl'))
models['parkinsons'] = safe_load(joblib.load, 'parkinsons_model',
                                os.path.join(models_path, 'parkinsons_model.pkl'))

# Load scalers
scalers['diabetes'] = safe_load(joblib.load, 'diabetes_scaler',
                                os.path.join(scalers_path, 'diabetes_scaler.pkl'))
scalers['heart'] = safe_load(joblib.load, 'heart_scaler',
                            os.path.join(scalers_path, 'heart_scaler.pkl'))
scalers['parkinsons'] = safe_load(joblib.load, 'parkinsons_scaler',
                                  os.path.join(scalers_path, 'parkinsons_scaler.pkl'))

print("\n" + "="*70)
print("📊 LOAD SUMMARY:")
print(f"Models loaded: {[k for k,v in models.items() if v is not None]}")
print(f"Models failed: {[k for k,v in models.items() if v is None]}")
print(f"Scalers loaded: {[k for k,v in scalers.items() if v is not None]}")
print(f"Scalers failed: {[k for k,v in scalers.items() if v is None]}")
print("="*70)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/debug')
def debug():
    """Detailed debug endpoint"""
    info = {
        'base_dir': BASE_DIR,
        'working_directory': os.getcwd(),
        'directory_contents': os.listdir('.'),
        'models_path_exists': os.path.exists(models_path),
        'models_folder': os.listdir(models_path) if os.path.exists(models_path) else 'NOT FOUND',
        'scalers_path_exists': os.path.exists(scalers_path),
        'scalers_folder': os.listdir(scalers_path) if os.path.exists(scalers_path) else 'NOT FOUND',
        'models_loaded': [k for k,v in models.items() if v is not None],
        'models_missing': [k for k,v in models.items() if v is None],
        'scalers_loaded': [k for k,v in scalers.items() if v is not None],
        'scalers_missing': [k for k,v in scalers.items() if v is None],
    }
    info_str = str(info).replace(', ', ',\n')
    return f"<pre>{info_str}</pre>"

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
            else:
                return f"Disease '{disease}' not found", 404
        except Exception as e:
            error_msg = f"❌ ERROR: {str(e)}\n\n{traceback.format_exc()}"
            return f"<pre>{error_msg}</pre>", 500
    return render_template(f'{disease}_form.html', disease=disease)

def predict_diabetes():
    if models['diabetes'] is None:
        return "❌ Diabetes model not loaded. Check /debug endpoint"
    
    try:
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
    except Exception as e:
        return f"❌ Prediction error: {str(e)}"

def predict_heart():
    if models['heart'] is None:
        return "❌ Heart model not loaded. Check /debug endpoint"
    
    try:
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
    except Exception as e:
        return f"❌ Prediction error: {str(e)}"

def predict_parkinsons():
    if models['parkinsons'] is None:
        return "❌ Parkinson's model not loaded. Check /debug endpoint"
    
    try:
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
    except Exception as e:
        return f"❌ Prediction error: {str(e)}"

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
