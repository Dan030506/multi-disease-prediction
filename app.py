from flask import Flask, render_template, request
import numpy as np
import joblib
import os
import sys
import traceback

app = Flask(__name__)

# Print debug info on startup
print("="*70)
print("🚀 STARTING APPLICATION IN DEBUG MODE")
print("="*70)
print(f"Current directory: {os.getcwd()}")
print(f"Python version: {sys.version}")
print(f"Files in current dir: {os.listdir('.')}")

# Check models folder
if os.path.exists('models'):
    print(f"\n📁 Models folder contents: {os.listdir('models')}")
else:
    print("\n❌ models folder not found!")

if os.path.exists('models/scalers'):
    print(f"📁 Scalers folder contents: {os.listdir('models/scalers')}")
else:
    print("❌ scalers folder not found!")

# Load models with detailed error handling
models = {}
scalers = {}

def safe_load(loader_func, name, path):
    try:
        print(f"\n🔄 Loading {name} from {path}...")
        obj = loader_func(path)
        print(f"✅ {name} loaded successfully")
        return obj
    except FileNotFoundError:
        print(f"❌ File not found: {path}")
        print(f"   Current working directory: {os.getcwd()}")
        print(f"   Files in current dir: {[f for f in os.listdir('.') if f.endswith('.pkl')]}")
        return None
    except Exception as e:
        print(f"❌ Error loading {name}: {str(e)}")
        traceback.print_exc()
        return None

# Load all models
models['diabetes'] = safe_load(joblib.load, 'diabetes_model', 'models/diabetes_model.pkl')
models['heart'] = safe_load(joblib.load, 'heart_model', 'models/heart_model.pkl')
models['parkinsons'] = safe_load(joblib.load, 'parkinsons_model', 'models/parkinsons_model.pkl')

scalers['diabetes'] = safe_load(joblib.load, 'diabetes_scaler', 'models/scalers/diabetes_scaler.pkl')
scalers['heart'] = safe_load(joblib.load, 'heart_scaler', 'models/scalers/heart_scaler.pkl')
scalers['parkinsons'] = safe_load(joblib.load, 'parkinsons_scaler', 'models/scalers/parkinsons_scaler.pkl')

print("\n" + "="*70)
print("📊 LOAD SUMMARY:")
print(f"Models loaded: {[k for k,v in models.items() if v is not None]}")
print(f"Scalers loaded: {[k for k,v in scalers.items() if v is not None]}")
print("="*70)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/debug')
def debug():
    """Super detailed debug endpoint - FIXED: No backslashes in f-string"""
    info = {
        'working_directory': os.getcwd(),
        'directory_contents': os.listdir('.'),
        'models_folder': os.listdir('models') if os.path.exists('models') else 'NOT FOUND',
        'scalers_folder': os.listdir('models/scalers') if os.path.exists('models/scalers') else 'NOT FOUND',
        'models_loaded': [k for k,v in models.items() if v is not None],
        'models_missing': [k for k,v in models.items() if v is None],
        'scalers_loaded': [k for k,v in scalers.items() if v is not None],
        'scalers_missing': [k for k,v in scalers.items() if v is None],
    }
    # FIXED: No backslash in f-string, using string replacement instead
    info_str = str(info).replace(', ', ',\n')
    return f"<pre>{info_str}</pre>"

@app.route('/predict/<disease>', methods=['GET', 'POST'])
def predict(disease):
    if request.method == 'POST':
        try:
            print(f"\n🔍 Processing {disease} prediction")
            print(f"Form data received: {dict(request.form)}")
            
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
            print(error_msg)
            return f"<pre>{error_msg}</pre>", 500
    
    return render_template(f'{disease}_form.html', disease=disease)

def predict_diabetes():
    """Handle diabetes prediction with detailed error checking"""
    # Check if model exists
    if models['diabetes'] is None:
        return "❌ Diabetes model not loaded. Check /debug endpoint"
    if scalers['diabetes'] is None:
        return "❌ Diabetes scaler not loaded. Check /debug endpoint"
    
    try:
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
        print(f"Features extracted: {features}")
        
        # Convert to numpy array
        features_array = np.array([features])
        print(f"Array shape: {features_array.shape}")
        
        # Scale features
        features_scaled = scalers['diabetes'].transform(features_array)
        print("Scaling successful")
        
        # Predict
        prediction = models['diabetes'].predict(features_scaled)[0]
        probability = models['diabetes'].predict_proba(features_scaled)[0]
        print(f"Prediction: {prediction}, Probabilities: {probability}")
        
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
        
    except KeyError as e:
        return f"❌ Missing form field: {e}. Available fields: {list(request.form.keys())}"
    except Exception as e:
        return f"❌ Prediction error: {str(e)}\n\n{traceback.format_exc()}"

def predict_heart():
    """Handle heart disease prediction"""
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
            'features': dict(request.form)
        }
        
        return render_template('result.html', result=result)
    except Exception as e:
        return f"❌ Heart prediction error: {str(e)}\n\n{traceback.format_exc()}"

def predict_parkinsons():
    """Handle Parkinson's prediction"""
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
            'features': dict(request.form)
        }
        
        return render_template('result.html', result=result)
    except Exception as e:
        return f"❌ Parkinson's error: {str(e)}\n\n{traceback.format_exc()}"

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/test/diabetes')
def test_diabetes():
    """Test endpoint with sample values"""
    return {
        'pregnancies': 8,
        'glucose': 180,
        'bloodpressure': 88,
        'skinthickness': 35,
        'insulin': 200,
        'bmi': 38.5,
        'dpf': 1.2,
        'age': 55
    }

if __name__ == '__main__':
    app.run(debug=True, port=5000)
