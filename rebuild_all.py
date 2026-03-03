# rebuild_all.py - COMPLETE REBUILD (RUN THIS ON RENDER)
import os
import sys
import subprocess
import shutil

print("="*70)
print("💥 NUCLEAR REBUILD - DELETE AND RETRAIN ALL MODELS")
print("="*70)

# Step 1: Delete all existing model files
print("\n🔴 Step 1: Deleting old model files...")
if os.path.exists('models'):
    shutil.rmtree('models')
    print("   ✅ Deleted models folder")
else:
    print("   ⚠️ No models folder found")

# Step 2: Recreate directories
print("\n🔴 Step 2: Creating fresh directories...")
os.makedirs('models/scalers', exist_ok=True)
print("   ✅ Created models/scalers folders")

# Step 3: Train new models
print("\n🔴 Step 3: Training new models...")

# Train diabetes
print("\n📊 Training Diabetes model...")
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imbalanced-learn.over_sampling import SMOTE
import joblib

# Diabetes
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv('data/raw/diabetes.csv', names=columns, header=0)

for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

X = df.drop('Outcome', axis=1)
y = df['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

joblib.dump(model, 'models/diabetes_model.pkl')
joblib.dump(scaler, 'models/scalers/diabetes_scaler.pkl')
print("   ✅ Diabetes model saved")

# Heart
print("\n📊 Training Heart Disease model...")
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df = pd.read_csv('data/raw/heart.csv', names=columns, header=0, na_values='?')

for col in ['ca', 'thal']:
    df[col] = df[col].fillna(df[col].median())

df['target'] = df['target'].apply(lambda x: 0 if x == 0 else 1)

X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = SVC(kernel='rbf', probability=True, random_state=42)
model.fit(X_scaled, y)

joblib.dump(model, 'models/heart_model.pkl')
joblib.dump(scaler, 'models/scalers/heart_scaler.pkl')
print("   ✅ Heart model saved")

# Parkinson's
print("\n📊 Training Parkinson's model...")
df = pd.read_csv('data/raw/parkinsons.csv')

if 'name' in df.columns:
    df = df.drop('name', axis=1)

X = df.drop('status', axis=1)
y = df['status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_resampled, y_resampled)

joblib.dump(model, 'models/parkinsons_model.pkl')
joblib.dump(scaler, 'models/scalers/parkinsons_scaler.pkl')
print("   ✅ Parkinson's model saved")

# Step 4: Verify files
print("\n🔴 Step 4: Verifying saved files...")
print("\n📁 Models folder:")
for f in os.listdir('models'):
    size = os.path.getsize(f'models/{f}') / 1024
    print(f"   📄 {f}: {size:.1f} KB")

print("\n📁 Scalers folder:")
for f in os.listdir('models/scalers'):
    size = os.path.getsize(f'models/scalers/{f}') / 1024
    print(f"   📄 {f}: {size:.1f} KB")

# Step 5: Test loading
print("\n🔴 Step 5: Testing model loading...")

try:
    test_model = joblib.load('models/diabetes_model.pkl')
    print("   ✅ Successfully loaded diabetes_model.pkl")
except Exception as e:
    print(f"   ❌ Failed: {e}")

print("\n" + "="*70)
print("✅ REBUILD COMPLETE!")
print("="*70)
