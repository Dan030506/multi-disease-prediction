# train_on_render.py - Run this during Render build
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os
import sys

print("="*60)
print("🔥 TRAINING MODELS ON RENDER")
print("="*60)

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('models/scalers', exist_ok=True)

# =====================================
# 1. DIABETES
# =====================================
print("\n📊 1. TRAINING DIABETES MODEL...")
print("-"*40)

# Load data
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv('data/raw/diabetes.csv', names=columns, header=0)

# Fix zeros
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Save
joblib.dump(model, 'models/diabetes_model.pkl')
joblib.dump(scaler, 'models/scalers/diabetes_scaler.pkl')
print("✅ Diabetes model saved")

# =====================================
# 2. HEART DISEASE
# =====================================
print("\n📊 2. TRAINING HEART DISEASE MODEL...")
print("-"*40)

columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df = pd.read_csv('data/raw/heart.csv', names=columns, header=0, na_values='?')

# Fix missing
for col in ['ca', 'thal']:
    df[col] = df[col].fillna(df[col].median())

# Convert target
df['target'] = df['target'].apply(lambda x: 0 if x == 0 else 1)

X = df.drop('target', axis=1)
y = df['target']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train
model = SVC(kernel='rbf', probability=True, random_state=42)
model.fit(X_scaled, y)

# Save
joblib.dump(model, 'models/heart_model.pkl')
joblib.dump(scaler, 'models/scalers/heart_scaler.pkl')
print("✅ Heart model saved")

# =====================================
# 3. PARKINSON'S
# =====================================
print("\n📊 3. TRAINING PARKINSON'S MODEL...")
print("-"*40)

df = pd.read_csv('data/raw/parkinsons.csv')

if 'name' in df.columns:
    df = df.drop('name', axis=1)

X = df.drop('status', axis=1)
y = df['status']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_resampled, y_resampled)

# Save
joblib.dump(model, 'models/parkinsons_model.pkl')
joblib.dump(scaler, 'models/scalers/parkinsons_scaler.pkl')
print("✅ Parkinson's model saved")

print("\n" + "="*60)
print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
print("="*60)
