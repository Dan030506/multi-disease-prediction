import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

print("="*60)
print("TRAINING ALL MODELS")
print("="*60)

os.makedirs('models/scalers', exist_ok=True)

# 1. DIABETES
print("\n1. Training Diabetes Model...")
df = pd.read_csv('data/raw/diabetes.csv', header=None)
df.columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigree','Age','Outcome']

for col in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
    df[col] = df[col].replace(0, np.nan).fillna(df[col].median())

X = df.drop('Outcome', axis=1)
y = df['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

joblib.dump(model, 'models/diabetes_model.pkl')
joblib.dump(scaler, 'models/scalers/diabetes_scaler.pkl')
print(f"✓ Diabetes model saved (accuracy: {model.score(X_scaled, y):.3f})")

# 2. HEART DISEASE
print("\n2. Training Heart Disease Model...")
df = pd.read_csv('data/raw/heart.csv', header=None, na_values='?')
df.columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']

for col in ['ca','thal']:
    df[col] = df[col].fillna(df[col].median())

df['target'] = (df['target'] > 0).astype(int)

X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = SVC(kernel='rbf', probability=True, random_state=42)
model.fit(X_scaled, y)

joblib.dump(model, 'models/heart_model.pkl')
joblib.dump(scaler, 'models/scalers/heart_scaler.pkl')
print(f"✓ Heart model saved (accuracy: {model.score(X_scaled, y):.3f})")

# 3. PARKINSON'S
print("\n3. Training Parkinson's Model...")
df = pd.read_csv('data/raw/parkinsons.csv')
if 'name' in df.columns:
    df = df.drop('name', axis=1)

print(f"\nOriginal distribution - Healthy: {sum(df['status']==0)}, Parkinson's: {sum(df['status']==1)}")

X = df.drop('status', axis=1)
y = df['status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
print(f"After SMOTE - Healthy: {sum(y_resampled==0)}, Parkinson's: {sum(y_resampled==1)}")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_resampled, y_resampled)

joblib.dump(model, 'models/parkinsons_model.pkl')
joblib.dump(scaler, 'models/scalers/parkinsons_scaler.pkl')
print("✓ Parkinson's model saved")

print("\n" + "="*60)
print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
print("="*60)
print("\nModels saved in 'models/' folder")
