# preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os

print("="*60)
print("🔄 PREPROCESSING PIPELINE")
print("="*60)

# Create directories
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models/scalers', exist_ok=True)

# =====================================
# 1. DIABETES
# =====================================
print("\n📊 1. PROCESSING DIABETES...")
print("-"*40)

# Load with correct columns
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv('data/raw/diabetes.csv', names=columns, header=0)
print(f"Loaded: {df.shape}")

# Fix zeros
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    zeros = (df[col] == 0).sum()
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())
    print(f"  {col}: fixed {zeros} zeros")

# Split features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split (70% train, 15% val, 15% test)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Save
np.save('data/processed/diabetes_X_train.npy', X_train)
np.save('data/processed/diabetes_X_val.npy', X_val)
np.save('data/processed/diabetes_X_test.npy', X_test)
np.save('data/processed/diabetes_y_train.npy', y_train)
np.save('data/processed/diabetes_y_val.npy', y_val)
np.save('data/processed/diabetes_y_test.npy', y_test)
joblib.dump(scaler, 'models/scalers/diabetes_scaler.pkl')

print(f"✅ Saved! Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# =====================================
# 2. HEART DISEASE
# =====================================
print("\n📊 2. PROCESSING HEART DISEASE...")
print("-"*40)

# Load with correct columns
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df = pd.read_csv('data/raw/heart.csv', names=columns, header=0, na_values='?')
print(f"Loaded: {df.shape}")

# Fix missing values
for col in ['ca', 'thal']:
    if col in df.columns:
        missing = df[col].isnull().sum()
        df[col] = df[col].fillna(df[col].median())
        print(f"  {col}: fixed {missing} missing")

# Convert target to binary
df['target'] = df['target'].apply(lambda x: 0 if x == 0 else 1)

# Split
X = df.drop('target', axis=1)
y = df['target']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Save
np.save('data/processed/heart_X_train.npy', X_train)
np.save('data/processed/heart_X_val.npy', X_val)
np.save('data/processed/heart_X_test.npy', X_test)
np.save('data/processed/heart_y_train.npy', y_train)
np.save('data/processed/heart_y_val.npy', y_val)
np.save('data/processed/heart_y_test.npy', y_test)
joblib.dump(scaler, 'models/scalers/heart_scaler.pkl')

print(f"✅ Saved! Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# =====================================
# 3. PARKINSON'S (WITH SMOTE)
# =====================================
print("\n📊 3. PROCESSING PARKINSON'S...")
print("-"*40)

df = pd.read_csv('data/raw/parkinsons.csv')
print(f"Loaded: {df.shape}")

# Drop name if exists
if 'name' in df.columns:
    df = df.drop('name', axis=1)
    print("  Dropped 'name' column")

print(f"\nOriginal distribution: {df['status'].value_counts().to_dict()}")

# Split
X = df.drop('status', axis=1)
y = df['status']

# Scale first
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE
print("Applying SMOTE to fix imbalance...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
print(f"After SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")

# Split
X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Save
np.save('data/processed/parkinsons_X_train.npy', X_train)
np.save('data/processed/parkinsons_X_val.npy', X_val)
np.save('data/processed/parkinsons_X_test.npy', X_test)
np.save('data/processed/parkinsons_y_train.npy', y_train)
np.save('data/processed/parkinsons_y_val.npy', y_val)
np.save('data/processed/parkinsons_y_test.npy', y_test)
joblib.dump(scaler, 'models/scalers/parkinsons_scaler.pkl')

print(f"✅ Saved! Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# =====================================
# SUMMARY
# =====================================
print("\n" + "="*60)
print("✅ PREPROCESSING COMPLETE!")
print("="*60)
print("\n📁 Files saved in:")
print("   - data/processed/ (train/val/test splits)")
print("   - models/scalers/ (scalers for deployment)")
