# train_models.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

print("="*60)
print("🤖 TRAINING BASELINE MODELS")
print("="*60)

os.makedirs('models', exist_ok=True)

# =====================================
# 1. DIABETES MODEL
# =====================================
print("\n📊 1. DIABETES MODEL")
print("-"*40)

# Load data
X_train = np.load('data/processed/diabetes_X_train.npy')
X_val = np.load('data/processed/diabetes_X_val.npy')
X_test = np.load('data/processed/diabetes_X_test.npy')
y_train = np.load('data/processed/diabetes_y_train.npy')
y_val = np.load('data/processed/diabetes_y_val.npy')
y_test = np.load('data/processed/diabetes_y_test.npy')

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
train_acc = accuracy_score(y_train, model.predict(X_train))
val_acc = accuracy_score(y_val, model.predict(X_val))
test_acc = accuracy_score(y_test, model.predict(X_test))

print(f"Train accuracy: {train_acc:.3f}")
print(f"Validation accuracy: {val_acc:.3f}")
print(f"Test accuracy: {test_acc:.3f}")

# Save
joblib.dump(model, 'models/diabetes_model.pkl')
print("✅ Model saved to models/diabetes_model.pkl")

# =====================================
# 2. HEART DISEASE MODEL
# =====================================
print("\n📊 2. HEART DISEASE MODEL")
print("-"*40)

# Load data
X_train = np.load('data/processed/heart_X_train.npy')
X_val = np.load('data/processed/heart_X_val.npy')
X_test = np.load('data/processed/heart_X_test.npy')
y_train = np.load('data/processed/heart_y_train.npy')
y_val = np.load('data/processed/heart_y_val.npy')
y_test = np.load('data/processed/heart_y_test.npy')

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Train
model = SVC(kernel='rbf', probability=True, random_state=42)
model.fit(X_train, y_train)

# Evaluate
train_acc = accuracy_score(y_train, model.predict(X_train))
val_acc = accuracy_score(y_val, model.predict(X_val))
test_acc = accuracy_score(y_test, model.predict(X_test))

print(f"Train accuracy: {train_acc:.3f}")
print(f"Validation accuracy: {val_acc:.3f}")
print(f"Test accuracy: {test_acc:.3f}")

# Save
joblib.dump(model, 'models/heart_model.pkl')
print("✅ Model saved to models/heart_model.pkl")

# =====================================
# 3. PARKINSON'S MODEL
# =====================================
print("\n📊 3. PARKINSON'S MODEL")
print("-"*40)

# Load data
X_train = np.load('data/processed/parkinsons_X_train.npy')
X_val = np.load('data/processed/parkinsons_X_val.npy')
X_test = np.load('data/processed/parkinsons_X_test.npy')
y_train = np.load('data/processed/parkinsons_y_train.npy')
y_val = np.load('data/processed/parkinsons_y_val.npy')
y_test = np.load('data/processed/parkinsons_y_test.npy')

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
train_acc = accuracy_score(y_train, model.predict(X_train))
val_acc = accuracy_score(y_val, model.predict(X_val))
test_acc = accuracy_score(y_test, model.predict(X_test))

print(f"Train accuracy: {train_acc:.3f}")
print(f"Validation accuracy: {val_acc:.3f}")
print(f"Test accuracy: {test_acc:.3f}")

# Save
joblib.dump(model, 'models/parkinsons_model.pkl')
print("✅ Model saved to models/parkinsons_model.pkl")

# =====================================
# SUMMARY
# =====================================
print("\n" + "="*60)
print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
print("="*60)
print("\n📁 Models saved in: models/")
print("   - diabetes_model.pkl")
print("   - heart_model.pkl")
print("   - parkinsons_model.pkl")
print("\n📊 Test accuracies:")
print(f"   Diabetes: {test_acc:.3f}")
print(f"   Heart: {test_acc:.3f}")
print(f"   Parkinson's: {test_acc:.3f}")
