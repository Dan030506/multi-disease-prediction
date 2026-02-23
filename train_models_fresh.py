#!/usr/bin/env python3
"""
Train models from preprocessed data
Run with: python3 train_models_fresh.py
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

print("="*60)
print("🤖 TRAINING MODELS FRESH")
print("="*60)

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('models/scalers', exist_ok=True)

# =====================================
# 1. DIABETES MODEL
# =====================================
print("\n📊 1. TRAINING DIABETES MODEL")
print("-"*40)

try:
    # Load data
    X_train = np.load('data/processed/diabetes_X_train.npy')
    X_val = np.load('data/processed/diabetes_X_val.npy')
    X_test = np.load('data/processed/diabetes_X_test.npy')
    y_train = np.load('data/processed/diabetes_y_train.npy')
    y_val = np.load('data/processed/diabetes_y_val.npy')
    y_test = np.load('data/processed/diabetes_y_test.npy')
    
    print(f"Training data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")
    print(f"Test data: {X_test.shape}")
    
    # Train model
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    print(f"\n✅ Results:")
    print(f"   Train accuracy: {train_acc:.3f} ({train_acc*100:.1f}%)")
    print(f"   Validation accuracy: {val_acc:.3f} ({val_acc*100:.1f}%)")
    print(f"   Test accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")
    
    # Save model
    joblib.dump(model, 'models/diabetes_model.pkl')
    print("✅ Model saved to models/diabetes_model.pkl")
    
except Exception as e:
    print(f"❌ Error: {e}")

# =====================================
# 2. HEART DISEASE MODEL
# =====================================
print("\n📊 2. TRAINING HEART DISEASE MODEL")
print("-"*40)

try:
    # Load data
    X_train = np.load('data/processed/heart_X_train.npy')
    X_val = np.load('data/processed/heart_X_val.npy')
    X_test = np.load('data/processed/heart_X_test.npy')
    y_train = np.load('data/processed/heart_y_train.npy')
    y_val = np.load('data/processed/heart_y_val.npy')
    y_test = np.load('data/processed/heart_y_test.npy')
    
    print(f"Training data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")
    print(f"Test data: {X_test.shape}")
    
    # Train model
    print("\nTraining SVM...")
    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    print(f"\n✅ Results:")
    print(f"   Train accuracy: {train_acc:.3f} ({train_acc*100:.1f}%)")
    print(f"   Validation accuracy: {val_acc:.3f} ({val_acc*100:.1f}%)")
    print(f"   Test accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")
    
    # Save model
    joblib.dump(model, 'models/heart_model.pkl')
    print("✅ Model saved to models/heart_model.pkl")
    
except Exception as e:
    print(f"❌ Error: {e}")

# =====================================
# 3. PARKINSON'S MODEL
# =====================================
print("\n📊 3. TRAINING PARKINSON'S MODEL")
print("-"*40)

try:
    # Load data
    X_train = np.load('data/processed/parkinsons_X_train.npy')
    X_val = np.load('data/processed/parkinsons_X_val.npy')
    X_test = np.load('data/processed/parkinsons_X_test.npy')
    y_train = np.load('data/processed/parkinsons_y_train.npy')
    y_val = np.load('data/processed/parkinsons_y_val.npy')
    y_test = np.load('data/processed/parkinsons_y_test.npy')
    
    print(f"Training data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")
    print(f"Test data: {X_test.shape}")
    
    # Train model
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    print(f"\n✅ Results:")
    print(f"   Train accuracy: {train_acc:.3f} ({train_acc*100:.1f}%)")
    print(f"   Validation accuracy: {val_acc:.3f} ({val_acc*100:.1f}%)")
    print(f"   Test accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")
    
    # Save model
    joblib.dump(model, 'models/parkinsons_model.pkl')
    print("✅ Model saved to models/parkinsons_model.pkl")
    
except Exception as e:
    print(f"❌ Error: {e}")

# =====================================
# SUMMARY
# =====================================
print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print("\n📁 Models saved:")
if os.path.exists('models/diabetes_model.pkl'):
    print("   ✅ diabetes_model.pkl")
if os.path.exists('models/heart_model.pkl'): 
    print("   ✅ heart_model.pkl")
if os.path.exists('models/parkinsons_model.pkl'):
    print("   ✅ parkinsons_model.pkl")
