#!/usr/bin/env python3
"""
Test all trained models
Run with: python3 test_models_fresh.py
"""

import joblib
import numpy as np
import os

print("="*60)
print("🧪 TESTING ALL MODELS")
print("="*60)

# =====================================
# 1. TEST DIABETES MODEL
# =====================================
print("\n📊 1. DIABETES MODEL TEST")
print("-"*40)

try:
    model = joblib.load('models/diabetes_model.pkl')
    scaler = joblib.load('models/scalers/diabetes_scaler.pkl')
    
    # Test cases
    test_cases = [
        [2, 120, 70, 20, 80, 32, 0.5, 35],  # Healthy
        [8, 180, 80, 35, 200, 40, 1.2, 50], # High risk
        [0, 90, 60, 15, 50, 25, 0.2, 25]    # Very healthy
    ]
    
    for i, sample in enumerate(test_cases):
        sample_scaled = scaler.transform([sample])
        pred = model.predict(sample_scaled)[0]
        prob = model.predict_proba(sample_scaled)[0]
        
        print(f"\nTest Case {i+1}: {sample}")
        print(f"   Prediction: {'🔴 DIABETES' if pred==1 else '🟢 NO DIABETES'}")
        print(f"   Confidence: {prob[pred]*100:.1f}%")
        
except Exception as e:
    print(f"❌ Error: {e}")

# =====================================
# 2. TEST HEART MODEL
# =====================================
print("\n📊 2. HEART DISEASE MODEL TEST")
print("-"*40)

try:
    model = joblib.load('models/heart_model.pkl')
    scaler = joblib.load('models/scalers/heart_scaler.pkl')
    
    test_cases = [
        [55, 1, 3, 130, 240, 0, 1, 150, 0, 1.0, 1, 0, 3],  # Healthy
        [60, 1, 4, 140, 280, 1, 2, 120, 1, 2.5, 2, 2, 7]   # High risk
    ]
    
    for i, sample in enumerate(test_cases):
        sample_scaled = scaler.transform([sample])
        pred = model.predict(sample_scaled)[0]
        prob = model.predict_proba(sample_scaled)[0]
        
        print(f"\nTest Case {i+1}")
        print(f"   Prediction: {'🔴 HEART DISEASE' if pred==1 else '🟢 NO HEART DISEASE'}")
        print(f"   Confidence: {prob[pred]*100:.1f}%")
        
except Exception as e:
    print(f"❌ Error: {e}")

# =====================================
# 3. TEST PARKINSON'S MODEL
# =====================================
print("\n📊 3. PARKINSON'S MODEL TEST")
print("-"*40)

try:
    model = joblib.load('models/parkinsons_model.pkl')
    scaler = joblib.load('models/scalers/parkinsons_scaler.pkl')
    
    # Simplified test with main features + zeros
    test_cases = [
        [120, 150, 100, 0.005, 0.03, 20, 0.5, 0.6] + [0]*14,  # Healthy
        [200, 250, 80, 0.008, 0.06, 15, 0.8, 0.7] + [0]*14    # Parkinson's
    ]
    
    for i, sample in enumerate(test_cases):
        sample_scaled = scaler.transform([sample])
        pred = model.predict(sample_scaled)[0]
        prob = model.predict_proba(sample_scaled)[0]
        
        print(f"\nTest Case {i+1}")
        print(f"   Prediction: {'🔴 PARKINSON\'S' if pred==1 else '🟢 HEALTHY'}")
        print(f"   Confidence: {prob[pred]*100:.1f}%")
        
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*60)
print("✅ Testing complete!")
