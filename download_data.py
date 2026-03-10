import pandas as pd
import os

print("="*60)
print("DOWNLOADING DATASETS")
print("="*60)

os.makedirs('data/raw', exist_ok=True)

# 1. DIABETES
print("\n1. Downloading Diabetes dataset...")
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
df = pd.read_csv(url, header=None)
df.to_csv('data/raw/diabetes.csv', index=False, header=False)
print(f"✓ Diabetes dataset saved ({df.shape[0]} rows)")

# 2. HEART DISEASE
print("\n2. Downloading Heart Disease dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
df = pd.read_csv(url, header=None, na_values='?')
df.to_csv('data/raw/heart.csv', index=False, header=False)
print(f"✓ Heart dataset saved ({df.shape[0]} rows)")

# 3. PARKINSON'S
print("\n3. Downloading Parkinson's dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
df = pd.read_csv(url)
df.to_csv('data/raw/parkinsons.csv', index=False)
print(f"✓ Parkinson's dataset saved ({df.shape[0]} rows)")

print("\n" + "="*60)
print("✅ ALL DATASETS DOWNLOADED SUCCESSFULLY!")
print("="*60)
print("\nFiles saved in: data/raw/")
print("- diabetes.csv")
print("- heart.csv")
print("- parkinsons.csv")
