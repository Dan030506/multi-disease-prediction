# download_datasets_wget.py
import os
import subprocess
import pandas as pd

print("="*60)
print("📥 DOWNLOADING DATASETS (using wget)")
print("="*60)

# Create folders
os.makedirs('data/raw', exist_ok=True)

# Function to download with wget
def download_file(url, filename):
    try:
        # Try wget first
        subprocess.run(['wget', '-O', filename, url], check=True)
        return True
    except:
        try:
            # If wget fails, try curl
            subprocess.run(['curl', '-o', filename, url], check=True)
            return True
        except:
            return False

# 1. DIABETES
print("\n1. Diabetes Dataset...")
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
filename = "data/raw/diabetes.csv"
download_file(url, filename)
print(f"   ✅ Saved: {filename}")

# 2. HEART DISEASE
print("\n2. Heart Disease Dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
filename = "data/raw/heart.csv"
download_file(url, filename)
print(f"   ✅ Saved: {filename}")

# 3. PARKINSON'S
print("\n3. Parkinson's Dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
filename = "data/raw/parkinsons.csv"
download_file(url, filename)
print(f"   ✅ Saved: {filename}")

print("\n" + "="*60)
print("✅ ALL DATASETS DOWNLOADED!")
print("📁 Location: data/raw/")
