import pandas as pd
import os
import ssl
import urllib.request

# DISABLE SSL VERIFICATION (temporary fix)
ssl._create_default_https_context = ssl._create_unverified_context

print("="*60)
print("DOWNLOADING DATASETS (SSL bypassed)")
print("="*60)

os.makedirs('data/raw', exist_ok=True)

# Helper function to download with SSL bypass
def download_csv(url, filename, **kwargs):
    print(f"Downloading {filename}...")
    try:
        # First try with SSL bypass
        df = pd.read_csv(url, **kwargs)
        df.to_csv(f'data/raw/{filename}', index=False)
        print(f"✓ Saved: {filename}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

# 1. DIABETES
print("\n1. Downloading Diabetes dataset...")
diabetes_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
download_csv(diabetes_url, 'diabetes.csv', header=None)

# 2. HEART DISEASE
print("\n2. Downloading Heart Disease dataset...")
heart_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
download_csv(heart_url, 'heart.csv', header=None, na_values='?')

# 3. PARKINSON'S
print("\n3. Downloading Parkinson's dataset...")
parkinsons_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
download_csv(parkinsons_url, 'parkinsons.csv')

print("\n" + "="*60)
print("DOWNLOAD COMPLETE")
print("="*60)
print("\nCheck files:")
print("ls -la data/raw/")
