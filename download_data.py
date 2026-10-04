import os
import urllib.request
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
os.makedirs(DATA_DIR, exist_ok=True)

BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
FILES = [
    "winequality-red.csv",
    "winequality-white.csv",
    "winequality.names"
]

def download_data():
    for filename in FILES:
        url = BASE_URL + filename
        dest_path = os.path.join(DATA_DIR, filename)
        print(f"Downloading {filename} from {url}...")
        urllib.request.urlretrieve(url, dest_path)
        print(f"Saved to {dest_path}")

    # Verify and summarize datasets
    red_df = pd.read_csv(os.path.join(DATA_DIR, "winequality-red.csv"), sep=";")
    white_df = pd.read_csv(os.path.join(DATA_DIR, "winequality-white.csv"), sep=";")

    print("\nDataset Verification:")
    print(f"Red wine dataset shape: {red_df.shape}")
    print(f"White wine dataset shape: {white_df.shape}")
    print(f"Features: {list(red_df.columns)}")

if __name__ == "__main__":
    download_data()
