import pandas as pd
import subprocess

from config import RAW_DATA_DIR

# Pulling data from DVC
subprocess.run(["dvc", "pull"], check=True)

# Loading data locally
print("Loading training data")
data = pd.read_csv(RAW_DATA_DIR / "raw_data.csv")
print("Total rows:", len(data)) #data.count()