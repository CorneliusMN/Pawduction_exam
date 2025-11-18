import datetime

import pandas as pd
import subprocess
import argparse

from config import RAW_DATA_DIR

# Pulling data from DVC
subprocess.run(["dvc", "pull"], check=True)

# Loading data locally
print("Loading training data")
data = pd.read_csv(RAW_DATA_DIR / "raw_data.csv")
print("Total rows:", len(data)) #data.count()

# Parameterize min and max date
parser = argparse.ArgumentParser()
parser.add_argument("--min_date", type=str, default=None)
parser.add_argument("--max_date", type=str, default=None)
args = parser.parse_args()

min_date = (
    pd.to_datetime(args.min_date).date() 
    if args.min_date 
    else pd.to_datetime("2024-01-01").date()
)
max_date = (
    pd.to_datetime(args.max_date).date() 
    if args.max_date 
    else pd.to_datetime(datetime.datetime.now().date()).date()
)