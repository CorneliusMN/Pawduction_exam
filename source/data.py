import datetime
import json

import pandas as pd
import subprocess
import argparse
import mlflow

from config import DATE_LIMITS_FILE, RAW_DATA_FILE

# Pulling data from DVC
subprocess.run(["dvc", "pull"], check=True)

# Loading data locally
print("Loading training data")
data = pd.read_csv(RAW_DATA_FILE)
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

# Start MLFlow and log parameters
with mlflow.start_run():
    mlflow.log_param("min_date", str(min_date))
    mlflow.log_param("max_date", str(max_date))

    # Filter data to specified date range
    def filter_by_date(df: pd.DataFrame, min_date: datetime.date, max_date: datetime.date) -> pd.DataFrame:
        """
        Takes data frame and min/ max dates and returns a data frame filtered for the specified date range
        """
        df = df.copy()
        df["date_part"] = pd.to_datetime(df["date_part"]).dt.date
        return df[(df["date_part"] >= min_date) & (df["date_part"] <= max_date)]
    filtered_data = filter_by_date(data, min_date, max_date)

    # Compute the actual min/max in the filtered dataset
    actual_min = filtered_data["date_part"].min()
    actual_max = filtered_data["date_part"].max()

    date_limits = {
        "requested_min_date": str(min_date),
        "requested_max_date": str(max_date),
        "actual_min_date": str(actual_min),
        "actual_max_date": str(actual_max),
    }

    # Save date artifact
    with open(DATE_LIMITS_FILE, "w") as f:
            json.dump(date_limits, f)

    # Log date artifact in MLFlow
    mlflow.log_artifact(DATE_LIMITS_FILE)