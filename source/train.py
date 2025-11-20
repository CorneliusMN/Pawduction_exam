import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import uniform, randint
import datetime
from typing import Iterable

from xgboost import XGBRFClassifier
from sklearn.linear_model import LogisticRegression

import mlflow
import dvc.api
import joblib

from config import PROCESSED_DATA_DIR
from wrappers import lr_wrapper 

def create_dummy_cols(df: pd.DataFrame, col: Iterable[str]) -> pd.DataFrame:
    """
    Converts a categorical column into dummy/one-hot encoded columns 
    and drops the original column.
    """
    df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    new_df = pd.concat([df, df_dummies], axis=1)
    new_df = new_df.drop(col, axis=1)
    return new_df

def onehot_encode(data: pd.DataFrame) -> pd.DataFrame:
    data = data.drop(["lead_id", "customer_code", "date_part"], axis=1)

    cat_cols = ["customer_group", "onboarding", "bin_source", "source"]
    cat_vars = data[cat_cols]

    other_vars = data.drop(cat_cols, axis=1)

    for col in cat_vars:
        cat_vars[col] = cat_vars[col].astype("category")
        cat_vars = create_dummy_cols(cat_vars, col)

    data = pd.concat([other_vars, cat_vars], axis=1)

    for col in data:
        data[col] = data[col].astype("float64")
    
    return data

# Load processed data
data = pd.read_csv(PROCESSED_DATA_DIR / "dataset.csv")
print(f"Training data length: {len(data)}")

y = data["lead_indicator"]
X = data.drop(["lead_indicator"], axis=1)

# Split train/test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.15, stratify=y
)

# Define MLflow experiment
current_date = datetime.datetime.now().strftime("%Y_%B_%d")
experiment_name = current_date
mlflow.set_experiment(experiment_name)
experiment = mlflow.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

# Track XGBoost with MLflow
with mlflow.start_run(experiment_id=experiment_id, run_name="xgboost_rf") as run:
    
    # Enable XGBoost autologging
    mlflow.xgboost.autolog()
    
    model = XGBRFClassifier(random_state=42)
    param_dist = {
        "learning_rate": uniform(1e-2, 3e-1),
        "min_split_loss": uniform(0, 10),
        "max_depth": randint(3, 10),
        "subsample": uniform(0, 1),
        "objective": ["reg:squarederror", "binary:logistic", "reg:logistic"],
        "eval_metric": ["aucpr", "error"]
    }

    # Hyperparameter search
    grid_xgb = RandomizedSearchCV(model, param_distributions=param_dist, verbose=3, n_iter=10, cv=3)
    grid_xgb.fit(X_train, y_train)
    
    best_xgb_model = grid_xgb.best_estimator_
    
    # Save the model locally
    xgb_model_path = Path("./artifacts/lead_model_xgboost.json")
    best_xgb_model.save_model(str(xgb_model_path))
    
    # Log data version from DVC
    mlflow.log_param("data_version", dvc.api.get_url(PROCESSED_DATA_DIR / "dataset.csv"))

# Track Logistic Regression with MLFlow
with mlflow.start_run(experiment_id=experiment_id, run_name="logistic_regression") as run:
    
    # Enable sklearn autologging
    mlflow.sklearn.autolog(log_input_examples=True, log_models=True)
    
    lr_model = LogisticRegression()
    param_dist_lr = {
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        "penalty": ["none", "l1", "l2", "elasticnet"],
        "C": [100, 10, 1.0, 0.1, 0.01]
    }
    
    # Hyperparameter search
    grid_lr = RandomizedSearchCV(lr_model, param_distributions=param_dist_lr, verbose=3, n_iter=10, cv=3)
    grid_lr.fit(X_train, y_train)
    
    best_lr_model = grid_lr.best_estimator_
    
    # Save the model locally
    lr_model_path = Path("./artifacts/lead_model_lr.pkl")
    joblib.dump(best_lr_model, lr_model_path)
    
    # Log data version from DVC
    mlflow.log_param("data_version", dvc.api.get_url(PROCESSED_DATA_DIR / "dataset.csv"))
    
    # Log as pyfunc model for inference
    mlflow.pyfunc.log_model("model", python_model=lr_wrapper(best_lr_model))