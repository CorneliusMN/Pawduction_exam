import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import uniform, randint
import datetime

from xgboost import XGBRFClassifier
from sklearn.linear_model import LogisticRegression

import mlflow
import dvc.api
import joblib

from config import TRAIN_GOLD_FILE, XGBOOST_MODEL_FILE, LR_MODEL_FILE, X_TEST_FILE, Y_TEST_FILE
from wrappers import lr_wrapper 

def create_dummy_cols(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Converts a categorical column into dummy/one-hot encoded columns 
    and drops the original column.
    """
    df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    new_df = pd.concat([df, df_dummies], axis=1)
    new_df = new_df.drop(col, axis=1)
    return new_df

def onehot_encode(data: pd.DataFrame) -> pd.DataFrame:
    """
    Drops unnecessary columns and applies one-hot encoding to specified categorical features.
    Converts all resulting columns to float64 for modeling compatibility.
    """
    data = data.drop(["lead_id", "customer_code", "date_part"], axis=1)

    cat_cols = ["customer_group", "onboarding", "bin_source", "source"]
    cat_vars = data[cat_cols].copy()

    other_vars = data.drop(cat_cols, axis=1)

    for col in cat_vars:
        cat_vars[col] = cat_vars[col].astype("category")
        cat_vars = create_dummy_cols(cat_vars, col)

    data = pd.concat([other_vars, cat_vars], axis=1)

    for col in data:
        data[col] = data[col].astype("float64")
    
    return data

def train_model(model_name: str, experiment_id: str, run_name: str, 
                autolog_type: str, param_dist: dict, 
                X_train: pd.DataFrame, y_train: pd.DataFrame, model_path: Path):
    """
    Takes model, random grid search parameters and train data 
    and saves best model while tracking parameters in MLFlow

    model_name: XGBRFClassifier / LogisticRegression
    experiment_id: experiment_id
    run_name: xgboost_rf / logistic_regression
    autolog_type: xgboost / sklearn
    param_dist: dict of parameters
    X_train: df
    y_train: df
    model_path: Path where best model is saved locally
    """

    # Map string to actual class
    model_classes = {
        "XGBRFClassifier": XGBRFClassifier,
        "LogisticRegression": LogisticRegression
    }
    model_class = model_classes[model_name]

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
    
        # Enable autologging
        getattr(mlflow, autolog_type).autolog(log_models=False)
        
        model = model_class(random_state=42)

        # Hyperparameter search
        grid = RandomizedSearchCV(model, param_distributions=param_dist, verbose=3, n_iter=10, cv=3)
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        
        # Save the model locally and log as pyfunc model for inference 
        if model_name == "LogisticRegression":
            mlflow.pyfunc.log_model("model", python_model=lr_wrapper(model))
            joblib.dump(model, model_path)
        else:
            best_model.save_model(str(model_path))

        # Log data version
        mlflow.log_param("data_version", "00000")

# Load processed data
data = pd.read_csv(TRAIN_GOLD_FILE)
print(f"Training data length: {len(data)}")

# Transform data with onehot-encoder
onehot_encoded_data = onehot_encode(data)

# Split target variable and features
y = onehot_encoded_data["lead_indicator"]
X = onehot_encoded_data.drop(["lead_indicator"], axis=1)

# Split train/test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.15, stratify=y
)

# Save test files
X_test.to_csv(X_TEST_FILE)
y_test.to_csv(Y_TEST_FILE)

# Define MLflow experiment
current_date = datetime.datetime.now().strftime("%Y_%B_%d")
experiment_name = current_date
mlflow.set_experiment(experiment_name)
experiment = mlflow.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

# Define model parameters
param_dist_xgb = {
        "learning_rate": uniform(1e-2, 3e-1),
        "min_split_loss": uniform(0, 10),
        "max_depth": randint(3, 10),
        "subsample": uniform(0, 1),
        "objective": ["reg:squarederror", "binary:logistic", "reg:logistic"],
        "eval_metric": ["aucpr", "error"]
    }

param_dist_lr = {
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        "penalty": ["none", "l1", "l2", "elasticnet"],
        "C": [100, 10, 1.0, 0.1, 0.01]
    }

# Train XGBoost model
train_model("XGBRFClassifier", experiment_id, "xgboost_rf", "xgboost", param_dist_xgb, X_train, y_train, XGBOOST_MODEL_FILE)

# Train LogisticRegression model
train_model("LogisticRegression", experiment_id, "logistic_regression", "sklearn", param_dist_lr, X_train, y_train, LR_MODEL_FILE)