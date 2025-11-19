import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

from config import PROCESSED_DATA_DIR

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