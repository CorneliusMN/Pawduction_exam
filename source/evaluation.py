import os
import json
import time

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np

from config import X_TEST_FILE, Y_TEST_FILE




# Load test data

def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load X_test and y_test from CSV files defined in conifg."""
    X_test = pd.read_csv(X_TEST_FILE)
    
    y_test_df = pd.read_csv(Y_TEST_FILE)
    y_test = y_test_df.iloc[:, 0]
    
    return X_test, y_test


# Model test accuracy

def report_best_xgboost_and_accuracy(
    model_grid: RandomizedSearchCV,
    X_test: pd.DataFrame,
    y_test: pd.Series) -> tuple[dict, float]:
    """Return best XGBoost parameters and test accuracy."""
    best_model_xgboost_params = model_grid.best_params_
    y_pred_test = model_grid.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred_test)
    
    return best_model_xgboost_params, acc_test


# XGBoost performance overview

def confusion_matrix_and_classification_report(
    y_test: pd.Series,
    y_pred_test: pd.Series) -> tuple[np.ndarray, str]:
    """
    Produce a performance overview consisting of confusion matrix
    and classification report for test.
    """
    # # Train metrics
    # conf_mat_train = confusion_matrix(y_train, y_pred_train)
    # class_rep_train = classification_report(y_train, y_pred_train)
    
    # Test metrics
    conf_mat_test = confusion_matrix(y_test, y_pred_test)
    class_rep_test = classification_report(y_test, y_pred_test)
    
    return conf_mat_test, class_rep_test


# Save columns and model results

def save_columns_and_model_results(
    X_train: pd.DataFrame,
    model_results: dict,
    out_dir: str = "artifacts") -> tuple[str, str]:
    """Save column list and model results to JSON files."""
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    
    column_list_path = os.path.join(out_dir, "columns_list.json")
    # Extract columns names from X_train
    with open(column_list_path, "w", encoding="utf-8") as columns_file:
        columns = {"column_names": list(X_train.columns)}
        json.dump(columns, columns_file)
    
    # Output path for the model results JSON
    model_results_path = os.path.join(out_dir, "model_results.json")
    with open(model_results_path, "w", encoding="utf-8") as results_file:
        json.dump(model_results, results_file)
    
    return column_list_path, model_results_path


# Model selection

def model_selection(
    experiment_name: str,
    metric: str = "f1_score",
    ascending: bool = False,
    max_results: int = 10) -> pd.DataFrame:
    """
    Return a Dataframe of runs for an experiment,
    sorted by a metric (f1_score by default).
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment {experiment_name} not found.")
    
    order = "ASC" if ascending else "DESC"
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=max_results,
        order_by=[f"metrics.{metric} {order}"]
    )
    
    return runs


# Getting experiment model results

def get_experiment_best_f1(experiment_name: str) -> pd.Series:
    """Return the top run by f1 from the given experiment."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment {experiment_name} not found.")
    
    experiment_ids = [experiment.experiment_id]
    experiment_best = mlflow.search_runs(
        experiment_ids=experiment_ids,
        order_by=["metric.f1_score DESC"],
        max_results=1).iloc[0]
    
    return experiment_best


def load_results_and_print_best_model(results_path: str = "./artifacts/model_results.json") -> str:
    """
    Load ./artifacts/model_results.json, build a DataFrame
    of weighted averages, and print the best model by f1-score.
    """
    with open(results_path, "r", encoding="utf-8") as file:
        model_results = json.load(file)
    
    results_df = pd.DataFrame(
        {model: val["weighted avg"] for model, val in model_results.items()}).T
    
    best_model = results_df.sort_values("f1-score", ascending=False).iloc[0].name
    
    return best_model


# Get production model

def get_production_model(model_name: str) -> tuple[bool, str | None]:
    """Load the latest production-ready model for a given name."""
    client = MlflowClient()
    prod_model = [model for model in
                  client.search_model_versions(f"name='{model_name}'")
                  if dict(model)["current_stage"] == "Production"]
    
    prod_model_exists = len(prod_model) > 0
    if prod_model_exists:
        prod_model_version = dict(prod_model[0])["version"]
        prod_model_run_id = dict(prod_model[0])["run_id"]
        print(f"Production model name: {model_name}.")
        print(f"Production model version: {prod_model_version}.")
        print(f"Production model run id: {prod_model_run_id}.")
        return prod_model_exists, prod_model_run_id
    else:
        print("No model in production.")


# Compare prod and best trained model
# Here model_status is not used

def compare_prod_and_best_trained(
    experiment_best: pd.Series,
    prod_model_exists: bool,
    prod_model_run_id: str | None) -> str | None:
    """
    Compare Production vs. best trained model
    (by f1_score) and decide run_id to register.
    """
    train_model_score = experiment_best["metrics.f1_score"]
    model_status = {}
    run_id: str | None = None
    
    if prod_model_exists and prod_model_run_id is not None:
        run = mlflow.get_run(prod_model_run_id)
        prod_model_score = run.data.metrics.get("f1_score")
        model_status["current"] = train_model_score
        model_status["prod"] = prod_model_score
        
        if train_model_score > prod_model_score:
            run_id = experiment_best["run_id"]
    else:
        run_id = experiment_best["run_id"]
    
    print(f"Registered model: {run_id}.")
    return run_id


# Register best model

def register_best_model(
    run_id: str | None,
    artifact_path: str,
    model_name: str) -> dict | None:
    """Register a model from a run into the MLFlow Model Registry."""
    if run_id is not None:
        print(f"Best model found: {run_id}.")
        model_uri = mlflow.get_artifact_uri(artifact_path=artifact_path, run_id = run_id)
        model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
        model_details = dict(model_details)
        return model_details
    else:
        print("No run id is provided.")
        return None


# Deploy

def wait_for_deployment(
    model_name: str,
    model_version: int,
    stage: str = "Staging") -> bool:
    """Wait until the given model version reaches the target stage."""
    client = MlflowClient()
    status = False
    
    while not status:
        model_version_details = dict(
            client.get_model_version(
                name=model_name,
                version=model_version
                )
            )
        if model_version_details["current_stage"] == stage:
            print(f"Transition completed to {stage}.")
            status = True
            break
        else:
            time.sleep(2)
    
    return status


def run_stage_transition(
    model_name: str,
    model_version: str) -> bool:
    client = MlflowClient()
    model_version_details = dict(
        client.get_model_version(
            name=model_name,
            version=model_version
            )
        )
    
    model_status = True
    if model_version_details["current_stage"] != "Staging":
        client.transition_model_version_stage(
            name=model_name,
            version=model_version, 
            stage="Staging",
            archive_existing_versions=True)
        model_status = wait_for_deployment(model_name, model_version, "Staging")
    else:
        print("Model already in staging.")
    
    return model_status