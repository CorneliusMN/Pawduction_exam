from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import json
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd




# Model test accuracy section cell 61 notebook

def report_best_xgboost_and_accuracy(
    model_grid,
    X_train,
    y_train,
    X_test,
    y_test):
    """Return best XGBoost parameters and train/test accuracy."""
    best_model_xgboost_params = model_grid.best_params_
    
    y_pred_train = model_grid.predict(X_train)
    y_pred_test = model_grid.predict(X_test)
    
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    
    return best_model_xgboost_params, accuracy_train, accuracy_test


# XGBoost performance overview section cell 63 notebook

def confusion_matrix_and_classification_report(
    y_train,
    y_pred_train,
    y_test,
    y_pred_test):
    """Produce a performance overview consisting of confusion matrix and classification report for train and test."""
    # Train
    conf_mat_train = confusion_matrix(y_train, y_pred_train)
    class_rep_train = classification_report(y_train, y_pred_train)
    # Test
    conf_mat_test = confusion_matrix(y_test, y_pred_test)
    class_rep_test = classification_report(y_test, y_pred_test)
    
    return conf_mat_train, class_rep_train, conf_mat_test, class_rep_test


# Save columns and model results section cell 69 notebook

def save_columns_and_model_results(
    X_train,
    model_results,
    out_dir = "artifacts"):
    """Save column list and model results to JSON files"""
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
        json.dump(model_results, model_results_path)
    
    return column_list_path, model_results_path


# Model selection section cell 71 notebook

def model_selection(
    experiment_name,
    metric = "f1_score",
    ascending = False,
    max_results = 10):
    """Return a Dataframe of runs for an experiment, sorted by a metric (f1_score by default)"""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment {experiment_name} not found.")
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=max_results,
        order_by=[f"metrics.{metric} {"ASC" if ascending else "DESC"}"]
    )
    
    return runs


# Getting experiment model results section cell 74 notebook

def get_experiment_best_f1(experiment_name):
    """Return the top run by f1 from the given experiment"""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment {experiment_name} not found.")
    experiment_ids = [experiment.experiment_id]
    experiment_best = mlflow.search_runs(
        experiment_ids=experiment_ids,
        order_by=["metric.f1_score DESC"],
        max_results=1).iloc[0]
    
    return experiment_best


def load_results_and_print_best_model(results_path = "./artifacts/model_results.json"):
    """
    Load ./artifacts/model_results.json, build a DataFrame of weighted averages,
    and print the best model by f1-score.
    """
    with open(results_path, "r", encoding="utf-8") as file:
        model_results = json.load(file)
    
    results_df = pd.DataFrame(
        {model: val["weighted avg"] for model, val in model_results.items()}).T
    
    best_model = results_df.sort_values("f1-score", ascending=False).iloc[0].name
    
    return best_model


# Get production model section cell 81 notebook

def get_production_model(model_name):
    """Load the latest production-ready model for a given name."""
    client = MlflowClient()
    production_model = [model for model in client.search_model_versions(f"name='{model_name}'") if dict(model)["current_stage"] == "Production"]
    production_model_exists = len(production_model) > 0
    
    if production_model_exists:
        production_model_version = dict(production_model[0])["version"]
        production_model_run_id = dict(production_model[0])["run_id"]
        print("Production model name: ", model_name)
        print("Production model version:" , production_model_version)
        print("Production model run id:" , production_model_run_id)
    else:
        print("No model in production.")


