from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import json




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