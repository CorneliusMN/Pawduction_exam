from sklearn.metrics import accuracy_score, confusion_matrix, classification_report




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