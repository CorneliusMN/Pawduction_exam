from sklearn.metrics import accuracy_score




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