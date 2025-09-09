import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.base import RegressorMixin, clone
from sklearn.model_selection import train_test_split

DEFAULT_TRAIN_SIZE = 0.8
DEFAULT_MODEL_DESCRIPTION = 'default'

def _adjusted_r2(r2, n_observations, k_features):
    """Calculate adjusted R-squared"""
    if n_observations - k_features == 1:
        return 0
        
    return 1 - ((1 - r2) * (n_observations - 1) / (n_observations - k_features - 1))

def run_regressor(model: RegressorMixin,
                  X: np.ndarray | pd.DataFrame, 
                  y: np.ndarray | pd.Series,
                  split_random_state: int | None = None,
                  model_description: str = DEFAULT_MODEL_DESCRIPTION) -> str:
    """
    Train and evaluate a regression model, returning CSV-formatted metrics for Excel.
    
    Returns: model_description,rmse_train,mae_train,mape_train,r2_train,adj_r2_train,rmse_test,mae_test,mape_test,r2_test,adj_r2_test
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        train_size=DEFAULT_TRAIN_SIZE, 
        random_state=split_random_state
    )
    
    # Clone and train model
    model = clone(model)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate training metrics
    rmse_train = metrics.root_mean_squared_error(y_train, y_train_pred)
    mae_train = metrics.mean_absolute_error(y_train, y_train_pred)
    mape_train = metrics.mean_absolute_percentage_error(y_train, y_train_pred)
    r2_train = metrics.r2_score(y_train, y_train_pred)
    adj_r2_train = _adjusted_r2(r2_train, len(X_train), X.shape[1])
    
    # Calculate test metrics
    rmse_test = metrics.root_mean_squared_error(y_test, y_test_pred)
    mae_test = metrics.mean_absolute_error(y_test, y_test_pred)
    mape_test = metrics.mean_absolute_percentage_error(y_test, y_test_pred)
    r2_test = metrics.r2_score(y_test, y_test_pred)
    adj_r2_test = _adjusted_r2(r2_test, len(X_test), X.shape[1])
    
    # Combine all metrics
    all_metrics = [
        rmse_train, mae_train, mape_train, r2_train, adj_r2_train,
        rmse_test, mae_test, mape_test, r2_test, adj_r2_test
    ]
    
    # Format metrics to 3 decimal places
    formatted_metrics = [f"{x:.3f}" for x in all_metrics]
    
    return f"{model_description}," + ','.join(formatted_metrics)