import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def evaluate_rfmodel(rf_model, X_test, y_test):
    # Make predictions using the provided model
    y_pred = rf_model.predict(X_test)
    
    # Evaluate the model
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    
    return {
        'r2': r2,
        'mse': mse,
        'mae': mae,
        'y_pred': y_pred
    }
    
    
    
def evaluate_model(model, X_test, y_test_diff,  df):
    y_pred_diff = model.predict(X_test)
    rmse_diff =np.sqrt(mean_squared_error(y_test_diff, y_pred_diff))
    r2 = r2_score(y_test_diff, y_pred_diff)
    
    return rmse_diff, r2
