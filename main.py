import data_preprocessing as dp
import stationarity_tests as st
import feature_engineering as fe
import model_training as mt
import evaluation as ev
from model_training import select_features_and_train
from evaluation import evaluate_rfmodel

# Load and preprocess data
df = dp.load_data('Train_Data_CSV.csv')
df = dp.encode_labels(df, "Dust")
df = dp.handle_outliers(df)

# Feature engineering
df = fe.feature_engineering(df)


# Feature selection
selected_features = ['Differential_pressure', 'Dust_feed', 'Flow_rate_lag1', 'Differential_pressure_lag1', 'Dust', 'Flow_mean', 'Time']
refined_data = df[selected_features + ['RUL']]

######
results = select_features_and_train(df)

# Access the trained model and evaluation metrics
rf_model = results['rf_model']
X_test = results['X_test']
y_test = results['y_test']

# Further evaluation
evaluation = evaluate_rfmodel(rf_model, X_test, y_test)

# # Print feature importance
# print(results['feature_importance'])

# Print Regression metrics
print(f"R^2 Score of rf model : {evaluation['r2']}" )
print(f"Mean Squared Error of rf Model: {evaluation['mse']}")
print(f"Mean Absolute Error of rf Model: {evaluation['mae']}")



# Train model
X = refined_data.drop(columns=['RUL'])
y_diff = refined_data['RUL']
model, X_train, X_test, y_train, y_test_diff = mt.train_model(X, y_diff)


# Evaluate model


last_observed_value = df["RUL"].iloc[-1]
rmse, r2 = ev.evaluate_model(model, X_test, y_test_diff, df)
print(f"R-squared (R²) of XGBoost: {r2}")
