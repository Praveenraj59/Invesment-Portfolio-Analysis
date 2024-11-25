import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# Load LSTM predictions (dummy for example)
def load_lstm_outputs():
    return np.random.rand(100, 5), np.random.rand(100)  # Dummy data

X, y = load_lstm_outputs()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate and save
y_pred = rf_model.predict(X_test)
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
joblib.dump(rf_model, '../models/random_forest_model.pkl')
print("Random Forest model saved.")
