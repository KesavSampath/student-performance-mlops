import pandas as pd
import numpy as np

def compute_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)
    return mae, rmse, r2

df = pd.read_csv("data/raw/students.csv")
features = [
    "study_hours",
    "attendance_percentage",
    "previous_exam_score",
    "assignment_completion_percentage",
    "sleep_hours",
    "extracurricular_hours"
]
X = df[features].values
y = df["final_exam_score"].values

# Add intercept
X_bias = np.c_[np.ones((X.shape[0], 1)), X]

# Calculate OLS weights: w = (X^T X)^-1 X^T y
w = np.linalg.inv(X_bias.T.dot(X_bias)).dot(X_bias.T).dot(y)

# Predict
y_pred = X_bias.dot(w)

mae, rmse, r2 = compute_metrics(y, y_pred)
print(f"OLS MAE: {mae:.4f}")
print(f"OLS RMSE: {rmse:.4f}")
print(f"OLS R2: {r2:.4f}")
