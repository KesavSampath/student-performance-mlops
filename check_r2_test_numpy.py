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

# Manually split exactly like train_test_split(random_state=42) does
# Wait, train_test_split uses random permutations. Let's just use sklearn's permutation via np.random.RandomState(42)
rng = np.random.RandomState(42)
indices = np.arange(X.shape[0])
rng.shuffle(indices)

test_size = int(X.shape[0] * 0.2)
train_indices = indices[:-test_size]
test_indices = indices[-test_size:]

X_train, y_train = X[train_indices], y[train_indices]
X_test, y_test = X[test_indices], y[test_indices]

# Add intercept
X_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]

# Calculate OLS weights: w = (X^T X)^-1 X^T y
w = np.linalg.inv(X_bias.T.dot(X_bias)).dot(X_bias.T).dot(y_train)

# Predict
y_pred = X_test_bias.dot(w)

mae, rmse, r2 = compute_metrics(y_test, y_pred)
print(f"Test OLS MAE: {mae:.4f}")
print(f"Test OLS RMSE: {rmse:.4f}")
print(f"Test OLS R2: {r2:.4f}")
