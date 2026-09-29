import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"Test Set R2: {r2:.4f}")
