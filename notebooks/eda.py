"""
eda.py — Data Exploration Script
=================================
Generates all exploratory data analysis visualizations.
This script replicates what the exploration.ipynb notebook does,
and can be run as a standalone script in CI.

Output: reports/ directory with PNG plots.

Usage:
    python notebooks/eda.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = os.path.join("data", "raw", "students.csv")
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

print("[EDA] Loading dataset ...")
df = pd.read_csv(DATA_PATH)

print(f"[EDA] Shape: {df.shape}")
print(f"\n[EDA] Basic stats:\n{df.describe().round(2)}")
print(f"\n[EDA] Missing values:\n{df.isnull().sum()}")

# ── 1. Feature distributions ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold")
features = [
    "study_hours", "attendance_percentage", "previous_exam_score",
    "assignment_completion_percentage", "sleep_hours", "extracurricular_hours"
]
for ax, feat in zip(axes.flat, features):
    ax.hist(df[feat], bins=25, edgecolor="white", color="steelblue", alpha=0.8)
    ax.set_title(feat.replace("_", " ").title())
    ax.set_ylabel("Count")
plt.tight_layout()
path = os.path.join(REPORTS_DIR, "feature_distributions.png")
plt.savefig(path, dpi=100)
plt.close(fig)
print(f"[EDA] Saved -> {path}")

# ── 2. Target distribution ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["final_exam_score"], bins=30, edgecolor="white", color="darkorange", alpha=0.85)
ax.axvline(df["final_exam_score"].mean(), color="red", linestyle="--", label=f"Mean = {df['final_exam_score'].mean():.1f}")
ax.axvline(df["final_exam_score"].median(), color="blue", linestyle="--", label=f"Median = {df['final_exam_score'].median():.1f}")
ax.set_xlabel("Final Exam Score")
ax.set_ylabel("Count")
ax.set_title("Target Variable: Final Exam Score Distribution")
ax.legend()
fig.tight_layout()
path = os.path.join(REPORTS_DIR, "target_distribution.png")
plt.savefig(path, dpi=100)
plt.close(fig)
print(f"[EDA] Saved -> {path}")

# ── 3. Correlation matrix ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
corr = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
    center=0, square=True, ax=ax, linewidths=0.5,
)
ax.set_title("Feature Correlation Matrix")
fig.tight_layout()
path = os.path.join(REPORTS_DIR, "correlation_matrix.png")
plt.savefig(path, dpi=100)
plt.close(fig)
print(f"[EDA] Saved -> {path}")

# ── 4. Study hours vs Final score ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(df["study_hours"], df["final_exam_score"], alpha=0.4, s=20)
z = np.polyfit(df["study_hours"], df["final_exam_score"], 1)
xline = np.linspace(df["study_hours"].min(), df["study_hours"].max(), 100)
axes[0].plot(xline, np.poly1d(z)(xline), "r--", linewidth=2)
axes[0].set_xlabel("Study Hours")
axes[0].set_ylabel("Final Exam Score")
axes[0].set_title("Study Hours vs Final Score")

axes[1].scatter(df["previous_exam_score"], df["final_exam_score"], alpha=0.4, s=20, color="green")
z2 = np.polyfit(df["previous_exam_score"], df["final_exam_score"], 1)
xline2 = np.linspace(df["previous_exam_score"].min(), df["previous_exam_score"].max(), 100)
axes[1].plot(xline2, np.poly1d(z2)(xline2), "r--", linewidth=2)
axes[1].set_xlabel("Previous Exam Score")
axes[1].set_ylabel("Final Exam Score")
axes[1].set_title("Previous Score vs Final Score")
fig.tight_layout()
path = os.path.join(REPORTS_DIR, "feature_vs_target.png")
plt.savefig(path, dpi=100)
plt.close(fig)
print(f"[EDA] Saved -> {path}")

# ── 5. Pairplot (key features only) ───────────────────────────────────────────
key_cols = ["study_hours", "previous_exam_score", "attendance_percentage", "final_exam_score"]
fig = plt.figure(figsize=(10, 9))
pd.plotting.scatter_matrix(df[key_cols], alpha=0.3, figsize=(10, 9), diagonal="hist", color="steelblue")
plt.suptitle("Scatter Matrix — Key Features", fontsize=12)
path = os.path.join(REPORTS_DIR, "scatter_matrix.png")
plt.savefig(path, dpi=100)
plt.close("all")
print(f"[EDA] Saved -> {path}")

print("\n[EDA] PASSED: All EDA plots saved to reports/")
