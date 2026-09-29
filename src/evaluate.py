"""
evaluate.py
===========
Model evaluation utilities.

Computes MAE, RMSE, R² and applies a quality gate.

Quality Gate Thresholds (documented):
  R² ≥ 0.75  — model explains at least 75% of variance
  MAE ≤ 8.0  — mean absolute error within 8 points (100-point scale)

These thresholds are set conservatively. On our synthetic dataset,
Random Forest typically achieves R² ≈ 0.85–0.92 and MAE ≈ 3–5.

Usage:
    from src.evaluate import compute_metrics, quality_gate
"""

import math
from typing import Dict

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Quality gate thresholds
# ---------------------------------------------------------------------------
MIN_R2 = 0.75
MAX_MAE = 8.0


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Compute regression evaluation metrics.

    Returns
    -------
    dict with keys: mae, rmse, r2
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(round(mae, 4)), "rmse": float(round(rmse, 4)), "r2": float(round(r2, 4))}


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------
def quality_gate(metrics: Dict[str, float], raise_on_fail: bool = True) -> bool:
    """
    Check if model meets minimum quality thresholds.

    Parameters
    ----------
    metrics       : dict from compute_metrics()
    raise_on_fail : if True, raise ValueError when gate fails

    Returns
    -------
    bool — True if passed
    """
    r2_ok = metrics["r2"] >= MIN_R2
    mae_ok = metrics["mae"] <= MAX_MAE

    passed = r2_ok and mae_ok

    print("\n[quality_gate] --- Model Quality Gate ---")
    print(f"  R2  : {metrics['r2']:.4f}  (threshold >= {MIN_R2})  {'PASS' if r2_ok else 'FAIL'}")
    print(f"  MAE : {metrics['mae']:.4f}  (threshold <= {MAX_MAE})  {'PASS' if mae_ok else 'FAIL'}")
    print(f"  Result: {'PASSED' if passed else 'FAILED'}")
    print("[quality_gate] ----------------------------------\n")

    if not passed and raise_on_fail:
        raise ValueError(
            f"Model quality gate FAILED. "
            f"R2={metrics['r2']:.4f} (need >={MIN_R2}), "
            f"MAE={metrics['mae']:.4f} (need <={MAX_MAE})"
        )

    return passed
