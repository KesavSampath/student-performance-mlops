"""
generate_data.py
================
Generates a realistic SYNTHETIC dataset for student performance prediction.

IMPORTANT: This dataset is entirely synthetic — no real student records are used.
The relationships between features and the target are deliberately realistic
(e.g., study hours and attendance positively correlate with final score) but
are generated via controlled random processes with a fixed seed.

Usage:
    python src/generate_data.py

Output:
    data/raw/students.csv  (1000 rows)
"""

import argparse
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_STUDENTS = 1000
OUTPUT_PATH = os.path.join("data", "raw", "students.csv")


def generate_dataset(n: int = N_STUDENTS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate a synthetic student performance dataset.

    Feature distributions (approximate):
        study_hours                   : N(5, 2)   clipped to [0, 14]
        attendance_percentage         : N(75, 15)  clipped to [30, 100]
        previous_exam_score           : N(65, 18)  clipped to [10, 100]
        assignment_completion_pct     : N(75, 20)  clipped to [0, 100]
        sleep_hours                   : N(6.5, 1.5) clipped to [3, 10]
        extracurricular_hours         : N(3, 2)    clipped to [0, 10]

    Target:
        final_exam_score = weighted linear combination + noise, clipped to [0, 100]
    """
    rng = np.random.default_rng(seed)

    study_hours = rng.normal(5.0, 2.0, n).clip(0, 14)
    attendance_pct = rng.normal(75.0, 15.0, n).clip(30, 100)
    previous_score = rng.normal(65.0, 18.0, n).clip(10, 100)
    assignment_pct = rng.normal(75.0, 20.0, n).clip(0, 100)
    sleep_hours = rng.normal(6.5, 1.5, n).clip(3, 10)
    extracurricular = rng.normal(3.0, 2.0, n).clip(0, 10)

    # --- Target construction (realistic domain knowledge) ---
    # study_hours and previous_score have the strongest positive effect.
    # Too many extracurricular hours slightly hurt performance.
    # Noise represents real-world variability (exam conditions, luck, etc.)
    noise = rng.normal(0, 5, n)

    final_score = (
        3.5 * study_hours
        + 0.25 * attendance_pct
        + 0.35 * previous_score
        + 0.15 * assignment_pct
        + 1.5 * sleep_hours
        - 0.8 * extracurricular
        + noise
        - 10  # intercept adjustment to centre the distribution around ~65
    )
    final_score = final_score.clip(0, 100)

    df = pd.DataFrame(
        {
            "study_hours": np.round(study_hours, 2),
            "attendance_percentage": np.round(attendance_pct, 2),
            "previous_exam_score": np.round(previous_score, 2),
            "assignment_completion_percentage": np.round(assignment_pct, 2),
            "sleep_hours": np.round(sleep_hours, 2),
            "extracurricular_hours": np.round(extracurricular, 2),
            "final_exam_score": np.round(final_score, 2),
        }
    )

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic student dataset")
    parser.add_argument("--n", type=int, default=N_STUDENTS, help="Number of records")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH, help="Output CSV path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    df = generate_dataset(n=args.n, seed=args.seed)
    df.to_csv(args.output, index=False)

    print(f"[generate_data] Dataset saved -> {args.output}")
    print(f"[generate_data] Shape: {df.shape}")
    print(f"[generate_data] final_exam_score stats:\n{df['final_exam_score'].describe().round(2)}")


if __name__ == "__main__":
    main()
