#!/usr/bin/env python3
"""Generate data distribution figure for EduSynX dataset."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Ensure we can import from the project directory
sys.path.insert(0, str(Path(__file__).parent))

try:
    from generate_data import generate_chunk
    HAS_GENERATOR = True
except ImportError:
    HAS_GENERATOR = False
    print("Warning: generate_data.py not found. Figure will use a minimal synthetic sample.")


def load_or_generate_sample(n_rows=10000, seed=42):
    """Load data from CSV files if present, otherwise generate a sample."""
    data_dir = Path("data")
    csv_files = sorted(data_dir.glob("student_chunk_*.csv"))
    if csv_files:
        print(f"Reading data from {csv_files[0]}")
        # Read only the first chunk (or part of it) for speed
        df = pd.read_csv(csv_files[0], nrows=n_rows)
        if len(df) > n_rows:
            df = df.sample(n=n_rows, random_state=seed)
    elif HAS_GENERATOR:
        print("Generating a small synthetic sample using generate_data...")
        # Generate a small chunk (10k rows) using the same logic
        rng = np.random.default_rng(seed)
        # We need a list of regions; use the same fallback as generate_data
        regions = ["North", "South", "East", "West", "Central"]
        df = generate_chunk(n_rows, rng, regions, start_id=0)
    else:
        # Minimal fallback: create random data for illustration only
        print("No data files and generator not found. Creating random fallback data.")
        rng = np.random.default_rng(seed)
        df = pd.DataFrame({
            "study_hours": rng.normal(15, 5, n_rows),
            "attendance": rng.normal(70, 10, n_rows),
            "internal_marks": rng.normal(60, 15, n_rows),
            "performance_index": rng.normal(70, 10, n_rows),
            "final_result": rng.choice(["Pass", "Fail", "Distinction", "Dropout"], n_rows, p=[0.4, 0.25, 0.2, 0.15]),
            "socioeconomic_status": rng.choice(["Low", "Middle", "High"], n_rows, p=[0.35, 0.5, 0.15]),
            "gender": rng.choice(["Male", "Female", "Other"], n_rows, p=[0.48, 0.48, 0.04]),
        })
    return df


def main():
    # Load sample
    df = load_or_generate_sample(n_rows=10000, seed=42)

    # Select numeric columns for correlation
    numeric_cols = [
        "study_hours", "attendance", "internal_marks", "previous_gpa",
        "assignments_score", "LMS_usage_time", "login_frequency",
        "performance_index", "engagement_score", "risk_score"
    ]
    # Keep only columns that exist in df
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    # Setup figure and GridSpec
    fig = plt.figure(figsize=(12, 9), dpi=300)
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Top left: histogram of study_hours
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df["study_hours"], bins=30, color="#4C72B0", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax1.set_title("Study Hours Distribution", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Hours per week")
    ax1.set_ylabel("Frequency")
    ax1.grid(axis="y", alpha=0.3)

    # Top middle: histogram of attendance
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df["attendance"], bins=30, color="#55A868", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax2.set_title("Attendance Distribution", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Attendance (%)")
    ax2.set_ylabel("Frequency")
    ax2.grid(axis="y", alpha=0.3)

    # Top right: histogram of performance_index
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(df["performance_index"], bins=30, color="#C44E52", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax3.set_title("Performance Index Distribution", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Score (0-100)")
    ax3.set_ylabel("Frequency")
    ax3.grid(axis="y", alpha=0.3)

    # Bottom left: bar chart of final_result
    ax4 = fig.add_subplot(gs[1, 0])
    target_counts = df["final_result"].value_counts()
    target_order = ["Pass", "Fail", "Distinction", "Dropout"]
    counts = [target_counts.get(cls, 0) for cls in target_order]
    ax4.bar(target_order, counts, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"], alpha=0.8, edgecolor="black")
    ax4.set_title("Target Class Distribution", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Count")
    ax4.grid(axis="y", alpha=0.3)

    # Bottom middle: bar chart of socioeconomic_status
    ax5 = fig.add_subplot(gs[1, 1])
    ses_counts = df["socioeconomic_status"].value_counts()
    ses_order = ["Low", "Middle", "High"]
    counts_ses = [ses_counts.get(cls, 0) for cls in ses_order]
    ax5.bar(ses_order, counts_ses, color=["#CCB974", "#64B5CD", "#8172B2"], alpha=0.8, edgecolor="black")
    ax5.set_title("Socioeconomic Status Distribution", fontsize=12, fontweight="bold")
    ax5.set_ylabel("Count")
    ax5.grid(axis="y", alpha=0.3)

    # Bottom right: correlation heatmap
    ax6 = fig.add_subplot(gs[1, 2])
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        im = ax6.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax6.set_xticks(np.arange(len(corr.columns)))
        ax6.set_yticks(np.arange(len(corr.columns)))
        ax6.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
        ax6.set_yticklabels(corr.columns, fontsize=8)
        ax6.set_title("Correlation Matrix (Numeric Features)", fontsize=12, fontweight="bold")
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
        cbar.set_label("Pearson correlation", fontsize=10)
        # Annotate correlation values
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                ax6.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    else:
        ax6.text(0.5, 0.5, "Not enough numeric columns", ha="center", va="center", fontsize=12)
        ax6.set_title("Correlation Matrix", fontsize=12)
        ax6.axis("off")

    # Main title
    fig.suptitle("Statistical Properties of the EduSynX Synthetic Dataset", fontsize=14, fontweight="bold", y=0.98)

    # Save figure
    plt.savefig("fig_data_distribution.png", dpi=300, bbox_inches="tight")
    print("Data distribution figure saved as fig_data_distribution.png and fig_data_distribution.pdf")


if __name__ == "__main__":
    main()