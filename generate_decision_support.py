#!/usr/bin/env python3
"""Generate decision support illustration for three sample students."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Create figure directory if it does not exist
output_dir = Path("figure")
output_dir.mkdir(exist_ok=True)

# Student data
students = [
    {
        "name": "Student 1",
        "prob_distinction": 0.78,
        "features": ["study_hours", "assignments_score", "stress_level", "sleep_hours", "LMS_usage_time"],
        "shap_values": [0.12, 0.08, -0.05, -0.03, 0.02],
        "recommendations": [
            "Increase study hours from 6.3 to 12 hours/week",
            "Reduce stress level from 7 to 5",
            "Improve sleep quality to 7-8 hours/night"
        ]
    },
    {
        "name": "Student 2",
        "prob_distinction": 0.42,
        "features": ["study_hours", "attendance", "submission_delay", "stress_level", "assignment_completion_rate"],
        "shap_values": [-0.15, -0.10, -0.08, -0.06, 0.04],
        "recommendations": [
            "Increase study hours from 4.2 to 10 hours/week",
            "Reduce submission delay from 5.3 to 1 day",
            "Improve attendance from 58% to 80%",
            "Reduce stress level from 8 to 6"
        ]
    },
    {
        "name": "Student 3",
        "prob_distinction": 0.95,
        "features": ["performance_index", "engagement_score", "consistency_score", "previous_gpa", "internal_marks"],
        "shap_values": [0.18, 0.15, 0.12, 0.10, 0.08],
        "recommendations": [
            "Maintain current study habits",
            "Continue high engagement with LMS",
            "No intervention needed"
        ]
    }
]

# Set up figure
fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=300)

for ax, student in zip(axes, students):
    # Predicted probability text
    prob_text = f"Predicted Distinction Probability: {student['prob_distinction']:.2f}"
    ax.text(0.5, 1.02, prob_text, transform=ax.transAxes, ha='center', va='bottom',
            fontsize=10, fontweight='bold', color='#333333')

    # SHAP bar chart (horizontal)
    y_pos = np.arange(len(student['features']))[::-1]  # reverse so first feature at top
    shap_vals = student['shap_values']
    colors = ['#4C72B0' if v >= 0 else '#C44E52' for v in shap_vals]
    ax.barh(y_pos, shap_vals, color=colors, edgecolor='black', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(student['features'], fontsize=9)
    ax.set_xlabel('SHAP value', fontsize=9)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(student['name'], fontsize=12, fontweight='bold')

    # Recommendations as text below the bar chart
    rec_text = "Recommendations:\n" + "\n".join(f"• {r}" for r in student['recommendations'])
    ax.text(0.5, -0.55, rec_text, transform=ax.transAxes, ha='center', va='top',
            fontsize=9, color='#555555', wrap=True)

    # Adjust layout
    ax.set_xlim(-0.2, 0.25)
    ax.grid(axis='x', linestyle='--', alpha=0.3)

# Main title
fig.suptitle('Adaptive Decision Support for Sample Students', fontsize=14, fontweight='bold', y=1.05)

plt.tight_layout()
output_path = output_dir / 'decision_support.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Decision support figure saved to {output_path}")