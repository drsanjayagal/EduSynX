#!/usr/bin/env python3
"""Generate discussion summary infographic for EduSynX."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Set up figure
fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis("off")

# Colors for each panel
colors = {
    "contributions": "#4C72B0",  # blue
    "limitations": "#C44E52",    # red
    "future": "#55A868",         # green
    "title": "#333333",
    "text": "#222222",
}

def draw_panel(x, y, w, h, title, items, color, title_color="white"):
    """Draw a rounded box with title and bullet items."""
    # Panel background
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1",
        linewidth=2,
        edgecolor="black",
        facecolor=color,
        alpha=0.85,
    )
    ax.add_patch(box)

    # Title
    ax.text(
        x + w / 2, y + h - 0.25, title,
        ha="center", va="center",
        fontsize=13, fontweight="bold", color=title_color,
    )

    # Items
    item_y = y + h - 0.8
    for item in items:
        # Bullet point
        ax.scatter(x + 0.3, item_y, s=20, color="white", zorder=5)
        ax.text(
            x + 0.5, item_y, item,
            ha="left", va="center",
            fontsize=10, color="white",
            wrap=True,
        )
        item_y -= 0.55

# Define panel positions (x, y, width, height)
panel_w = 3.0
panel_h = 5.5
gap = 0.3
start_x = 0.5
y_pos = 0.8

# Contributions panel
contrib_items = [
    "Very large synthetic dataset (10M rows)",
    "45 diverse features, class imbalance",
    "Chunk-based generation, scalable to 100M+",
    "Full ML pipeline (RF, XGB, LightGBM)",
    "SHAP-based explainability",
    "Actionable decision support insights",
]
draw_panel(start_x, y_pos, panel_w, panel_h, "Contributions", contrib_items, colors["contributions"])

# Limitations panel
limit_items = [
    "Parametric assumptions may not capture all complexity",
    "Target defined via composite score",
    "Models trained on a sample, not full data",
    "SHAP analysis on limited sample",
    "Decision support uses static mapping",
    "No real-world validation yet",
]
draw_panel(start_x + panel_w + gap, y_pos, panel_w, panel_h, "Limitations", limit_items, colors["limitations"])

# Future Directions panel
future_items = [
    "GANs and VAEs for more realistic data",
    "Deep learning models for full dataset",
    "Other XAI methods (LIME, counterfactuals)",
    "Personalized recommendation engine",
    "Validation with real institutional data",
    "Hybrid simulation + data-driven generation",
]
draw_panel(start_x + 2 * (panel_w + gap), y_pos, panel_w, panel_h, "Future Directions", future_items, colors["future"])

# Add title
ax.text(5, 6.7, "EduSynX Framework: Summary of Key Aspects",
        ha="center", va="center", fontsize=16, fontweight="bold", color=colors["title"])

# Save figure
plt.tight_layout()
plt.savefig("fig_discussion_summary.png", dpi=300, bbox_inches="tight")
plt.savefig("fig_discussion_summary.pdf", bbox_inches="tight")
print("Discussion summary figure saved as fig_discussion_summary.png and fig_discussion_summary.pdf")