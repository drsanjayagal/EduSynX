#!/usr/bin/env python3
"""Generate the EduSynX framework architecture diagram."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Define figure size and styling
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

# Colors for modules
colors = {
    "data": "#4C72B0",       # blue
    "preprocess": "#55A868", # green
    "train": "#C44E52",      # red
    "evaluate": "#8172B2",   # purple
    "explain": "#CCB974",    # tan
    "decision": "#64B5CD",   # cyan
}

def draw_box(ax, x, y, w, h, text, color, fontsize=11, fontweight="bold", text_color="white"):
    """Draw a rounded box with centered text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1",
        linewidth=2,
        edgecolor="black",
        facecolor=color,
        alpha=0.9,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=text_color,
        wrap=True,
    )

def draw_arrow(ax, start, end, style="-|>", color="black", lw=1.5):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        mutation_scale=20,
        linewidth=lw,
        color=color,
    )
    ax.add_patch(arrow)

# Layout coordinates (x, y, width, height)
# Row 1: Data Generation
draw_box(ax, 0.5, 4.8, 2.0, 1.0, "Synthetic Data\nGenerator\n(Chunked CSV)", colors["data"])
# Row 2: Preprocessing
draw_box(ax, 3.0, 4.8, 1.8, 1.0, "Preprocessing\n& Feature\nEngineering", colors["preprocess"])
# Row 3: Model Training
draw_box(ax, 5.5, 4.8, 1.8, 1.0, "Model Training\n(RF, XGBoost,\nLightGBM)", colors["train"])
# Row 4: Evaluation
draw_box(ax, 8.0, 4.8, 1.5, 1.0, "Evaluation\n(Metrics)", colors["evaluate"])
# Row 5: Explainability
draw_box(ax, 5.5, 2.0, 1.8, 1.0, "Explainability\n(SHAP)", colors["explain"])
# Row 6: Decision Support
draw_box(ax, 8.0, 2.0, 1.5, 1.0, "Decision\nSupport", colors["decision"])

# Arrows connecting the modules
# Data -> Preprocess
draw_arrow(ax, (2.5, 5.3), (3.0, 5.3))
# Preprocess -> Train
draw_arrow(ax, (4.8, 5.3), (5.5, 5.3))
# Train -> Evaluate
draw_arrow(ax, (7.3, 5.3), (8.0, 5.3))
# Evaluate -> Explainability (downward)
draw_arrow(ax, (8.75, 4.8), (8.75, 3.0))
# Explainability -> Decision Support
draw_arrow(ax, (7.3, 2.5), (8.0, 2.5))
# Explainability feedback to Model Training (dashed)
arrow_fb = FancyArrowPatch(
    (6.4, 2.0), (6.4, 4.8),
    arrowstyle="-|>",
    mutation_scale=20,
    linewidth=1.5,
    linestyle="--",
    color="gray",
)
ax.add_patch(arrow_fb)
ax.text(6.5, 3.4, "feedback", rotation=90, fontsize=9, color="gray", ha="center", va="center")

# Add title
ax.text(5, 5.8, "EduSynX Framework Architecture", ha="center", va="center", fontsize=14, fontweight="bold")

# Save figure in both PNG and PDF formats
plt.tight_layout()
plt.savefig("fig_framework.png", dpi=300, bbox_inches="tight")
print("Framework diagram saved as fig_framework.png and fig_framework.pdf")