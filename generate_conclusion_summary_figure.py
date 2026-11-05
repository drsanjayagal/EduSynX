#!/usr/bin/env python3
"""Generate conclusion summary figure (visual abstract) for EduSynX."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Set up figure
fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
ax.set_xlim(0, 8)
ax.set_ylim(0, 8)
ax.axis("off")

# Define colors
colors = {
    "main": "#4C72B0",          # blue for central box
    "data": "#55A868",          # green
    "ml": "#C44E52",            # red
    "xai": "#8172B2",           # purple
    "decision": "#CCB974",      # tan
    "contribution": "#64B5CD",  # cyan
    "text": "#222222",
}

def draw_rounded_box(ax, x, y, w, h, text, facecolor, text_color="white", fontsize=10, fontweight="bold"):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15",
        linewidth=2,
        edgecolor="black",
        facecolor=facecolor,
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

def draw_arrow(ax, start, end, style="-|>", color="black", lw=1.5, linestyle="-"):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        mutation_scale=20,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
    )
    ax.add_patch(arrow)

# Central box: EduSynX Framework
central_x, central_y, central_w, central_h = 2.5, 3.0, 3.0, 2.0
draw_rounded_box(ax, central_x, central_y, central_w, central_h,
                 "EduSynX Framework", colors["main"], fontsize=12, fontweight="bold")

# Surrounding modules
# Data Generation (top left)
data_x, data_y, data_w, data_h = 0.5, 5.8, 2.0, 1.2
draw_rounded_box(ax, data_x, data_y, data_w, data_h,
                 "Synthetic Data\nGeneration\n(10M+ rows)", colors["data"], fontsize=9)

# ML Pipeline (top right)
ml_x, ml_y, ml_w, ml_h = 5.5, 5.8, 2.0, 1.2
draw_rounded_box(ax, ml_x, ml_y, ml_w, ml_h,
                 "Machine Learning\nPipeline\n(RF, XGB, LightGBM)", colors["ml"], fontsize=9)

# Explainability (bottom left)
xai_x, xai_y, xai_w, xai_h = 0.5, 1.5, 2.0, 1.2
draw_rounded_box(ax, xai_x, xai_y, xai_w, xai_h,
                 "Explainability\n(SHAP)", colors["xai"], fontsize=9)

# Decision Support (bottom right)
dec_x, dec_y, dec_w, dec_h = 5.5, 1.5, 2.0, 1.2
draw_rounded_box(ax, dec_x, dec_y, dec_w, dec_h,
                 "Decision Support\n(Actionable Insights)", colors["decision"], fontsize=9)

# Arrows connecting modules to central box
# From data generation to central
draw_arrow(ax, (data_x + data_w, data_y + data_h/2), (central_x, central_y + central_h*0.75))
# From ML pipeline to central
draw_arrow(ax, (ml_x, ml_y + ml_h/2), (central_x + central_w, central_y + central_h*0.75))
# From central to explainability
draw_arrow(ax, (central_x, central_y + central_h*0.25), (xai_x + xai_w, xai_y + xai_h/2))
# From central to decision support
draw_arrow(ax, (central_x + central_w, central_y + central_h*0.25), (dec_x, dec_y + dec_h/2))

# Contributions summary box at the very bottom or side
contrib_x, contrib_y, contrib_w, contrib_h = 1.0, 0.2, 6.0, 1.0
draw_rounded_box(ax, contrib_x, contrib_y, contrib_w, contrib_h,
                 "Key Contributions:\n"
                 "• Very large, realistic synthetic dataset  •  Full ML pipeline with explainability  •  Actionable decision support",
                 colors["contribution"], fontsize=9, fontweight="normal")

# Title at top
ax.text(4, 7.2, "EduSynX: A Machine Learning Framework for Academic Success Prediction",
        ha="center", va="center", fontsize=14, fontweight="bold", color=colors["text"])

# Save figure
plt.tight_layout()
plt.savefig("fig_conclusion_summary.png", dpi=300, bbox_inches="tight")
plt.savefig("fig_conclusion_summary.pdf", bbox_inches="tight")
print("Conclusion summary figure saved as fig_conclusion_summary.png and fig_conclusion_summary.pdf")