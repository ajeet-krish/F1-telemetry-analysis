"""
Mercedes-themed matplotlib style configuration.

Dark background (#1a1a1a), teal accent (#00D2BE), white text,
and consistent formatting for all F1 aerodynamics plots.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cycler


MERCEDES_TEAL = "#00D2BE"
MERCEDES_DARK = "#1A1A1A"
MERCEDES_CARD = "#2A2A2A"
MERCEDES_GRAY = "#D4D4D4"
MERCEDES_WHITE = "#FFFFFF"
MERCEDES_RED = "#E94560"
MERCEDES_AMBER = "#FFB347"

COLORS = [
    MERCEDES_TEAL,
    MERCEDES_RED,
    MERCEDES_AMBER,
    "#4ECDC4",
    "#A855F7",
    "#22C55E",
    "#F97316",
    "#3B82F6",
]


def set_f1_style():
    """Apply the Mercedes-themed F1 dark style to matplotlib."""
    plt.style.use("dark_background")
    mpl.rcParams.update(
        {
            "figure.facecolor": MERCEDES_DARK,
            "axes.facecolor": MERCEDES_DARK,
            "axes.edgecolor": MERCEDES_GRAY,
            "axes.labelcolor": MERCEDES_WHITE,
            "axes.titlecolor": MERCEDES_WHITE,
            "axes.grid": True,
            "axes.grid.which": "both",
            "axes.prop_cycle": cycler("color", COLORS),
            "grid.color": MERCEDES_CARD,
            "grid.alpha": 0.5,
            "text.color": MERCEDES_WHITE,
            "xtick.color": MERCEDES_GRAY,
            "ytick.color": MERCEDES_GRAY,
            "legend.facecolor": MERCEDES_CARD,
            "legend.edgecolor": MERCEDES_GRAY,
            "legend.labelcolor": MERCEDES_WHITE,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.facecolor": MERCEDES_DARK,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "lines.linewidth": 2,
            "lines.markersize": 6,
        }
    )


def teal_colormap():
    """Monochrome teal colormap for contour/heatmap plots."""
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["#1A1A1A", "#004D47", "#00857A", "#00B3A0", "#00D2BE", "#66E3D5"]
    return LinearSegmentedColormap.from_list("mercedes_teal", colors, N=256)
