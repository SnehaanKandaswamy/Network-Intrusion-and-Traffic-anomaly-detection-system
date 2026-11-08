"""Shared Matplotlib style for all project figures (static PNGs for the report).

Colours: a fixed categorical order (never cycled), recessive grid and axes, text
in ink colours rather than series colours, and a reserved 'critical' red that is
used only for alarms and thresholds.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CRITICAL = "#d03b3b"      # alarms / threshold (status colour, not a series colour)
WARNING_FILL = "#fab219"  # ground-truth anomaly bands
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 9.5, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5, "legend.frameon": False, "legend.fontsize": 8.5, "text.color": INK,
        "lines.linewidth": 1.6, "lines.markersize": 5, "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "font.size": 9.5,
    })


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")
