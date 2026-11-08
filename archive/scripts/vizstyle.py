"""Shared chart styling: one validated palette, one place.

Categorical slots and the sequential blue ramp are the reference palette from
the project's data-viz guidance, validated with its checker for 5 categorical
slots on the light surface: lightness band PASS, chroma floor PASS, worst
adjacent CVD dE 9.1 PASS, worst adjacent normal-vision dE 19.6 PASS, contrast
WARN on 3 slots -> **relief is required**, which is why every chart here ships
direct value labels and REPORT.md carries the numeric tables.

Output is light-mode only: these are static PNGs embedded in a Markdown report,
so there is no theme to follow.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# -- surfaces and ink -------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
INK_INVERSE = "#ffffff"
GRID = "#e6e5e1"

# -- categorical slots, in fixed order (never cycled) -----------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MODEL_COLOR = {
    "bdh": SERIES[0],            # baseline BDH
    "gpt": SERIES[1],            # matched Transformer
    "bdh_den_gated": SERIES[2],  # dendritic variants
    "bdh_den_sum": SERIES[3],
    "bdh_den_max": SERIES[4],
    "bdh_narrow": INK_MUTED,     # a control, deliberately recessive
}
STRATEGY_COLOR = {"naive": SERIES[0], "replay": SERIES[2], "joint": SERIES[3]}
STREAM_COLOR = {"x_sparse": SERIES[0], "y_sparse": SERIES[2], "mlp_hidden": SERIES[1]}

# -- sequential ramp: one hue, light -> dark --------------------------------
BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SEQ_CMAP = LinearSegmentedColormap.from_list("bdh_blue", BLUE_RAMP)


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK,
            "axes.labelsize": 9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.labelcolor": INK_SECONDARY,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "font.size": 9.5,
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def strip_spines(ax, keep=("left", "bottom")) -> None:
    """Recessive axes: drop the frame, keep only what orients the reader."""
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


def cell_ink(value: float, vmin: float, vmax: float) -> str:
    """Readable annotation color for a heatmap cell (primary or inverse ink)."""
    if vmax == vmin:
        return INK
    t = (value - vmin) / (vmax - vmin)
    return INK_INVERSE if t > 0.55 else INK


def save(fig, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")
