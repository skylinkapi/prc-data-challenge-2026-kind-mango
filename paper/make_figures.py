"""Build the figures of the JOAS paper from the CSV files in paper/data."""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
WIDTH_IN = 6.3
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
BOX_FILL = "#f3f2ef"
BLUE = "#2a78d6"
GRAY = "#a3a29c"
RAMP = {"stand_clock": "#1c5cab", "segment_only": "#86b6ef"}
# v1 to v10 scored about 561 s, before the label filter was removed; they flatten the scale.
FIRST_SHOWN = 13
STYLE = {
    "font.size": 8,
    "axes.edgecolor": INK_MUTED,
    "axes.labelcolor": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
}

# Pipeline figure: boxes as (x, y, width, height, text), lane titles and arrows, in axis units of 10 by 10.6.
PIPELINE_BOXES = [
    (0.2, 9.4, 4.5, 1.0, "Organiser data\nmovements joined to NM flight records\n2025 with labels, 2026 without"),
    (5.3, 9.4, 4.5, 1.0, "Open data\nMETAR, OurAirports, OSM, X-Plane stands,\nEUROCONTROL files, OPDI, ADSB.lol"),
    (
        1.5,
        7.9,
        7.0,
        0.9,
        "19 tables, 158 columns: schedule, NM clocks, traffic,\nweather, geometry, ground-track clocks",
    ),
    (0.2, 6.2, 3.0, 1.0, "Tree members\nLightGBM, CatBoost\nplain and anchored"),
    (3.5, 6.2, 3.0, 1.0, "Neural members\nMLP, five TabM\nfamilies"),
    (1.0, 4.85, 4.7, 0.8, "Blend: 0.28 trees + 0.72 neural"),
    (1.0, 3.45, 4.7, 0.9, "Pushback stack on label minus blend\n(rows with a ground track), stand bound"),
    (1.0, 2.2, 4.7, 0.75, "Month level: June and July"),
    (7.0, 6.2, 2.5, 1.0, "Gate: probability p\nof a schedule copy"),
    (7.0, 4.75, 2.5, 1.0, "p * s\n+ (1 - p) * normal model"),
    (7.0, 3.3, 2.5, 1.0, "Step A band table\n(no NM record,\ns above 4 h)"),
    (1.5, 0.95, 7.0, 0.75, "Row rules: Step A date, LOBT window, LFPG precision"),
    (3.25, 0.0, 3.5, 0.6, "Submission file"),
]
PIPELINE_LANES = [(3.3, 7.45, "outside LIRF"), (8.25, 7.45, "LIRF")]
PIPELINE_ARROWS = [
    ((2.45, 9.4), (3.5, 8.8)),
    ((7.55, 9.4), (6.5, 8.8)),
    ((3.0, 7.9), (1.7, 7.2)),
    ((4.0, 7.9), (5.0, 7.2)),
    ((7.0, 7.9), (8.25, 7.2)),
    ((1.7, 6.2), (2.6, 5.65)),
    ((5.0, 6.2), (4.1, 5.65)),
    ((3.35, 4.85), (3.35, 4.35)),
    ((3.35, 3.45), (3.35, 2.95)),
    ((8.25, 6.2), (8.25, 5.75)),
    ((8.25, 4.75), (8.25, 4.3)),
    ((3.35, 2.2), (3.35, 1.7)),
    ((8.25, 3.3), (7.0, 1.7)),
    ((5.0, 0.95), (5.0, 0.6)),
]

log = logging.getLogger(__name__)


def plot_score_path(scores: pd.DataFrame) -> plt.Figure:
    """Plot the live RMSE of each upload and the best score so far."""
    shown = scores[scores.version >= FIRST_SHOWN].reset_index(drop=True)
    best = shown.rmse_s.cummin()
    is_new_best = shown.rmse_s.lt(best.shift(fill_value=float("inf")))
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 3.0))
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.scatter(
        shown.version[~is_new_best], shown.rmse_s[~is_new_best], s=10, color=GRAY, label="upload, no new best", zorder=2
    )
    ax.step(shown.version, best, where="post", color=BLUE, lw=2, label="best live score", zorder=3)
    ax.scatter(
        shown.version[is_new_best], shown.rmse_s[is_new_best], s=14, color=BLUE, edgecolor="white", lw=0.6, zorder=4
    )
    for row in shown.dropna(subset=["note"]).itertuples():
        ax.annotate(
            f"v{row.version}: {row.note}",
            (row.version, row.rmse_s),
            xytext=(row.note_x, row.note_y),
            va="center",
            fontsize=7,
            color=INK_MUTED,
            arrowprops={"arrowstyle": "-", "color": GRAY, "lw": 0.6, "shrinkA": 1, "shrinkB": 2},
        )
    ax.set_xlabel("Upload version")
    ax.set_ylabel("Live RMSE, s")
    ax.set_xlim(FIRST_SHOWN - 3, shown.version.max() + 3)
    ax.set_ylim(190, 450)
    ax.legend(frameon=False, loc="upper right")
    return fig


def plot_coverage(coverage: pd.DataFrame) -> plt.Figure:
    """Plot the share of 2026 ranking departures with an ADS-B clock, per airport."""
    order = coverage.assign(track=1 - coverage.no_adsb).sort_values("track")
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.8))
    left = pd.Series(0.0, index=order.index)
    labels = {"stand_clock": "last point at the assigned stand", "segment_only": "ground segment only"}
    for column, label in labels.items():
        ax.barh(
            order.airport,
            order[column] * 100,
            left=left * 100,
            height=0.62,
            color=RAMP[column],
            edgecolor="white",
            lw=1,
            label=label,
        )
        left += order[column]
    for y, share in enumerate(order.track):
        ax.text(share * 100 + 1, y, f"{share:.0%}", va="center", fontsize=7, color=INK_MUTED)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Share of ranking departures, % (January and July 2026)")
    ax.legend(frameon=False, ncols=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return fig


def draw_box(ax: plt.Axes, x: float, y: float, w: float, h: float, text: str) -> None:
    """Draw one stage of the pipeline as a rounded box with centred text."""
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc=BOX_FILL, ec=INK_MUTED, lw=0.7)
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7, color=INK, linespacing=1.3)


def plot_pipeline() -> plt.Figure:
    """Draw the stages of the served model, from the input files to the submission."""
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 5.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.6)
    ax.axis("off")
    for box in PIPELINE_BOXES:
        draw_box(ax, *box)
    for x, y, text in PIPELINE_LANES:
        ax.text(x, y, text, ha="center", fontsize=7, color=INK_MUTED, style="italic")
    arrow = {"arrowstyle": "-|>", "color": INK_MUTED, "lw": 0.8, "shrinkA": 0, "shrinkB": 0}
    for start, end in PIPELINE_ARROWS:
        ax.annotate("", end, start, arrowprops=arrow)
    return fig


def save(fig: plt.Figure, name: str) -> None:
    """Write one figure as a PDF file to the figures folder of the paper."""
    path = FIGURES / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    log.info("Wrote %s", path)


def main() -> None:
    logging.basicConfig(format="%(message)s")
    log.setLevel(logging.INFO)
    plt.rcParams.update(STYLE)
    FIGURES.mkdir(exist_ok=True)
    save(plot_pipeline(), "pipeline")
    save(plot_score_path(pd.read_csv(DATA / "live_scores.csv")), "score_path")
    save(plot_coverage(pd.read_csv(DATA / "coverage_2026.csv")), "coverage")


if __name__ == "__main__":
    main()
