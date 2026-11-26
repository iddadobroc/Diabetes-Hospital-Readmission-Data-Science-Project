"""Charts in the IBCS notation, shared by the notebooks, the app and the slides.

The rules come from the IBCS Standards 2.0 and the course slides (W8, "Reporting with
IBCS"). What this module enforces:

- A page has a **message** on top (one full sentence that can be checked), then a
  **three-line title**: who (the unit and filters), what (the measure in bold and its
  unit), when (period, scenarios). The title never judges; the message does.
- **Scenarios**: AC (observed) solid dark grey, FC (the model's estimate) hatched, PL
  outlined. Models and baselines are not scenarios: they are structure categories, so
  they are listed top to bottom as horizontal bars.
- **Variances**: desirable bright green, undesirable dark red, neutral blue; a "+" on
  positive variances only; the variance of a percentage is in pp.
- **No decoration**: no value axis, no grid lines, no frame; data labels instead. Value
  axes start at zero. No pies, gauges or traffic lights anywhere.
- A **footnote** names the source, the data used and what was left out.
"""

import matplotlib.pyplot as plt
import numpy as np

# Colours (IBCS gives no hex codes; these follow its rules)
AC = "#404040"  # actual / observed: solid dark grey
PY = "#A6A6A6"  # reference structure, other categories: lighter grey
GOOD = "#5AA02C"  # desirable variance: bright green
BAD = "#B00000"  # undesirable variance: dark red
NEUTRAL = "#2F6EBA"  # neutral variance, highlighting ellipse and markers
TEXT = "#202020"
MUTED = "#707070"
FONT = "DejaVu Sans"

plt.rcParams.update(
    {
        "font.family": FONT,
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "text.color": TEXT,
        "hatch.linewidth": 1.2,
        "savefig.dpi": 160,
        "figure.dpi": 100,
    }
)


def fmt_number(value, decimals=0, unit="", sign=False):
    """A number in the IBCS format: thin-space thousands, "−" for negatives.

    Args:
        value (float): The number.
        decimals (int): Decimal places.
        unit (str): Appended after a thin space, such as "%" or "pp" ("" for none).
        sign (bool): Put a "+" on positive values (variances only).

    Returns:
        str: The formatted number, such as "+1 234" or "−2.5 pp".
    """
    text = f"{abs(value):,.{decimals}f}".replace(",", " ")
    if value < 0 and float(text.replace(" ", "")) != 0:
        text = "−" + text
    elif sign and value > 0:
        text = "+" + text
    if unit:
        text += ("" if unit == "%" else " ") + unit
    return text


def clean_axes(ax, keep_category_axis=True):
    """Remove frame, value axis and grid lines (IBCS SIMPLIFY)."""
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(False)
    ax.tick_params(length=0)
    if not keep_category_axis:
        ax.set_yticks([])


def page(fig, message, who, what, unit, when, top=0.80):
    """Write the message and the three-line title above the charts of `fig`.

    Args:
        fig (matplotlib.figure.Figure): The figure (the "page").
        message (str): One full sentence with numbers: the key message.
        who (str): Line 1, the reporting unit and filters.
        what (str): Line 2, the measure, written in bold.
        unit (str): Line 2, the unit, written in regular after the measure.
        when (str): Line 3, the period and the scenarios.
        top (float): The top of the chart area, in figure coordinates.
    """
    fig.text(0.02, 0.985, message, ha="left", va="top", fontsize=12.5, weight="bold")
    y = 0.985 - 0.075 * (1 + message.count("\n"))
    fig.text(0.02, y, who, ha="left", va="top", fontsize=10)
    measure = fig.text(
        0.02, y - 0.045, what, ha="left", va="top", fontsize=10, weight="bold"
    )
    fig.canvas.draw()
    box = measure.get_window_extent().transformed(fig.transFigure.inverted())
    fig.text(box.x1 + 0.006, y - 0.045, unit, ha="left", va="top", fontsize=10)
    fig.text(0.02, y - 0.09, when, ha="left", va="top", fontsize=10)
    fig.subplots_adjust(top=top)


def footnote(fig, text, bottom=0.14):
    """Write the footnote (source, data used, what was left out) under the charts."""
    fig.text(0.02, 0.015, text, ha="left", va="bottom", fontsize=7.5, color=MUTED)
    fig.subplots_adjust(bottom=bottom)


def bars_h(
    ax,
    labels,
    values,
    decimals=0,
    unit="",
    highlight=None,
    hatch=None,
    color=AC,
    xmax=None,
    width=0.66,
):
    """Horizontal bars for structure categories (models, groups, features).

    Labels on the left, values at the end of each bar, no value axis. The axis starts
    at zero; pass the same `xmax` to charts of the same unit on one page (IBCS CHECK:
    same unit, same scale).

    Args:
        ax: The matplotlib axes.
        labels (list[str]): Category names, top to bottom.
        values (list[float]): One value per category (same unit).
        decimals (int): Decimal places of the labels.
        unit (str): Unit of the labels ("" when it is in the title).
        highlight (int, optional): Index of the bar the message is about: it keeps the
            dark AC fill while the others turn lighter grey.
        hatch (list[bool], optional): Bars that are estimates (FC), drawn hatched.
        color (str): Fill of the bars.
        xmax (float, optional): Right end of the scale.
        width (float): Bar thickness as a share of the category (2/3 for absolute
            measures, 1/3 for ratios).
    """
    values = np.asarray(values, dtype=float)
    y = np.arange(len(values))
    for i, value in enumerate(values):
        fill = color if highlight is None or i == highlight else PY
        is_fc = bool(hatch[i]) if hatch is not None else False
        ax.barh(
            y[i],
            value,
            height=width,
            color="white" if is_fc else fill,
            edgecolor=fill,
            hatch="////" if is_fc else None,
            linewidth=1.0,
        )
        ax.text(
            value if value >= 0 else 0,
            y[i],
            " " + fmt_number(value, decimals, unit),
            va="center",
            ha="left",
            fontsize=9.5,
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xticks([])
    top = xmax if xmax is not None else max(values.max(), 0) * 1.18
    ax.set_xlim(min(values.min(), 0), top)
    clean_axes(ax)


def variance_h(ax, values, decimals=0, unit="", higher_is_better=True, xmax=None):
    """Absolute variances as horizontal bars, beside the base bars, same scale.

    Green where the variance is desirable, red where it is not; labels outside the
    bar on the side of growth; "+" on positive values only. Share the y order with the
    base chart (`sharey`).
    """
    values = np.asarray(values, dtype=float)
    y = np.arange(len(values))
    for i, value in enumerate(values):
        if np.isnan(value):
            continue
        good = value >= 0 if higher_is_better else value <= 0
        ax.barh(y[i], value, height=0.66, color=GOOD if good else BAD)
        ax.text(
            value,
            y[i],
            (" " if value >= 0 else "") + fmt_number(value, decimals, unit, True)
            + ("" if value >= 0 else " "),
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9.5,
        )
    ax.axvline(0, color=AC, linewidth=1.2)
    span = xmax if xmax is not None else np.nanmax(np.abs(values)) * 1.4
    ax.set_xlim(-span, span)
    ax.set_xticks([])
    ax.tick_params(labelleft=False)
    clean_axes(ax)


def columns_ac_fc(ax, labels, ac, fc, decimals=1, unit="%"):
    """Observed (AC, solid) against estimated (FC, hatched) columns, per category.

    The FC column stands slightly behind and to the left, the AC column in front; used
    for calibration by risk group (the categories are ordered groups, not periods).
    """
    x = np.arange(len(labels))
    ax.bar(
        x - 0.12,
        fc,
        width=0.5,
        color="white",
        edgecolor=AC,
        hatch="////",
        linewidth=1.0,
        label="FC",
    )
    ax.bar(x + 0.06, ac, width=0.5, color=AC, label="AC")
    for i, value in enumerate(ac):
        ax.text(
            x[i] + 0.06,
            value,
            fmt_number(value, decimals, unit),
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax.set_xticks(x, labels)
    ax.set_yticks([])
    ax.set_ylim(0, max(np.max(ac), np.max(fc)) * 1.2)
    clean_axes(ax)
    # Legend integrated next to the last pair of columns (IBCS UNIFY 2.3)
    last = len(labels) - 1
    ax.text(last + 0.45, ac[last] * 0.5, "AC", fontsize=8.5, va="center")
    ax.text(last - 0.12, fc[last] * 1.06 + 0.002, "FC", fontsize=8.5, ha="center")


def variance_columns(ax, labels, values, decimals=1, unit="pp", higher_is_better=False):
    """A variance tier of columns (e.g. ΔFC = AC − FC in pp) above a column chart."""
    x = np.arange(len(labels))
    values = np.asarray(values, dtype=float)
    for i, value in enumerate(values):
        good = value >= 0 if higher_is_better else value <= 0
        ax.bar(x[i], value, width=0.5, color=GOOD if good else BAD)
        ax.text(
            x[i],
            value,
            fmt_number(value, decimals, unit, True),
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )
    ax.axhline(0, color=AC, linewidth=1.2)
    span = np.nanmax(np.abs(values)) * 1.6 if len(values) else 1
    ax.set_ylim(-span, span)
    ax.set_xticks([])
    ax.set_yticks([])
    clean_axes(ax)


def save(fig, path):
    """Save a page as PNG with a white background."""
    fig.savefig(path, facecolor="white")
    plt.close(fig)
