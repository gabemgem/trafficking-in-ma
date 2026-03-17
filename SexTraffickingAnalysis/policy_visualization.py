"""
policy_visualization.py
─────────────────────────────────────────────────────────────────────────────
Visualizes policy scenario data from CSWPolicyScenarioOutput.xlsx.

Five figure options (--plot):
  heatmap        — % change heatmap (PolicyPercentChanges)
  grouped_bar    — % change clustered vertical bars (PolicyPercentChanges)
  diverging_bar  — % change small multiples, one panel per metric (PolicyPercentChanges)
  changes_panels — absolute difference small multiples (PolicyChanges)
  vals_panels    — absolute values small multiples with baseline (PolicyVals)

COMMAND LINE:
  python policy_visualization.py                         # all five figures
  python policy_visualization.py --plot heatmap
  python policy_visualization.py --output results/ --dpi 300
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
EXCEL_FILE = "CSWPolicyScenarioOutput.xlsx"
OUTPUT_DIR = "policy_figures"
DPI        = 150

# Short display labels — shared across all three sheets
METRIC_LABELS_PCT = {
    "TotalTSS % Change":                                   "Total TSS",
    "TotalSS % Change":                                    "Total SS",
    "% Trafficked vs. Nontrafficked Sex Sellers % Change": "% Trafficked\nvs Non-Trafficked",
    "Buyers % Change":                                     "Buyers",
    "Demand % Change":                                     "Demand",
    "Traffickers % Change":                                "Traffickers",
}

METRIC_LABELS_ABS = {
    "TotalTSS":                                        "Total TSS",
    "TotalSS":                                         "Total SS",
    "% Trafficked vs. Nontrafficked Sex Sellers":      "% Trafficked\nvs Non-Trafficked",
    "Buyers":                                          "Buyers",
    "Demand":                                          "Demand",
    "Traffickers":                                     "Traffickers",
}

# Scenario colors (policy scenarios only — not used for Default)
SCENARIO_COLORS = [
    "#2166ac",   # blue
    "#4dac26",   # green
    "#d01c8b",   # magenta
    "#f1a340",   # orange
]
DEFAULT_COLOR = "#888888"


# ── Data loading ──────────────────────────────────────────────────────────────
def _read_sheet(sheet: str, label_map: dict) -> tuple[list[str], list[str], np.ndarray]:
    df = pd.read_excel(EXCEL_FILE, sheet_name=sheet).dropna(how="all")
    scenarios   = df["scenario"].tolist()
    metric_cols = [c for c in df.columns if c != "scenario"]
    metrics     = [label_map.get(c, c) for c in metric_cols]
    values      = df[metric_cols].values
    return scenarios, metrics, values


def load_pct():
    scenarios, metrics, values = _read_sheet("PolicyPercentChanges", METRIC_LABELS_PCT)
    return scenarios, metrics, values * 100     # → percentage points


def load_changes():
    return _read_sheet("PolicyChanges", METRIC_LABELS_ABS)


def load_vals():
    return _read_sheet("PolicyVals", METRIC_LABELS_ABS)


# ── Shared helpers ────────────────────────────────────────────────────────────
def _smart_fmt(v: float) -> str:
    """Format a number compactly: use k suffix for thousands."""
    if abs(v) >= 1000:
        return f"{v/1000:+.1f}k" if v != 0 else "0"
    if abs(v) < 0.1:
        return f"{v:+.4f}"
    return f"{v:+.2f}"


def _bar_label(ax, bars, vals, fontsize=10, fmt_fn=None):
    """Add value labels above/below bars after y-limits are finalized."""
    fmt = fmt_fn or (lambda v: f"{v:+.1f}%")
    ylo, yhi = ax.get_ylim()
    pad = (yhi - ylo) * 0.015
    for bar, val in zip(bars, vals):
        y  = bar.get_height() if val >= 0 else val
        va = "bottom" if val >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2,
                y + (pad if val >= 0 else -pad),
                fmt(val), ha="center", va=va, fontsize=fontsize, zorder=5)


# ── Figure 1: Heatmap (% change) ──────────────────────────────────────────────
def plot_heatmap(scenarios, metrics, values, out_dir: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(11, 3.8))

    vmax = max(0.01, np.abs(values).max())
    im = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("% Change from Baseline", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios, fontsize=12)

    for r in range(len(scenarios)):
        for c in range(len(metrics)):
            val = values[r, c]
            txt_color = "white" if abs(val) > vmax * 0.65 else "black"
            ax.text(c, r, f"{val:+.1f}%", ha="center", va="center",
                    fontsize=11, color=txt_color, fontweight="bold")

    ax.set_xticks(np.arange(-0.5, len(metrics)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(scenarios)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    ax.set_title("Policy Scenario Impact — % Change from Baseline",
                 fontsize=14, pad=10)
    fig.tight_layout()
    path = out_dir / "policy_heatmap.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 2: Grouped bar chart (% change) ────────────────────────────────────
def plot_grouped_bar(scenarios, metrics, values, out_dir: Path, dpi: int):
    n_scenarios = len(scenarios)
    n_metrics   = len(metrics)
    x     = np.arange(n_metrics)
    width = 0.72 / n_scenarios

    fig, ax = plt.subplots(figsize=(12, 5))

    all_bars = []
    for i, (scenario, color) in enumerate(zip(scenarios, SCENARIO_COLORS)):
        offsets = x + (i - (n_scenarios - 1) / 2) * width
        bars = ax.bar(offsets, values[i], width=width * 0.88,
                      color=color, edgecolor="black", linewidth=0.5,
                      label=scenario, zorder=3)
        all_bars.append((bars, values[i]))

    ax.axhline(0, color="black", linewidth=0.8, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("\n", " ") for m in metrics], fontsize=12)
    ax.set_ylabel("% Change from Baseline", fontsize=13)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.1f}%"))
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo - (yhi - ylo) * 0.12, yhi + (yhi - ylo) * 0.12)

    for bars, vals in all_bars:
        _bar_label(ax, bars, vals, fontsize=9)

    ax.legend(fontsize=11, framealpha=0.9, loc="lower right")
    ax.set_title("Policy Scenario Impact — % Change from Baseline",
                 fontsize=14, pad=10)
    fig.tight_layout()
    path = out_dir / "policy_grouped_bar.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 3: Diverging horizontal bar, one panel per metric (% change) ───────
def plot_diverging_bar(scenarios, metrics, values, out_dir: Path, dpi: int):
    n_metrics = len(metrics)
    ncols = 3
    nrows = int(np.ceil(n_metrics / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 3.2 * nrows),
                             sharey=False)
    axes_flat = axes.flatten()

    # Reverse so first scenario in data appears at top
    scenarios_r = scenarios[::-1]
    values_r    = values[::-1]

    for c_idx, (metric, ax) in enumerate(zip(metrics, axes_flat)):
        vals   = values_r[:, c_idx]
        colors = SCENARIO_COLORS[:len(scenarios_r)][::-1]
        bars   = ax.barh(range(len(scenarios_r)), vals,
                         color=colors, edgecolor="black", linewidth=0.5, height=0.6)

        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(range(len(scenarios_r)))
        ax.set_yticklabels(scenarios_r, fontsize=11)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8, steps=[1, 2, 5, 10]))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.1f}%"))
        ax.tick_params(axis="x", labelsize=10)
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)
        ax.set_title(metric.replace("\n", " "), fontsize=12)

        # Labels inside large bars, outside small bars
        xlo, xhi = ax.get_xlim()
        xrange   = xhi - xlo
        threshold = xrange * 0.18   # bar must span >18% of axis range to fit label inside
        pad = xrange * 0.02
        for bar, val in zip(bars, vals):
            if abs(val) >= threshold:
                # Centered inside bar — always white
                ax.text(val / 2, bar.get_y() + bar.get_height() / 2,
                        f"{val:+.1f}%", va="center", ha="center",
                        fontsize=10, color="white", fontweight="bold")
            else:
                # Outside bar tip — always black
                ha = "left" if val >= 0 else "right"
                x  = val + (pad if val >= 0 else -pad)
                ax.text(x, bar.get_y() + bar.get_height() / 2,
                        f"{val:+.1f}%", va="center", ha=ha,
                        fontsize=10, color="black", fontweight="bold")

    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)

    fig.suptitle("Policy Scenario Impact — % Change from Baseline",
                 fontsize=15, y=1.01)
    fig.tight_layout()
    path = out_dir / "policy_diverging_bar.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 4: Absolute change small multiples (PolicyChanges) ─────────────────
def plot_changes_panels(scenarios, metrics, values, out_dir: Path, dpi: int):
    """
    One panel per metric. Scales differ too much for a single axis.
    Horizontal diverging bars; zero line marks no change from baseline.
    """
    n_metrics = len(metrics)
    ncols = 3
    nrows = int(np.ceil(n_metrics / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 3.4 * nrows),
                             sharey=False)
    axes_flat = axes.flatten()

    for c_idx, (metric, ax) in enumerate(zip(metrics, axes_flat)):
        vals   = values[:, c_idx]
        colors = SCENARIO_COLORS[:len(scenarios)]
        bars   = ax.barh(range(len(scenarios)), vals,
                         color=colors, edgecolor="black", linewidth=0.5, height=0.6)

        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(range(len(scenarios)))
        ax.set_yticklabels(scenarios, fontsize=11)
        ax.tick_params(axis="x", labelsize=10)
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)
        ax.set_title(metric.replace("\n", " "), fontsize=12)

        # Smart number formatting per panel
        pad = max(abs(vals).max() * 0.03, abs(vals).max() * 1e-6 + 1e-9)
        for bar, val in zip(bars, vals):
            ha = "left" if val >= 0 else "right"
            ax.text(val + (pad if val >= 0 else -pad),
                    bar.get_y() + bar.get_height() / 2,
                    _smart_fmt(val), va="center", ha=ha, fontsize=10)

    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)

    fig.suptitle("Policy Scenario Impact — Absolute Change from Baseline",
                 fontsize=15, y=1.01)
    fig.tight_layout()
    path = out_dir / "policy_changes_panels.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 5: Absolute values small multiples (PolicyVals) ────────────────────
def plot_vals_panels(scenarios, metrics, values, out_dir: Path, dpi: int):
    """
    One panel per metric. Includes Default as a gray reference bar.
    Y-axis starts near (not at) zero to show differences clearly.
    """
    n_metrics = len(metrics)
    ncols = 3
    nrows = int(np.ceil(n_metrics / ncols))

    # First row is Default; remaining rows are policy scenarios
    default_vals = values[0]
    policy_scenarios = scenarios[1:]
    policy_values    = values[1:]

    all_scenarios = scenarios          # Default + policy
    all_colors    = [DEFAULT_COLOR] + SCENARIO_COLORS[:len(policy_scenarios)]

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 3.6 * nrows),
                             sharey=False)
    axes_flat = axes.flatten()

    for c_idx, (metric, ax) in enumerate(zip(metrics, axes_flat)):
        vals   = values[:, c_idx]
        colors = all_colors[:len(all_scenarios)]
        x      = np.arange(len(all_scenarios))

        bars = ax.bar(x, vals, color=colors, edgecolor="black",
                      linewidth=0.5, width=0.6, zorder=3)

        # Baseline reference line
        ax.axhline(default_vals[c_idx], color=DEFAULT_COLOR,
                   linewidth=1.2, linestyle="--", zorder=4, alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels([s.replace(" ", "\n") for s in all_scenarios],
                           fontsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(metric.replace("\n", " "), fontsize=14)

        # Zoom y-axis to show differences (start at 90% of min value)
        ylo = min(vals) * 0.97
        yhi = max(vals) * 1.06
        ax.set_ylim(ylo, yhi)

        # Value labels above each bar
        yrange = yhi - ylo
        pad    = yrange * 0.012
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + pad,
                    _smart_fmt(val).lstrip("+"), ha="center", va="bottom",
                    fontsize=12, zorder=5)

    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)

    # Legend for Default vs scenario colors
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=DEFAULT_COLOR, edgecolor="black", label="Default")] + \
                     [Patch(facecolor=c, edgecolor="black", label=s)
                      for c, s in zip(SCENARIO_COLORS, policy_scenarios)]
    # fig.legend(handles=legend_handles, fontsize=11, loc="lower center",
    #            ncol=len(all_scenarios), bbox_to_anchor=(0.5, -0.02),
    #            framealpha=0.9)

    fig.suptitle("Policy Scenario Outcomes — Absolute Values",
                 fontsize=17, y=1.01)
    fig.tight_layout()
    path = out_dir / "policy_vals_panels.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Policy scenario visualizations")
    parser.add_argument(
        "--plot",
        choices=["heatmap", "grouped_bar", "diverging_bar", "changes_panels", "vals_panels"],
        default=None,
        help="Which plot to generate (default: all)",
    )
    parser.add_argument("--output", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--dpi",    default=DPI, type=int, help="Figure DPI")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)

    pct_scenarios, pct_metrics, pct_values         = load_pct()
    chg_scenarios, chg_metrics, chg_values         = load_changes()
    val_scenarios, val_metrics, val_values         = load_vals()

    plots = {
        "heatmap":        lambda: plot_heatmap(pct_scenarios, pct_metrics, pct_values, out_dir, args.dpi),
        "grouped_bar":    lambda: plot_grouped_bar(pct_scenarios, pct_metrics, pct_values, out_dir, args.dpi),
        "diverging_bar":  lambda: plot_diverging_bar(pct_scenarios, pct_metrics, pct_values, out_dir, args.dpi),
        "changes_panels": lambda: plot_changes_panels(chg_scenarios, chg_metrics, chg_values, out_dir, args.dpi),
        "vals_panels":    lambda: plot_vals_panels(val_scenarios, val_metrics, val_values, out_dir, args.dpi),
    }

    to_run = [args.plot] if args.plot else list(plots.keys())
    for name in to_run:
        print(f"Generating: {name}")
        plots[name]()


if __name__ == "__main__":
    main()
