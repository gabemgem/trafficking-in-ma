"""
Sensitivity Analysis for LHS System Dynamics Model Outputs
===========================================================
Uses Partial Rank Correlation Coefficients (PRCC) and
Spearman Rank Correlation (SRCC) to identify which input
parameters most influence key outputs.

PRCC is the standard method for system dynamics LHS sensitivity:
it removes the effect of all other inputs before computing the
correlation, handling non-linear monotonic relationships well.

─────────────────────────────────────────────────────────────
HOW TO USE:

  Option A — Separate files (typical LHS workflow):
    1. Set LHS_PARAMS_FILE  = "lhs_samples.xlsx"   (from sensitivity_analysis_param_generation.py)
    2. Set LHS_OUTPUTS_FILE = "lhs_outputs.csv"   (model outputs, row-matched)
    3. Leave COMBINED_FILE  = None

  Option B — Single combined CSV (inputs + outputs in one file):
    1. Set COMBINED_FILE = "CSWOutput_General.csv" (or similar)
    2. Leave LHS_PARAMS_FILE = LHS_OUTPUTS_FILE = None

  Then run:  python sensitivity_analysis.py
─────────────────────────────────────────────────────────────
"""

import math
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from scipy.stats import nct
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Option A: separate files (rows must be aligned 1-to-1)
LHS_PARAMS_FILE  = "lhs_samples.xlsx"
LHS_OUTPUTS_FILE = "../Sex_Trafficking_SDM/LHSVariationOutput.csv"

# Option B: single combined file — set path here, leave None for Option A
COMBINED_FILE = None   # e.g., "CSWOutput_General.csv"

# Outputs to analyze (must exist as columns in your data)
OUTPUTS_OF_INTEREST = [
    "TotalTSS",
    "pctTraffickedVsNontrafficked",
    "Buyers",
    "demand",
    "Traffickers",
    "pctSSCrimRecord",
]

# Columns to always exclude as inputs (identifiers, truly fixed params, etc.)
EXCLUDE_FROM_INPUTS = [
    "total_adult_population",
    "total_youth_population",
    "run_id",
    "scenario",
]

# Human-readable display names for output variables (used in all plots/tables)
OUTPUT_LABELS = {
    "TotalTSS":                    "Total Sex Sellers",
    "pctTraffickedVsNontrafficked":"% Trafficked vs Non-Trafficked",
    "Buyers":                      "Buyers",
    "demand":                      "Demand",
    "Traffickers":                 "Traffickers",
    "pctSSCrimRecord":             "% with Criminal Record",
    "TotalSS":                     "Total Sex Sellers (All)",
    "TotalYouthTSS":               "Total Youth Sex Sellers",
    "TotalNonTraffickedSS":        "Non-Trafficked Sex Sellers",
    "EmServicesAdult":             "Adult Em Services",
    "STServicesAdult":             "Adult ST Services",
    "LTServicesAdult":             "Adult LT Services",
    "pctExitedCrimRecord":         "% Exited with Crim Record",
    "cost":                        "Cost",
}

# Significance level for PRCC p-values
ALPHA = 0.05

# Max parameters shown per tornado plot (sorted by |PRCC|); 0 = show all
TOP_N = 20

# Directory where figures and CSVs are saved
OUTPUT_DIR = "sensitivity_results"


# ─────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────

def _snake(name: str) -> str:
    """Convert camelCase / PascalCase param names to snake_case for display."""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
    return s.lower()


def _out_label(name: str) -> str:
    """Return a human-readable display label for an output variable."""
    return OUTPUT_LABELS.get(name, name)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (X, Y) where X = input parameters DataFrame, Y = outputs DataFrame.
    Columns in Y are limited to OUTPUTS_OF_INTEREST that actually exist.
    """
    if COMBINED_FILE:
        df = pd.read_csv(COMBINED_FILE, index_col=None)
        print(f"Loaded combined file: {COMBINED_FILE}  ({len(df)} rows, {len(df.columns)} cols)")
        outputs_present = [c for c in OUTPUTS_OF_INTEREST if c in df.columns]
        _warn_missing(outputs_present, "combined file")
        input_cols = [
            c for c in df.columns
            if c not in outputs_present
            and c not in EXCLUDE_FROM_INPUTS
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        return df[input_cols].reset_index(drop=True), df[outputs_present].reset_index(drop=True)

    # Option A: separate files
    params_df  = pd.read_excel(LHS_PARAMS_FILE, sheet_name="lhs_samples", index_col=0)
    outputs_df = pd.read_csv(LHS_OUTPUTS_FILE, index_col=None)
    print(f"Loaded params:  {LHS_PARAMS_FILE}   ({len(params_df)} rows, {params_df.shape[1]} cols)")
    print(f"Loaded outputs: {LHS_OUTPUTS_FILE}  ({len(outputs_df)} rows, {outputs_df.shape[1]} cols)")

    if len(params_df) != len(outputs_df):
        raise ValueError(
            f"Row mismatch: {len(params_df)} param rows vs {len(outputs_df)} output rows.\n"
            "Files must be row-aligned (same run order)."
        )

    outputs_present = [c for c in OUTPUTS_OF_INTEREST if c in outputs_df.columns]
    _warn_missing(outputs_present, LHS_OUTPUTS_FILE)

    input_cols = [
        c for c in params_df.columns
        if c not in EXCLUDE_FROM_INPUTS
        and pd.api.types.is_numeric_dtype(params_df[c])
    ]
    return (
        params_df[input_cols].reset_index(drop=True),
        outputs_df[outputs_present].reset_index(drop=True),
    )


def _warn_missing(found: list, source: str):
    missing = set(OUTPUTS_OF_INTEREST) - set(found)
    if missing:
        print(f"  WARNING: these outputs were not found in {source}: {sorted(missing)}")
    if not found:
        raise ValueError("No output columns found — check OUTPUTS_OF_INTEREST and your data file.")


def drop_constant_cols(X: pd.DataFrame) -> pd.DataFrame:
    """Remove columns with zero variance (fixed across all runs)."""
    constant = [c for c in X.columns if X[c].nunique() <= 1]
    if constant:
        print(f"  Dropping {len(constant)} constant input column(s): {constant}")
    return X.drop(columns=constant)


# ─────────────────────────────────────────────
# SENSITIVITY METHODS
# ─────────────────────────────────────────────

def compute_prcc(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Partial Rank Correlation Coefficient between each column of X and y.

    Algorithm:
      1. Rank-transform all variables.
      2. For each input Xi, regress both Xi and Y on all other inputs (OLS).
      3. PRCC = Pearson r of the two residual vectors.

    Returns DataFrame indexed by parameter name with columns:
      prcc, p_value, significant
    """
    n, k = X.shape
    Xr = X.apply(lambda col: stats.rankdata(col), axis=0).values  # (n, k)
    yr = stats.rankdata(y.values)
    ones = np.ones(n)
    records = []

    for i, col in enumerate(X.columns):
        others_idx = [j for j in range(k) if j != i]
        xi = Xr[:, i]

        if not others_idx:
            # Only one input: PRCC collapses to Spearman r
            r, p = stats.pearsonr(xi, yr)
        else:
            others = Xr[:, others_idx]
            design = np.column_stack([ones, others])

            coef_x, *_ = np.linalg.lstsq(design, xi, rcond=None)
            xi_res = xi - design @ coef_x

            coef_y, *_ = np.linalg.lstsq(design, yr, rcond=None)
            yr_res = yr - design @ coef_y

            r, p = stats.pearsonr(xi_res, yr_res)

        records.append({
            "parameter":   col,
            "prcc":        r,
            "p_value":     p,
            "significant": p < ALPHA,
        })

    return pd.DataFrame(records).set_index("parameter")


def compute_srcc(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Spearman Rank Correlation Coefficient for each input column vs y."""
    records = []
    for col in X.columns:
        r, p = stats.spearmanr(X[col], y)
        records.append({
            "parameter":   col,
            "srcc":        r,
            "p_value":     p,
            "significant": p < ALPHA,
        })
    return pd.DataFrame(records).set_index("parameter")


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def tornado_plot(prcc_df: pd.DataFrame, output_name: str, out_dir: Path):
    """Horizontal bar chart of PRCC values sorted by magnitude."""
    df = prcc_df.copy().sort_values("prcc", key=abs, ascending=True)
    if TOP_N > 0:
        df = df.tail(TOP_N)

    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.38)))

    for idx, (param, row) in enumerate(df.iterrows()):
        color = "#2166ac" if row["prcc"] > 0 else "#d6604d"   # blue / red
        alpha = 1.0 if row["significant"] else 0.35
        ax.barh(idx, row["prcc"], color=color, alpha=alpha,
                edgecolor="black" if row["significant"] else "gray", linewidth=0.5)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([_snake(p) for p in df.index], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("PRCC", fontsize=10)
    ax.set_xlim(-1.05, 1.05)
    ax.set_title(
        f"Sensitivity (PRCC): {_out_label(output_name)}\n"
        f"solid = p < {ALPHA} | faded = not significant   "
        f"[blue = positive, red = negative]",
        fontsize=10,
    )
    # Legend patch
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2166ac", label="Positive effect"),
        Patch(facecolor="#d6604d", label="Negative effect"),
        Patch(facecolor="gray", alpha=0.35, label=f"p ≥ {ALPHA}"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    fig.tight_layout()

    path = out_dir / f"tornado_{output_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def heatmap_plot(all_prcc: dict, out_dir: Path):
    """
    Heatmap: rows = input parameters, columns = outputs.
    Top parameters (by max |PRCC| across outputs) shown first.
    Significant cells marked with a dot.
    """
    outputs = list(all_prcc.keys())
    prcc_matrix = pd.DataFrame(
        {out: all_prcc[out]["prcc"] for out in outputs}
    )
    sig_matrix = pd.DataFrame(
        {out: all_prcc[out]["significant"] for out in outputs}
    )

    # Sort by strongest effect across any output
    prcc_matrix["_sort"] = prcc_matrix.abs().max(axis=1)
    prcc_matrix = prcc_matrix.sort_values("_sort", ascending=False).drop(columns="_sort")
    if TOP_N > 0:
        prcc_matrix = prcc_matrix.head(TOP_N)
    sig_matrix = sig_matrix.loc[prcc_matrix.index]

    vmax = max(0.01, prcc_matrix.abs().max().max())
    fig, ax = plt.subplots(figsize=(max(6, len(outputs) * 1.5), max(4, len(prcc_matrix) * 0.4)))
    im = ax.imshow(prcc_matrix.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="PRCC", shrink=0.8)

    ax.set_xticks(range(len(outputs)))
    ax.set_xticklabels([_out_label(o) for o in outputs], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(prcc_matrix)))
    ax.set_yticklabels([_snake(p) for p in prcc_matrix.index], fontsize=8)

    for r, param in enumerate(prcc_matrix.index):
        for c, out in enumerate(outputs):
            val = prcc_matrix.at[param, out]
            marker = "●" if sig_matrix.at[param, out] else "○"
            color = "black" #"white" if abs(val) > vmax * 0.6 else "black"
            ax.text(c, r, marker, ha="center", va="center", fontsize=7, color=color)

    ax.set_title(
        f"PRCC Heatmap — top {min(TOP_N, len(prcc_matrix))} inputs × all outputs\n"
        f"● = p < {ALPHA}   ○ = not significant",
        fontsize=10,
    )
    fig.tight_layout()
    path = out_dir / "heatmap_prcc.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def scatter_top_params(X: pd.DataFrame, Y: pd.DataFrame, all_prcc: dict, out_dir: Path, n_top: int = 3):
    """Scatter plots of raw input vs output for the top N most influential parameters."""
    for out_name, prcc_df in all_prcc.items():
        top_params = prcc_df["prcc"].abs().nlargest(n_top).index.tolist()
        y = Y[out_name]

        fig, axes = plt.subplots(1, len(top_params), figsize=(5 * len(top_params), 4))
        if len(top_params) == 1:
            axes = [axes]

        for ax, param in zip(axes, top_params):
            ax.scatter(X[param], y, alpha=0.25, s=8, color="#2166ac", rasterized=True)
            row = prcc_df.loc[param]
            ax.set_xlabel(_snake(param), fontsize=8)
            ax.set_ylabel(_out_label(out_name), fontsize=8)
            sig_str = f"p={row['p_value']:.2g}" + ("*" if row["significant"] else "")
            ax.set_title(f"PRCC={row['prcc']:+.3f}  {sig_str}", fontsize=9)

        fig.suptitle(f"Top {n_top} parameters → {_out_label(out_name)}", fontsize=11)
        fig.tight_layout()
        path = out_dir / f"scatter_{out_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path.name}")


def prcc_vs_srcc_plot(all_prcc: dict, all_srcc: dict, out_dir: Path):
    """
    Scatter of PRCC vs SRCC for each parameter-output pair.
    Points near the diagonal mean that controlling for other parameters
    doesn't change the raw Spearman correlation much (low multicollinearity).
    """
    fig, axes = plt.subplots(
        1, len(all_prcc),
        figsize=(4.5 * len(all_prcc), 4),
        sharey=False,
    )
    if len(all_prcc) == 1:
        axes = [axes]

    for ax, (out_name, prcc_df) in zip(axes, all_prcc.items()):
        srcc_df = all_srcc[out_name]
        common = prcc_df.index.intersection(srcc_df.index)
        x = srcc_df.loc[common, "srcc"]
        y = prcc_df.loc[common, "prcc"]
        sig = prcc_df.loc[common, "significant"]

        ax.scatter(x[~sig], y[~sig], alpha=0.4, s=15, color="gray", label="not sig")
        ax.scatter(x[sig],  y[sig],  alpha=0.7, s=20, color="#2166ac", label=f"p<{ALPHA}")
        lim = max(x.abs().max(), y.abs().max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.7, label="PRCC=SRCC")
        ax.axhline(0, color="gray", linewidth=0.4)
        ax.axvline(0, color="gray", linewidth=0.4)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("SRCC", fontsize=9)
        ax.set_ylabel("PRCC", fontsize=9)
        ax.set_title(_out_label(out_name), fontsize=10)
        ax.legend(fontsize=7)

    fig.suptitle("PRCC vs SRCC — deviation from diagonal indicates partial-out effect", fontsize=10)
    fig.tight_layout()
    path = out_dir / "prcc_vs_srcc.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ─────────────────────────────────────────────
# SAMPLE SIZE / POWER ANALYSIS
# ─────────────────────────────────────────────

def _prcc_power(n: int, k: int, rho: float, alpha: float = 0.05) -> float:
    """
    Exact power for a two-tailed PRCC test.

    The PRCC test statistic is  T = r·√(df) / √(1−r²)  ~ t(df) under H0,
    where df = n − 2 − k  (k = number of *other* inputs partialed out).
    Under H1 (true PRCC = ρ), T follows a non-central t with
    non-centrality  λ = ρ·√df / √(1−ρ²).

    Fisher-z quick approximation (Marino et al. 2008):
        n ≥ ⌈((z_α/2 + z_β) / atanh(ρ))²⌉ + k + 3
    """
    df = n - 2 - k
    if df <= 0:
        return 0.0
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ncp = rho * math.sqrt(df) / math.sqrt(1 - rho ** 2)
    # P(|T| > t_crit | ncp)  =  upper tail + lower tail
    power = 1 - nct.cdf(t_crit, df, ncp) + nct.cdf(-t_crit, df, ncp)
    return float(power)


def _min_detectable_rho(n: int, k: int, alpha: float = 0.05, target_power: float = 0.80) -> float:
    """
    Minimum |PRCC| detectable with `target_power` given n runs and k other inputs.
    Solved by bisection on _prcc_power.
    """
    df = n - 2 - k
    if df <= 1:
        return float("nan")
    lo, hi = 1e-6, 1.0 - 1e-9
    for _ in range(60):
        mid = (lo + hi) / 2
        if _prcc_power(n, k, mid, alpha) < target_power:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _required_n(rho: float, k: int, alpha: float = 0.05, target_power: float = 0.80) -> int:
    """
    Minimum n to detect |PRCC| = rho at given power.
    Iterates from n = k+3 upward (exact non-central t, not the Fisher-z approx).
    """
    for n in range(k + 3, 50_001):
        if _prcc_power(n, k, rho, alpha) >= target_power:
            return n
    return 50_001   # effectively "infeasible"


def sample_size_report(k: int, current_n: int, out_dir: Path):
    """
    Print a power/sample-size summary for PRCC with k active input parameters,
    and save a power-curve plot.

    Parameters
    ----------
    k           : number of active (non-fixed) input parameters
    current_n   : the N_SAMPLES value you are currently using
    out_dir     : directory to save the figure
    """
    alpha = ALPHA
    power_levels = [0.70, 0.80, 0.90]
    rho_grid = np.arange(0.05, 0.55, 0.05)      # effect sizes to tabulate

    # ── Console table: required n vs effect size ────────────────
    print(f"\n── PRCC Sample Size Analysis ───────────────────────────")
    print(f"   k (active inputs) = {k}   α = {alpha}   (two-tailed)")
    print(f"   df = n − 2 − {k}  →  need n > {k + 2} at minimum")
    print()
    header = f"  {'|ρ_min|':>7}" + "".join(f"  {'pwr='+str(int(p*100))+'%':>9}" for p in power_levels)
    print(header)
    print("  " + "─" * (8 + 11 * len(power_levels)))
    for rho in rho_grid:
        row = f"  {rho:>7.2f}"
        for pw in power_levels:
            n_req = _required_n(rho, k - 1, alpha, pw)
            row += f"  {n_req:>9,}"
        print(row)

    # ── What can current_n detect? ──────────────────────────────
    print()
    print(f"  Your current N = {current_n:,}")
    for pw in power_levels:
        mdr = _min_detectable_rho(current_n, k - 1, alpha, pw)
        if math.isnan(mdr):
            msg = "indeterminate (n too close to k)"
        else:
            msg = f"|PRCC| ≥ {mdr:.3f}"
        print(f"    Power {int(pw*100)}% → detects {msg}")

    # ── Power curves: required n vs |ρ_min| ─────────────────────
    rho_fine = np.linspace(0.05, 0.50, 200)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4dac26", "#2166ac", "#d6604d"]
    for pw, color in zip(power_levels, colors):
        n_vals = [_required_n(r, k - 1, alpha, pw) for r in rho_fine]
        ax.plot(rho_fine, n_vals, color=color, linewidth=1.8, label=f"Power = {int(pw*100)}%")

    ax.axhline(current_n, color="black", linewidth=1.2, linestyle="--",
               label=f"Current N = {current_n:,}")

    # Annotate where current N intersects each power curve
    for pw, color in zip(power_levels, colors):
        mdr = _min_detectable_rho(current_n, k - 1, alpha, pw)
        if not math.isnan(mdr):
            ax.axvline(mdr, color=color, linewidth=0.7, linestyle=":")
            ax.text(mdr + 0.003, current_n * 1.03, f"{mdr:.2f}",
                    color=color, fontsize=8, va="bottom")

    ax.set_xlabel("|ρ_min|  (minimum detectable |PRCC|)", fontsize=10)
    ax.set_ylabel("Required sample size  n", fontsize=10)
    ax.set_title(
        f"PRCC Power Analysis  (k = {k} inputs, α = {alpha}, two-tailed)\n"
        f"Vertical dotted lines = min detectable |PRCC| at current N = {current_n:,}",
        fontsize=10,
    )
    ax.set_xlim(rho_fine[0], rho_fine[-1])
    ax.set_ylim(0, min(current_n * 4, 5000))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = out_dir / "power_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Power curve saved: {path.name}")

    # ── Heatmap: power(n, rho) ───────────────────────────────────
    rho_heat  = np.round(np.arange(0.05, 0.50, 0.05), 2)
    n_heat    = [100, 200, 300, 500, 750, 1000, 1500, 2000]
    power_mat = np.array(
        [[_prcc_power(n, k - 1, r, alpha) for r in rho_heat] for n in n_heat]
    )

    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
    im = ax2.imshow(power_mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax2, label="Power (1 − β)")
    ax2.set_xticks(range(len(rho_heat)))
    ax2.set_xticklabels([f"{r:.2f}" for r in rho_heat], fontsize=9)
    ax2.set_yticks(range(len(n_heat)))
    ax2.set_yticklabels([str(n) for n in n_heat], fontsize=9)
    ax2.set_xlabel("|ρ_min|  (true PRCC effect size)", fontsize=10)
    ax2.set_ylabel("Sample size  n", fontsize=10)
    # Annotate each cell
    for i, n in enumerate(n_heat):
        for j, r in enumerate(rho_heat):
            pwr = power_mat[i, j]
            txt_color = "black" if 0.2 < pwr < 0.85 else "white"
            ax2.text(j, i, f"{pwr:.2f}", ha="center", va="center",
                     fontsize=8, color=txt_color)
    # Highlight current N row if it's in the list
    if current_n in n_heat:
        idx = n_heat.index(current_n)
        ax2.add_patch(mpatches.Rectangle(
            (-0.5, idx - 0.5), len(rho_heat), 1,
            fill=False, edgecolor="black", linewidth=2,
        ))
    ax2.set_title(
        f"Power Heatmap  (k = {k} inputs, α = {alpha})\n"
        f"Each cell = P(detect |PRCC| ≥ ρ_min at sample size n)",
        fontsize=10,
    )
    fig2.tight_layout()
    path2 = out_dir / "power_heatmap.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Power heatmap saved: {path2.name}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_sensitivity():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(exist_ok=True)

    print("\n── Loading data ────────────────────────────────────────")
    X, Y = load_data()
    X = drop_constant_cols(X)
    print(f"  Active inputs : {X.shape[1]}")
    print(f"  Outputs found : {Y.columns.tolist()}")
    print(f"  Runs (rows)   : {len(X)}")

    sample_size_report(k=X.shape[1], current_n=len(X), out_dir=out_dir)

    all_prcc: dict = {}
    all_srcc: dict = {}

    print("\n── Computing PRCC & SRCC ───────────────────────────────")
    for out_name in Y.columns:
        y = Y[out_name].dropna()
        X_aligned = X.loc[y.index]
        print(f"  {out_name} ({len(y)} valid runs) ...", end=" ", flush=True)
        all_prcc[out_name] = compute_prcc(X_aligned, y)
        all_srcc[out_name] = compute_srcc(X_aligned, y)
        n_sig = all_prcc[out_name]["significant"].sum()
        print(f"{n_sig}/{X.shape[1]} params significant")

    # ── Save result tables ──────────────────────────────────────
    prcc_wide = pd.concat({k: v["prcc"]    for k, v in all_prcc.items()}, axis=1)
    pval_wide = pd.concat({k: v["p_value"] for k, v in all_prcc.items()}, axis=1)
    srcc_wide = pd.concat({k: v["srcc"]    for k, v in all_srcc.items()}, axis=1)

    prcc_wide.to_csv(out_dir / "prcc_table.csv")
    pval_wide.to_csv(out_dir / "prcc_pvalues.csv")
    srcc_wide.to_csv(out_dir / "srcc_table.csv")
    print(f"\n  Tables saved to '{OUTPUT_DIR}/'")

    # ── Print console summary ───────────────────────────────────
    print(f"\n── Top {min(TOP_N, X.shape[1])} parameters by |PRCC| ──────────────────────────")
    for out_name, prcc_df in all_prcc.items():
        top = prcc_df.reindex(
            prcc_df["prcc"].abs().sort_values(ascending=False).index
        ).head(10)
        print(f"\n  {_out_label(out_name)}:")
        print(f"    {'*':1}  {'Parameter':<44} {'PRCC':>7}  {'p-value':>9}")
        print(f"    {'─' * 65}")
        for param, row in top.iterrows():
            flag = "*" if row["significant"] else " "
            print(f"    {flag}  {_snake(param):<44} {row['prcc']:+.4f}  {row['p_value']:.3g}")

    # ── Plots ───────────────────────────────────────────────────
    print("\n── Generating plots ────────────────────────────────────")
    for out_name, prcc_df in all_prcc.items():
        tornado_plot(prcc_df, out_name, out_dir)

    heatmap_plot(all_prcc, out_dir)
    scatter_top_params(X, Y, all_prcc, out_dir, n_top=3)
    prcc_vs_srcc_plot(all_prcc, all_srcc, out_dir)

    print(f"\nDone. All results saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    run_sensitivity()
