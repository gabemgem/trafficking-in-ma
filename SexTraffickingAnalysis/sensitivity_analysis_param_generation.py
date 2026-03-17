"""
Parameter set generation for system dynamics sensitivity analysis.

Supports two sampling modes:

  LHS  — Latin Hypercube Sampling: N quasi-random samples covering the full
          parameter space simultaneously. Use for PRCC/SRCC sensitivity analysis.

  OAT  — One-At-a-Time: vary each parameter across its range while holding all
          others at their defaults. Use for response-curve plots and communicating
          individual parameter effects.

HOW TO CUSTOMIZE:
  - Add/remove parameters in the PARAMETERS dict below
  - Set bounds: [min, max] to specify a range manually
  - Leave bounds as None to auto-compute +/-DEFAULT_RANGE_PCT around the default
  - Set "fixed": True to hold a parameter constant across all runs
  - NOTE: params with default=0 AND bounds=None cannot be auto-sampled

COMMAND LINE:
  python sensitivity_analysis_param_generation.py
      LHS mode, default output file (lhs_samples.xlsx)

  python sensitivity_analysis_param_generation.py --mode oat
      OAT mode, default output file (oat_samples.xlsx), 10 steps per parameter

  python sensitivity_analysis_param_generation.py --mode oat --oat-steps 20
      OAT mode with 20 evenly-spaced values per parameter

  python sensitivity_analysis_param_generation.py --auto-bounds
      LHS mode, ignore explicit bounds and auto-compute ±DEFAULT_RANGE_PCT for all

  python sensitivity_analysis_param_generation.py -o my_file.xlsx
      Write output to a custom filename
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import qmc

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
N_SAMPLES        = 5000          # LHS: number of samples to generate
DEFAULT_RANGE_PCT = 0.20         # +/- 20% of default when bounds=None
OAT_STEPS        = 10            # OAT: evenly-spaced steps per parameter (default)

LHS_OUTPUT_FILE  = "lhs_samples.xlsx"
OAT_OUTPUT_FILE  = "oat_samples.xlsx"

RANDOM_SEED = 42

# ─────────────────────────────────────────────
# PARAMETERS
#
# Format: "param_name": {"default": <value>, "bounds": [min, max]}
#
# bounds: [min, max]  → use this exact range
#         None        → auto-compute +/-DEFAULT_RANGE_PCT around default
#                       (params with default=0 AND no bounds will be treated as fixed)
# fixed:  True        → hold at default, exclude from sampling
#
# ADD a parameter:    insert a new line anywhere in the dict
# REMOVE a parameter: delete or comment out (#) its line
# ─────────────────────────────────────────────
PARAMETERS = {

    # ── Recruitment & Crime ──────────────────────────────────────────────────
    "st_recruitment":                       {"default": 0.1,       "bounds": [0,1]},
    "crim_record_obstacle":                 {"default": 0.6,       "bounds": [0,1]},
    "org_crime_prevalence":                 {"default": 0.1,       "bounds": [0,1]},
    "trafficker_exposure":                  {"default": 0.05,      "bounds": [0,1]},
    "trafficking_educ":                     {"default": 0.5,       "bounds": [0,1]},

    # ── Population ───────────────────────────────────────────────────────────
    "total_adult_population":               {"default": 5656000.0, "bounds": None, "fixed": True},
    "total_youth_population":               {"default": 1344000.0, "bounds": None, "fixed": True},
    "pop_pct_adult_homeless":               {"default": 0.0048,    "bounds": None, "fixed": True},
    "pop_pct_adult_poverty":                {"default": 0.104,     "bounds": None, "fixed": True},
    "pop_pct_adult_lack_support_network":   {"default": 0.2,       "bounds": None, "fixed": True},
    "pop_pct_youth_homeless":               {"default": 0.00166,   "bounds": None, "fixed": True},
    "pop_pct_youth_poverty":                {"default": 0.115,     "bounds": None, "fixed": True},
    "pop_pct_youth_lack_support_network":   {"default": 0.1,       "bounds": None, "fixed": True},
    "trafficked_pct_youth":                 {"default": 0.25,      "bounds": None},
    "childhood_violence":                   {"default": 0.2,       "bounds": None},
    "geographic_csw_normalization":         {"default": 0.1,       "bounds": None},

    # ── Prosecution & Policy ─────────────────────────────────────────────────
    "prosecution_culture":                  {"default": 0.05,      "bounds": [0,0.5]},
    "prosecution_culture_v":                {"default": 0.2,       "bounds": [0,0.5]},
    "prosecution_culture_s":                {"default": 0.1,       "bounds": [0,0.5]},
    "prosecution_culture_youth":            {"default": 0.0,       "bounds": [0.0, 0.2]},
    "prosecution_culture_youth_v":          {"default": 0.1,       "bounds": [0,0.5]},
    "prosecution_culture_youth_s":          {"default": 0.05,      "bounds": [0,0.5]},
    "buyer_prosecution_culture":            {"default": 0.01,      "bounds": [0,0.2]},
    "prosecution_mult":                     {"default": 0.5,       "bounds": [0,1]},
    "pctExpungement":                       {"default": 0.0,       "bounds": [0.0, 1]},

    # ── Services — Adult ─────────────────────────────────────────────────────
    "em_services_capacity":                 {"default": 2000.0,    "bounds": [1000, 10000]},
    "st_services_capacity":                 {"default": 250.0,     "bounds": [100, 1000]},
    "lt_services_capacity":                 {"default": 100.0,     "bounds": [50,500]},
    "em_services_duration":                 {"default": 1.0,       "bounds": None, "fixed": True},
    "st_services_duration":                 {"default": 12.0,      "bounds": None, "fixed": True},
    "lt_services_duration":                 {"default": 24.0,      "bounds": None, "fixed": True},
    "em_services_recovery_rate":            {"default": 0.85,      "bounds": [0,1]},
    "st_services_recovery_rate":            {"default": 0.65,      "bounds": [0,1]},
    "lt_services_recovery_rate":            {"default": 0.5,       "bounds": [0,1]},

    # ── Services — Youth ─────────────────────────────────────────────────────
    "em_services_capacity_youth":           {"default": 2000.0,    "bounds": [1000,10000]},
    "st_services_capacity_youth":           {"default": 500.0,     "bounds": [250,2500]},
    "lt_services_capacity_youth":           {"default": 200.0,     "bounds": [100,1000]},
    "em_services_duration_youth":           {"default": 1.0,       "bounds": None, "fixed": True},
    "st_services_duration_youth":           {"default": 12.0,      "bounds": None, "fixed": True},
    "lt_services_duration_youth":           {"default": 24.0,      "bounds": None, "fixed": True},
    "em_services_recovery_rate_youth":      {"default": 0.85,      "bounds": [0,1]},
    "st_services_recovery_rate_youth":      {"default": 0.65,      "bounds": [0,1]},
    "lt_services_recovery_rate_youth":      {"default": 0.5,       "bounds": [0,1]},

    # ── Recovery & Recidivism — Adult ────────────────────────────────────────
    "base_adult_recovery_rate":             {"default": 0.1,       "bounds": [0,1]},
    "post_service_vulnerability":           {"default": 0.1,       "bounds": [0,1]},
    "p_AdultExiting_to_LTServices":         {"default": 0.05,      "bounds": None, "fixed": True},
    "p_AdultExiting_to_STServices":         {"default": 0.1,       "bounds": None, "fixed": True},
    "p_AdultExiting_to_EmServices":         {"default": 0.75,      "bounds": None, "fixed": True},
    "p_non_trafficked_recidivism":          {"default": 0.25,      "bounds": None, "fixed": True},
    "vulnerable_recidivism":                {"default": 0.1,       "bounds": [0,0.5]},

    # ── Recovery & Recidivism — Youth ────────────────────────────────────────
    "base_youth_recovery_rate":             {"default": 0.1,       "bounds": [0,1]},
    "post_service_vulnerability_youth":     {"default": 0.1,       "bounds": [0,1]},
    "p_YouthExiting_to_LTServices":         {"default": 0.05,      "bounds": None, "fixed": True},
    "p_YouthExiting_to_STServices":         {"default": 0.1,       "bounds": None, "fixed": True},
    "p_YouthExiting_to_EmServices":         {"default": 0.75,      "bounds": None, "fixed": True},
    "p_non_trafficked_recidivism_youth":    {"default": 0.0,       "bounds": None, "fixed": True},
    "vulnerable_recidivism_youth":          {"default": 0.25,      "bounds": [0,0.5]},

    # ── Supply/Demand & Economics ────────────────────────────────────────────
    "csw_acceptance":                       {"default": 0.5,       "bounds": [0,1]},
    "csw_exhaustion":                       {"default": 0.2,       "bounds": [0,1]},
    "sb_benefit":                           {"default": 200.0,     "bounds": [1,1000]},
    "sb_base_rate":                         {"default": 23000.0,   "bounds": None, "fixed": True},
    "ss_trns_pm":                           {"default": 80.0,      "bounds": [60,100]},
    "ss_desired_profit":                    {"default": 4000.0,    "bounds": [3000,5000]},
    "trafficked_to_nontrafficked_pct":      {"default": 0.01,      "bounds": [0,0.1]},
    "risk_cross_effect":                    {"default": 0.1,       "bounds": [0,1]},
    "sb_trns_risk_weight":                  {"default": 3.0,       "bounds": [1, 10]},
    "buyer_base_trns":                      {"default": 10.0,      "bounds": [1,20]},
}


# ─────────────────────────────────────────────
# SAMPLING LOGIC — no need to edit below
# ─────────────────────────────────────────────

def resolve_bounds(name, spec, auto_bounds=False):
    """
    Return [lo, hi] for a parameter, or None if it should be treated as fixed.

    Priority:
      1. auto_bounds=True  → always compute ±DEFAULT_RANGE_PCT (error if default=0)
      2. explicit bounds   → use as-is
      3. bounds=None       → compute ±DEFAULT_RANGE_PCT (error if default=0)
    """
    default = spec["default"]
    bounds  = spec.get("bounds")

    if auto_bounds:
        if default == 0.0:
            print(f"  WARNING: '{name}' default=0 — cannot auto-compute bounds, treating as fixed.")
            return None
        return [default * (1 - DEFAULT_RANGE_PCT), default * (1 + DEFAULT_RANGE_PCT)]

    if bounds is not None:
        return bounds

    if default == 0.0:
        print(f"  WARNING: '{name}' default=0 and no bounds — treating as fixed. "
              "Add bounds=[min, max] to include in sampling.")
        return None

    return [default * (1 - DEFAULT_RANGE_PCT), default * (1 + DEFAULT_RANGE_PCT)]


def _partition_params(auto_bounds=False):
    """
    Returns (fixed_params, sampled_params) where:
      fixed_params   : {name: default_value}
      sampled_params : {name: [lo, hi]}
    """
    fixed   = {}
    sampled = {}
    for name, spec in PARAMETERS.items():
        if spec.get("fixed", False):
            fixed[name] = spec["default"]
            continue
        bounds = resolve_bounds(name, spec, auto_bounds)
        if bounds is None:
            fixed[name] = spec["default"]
        else:
            sampled[name] = bounds
    return fixed, sampled


def _save(df, output_file, sheet_name):
    df.index.name = "run_id"
    df.to_excel(output_file, sheet_name=sheet_name)
    print(f"\nSaved -> '{output_file}'")


def _print_bounds_table(sampled_params):
    print(f"\n{'Parameter':<45} {'Lo':>12}  {'Default':>12}  {'Hi':>12}")
    print("-" * 87)
    for name, (lo, hi) in sampled_params.items():
        print(f"{name:<45} {lo:>12.4g}  {PARAMETERS[name]['default']:>12.4g}  {hi:>12.4g}")


# ─────────────────────────────────────────────
# LHS
# ─────────────────────────────────────────────

def run_lhs(auto_bounds=False, output_file=None):
    """Generate N_SAMPLES Latin Hypercube parameter sets."""
    fixed, sampled = _partition_params(auto_bounds)

    param_names = list(sampled.keys())
    lo = np.array([sampled[p][0] for p in param_names])
    hi = np.array([sampled[p][1] for p in param_names])

    bounds_mode = (
        f"auto (±{DEFAULT_RANGE_PCT:.0%}) for all" if auto_bounds
        else f"explicit where specified, auto (±{DEFAULT_RANGE_PCT:.0%}) otherwise"
    )
    print(f"\nMode        : LHS")
    print(f"Bounds mode : {bounds_mode}")
    print(f"Sampling    : {len(param_names)} parameters, {N_SAMPLES} runs")
    if fixed:
        print(f"Fixed params: {list(fixed.keys())}")

    sampler = qmc.LatinHypercube(d=len(param_names), seed=RANDOM_SEED)
    samples = qmc.scale(sampler.random(n=N_SAMPLES), lo, hi)

    df = pd.DataFrame(samples, columns=param_names)
    for name, val in fixed.items():
        df[name] = val
    df = df[list(PARAMETERS.keys())]

    out = output_file or LHS_OUTPUT_FILE
    _save(df, out, sheet_name="lhs_samples")
    _print_bounds_table(sampled)
    return df


# ─────────────────────────────────────────────
# OAT
# ─────────────────────────────────────────────

def run_oat(oat_steps=OAT_STEPS, auto_bounds=False, output_file=None):
    """
    Generate One-At-a-Time parameter sets.

    Output structure:
      Row 0           : baseline — all parameters at their defaults
      Rows 1..k*steps : for each sampled parameter, oat_steps rows where that
                        parameter sweeps from lo to hi; all others at default

    A 'varied_param' column records which parameter was swept in each row
    ('baseline' for row 0). This column is non-numeric and is automatically
    ignored by sensitivity_analysis.py's input detection.
    """
    fixed, sampled = _partition_params(auto_bounds)

    defaults = {name: spec["default"] for name, spec in PARAMETERS.items()}
    all_cols  = list(PARAMETERS.keys())

    print(f"\nMode        : OAT")
    print(f"Steps/param : {oat_steps}")
    print(f"Parameters  : {len(sampled)}")
    print(f"Total runs  : 1 baseline + {len(sampled)} × {oat_steps} = "
          f"{1 + len(sampled) * oat_steps}")

    rows        = []
    varied_tags = []

    # Baseline row
    rows.append({name: defaults[name] for name in all_cols})
    varied_tags.append("baseline")

    # One sweep per sampled parameter
    for param_name, (lo, hi) in sampled.items():
        for value in np.linspace(lo, hi, oat_steps):
            row = {name: defaults[name] for name in all_cols}
            row[param_name] = value
            rows.append(row)
            varied_tags.append(param_name)

    df = pd.DataFrame(rows, columns=all_cols)
    df.insert(0, "varied_param", varied_tags)

    out = output_file or OAT_OUTPUT_FILE
    _save(df, out, sheet_name="oat_params")
    _print_bounds_table(sampled)
    return df


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate LHS or OAT parameter sets for sensitivity analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["lhs", "oat"],
        default="lhs",
        help="Sampling mode: 'lhs' (default) or 'oat'.",
    )
    parser.add_argument(
        "--auto-bounds",
        action="store_true",
        help=(
            f"Ignore all explicit bounds and auto-compute "
            f"±{DEFAULT_RANGE_PCT:.0%} around each default."
        ),
    )
    parser.add_argument(
        "--oat-steps",
        type=int,
        default=OAT_STEPS,
        metavar="N",
        help=f"OAT mode: number of evenly-spaced values per parameter (default: {OAT_STEPS}).",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        metavar="FILE",
        help=(
            f"Output .xlsx filename "
            f"(default: {LHS_OUTPUT_FILE} for LHS, {OAT_OUTPUT_FILE} for OAT)."
        ),
    )

    args = parser.parse_args()

    if args.mode == "lhs":
        run_lhs(auto_bounds=args.auto_bounds, output_file=args.output)
    else:
        run_oat(oat_steps=args.oat_steps, auto_bounds=args.auto_bounds, output_file=args.output)
