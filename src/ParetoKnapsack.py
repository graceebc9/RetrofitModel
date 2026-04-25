"""
src/ParetoKnapsack.py
=====================
Multi-choice knapsack with ε-constraint on equity spend.

For each building, pick at most one intervention package to maximise
total CO2 abatement subject to:
  - total spend <= budget
  - at least X% of spend goes to high-equity personas (ε-constraint)

The equity floor is swept to trace the Pareto front between abatement
and equity.

Changes vs. previous version
----------------------------
- Shared Gurobi environment across calls (one license check, not N).
- Bulk `addConstrs` for the per-building max-one constraint
  (materially faster model build at scale).
- `OutputFlag=0` by default — rely on the logger, not Gurobi stdout.
- `carbon_col > 0` filter applied pre-solve (was post-solve only).
- `sys.exit()` replaced with a proper exception.
- Typo fix: `percntile_col` -> `percentile_col`.
- Thread count respects SLURM if set.
- Returns `float('nan')` for cpex when undefined (not `None`) so
  downstream formatting is safer.
- Unknown-persona rows now raise rather than silently being dropped
  from the breakdown while still counting in totals.
- `DEFAULT_HIGH_EQUITY_PERSONAS` is now a `frozenset` (immutable).
- `pareto_sweep` kept and documented as a convenience; the main
  pipeline is expected to use its own sweep loop for richer per-step
  bookkeeping.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB


# ---------------------------------------------------------------------------
# DEFAULTS
# ---------------------------------------------------------------------------

DEFAULT_HIGH_EQUITY_PERSONAS: frozenset[str] = frozenset({"high_risk", "med_risk"})
ALL_PERSONAS: tuple[str, ...] = (
    "high_risk", "med_risk", "middle_risk", "low_risk", "v_low_risk",
)


# ---------------------------------------------------------------------------
# SHARED GUROBI ENVIRONMENT
# ---------------------------------------------------------------------------
# Creating a Gurobi environment takes a license check + thread pool init
# which is non-trivial overhead when sweeping many equity floors. We
# share one environment across calls within a single process.
#
# On HPC with per-process licenses this is ideal. With a floating token
# license, set RETROFIT_GUROBI_FRESH_ENV=Y to force a new env per call.

_GUROBI_ENV: Optional[gp.Env] = None
_FRESH_ENV_PER_CALL = os.getenv("RETROFIT_GUROBI_FRESH_ENV", "N").upper() == "Y"


def _get_gurobi_env(output_flag: int = 0) -> gp.Env:
    """Return a shared Gurobi environment (or a fresh one if requested)."""
    global _GUROBI_ENV
    if _FRESH_ENV_PER_CALL:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", output_flag)
        env.start()
        return env

    if _GUROBI_ENV is None:
        _GUROBI_ENV = gp.Env(empty=True)
        _GUROBI_ENV.setParam("OutputFlag", output_flag)
        _GUROBI_ENV.start()
    return _GUROBI_ENV


def dispose_shared_env() -> None:
    """Release the shared Gurobi env. Call at end of pipeline if desired."""
    global _GUROBI_ENV
    if _GUROBI_ENV is not None:
        _GUROBI_ENV.dispose()
        _GUROBI_ENV = None


def _resolve_thread_count() -> int:
    """
    Thread count for Gurobi. Respect SLURM's CPUS_PER_TASK so we don't
    oversubscribe a shared node. 0 = Gurobi picks (usually all cores).
    """
    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass
    return 0


# ---------------------------------------------------------------------------
# SOLVER
# ---------------------------------------------------------------------------

def multichoice_knapsack(
    df_all_packages: pd.DataFrame,
    budget: float,
        mip_gap: float ,
    equity_floor_pct: float = 0.0,
    high_equity_personas: Optional[Set[str]] = None,
    upn_col: str = "upn",
    persona_col: str = "meta_socio_persona",
    cost_col: str = "mean_total_capex",
    carbon_col: str = "mean_total_co2_saved",
    intervention_col: str = "intervention",
    percentile_col: str = "avg_gas_percentile",
    time_limit_seconds: int = 1200,

    gurobi_output_flag: int = 0,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Multi-choice knapsack: pick at most one package per building to
    maximise total CO2 abatement, subject to budget and optional equity
    spend floor.

    Parameters
    ----------
    df_all_packages : DataFrame
        One row per (building, candidate intervention package).
    budget : float
        Total spend cap (£).
    equity_floor_pct : float
        Minimum % of total spend that must go to `high_equity_personas`.
        0 = no floor. Expressed as a percent (e.g. 25, not 0.25).
    high_equity_personas : set[str] or None
        Personas that count toward the equity floor.
    mip_gap : float
        Relative MIP gap. 0.01 = 1% (fast, good for sweeps).
        Tighten to 0.001 for headline numbers.
    gurobi_output_flag : int
        0 = silent (default), 1 = Gurobi prints to stdout.

    Returns
    -------
    selected_df : DataFrame
        Rows selected by the solver, with carbon > 0.
    stats : dict
        Summary statistics (status, totals, breakdowns).
    """
    if high_equity_personas is None:
        high_equity_personas = DEFAULT_HIGH_EQUITY_PERSONAS

    t0 = time.time()

    # --- Preprocessing ---
    n_buildings_input = df_all_packages[upn_col].nunique()

    df = df_all_packages.copy().reset_index(drop=True)
    # Filter strictly at the start: carbon > 0 (zero-carbon packages
    # can never help an abatement-maximising objective) and cost fits.
    df = df[(df[cost_col] <= budget) & (df[carbon_col] > 0)]
    df = df.reset_index(drop=True)
    df["_is_high_eq"] = df[persona_col].isin(high_equity_personas).astype(int)

    n_buildings_feasible = df[upn_col].nunique()

    if logger:
        logger.info(
            f"Solve: {len(df):,} rows | "
            f"{n_buildings_feasible:,} / {n_buildings_input:,} buildings feasible | "
            f"budget=£{budget:,.0f} | equity_spend_floor={equity_floor_pct:.0f}%"
        )

    if df.empty:
        return df_all_packages.iloc[:0].copy(), {
            "status": "empty",
            "equity_floor_pct": equity_floor_pct,
            "n_retrofitted": 0,
            "total_cost": 0,
            "total_abatement": 0,
            "cpex_per_ton": float("nan"),
            "high_eq_spend_pct": 0,
            "high_eq_abatement_pct": 0,
            "solve_time_s": round(time.time() - t0, 2),
            "persona_breakdown": {},
            "intervention_breakdown": {},
            "percentile_breakdown": {},
        }

    # --- Build ILP ---
    env = _get_gurobi_env(output_flag=gurobi_output_flag)
    model = gp.Model("retrofit_multichoice", env=env)
    model.Params.OutputFlag = gurobi_output_flag
    model.Params.TimeLimit = time_limit_seconds
    model.Params.MIPGap = mip_gap
    model.Params.Threads = _resolve_thread_count()

    n = len(df)

    # Positional indexing throughout — df is reset_index'd above so
    # positional i matches df.iloc[i].
    x = model.addVars(n, vtype=GRB.BINARY, name="sel")

    carbon = df[carbon_col].to_numpy()
    costs = df[cost_col].to_numpy()
    is_high = df["_is_high_eq"].to_numpy()

    # OBJECTIVE: maximise total CO2 abatement
    model.setObjective(
        gp.quicksum(carbon[i] * x[i] for i in range(n)),
        GRB.MAXIMIZE,
    )

    # C1: total spend <= budget
    model.addConstr(
        gp.quicksum(costs[i] * x[i] for i in range(n)) <= budget,
        name="budget_limit",
    )
    # C2: at most one package per building.
    # groupby().indices gives positional indices per building up-front,
    # which is faster than re-groupby'ing inside the loop. We still need
    # a Python-level loop because gp.addConstrs doesn't accept unhashable
    # (numpy-array) keys in its generator form.
    group_idx = df.groupby(upn_col).indices  # dict[upn, np.ndarray[int]]
    for upn, idxs in group_idx.items():
        if len(idxs) == 1:
            # Trivially satisfied — a single binary is always <= 1.
            continue
        model.addConstr(
            gp.quicksum(x[int(i)] for i in idxs) <= 1,
            name=f"max_one_{upn}",
        )
    # C3: equity floor on spend (ε-constraint form)
    #   sum_i (is_high[i] - frac) * cost[i] * x[i] >= 0
    #   <=> sum_high cost*x  >=  frac * sum_all cost*x
    if equity_floor_pct > 0:
        frac = equity_floor_pct / 100.0
        model.addConstr(
            gp.quicksum(
                (is_high[i] - frac) * costs[i] * x[i] for i in range(n)
            ) >= 0,
            name="equity_floor",
        )

    # --- Solve ---
    model.optimize()
    solve_time = time.time() - t0

    # Map Gurobi status to PuLP-style strings for downstream compatibility.
    if model.Status == GRB.OPTIMAL or (
        model.Status == GRB.TIME_LIMIT and model.SolCount > 0
    ):
        status_str = "Optimal"
    elif model.Status == GRB.INFEASIBLE:
        status_str = "Infeasible"
    else:
        status_str = "Not Solved"

    if logger:
        logger.info(f"  Status: {status_str} ({solve_time:.1f}s)")

    # --- Extract selected rows ---
    if model.SolCount > 0:
        sel_vals = np.array([x[i].X for i in range(n)])
        selected_mask = sel_vals > 0.5
        selected_df = df.loc[selected_mask].copy()
    else:
        selected_df = df.iloc[:0].copy()

    selected_df = selected_df.drop(columns=["_is_high_eq"], errors="ignore")

    # --- Compute stats ---
    stats = _build_stats(
        selected_df=selected_df,
        equity_floor_pct=equity_floor_pct,
        status_str=status_str,
        solve_time=solve_time,
        budget=budget,
        high_equity_personas=high_equity_personas,
        persona_col=persona_col,
        cost_col=cost_col,
        carbon_col=carbon_col,
        intervention_col=intervention_col,
        percentile_col=percentile_col,
        logger=logger,
    )

    # Dispose the model; keep the shared env alive for the next call.
    model.dispose()
    if _FRESH_ENV_PER_CALL:
        env.dispose()

    return selected_df, stats


# ---------------------------------------------------------------------------
# STATS BUILDER
# ---------------------------------------------------------------------------

def _build_stats(
    selected_df: pd.DataFrame,
    equity_floor_pct: float,
    status_str: str,
    solve_time: float,
    budget: float,
    high_equity_personas: Set[str],
    persona_col: str,
    cost_col: str,
    carbon_col: str,
    intervention_col: str,
    percentile_col: str,
    logger: Optional[logging.Logger],
) -> Dict:
    """Compute summary statistics for a solved knapsack instance."""
    empty = selected_df.empty

    total_cost = float(selected_df[cost_col].sum()) if not empty else 0.0
    total_carbon = float(selected_df[carbon_col].sum()) if not empty else 0.0

    if empty:
        high_eq_mask = pd.Series(dtype=bool)
    else:
        high_eq_mask = selected_df[persona_col].isin(high_equity_personas)

    high_eq_spend = (
        float(selected_df.loc[high_eq_mask, cost_col].sum()) if not empty else 0.0
    )
    high_eq_carbon = (
        float(selected_df.loc[high_eq_mask, carbon_col].sum()) if not empty else 0.0
    )

    # --- Persona breakdown ---
    # Guard against unexpected personas leaking past validation; totals
    # and the breakdown must agree.
    persona_breakdown: Dict[str, Dict] = {}
    if not empty:
        observed = set(selected_df[persona_col].dropna().unique())
        unknown = observed - set(ALL_PERSONAS)
        if unknown:
            raise ValueError(
                f"Unknown personas in selection: {sorted(unknown)}. "
                f"Expected one of {ALL_PERSONAS}."
            )
        for p in ALL_PERSONAS:
            pmask = selected_df[persona_col] == p
            persona_breakdown[p] = {
                "buildings": int(pmask.sum()),
                "spend": float(selected_df.loc[pmask, cost_col].sum()),
                "abatement": float(selected_df.loc[pmask, carbon_col].sum()),
            }

    # --- Intervention breakdown ---
    intervention_breakdown: Dict[str, Dict] = {}
    if not empty and intervention_col in selected_df.columns:
        for intv, grp in selected_df.groupby(intervention_col):
            intervention_breakdown[intv] = {
                "buildings": int(len(grp)),
                "spend": float(grp[cost_col].sum()),
                "abatement": float(grp[carbon_col].sum()),
            }

    # --- Percentile/decile breakdown ---
    percentile_breakdown: Dict = {}
    if not empty:
        if percentile_col not in selected_df.columns:
            raise KeyError(
                f"Expected column '{percentile_col}' not in selected_df. "
                f"Columns available: {selected_df.columns.tolist()}"
            )
        for pctl, grp in selected_df.groupby(percentile_col):
            # Preserve original key type (int/float/str) — downstream
            # plotting indexes by decile 1..10.
            key = pctl.item() if hasattr(pctl, "item") else pctl
            percentile_breakdown[key] = {
                "buildings": int(len(grp)),
                "spend": float(grp[cost_col].sum()),
                "abatement": float(grp[carbon_col].sum()),
            }

    cpex_per_ton = (total_cost / total_carbon) if total_carbon > 0 else float("nan")
    high_eq_spend_pct = (100 * high_eq_spend / total_cost) if total_cost > 0 else 0.0
    high_eq_abatement_pct = (
        100 * high_eq_carbon / total_carbon if total_carbon > 0 else 0.0
    )

    stats = {
        "status": status_str,
        "solve_time_s": round(solve_time, 2),
        "equity_floor_pct": equity_floor_pct,
        "n_retrofitted": int(len(selected_df)),
        "n_high_equity": int(high_eq_mask.sum()) if not empty else 0,
        "total_cost": total_cost,
        "remaining_budget": budget - total_cost,
        "total_abatement": total_carbon,
        "cpex_per_ton": cpex_per_ton,
        "high_eq_spend": high_eq_spend,
        "high_eq_spend_pct": high_eq_spend_pct,
        "high_eq_abatement": high_eq_carbon,
        "high_eq_abatement_pct": high_eq_abatement_pct,
        "persona_breakdown": persona_breakdown,
        "intervention_breakdown": intervention_breakdown,
        "percentile_breakdown": percentile_breakdown,
    }

    if logger and status_str == "Optimal":
        cpex_display = (
            f"£{cpex_per_ton:,.0f}/t" if np.isfinite(cpex_per_ton) else "n/a"
        )
        logger.info(
            f"  {stats['n_retrofitted']:,} bldgs | "
            f"£{total_cost:,.0f} | {total_carbon:,.1f} tCO2 | "
            f"{cpex_display} | "
            f"high-eq spend: {high_eq_spend_pct:.1f}%"
        )

    return stats


# ---------------------------------------------------------------------------
# PARETO SWEEP (convenience wrapper)
# ---------------------------------------------------------------------------

# def pareto_sweep(
#     df_all_packages: pd.DataFrame,
#     budget: float,
#     mip_gap: float, 
#     equity_floors: Optional[List[float]] = None,
#     high_equity_personas: Optional[Set[str]] = None,
#     upn_col: str = "upn",
#     persona_col: str = "meta_socio_persona",
#     cost_col: str = "mean_total_capex",
#     carbon_col: str = "mean_total_co2_saved",
#     time_limit_seconds: int = 1200,
    
#     logger: Optional[logging.Logger] = None,
# ) -> Tuple[pd.DataFrame, List[Dict]]:
#     """
#     Sweep equity floor from 0% upward to trace the Pareto front.

#     Stops early if a solve comes back infeasible — higher floors will
#     also be infeasible.

#     The main pipeline typically implements its own sweep so it can
#     save per-step outputs; this wrapper is useful for notebooks and
#     one-off analyses.
#     """
#     if equity_floors is None:
#         equity_floors = list(range(0, 105, 5))

#     all_stats: List[Dict] = []
#     for eps in equity_floors:
#         if logger:
#             logger.info(f"\n{'=' * 50}")

#         _, stats = multichoice_knapsack(
#             df_all_packages=df_all_packages,
#             budget=budget,
#             equity_floor_pct=eps,
#             high_equity_personas=high_equity_personas,
#             upn_col=upn_col,
#             persona_col=persona_col,
#             cost_col=cost_col,
#             carbon_col=carbon_col,
#             time_limit_seconds=time_limit_seconds,
#             mip_gap=mip_gap,
#             logger=logger,
#         )
#         all_stats.append(stats)

#         if stats["status"] not in ("Optimal", "Not Solved"):
#             if logger:
#                 logger.info(f"  Infeasible at {eps}% — stopping sweep.")
#             break

#     return pd.DataFrame(all_stats), all_stats


# ---------------------------------------------------------------------------
# BASELINE: greedy pre-selection for comparison
# ---------------------------------------------------------------------------
def preselect_best_cpt(
    df_all_packages: pd.DataFrame,
    upn_col: str = "upn",
    cost_col: str = "mean_total_capex",
    carbon_col: str = "mean_total_co2_saved",
    score_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    For each building, keep only the single package with the lowest
    score. Default score is mean £/tCO2 (cost_col / carbon_col); if
    `score_col` is provided, rank by that column directly (used to pass
    the aleatoric-penalised cost-per-tonne from preprocessing).
    """
    df = df_all_packages[df_all_packages[carbon_col] > 0].copy()
    if score_col is not None:
        if score_col not in df.columns:
            raise KeyError(
                f"score_col='{score_col}' not in dataframe; "
                f"available: {df.columns.tolist()[:20]}..."
            )
        rank_col = score_col
    else:
        df["_cpt"] = df[cost_col] / df[carbon_col]
        rank_col = "_cpt"
    best_idx = df.groupby(upn_col)[rank_col].idxmin()
    out = df.loc[best_idx]
    return out.drop(columns=["_cpt"], errors="ignore")