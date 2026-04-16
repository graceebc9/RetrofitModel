"""
src/ParetoKnapsack.py
=====================
Multi-choice knapsack with ε-constraint on equity spend.

Replaces the old approach of:
  1. Pre-selecting best £/tCO2 package per building
  2. Distorting £/tCO2 with equity_factor weighting
  3. Running single-objective knapsack

New approach:
  1. Feed ALL packages per building to the solver
  2. Solver jointly picks buildings AND packages
  3. Constraint: at least X% of spend goes to high-equity personas
  4. Sweep X to trace the Pareto front

Usage:
    from src.ParetoKnapsack import multichoice_knapsack, pareto_sweep

    # Single solve
    selected_df, stats = multichoice_knapsack(
        df_all_packages=df,       # all packages, multiple rows per upn
        budget=10_000_000,
        equity_floor_pct=60,      # at least 60% of spend to high/med risk
        cost_col='mean_total_capex',
        carbon_col='mean_total_co2_saved',
    )

    # Sweep to get Pareto front
    pareto_df, all_stats = pareto_sweep(
        df_all_packages=df,
        budget=10_000_000,
        equity_floors=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    )
"""

import logging
import time
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import pulp


# ---------------------------------------------------------------------------
# DEFAULTS
# ---------------------------------------------------------------------------

DEFAULT_HIGH_EQUITY_PERSONAS = {"high_risk", "med_risk"}
ALL_PERSONAS = ["high_risk", "med_risk", "middle_risk", "low_risk", "v_low_risk"]


# ---------------------------------------------------------------------------
# SOLVER
# ---------------------------------------------------------------------------

def multichoice_knapsack(
    df_all_packages: pd.DataFrame,
    budget: float,
    equity_floor_pct: float = 0.0,
    high_equity_personas: Set[str] = None,
    upn_col: str = "upn",
    persona_col: str = "meta_socio_persona",
    cost_col: str = "mean_total_capex",
    carbon_col: str = "mean_total_co2_saved",
    time_limit_seconds: int = 600,
    logger: logging.Logger = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Multiple-choice knapsack: pick at most one package per building
    to maximise total CO2 abatement, subject to:
      - total cost <= budget
      - at least equity_floor_pct % of total spend goes to buildings
        whose meta_socio_persona is in high_equity_personas

    Parameters
    ----------
    df_all_packages : DataFrame
        All packages for all buildings. Multiple rows per upn
        (one per intervention). Should include a do_nothing option
        if you want the solver to be able to skip a building.
    budget : float
        Total budget cap (£).
    equity_floor_pct : float
        Minimum % of total spend that must go to high-equity
        persona buildings. Range: 0–100.
    high_equity_personas : set of str
        Which meta_socio_persona values count as "high equity".
        Default: {"high_risk", "med_risk"}.
    upn_col : str
        Column identifying each building.
    persona_col : str
        Column containing the persona category.
    cost_col : str
        Column with intervention cost.
    carbon_col : str
        Column with CO2 saved.
    time_limit_seconds : int
        Max solver time per solve.
    logger : logging.Logger, optional

    Returns
    -------
    selected_df : DataFrame of selected packages (do-nothing excluded).
    stats : dict with summary statistics including persona and
            intervention breakdowns.
    """
    if high_equity_personas is None:
        high_equity_personas = DEFAULT_HIGH_EQUITY_PERSONAS

    t0 = time.time()

    # --- Preprocessing ---
    df = df_all_packages.copy().reset_index(drop=True)
    df = df[(df[cost_col] <= budget) & (df[carbon_col] >= 0)]
    df["_is_high_eq"] = df[persona_col].isin(high_equity_personas).astype(int)

    n_buildings = df[upn_col].nunique()

    if logger:
        logger.info(
            f"Solve: {len(df):,} rows | {n_buildings:,} buildings | "
            f"budget=£{budget:,.0f} | equity_spend_floor={equity_floor_pct:.0f}%"
        )

    if df.empty:
        return df_all_packages.iloc[:0].copy(), {"status": "empty"}

    # --- Build ILP ---
    prob = pulp.LpProblem("retrofit_multichoice", pulp.LpMaximize)
    indices = df.index.tolist()
    x = pulp.LpVariable.dicts("sel", indices, cat=pulp.LpBinary)

    carbon = df[carbon_col].to_dict()
    costs = df[cost_col].to_dict()
    is_high = df["_is_high_eq"].to_dict()

    # OBJECTIVE: maximise total CO2 abatement
    prob += pulp.lpSum(carbon[i] * x[i] for i in indices)

    # C1: total spend <= budget
    prob += pulp.lpSum(costs[i] * x[i] for i in indices) <= budget

    # C2: at most one package per building
    for upn, group in df.groupby(upn_col):
        prob += pulp.lpSum(x[i] for i in group.index) <= 1

    # C3: equity floor on SPEND
    #   spend_high_eq >= (equity_floor_pct / 100) * spend_total
    #   Rearranged: sum((is_high_i - frac) * cost_i * x_i) >= 0
    if equity_floor_pct > 0:
        frac = equity_floor_pct / 100.0
        prob += pulp.lpSum(
            (is_high[i] - frac) * costs[i] * x[i] for i in indices
        ) >= 0

    # --- Solve ---
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_seconds)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    solve_time = time.time() - t0

    if logger:
        logger.info(f"  Status: {status} ({solve_time:.1f}s)")

    # --- Extract selected rows ---
    selected_idx = [i for i in indices if pulp.value(x[i]) > 0.5]
    selected_df = df.loc[selected_idx].copy()
    # Remove do-nothing selections (zero carbon) from output
    selected_df = selected_df[selected_df[carbon_col] > 0]
    selected_df = selected_df.drop(columns=["_is_high_eq"], errors="ignore")

    # --- Compute stats ---
    total_cost = selected_df[cost_col].sum()
    total_carbon = selected_df[carbon_col].sum()
    high_eq_mask = selected_df[persona_col].isin(high_equity_personas)
    high_eq_spend = selected_df.loc[high_eq_mask, cost_col].sum()
    high_eq_carbon = selected_df.loc[high_eq_mask, carbon_col].sum()

    # Per-persona breakdown
    persona_breakdown = {}
    for p in ALL_PERSONAS:
        pmask = selected_df[persona_col] == p
        persona_breakdown[p] = {
            "buildings": int(pmask.sum()),
            "spend": round(selected_df.loc[pmask, cost_col].sum(), 0),
            "abatement": round(selected_df.loc[pmask, carbon_col].sum(), 2),
        }

    # Per-intervention breakdown
    intervention_col = "intervention"
    intervention_breakdown = {}
    if intervention_col in selected_df.columns:
        for intv, grp in selected_df.groupby(intervention_col):
            intervention_breakdown[intv] = {
                "buildings": len(grp),
                "spend": round(grp[cost_col].sum(), 0),
                "abatement": round(grp[carbon_col].sum(), 2),
            }


    # Per-intervention breakdown
    percntile_col = "avg_gas_percentile"
    percentile_breakdown = {}
    if percntile_col in selected_df.columns:
        print('col present') 
        for intv, grp in selected_df.groupby(percntile_col):
            percentile_breakdown[intv] = {
                "buildings": len(grp),
                "spend": round(grp[cost_col].sum(), 0),
                "abatement": round(grp[carbon_col].sum(), 2),
            }
    else:
        import sys 
        print('missing percentile')
        sys.exit() 


    stats = {
        "status": status,
        "solve_time_s": round(solve_time, 2),
        "equity_floor_pct": equity_floor_pct,
        "n_retrofitted": len(selected_df),
        "n_high_equity": int(high_eq_mask.sum()),
        "total_cost": round(total_cost, 2),
        "remaining_budget": round(budget - total_cost, 2),
        "total_abatement": round(total_carbon, 4),
        "cpex_per_ton": (
            round(total_cost / total_carbon, 2) if total_carbon > 0 else None
        ),
        "high_eq_spend": round(high_eq_spend, 2),
        "high_eq_spend_pct": (
            round(100 * high_eq_spend / total_cost, 2) if total_cost > 0 else 0
        ),
        "high_eq_abatement": round(high_eq_carbon, 4),
        "high_eq_abatement_pct": (
            round(100 * high_eq_carbon / total_carbon, 2)
            if total_carbon > 0 else 0
        ),
        "persona_breakdown": persona_breakdown,
        "intervention_breakdown": intervention_breakdown,
        "percentile_breakdown": percentile_breakdown , 
    }

    if logger:
        logger.info(
            f"  {stats['n_retrofitted']:,} bldgs | "
            f"£{total_cost:,.0f} | {total_carbon:,.1f} tCO2 | "
            f"£{stats['cpex_per_ton']:,.0f}/t | "
            f"high-eq spend: {stats['high_eq_spend_pct']:.1f}%"
        )

    return selected_df, stats


# ---------------------------------------------------------------------------
# PARETO SWEEP
# ---------------------------------------------------------------------------

def pareto_sweep(
    df_all_packages: pd.DataFrame,
    budget: float,
    equity_floors: List[float] = None,
    high_equity_personas: Set[str] = None,
    upn_col: str = "upn",
    persona_col: str = "meta_socio_persona",
    cost_col: str = "mean_total_capex",
    carbon_col: str = "mean_total_co2_saved",
    time_limit_seconds: int = 600,
    logger: logging.Logger = None,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Sweep equity floor from 0% upward to trace the Pareto front.

    At each equity_floor value, solves the multi-choice knapsack.
    Stops early if a value is infeasible (all higher values will
    also be infeasible).

    Returns
    -------
    pareto_df : DataFrame with one row per feasible equity floor.
    all_stats : list of full stat dicts (including breakdowns).
    """
    if equity_floors is None:
        equity_floors = list(range(0, 105, 5))

    all_stats = []
    for eps in equity_floors:
        if logger:
            logger.info(f"\n{'='*50}")

        _, stats = multichoice_knapsack(
            df_all_packages=df_all_packages,
            budget=budget,
            equity_floor_pct=eps,
            high_equity_personas=high_equity_personas,
            upn_col=upn_col,
            persona_col=persona_col,
            cost_col=cost_col,
            carbon_col=carbon_col,
            time_limit_seconds=time_limit_seconds,
            logger=logger,
        )
        all_stats.append(stats)

        # Stop if infeasible — higher floors will also fail
        if stats["status"] not in ("Optimal", "Not Solved"):
            if logger:
                logger.info(f"  Infeasible at {eps}% — stopping sweep.")
            break

    pareto_df = pd.DataFrame(all_stats)
    return pareto_df, all_stats


# ---------------------------------------------------------------------------
# BASELINE: old pre-select approach (for comparison)
# ---------------------------------------------------------------------------

def preselect_best_cpt(
    df_all_packages: pd.DataFrame,
    upn_col: str = "upn",
    cost_col: str = "mean_total_capex",
    carbon_col: str = "mean_total_co2_saved",
) -> pd.DataFrame:
    """
    Old method: for each building, pick the package with the lowest
    £/tCO2, returning one row per building.

    Use this to benchmark against the new multi-choice approach.
    """
    df = df_all_packages[df_all_packages[carbon_col] > 0].copy()
    df["_cpt"] = df[cost_col] / df[carbon_col]
    best_idx = df.groupby(upn_col)["_cpt"].idxmin()
    return df.loc[best_idx].drop(columns=["_cpt"])