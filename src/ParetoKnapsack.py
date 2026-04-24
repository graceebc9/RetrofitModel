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
"""

import logging
import time
import sys
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB


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
    time_limit_seconds: int = 1200,
    logger: logging.Logger = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Multiple-choice knapsack: pick at most one package per building
    to maximise total CO2 abatement.
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

    # --- Build ILP (Gurobi) ---
    
    # Create environment and model
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1) # Set to 0 to suppress Gurobi console output
    env.start()
    model = gp.Model("retrofit_multichoice", env=env)

    # Set Gurobi Parameters (matching your previous HiGHS config)
    model.Params.TimeLimit = time_limit_seconds
    model.Params.MIPGap = 0.01  # 1% relative gap
    model.Params.Threads = 0   # 0 = use all available cores
    
    indices = df.index.tolist()

    # Decision Variables
    x = model.addVars(indices, vtype=GRB.BINARY, name="sel")

    carbon = df[carbon_col].to_dict()
    costs = df[cost_col].to_dict()
    is_high = df["_is_high_eq"].to_dict()

    # OBJECTIVE: maximise total CO2 abatement
    model.setObjective(gp.quicksum(carbon[i] * x[i] for i in indices), GRB.MAXIMIZE)

    # C1: total spend <= budget
    model.addConstr(gp.quicksum(costs[i] * x[i] for i in indices) <= budget, name="budget_limit")

    # C2: at most one package per building
    for upn, group in df.groupby(upn_col):
        model.addConstr(gp.quicksum(x[i] for i in group.index) <= 1, name=f"max_one_{upn}")

    # C3: equity floor on SPEND
    if equity_floor_pct > 0:
        frac = equity_floor_pct / 100.0
        model.addConstr(
            gp.quicksum((is_high[i] - frac) * costs[i] * x[i] for i in indices) >= 0,
            name="equity_floor"
        )

    # --- Solve ---
    model.optimize()

    solve_time = time.time() - t0

    # --- Map Gurobi Status to PuLP-style strings for compatibility ---
    if model.Status == GRB.OPTIMAL or (model.Status == GRB.TIME_LIMIT and model.SolCount > 0):
        status_str = "Optimal"
    elif model.Status == GRB.INFEASIBLE:
        status_str = "Infeasible"
    else:
        status_str = "Not Solved"

    if logger:
        logger.info(f"  Status: {status_str} ({solve_time:.1f}s)")

    # --- Extract selected rows ---
    if model.SolCount > 0:
        selected_idx = [i for i in indices if x[i].X > 0.5]
        selected_df = df.loc[selected_idx].copy()
    else:
        selected_df = df.iloc[:0].copy()

    # Remove do-nothing selections (zero carbon) from output
    if not selected_df.empty:
        selected_df = selected_df[selected_df[carbon_col] > 0]
        selected_df = selected_df.drop(columns=["_is_high_eq"], errors="ignore")

    # --- Compute stats ---
    total_cost = selected_df[cost_col].sum() if not selected_df.empty else 0
    total_carbon = selected_df[carbon_col].sum() if not selected_df.empty else 0
    high_eq_mask = selected_df[persona_col].isin(high_equity_personas) if not selected_df.empty else pd.Series(dtype=bool)
    high_eq_spend = selected_df.loc[high_eq_mask, cost_col].sum() if not selected_df.empty else 0
    high_eq_carbon = selected_df.loc[high_eq_mask, carbon_col].sum() if not selected_df.empty else 0

    # Per-persona breakdown
    persona_breakdown = {}
    if not selected_df.empty:
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
    if not selected_df.empty and intervention_col in selected_df.columns:
        for intv, grp in selected_df.groupby(intervention_col):
            intervention_breakdown[intv] = {
                "buildings": len(grp),
                "spend": round(grp[cost_col].sum(), 0),
                "abatement": round(grp[carbon_col].sum(), 2),
            }

    # Per-percentile breakdown
    percntile_col = "avg_gas_percentile"
    percentile_breakdown = {}
    if not selected_df.empty:
        if percntile_col in selected_df.columns:
            for intv, grp in selected_df.groupby(percntile_col):
                percentile_breakdown[intv] = {
                    "buildings": len(grp),
                    "spend": round(grp[cost_col].sum(), 0),
                    "abatement": round(grp[carbon_col].sum(), 2),
                }
        else:
            print('missing percentile')
            sys.exit() 

    stats = {
        "status": status_str,
        "solve_time_s": round(solve_time, 2),
        "equity_floor_pct": equity_floor_pct,
        "n_retrofitted": len(selected_df),
        "n_high_equity": int(high_eq_mask.sum()) if not selected_df.empty else 0,
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

    if logger and status_str == "Optimal":
        logger.info(
            f"  {stats['n_retrofitted']:,} bldgs | "
            f"£{total_cost:,.0f} | {total_carbon:,.1f} tCO2 | "
            f"£{stats['cpex_per_ton'] if stats['cpex_per_ton'] else 0:,.0f}/t | "
            f"high-eq spend: {stats['high_eq_spend_pct']:.1f}%"
        )

    # Clean up the Gurobi environment to free up the license token when done
    model.dispose()
    env.dispose()

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
    time_limit_seconds: int = 1200,
    logger: logging.Logger = None,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Sweep equity floor from 0% upward to trace the Pareto front.
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
    df = df_all_packages[df_all_packages[carbon_col] > 0].copy()
    df["_cpt"] = df[cost_col] / df[carbon_col]
    best_idx = df.groupby(upn_col)["_cpt"].idxmin()
    return df.loc[best_idx].drop(columns=["_cpt"])