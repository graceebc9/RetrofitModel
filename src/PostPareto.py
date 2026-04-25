"""
src/PostPareto.py
=================

Slim post-processing for the Pareto knapsack pipeline.

Reduced to the essential plots and aligned with the new decomposed
uncertainty schema produced upstream:

  - pareto_summary.csv now carries portfolio-level aleatoric and
    epistemic stds (closed-form + per-run propagation) plus a
    percentile-based £/tCO2 envelope. We use these directly rather
    than recomputing them from row-level stds.

Plots produced (per loft value):
  1. Pareto front overlay — abatement vs equity floor across budgets,
     with aleatoric (narrow) and epistemic (wide) uncertainty bands.
  2. £/tCO2 overlay — median + P16/P84 envelope across budgets,
     using the upstream per-run-derived percentiles.
  3. Trade-off scatter — single-glance summary of every
     (budget, equity_floor) combination.
  4. Epistemic share — diagnostic showing that aleatoric uncertainty
     washes out and epistemic dominates as the portfolio grows.
     This is the headline figure that justifies the nested sampling.

CSV outputs:
  - pareto_summaries_all_budgets.csv (raw upstream concatenation)
  - equity_persona_shares.csv        (persona shares per scenario)

Folder layout expected (matches upstream optimiser):
    base_path/
        budget_{label}M__loft_{loft}__mip_{mip_gap}/
            selected_projects_eq{ef}.csv
            pareto_summary.csv
            ...
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# CONSTANTS
# ============================================================================

MILLION_FACTOR = 1_000_000


def budget_label(budget: float) -> str:
    """
    Canonical folder-name label for a budget in pounds.

    £1_000_000  -> '1'
    £2_500_000  -> '2.5'
    £10_000_000 -> '10'
    £100_000_000-> '100'
    £500_000    -> '0.5'

    Mirrors the helper in pareto_knapsack.py — keep them in sync.
    """
    return f"{budget / MILLION_FACTOR:g}"


# ============================================================================
# DATA LOADING
# ============================================================================

def _budget_dir(base_path: str, budget: float, loft_val: float,
                mip_gap: float) -> str:
    """Reconstruct the per-budget folder path written by the optimiser."""
    name = f'budget_{budget_label(budget)}M__loft_{loft_val}__mip_{mip_gap}'
    return os.path.join(base_path, name)


def load_pareto_summaries(
    budgets: list,
    loft_val: float,
    base_path: str,
    mip_gap: float,
) -> pd.DataFrame:
    """
    Concatenate pareto_summary.csv from each budget folder into one frame.

    Columns include the new uncertainty fields written by the upstream
    optimiser (total_*_aleatoric_std, total_*_epistemic_std,
    epistemic_share_*, cpex_per_ton_p16/median/p84) — we don't recompute
    these here, just consume them.
    """
    summaries = []
    for budget in budgets:
        summary_file = os.path.join(
            _budget_dir(base_path, budget, loft_val, mip_gap),
            'pareto_summary.csv',
        )
        if not os.path.isfile(summary_file):
            print(f"  ⚠️ Missing summary: {summary_file}")
            continue
        s = pd.read_csv(summary_file)
        s['budget'] = budget
        s['budget_label'] = f'£{budget/1e6:.0f}M'
        summaries.append(s)

    if not summaries:
        return pd.DataFrame()
    return pd.concat(summaries, ignore_index=True)


def load_equity_shares(
    budgets: list,
    equity_floors: list,
    loft_val: float,
    base_path: str,
    mip_gap: float,
) -> pd.DataFrame:
    """
    Build a (budget, equity_floor) table of persona shares from the
    selected_projects CSVs. Only used to enrich the summary CSV — not
    needed for any of the headline plots, but kept because reviewers
    sometimes ask "who's in the portfolio at floor=X?".
    """
    rows = []
    for budget in budgets:
        bdir = _budget_dir(base_path, budget, loft_val, mip_gap)
        if not os.path.isdir(bdir):
            continue
        for ef in equity_floors:
            f = os.path.join(bdir, f'selected_projects_eq{int(ef)}.csv')
            if not os.path.isfile(f):
                continue
            sel = pd.read_csv(f)
            if sel.empty or 'meta_socio_persona' not in sel.columns:
                continue
            counts = sel.groupby('meta_socio_persona')['upn'].count()
            n_total = counts.sum()
            if n_total == 0:
                continue
            row = {
                'budget': budget,
                'equity_floor_pct': ef,
                'n_total': int(n_total),
            }
            for p in ('high_risk', 'med_risk', 'middle_risk',
                      'low_risk', 'v_low_risk'):
                row[f'{p}_count'] = int(counts.get(p, 0))
                row[f'{p}_pct'] = float(counts.get(p, 0) / n_total)
            rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================
# PLOTS
# ============================================================================

def _budget_colors(budgets: list) -> dict:
    """Consistent colour per budget across plots."""
    cmap = plt.colormaps.get_cmap('viridis')
    n = max(len(budgets) - 1, 1)
    return {b: cmap(i / n) for i, b in enumerate(budgets)}


def plot_pareto_front_overlay(
    summaries: pd.DataFrame,
    output_dir: str,
    loft_val: float,
) -> None:
    """
    Abatement vs equity floor, one curve per budget, with uncertainty
    bands. Aleatoric band is narrow (washes out at scale); epistemic
    band is wider (does not wash out). Both are ±1 σ envelopes.
    """
    if summaries.empty:
        print("  No Pareto summaries to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    budgets = sorted(summaries['budget'].unique())
    colors = _budget_colors(budgets)

    have_ale = 'total_abatement_aleatoric_std' in summaries.columns
    have_epi = 'total_abatement_epistemic_std' in summaries.columns

    for budget in budgets:
        sub = summaries[summaries['budget'] == budget].copy()
        sub = sub[sub['status'].isin(['Optimal', 'Not Solved'])]
        sub = sub.sort_values('equity_floor_pct')
        if sub.empty:
            continue

        x = sub['equity_floor_pct'].to_numpy()
        y = sub['total_abatement'].to_numpy()
        c = colors[budget]

        # Epistemic band (wider, lighter) — drawn first so it sits behind.
        if have_epi:
            epi = sub['total_abatement_epistemic_std'].fillna(0).to_numpy()
            ax.fill_between(x, y - epi, y + epi, color=c, alpha=0.12)

        # Aleatoric band (narrow, darker shade of same colour).
        if have_ale:
            ale = sub['total_abatement_aleatoric_std'].fillna(0).to_numpy()
            ax.fill_between(x, y - ale, y + ale, color=c, alpha=0.30)

        ax.plot(
            x, y, 'o-', color=c, linewidth=2, markersize=6,
            label=f'£{budget/1e6:.0f}M',
        )

    ax.set_xlabel('Equity floor (% of spend to high/med risk)', fontsize=11)
    ax.set_ylabel('Total CO₂ abatement (tonnes)', fontsize=11)
    ax.set_title(
        f'Pareto fronts across budgets — Loft {loft_val}\n'
        f'Bands: ±1σ aleatoric (dark) and ±1σ epistemic (light)',
        fontsize=12, fontweight='bold',
    )
    ax.legend(title='Budget', fontsize=9, loc='best')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, f'01_pareto_fronts_loft_{loft_val}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {os.path.basename(path)}")


def plot_cpex_vs_equity_overlay(
    summaries: pd.DataFrame,
    output_dir: str,
    loft_val: float,
) -> None:
    """
    £/tCO2 vs equity floor, one curve per budget. Uses the per-run-
    derived median + P16/P84 envelope where available — sidesteps the
    ratio-of-Gaussians problem of computing std on a ratio of means.
    Falls back to the mean line if percentile columns are missing.
    """
    if summaries.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    budgets = sorted(summaries['budget'].unique())
    colors = _budget_colors(budgets)

    have_pct = all(
        c in summaries.columns
        for c in ('cpex_per_ton_median', 'cpex_per_ton_p16', 'cpex_per_ton_p84')
    )

    for budget in budgets:
        sub = summaries[summaries['budget'] == budget].copy()
        sub = sub[sub['status'].isin(['Optimal', 'Not Solved'])]
        sub = sub.sort_values('equity_floor_pct')
        if sub.empty:
            continue

        x = sub['equity_floor_pct'].to_numpy()
        c = colors[budget]

        if have_pct and sub['cpex_per_ton_median'].notna().any():
            med = sub['cpex_per_ton_median'].to_numpy()
            p16 = sub['cpex_per_ton_p16'].to_numpy()
            p84 = sub['cpex_per_ton_p84'].to_numpy()
            ax.fill_between(x, p16, p84, color=c, alpha=0.18)
            ax.plot(
                x, med, 'o-', color=c, linewidth=2, markersize=6,
                label=f'£{budget/1e6:.0f}M',
            )
        else:
            # Fallback: mean line only.
            y = sub['cpex_per_ton'].to_numpy()
            ax.plot(
                x, y, 's-', color=c, linewidth=2, markersize=6,
                label=f'£{budget/1e6:.0f}M (mean)',
            )

    ax.set_xlabel('Equity floor (% of spend to high/med risk)', fontsize=11)
    ax.set_ylabel('Portfolio £/tCO₂', fontsize=11)
    title = f'Cost-effectiveness vs equity — Loft {loft_val}'
    if have_pct:
        title += '\nLines: median across epistemic runs; bands: P16–P84'
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(title='Budget', fontsize=9, loc='best')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, f'02_cpex_vs_equity_loft_{loft_val}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {os.path.basename(path)}")


def plot_tradeoff_scatter(
    summaries: pd.DataFrame,
    output_dir: str,
    loft_val: float,
) -> None:
    """
    Scatter of £/tCO2 vs total abatement across all (budget, equity_floor)
    combinations. Colour = budget, marker size = equity floor.
    Useful single-glance summary; uncertainty is intentionally suppressed
    here — see the overlay plots for that.
    """
    if summaries.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    budgets = sorted(summaries['budget'].unique())
    colors = _budget_colors(budgets)

    for budget in budgets:
        sub = summaries[summaries['budget'] == budget].copy()
        sub = sub[sub['status'].isin(['Optimal', 'Not Solved'])]
        if sub.empty:
            continue
        sizes = 30 + (sub['equity_floor_pct'] / 100.0) * 150
        ax.scatter(
            sub['total_abatement'], sub['cpex_per_ton'],
            s=sizes, color=colors[budget], alpha=0.7,
            edgecolor='white', linewidth=0.5,
            label=f'£{budget/1e6:.0f}M',
        )

    ax.set_xlabel('Total CO₂ abatement (tonnes)', fontsize=11)
    ax.set_ylabel('Portfolio £/tCO₂', fontsize=11)
    ax.set_title(
        f'Trade-off space (marker size = equity floor) — Loft {loft_val}',
        fontsize=12, fontweight='bold',
    )
    ax.legend(title='Budget', fontsize=9, loc='best')
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, f'03_tradeoff_scatter_loft_{loft_val}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {os.path.basename(path)}")


def plot_epistemic_share(
    summaries: pd.DataFrame,
    output_dir: str,
    loft_val: float,
) -> None:
    """
    Headline diagnostic: epistemic share of total portfolio uncertainty
    vs portfolio size, for both cost and carbon.

    The expected story: at small portfolios aleatoric matters; as the
    portfolio grows, aleatoric averages out (sqrt(n) scaling) while
    epistemic does not, so the share rises toward 1.

    This is the figure that justifies bothering with the nested
    sampling / Eve's-law decomposition — without it you can't distinguish
    irreducible global-parameter uncertainty from washable building noise.
    """
    if summaries.empty:
        return

    needed = (
        'epistemic_share_cost', 'epistemic_share_carbon',
        'n_retrofitted', 'budget',
    )
    if not all(c in summaries.columns for c in needed):
        print("  Skipping epistemic-share plot: required columns missing.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    budgets = sorted(summaries['budget'].unique())
    colors = _budget_colors(budgets)

    for ax, share_col, ylabel in [
        (axes[0], 'epistemic_share_carbon', 'CO₂ abatement'),
        (axes[1], 'epistemic_share_cost', 'Cost'),
    ]:
        for budget in budgets:
            sub = summaries[summaries['budget'] == budget].copy()
            sub = sub[sub['status'].isin(['Optimal', 'Not Solved'])]
            sub = sub.dropna(subset=[share_col])
            if sub.empty:
                continue
            ax.scatter(
                sub['n_retrofitted'], sub[share_col],
                s=60, color=colors[budget], alpha=0.8,
                edgecolor='white', linewidth=0.5,
                label=f'£{budget/1e6:.0f}M',
            )

        ax.set_xscale('log')
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(0.5, color='#888', linestyle=':', linewidth=0.8)
        ax.set_xlabel('Buildings retrofitted (log scale)', fontsize=11)
        ax.set_title(f'{ylabel}', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(
        r'$\sigma^2_{epi} / (\sigma^2_{ale} + \sigma^2_{epi})$',
        fontsize=11,
    )
    axes[1].legend(title='Budget', fontsize=9, loc='best')
    fig.suptitle(
        f'Epistemic share of portfolio uncertainty — Loft {loft_val}\n'
        f'Aleatoric washes out as portfolio grows; epistemic does not.',
        fontsize=12, fontweight='bold',
    )
    fig.tight_layout()

    path = os.path.join(output_dir, f'04_epistemic_share_loft_{loft_val}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {os.path.basename(path)}")


# ============================================================================
# MAIN
# ============================================================================

def post_proc_pareto(
    BUDGETS: list,
    EQUITY_FLOORS: list,
    LOFT_VALUE: float,
    BASE_PATH: str,
    OUTPUT_PATH: str,
    MIP_GAP: float,
) -> None:
    """
    Slim post-processor. Loads upstream summaries, writes essential plots
    and CSVs. Backwards-compatible signature: RHO is accepted but unused
    (the new pipeline does not need a hand-tuned correlation parameter).

    Parameters
    ----------
    BUDGETS : list[float]    Budgets in raw £.
    EQUITY_FLOORS : list[float]   Equity floor percentages.
    LOFT_VALUE : float       Loft fraction used in folder names.
    BASE_PATH : str          Folder containing budget_{label}M__loft_{val}__mip_{gap}/.
    OUTPUT_PATH : str        Where to write plots and CSVs.
    MIP_GAP : float          Used to reconstruct the per-budget folder name.
    """
     

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # ── Load summaries ────────────────────────────────────────────────────
    summaries = load_pareto_summaries(
        budgets=BUDGETS, loft_val=LOFT_VALUE,
        base_path=BASE_PATH, mip_gap=MIP_GAP,
    )
    if summaries.empty:
        print("Critical: no pareto_summary.csv files found — exiting.")
        return

    summaries.to_csv(
        os.path.join(OUTPUT_PATH, 'pareto_summaries_all_budgets.csv'),
        index=False,
    )

    # ── Equity persona shares (CSV only — no plot) ────────────────────────
    persona_df = load_equity_shares(
        budgets=BUDGETS, equity_floors=EQUITY_FLOORS, loft_val=LOFT_VALUE,
        base_path=BASE_PATH, mip_gap=MIP_GAP,
    )
    if not persona_df.empty:
        persona_df.to_csv(
            os.path.join(OUTPUT_PATH, 'equity_persona_shares.csv'),
            index=False,
        )

    # ── Plots ─────────────────────────────────────────────────────────────
    print(f"\n--- Generating essential plots in: {OUTPUT_PATH} ---")
    plot_pareto_front_overlay(summaries, OUTPUT_PATH, LOFT_VALUE)
    plot_cpex_vs_equity_overlay(summaries, OUTPUT_PATH, LOFT_VALUE)
    plot_tradeoff_scatter(summaries, OUTPUT_PATH, LOFT_VALUE)
    plot_epistemic_share(summaries, OUTPUT_PATH, LOFT_VALUE)
    print("  Plotting complete.")