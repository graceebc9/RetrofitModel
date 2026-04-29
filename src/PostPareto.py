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


 
 
# ----------------------------------------------------------------------------
# Small helper to DRY up the status filter that's repeated in every plot.
# ----------------------------------------------------------------------------

_SOLVED_STATUSES = ('Optimal', 'Not Solved')


def _solved(df: pd.DataFrame) -> pd.DataFrame:
    """Rows the optimiser actually returned a portfolio for."""
    if 'status' not in df.columns:
        return df
    return df[df['status'].isin(_SOLVED_STATUSES)]


# ----------------------------------------------------------------------------
# Plot A — Equity concentration heatmaps
# ----------------------------------------------------------------------------

def plot_equity_concentration_heatmaps(
    summaries: pd.DataFrame,
    output_dir: str,
    loft_val: float,
) -> None:
    """
    Two heatmaps side by side: high-equity share of spend (left) and of
    abatement (right), across the full (budget × equity_floor) grid.

    Colour is centred on the equity_floor value of each cell — i.e. the
    floor is treated as the neutral reference and we show *deviation*
    from it. Cells warmer than neutral mean the optimiser allocated
    more than the floor required (over-delivering on equity); cooler
    cells mean the constraint was binding and the optimiser sat right
    on it. Same colour scale across both panels so they're comparable.
    """
    needed = ('high_eq_spend_pct', 'high_eq_abatement_pct',
              'budget', 'equity_floor_pct')
    if summaries.empty or not all(c in summaries.columns for c in needed):
        print("  Skipping equity-concentration heatmap: required columns missing.")
        return

    df = _solved(summaries).copy()
    if df.empty:
        return

    # Pivot to (equity_floor × budget) with budget on x so it reads
    # left-to-right small→large. Equity floor on y, ascending bottom→top.
    def _pivot(col):
        p = df.pivot_table(
            index='equity_floor_pct', columns='budget',
            values=col, aggfunc='mean',
        ).sort_index(ascending=True)
        return p

    spend = _pivot('high_eq_spend_pct')
    abate = _pivot('high_eq_abatement_pct')

    # Floor reference matrix — same shape, value = the row's equity floor.
    # We compute deviation (delivered − floor) so 0 = exactly on floor.
    floor_ref = np.broadcast_to(
        spend.index.values[:, None], spend.shape,
    ).astype(float)
    spend_dev = spend.values - floor_ref
    abate_dev = abate.values - floor_ref

    # Symmetric colour limits across both panels for comparability.
    vmax = float(np.nanmax(np.abs(np.concatenate([spend_dev, abate_dev]))))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    budgets = list(spend.columns)
    floors = list(spend.index)

    for ax, mat, raw, title in [
        (axes[0], spend_dev, spend.values, 'Spend share − floor'),
        (axes[1], abate_dev, abate.values, 'Abatement share − floor'),
    ]:
        im = ax.imshow(
            mat, aspect='auto', origin='lower',
            cmap='RdBu_r', vmin=-vmax, vmax=vmax,
        )
        ax.set_xticks(range(len(budgets)))
        ax.set_xticklabels([f'£{b/1e6:g}M' for b in budgets],
                           rotation=30, ha='right')
        ax.set_yticks(range(len(floors)))
        ax.set_yticklabels([f'{f:g}%' for f in floors])
        ax.set_xlabel('Budget', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')

        # Annotate each cell with the raw delivered share, not the
        # deviation — the deviation is what the colour encodes, the
        # raw number is what the reader actually wants to read off.
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = raw[i, j]
                if np.isnan(v):
                    continue
                # Pick text colour for contrast against the cell.
                contrast = 'white' if abs(mat[i, j]) > 0.55 * vmax else 'black'
                ax.text(j, i, f'{v:.0f}%', ha='center', va='center',
                        fontsize=8, color=contrast)

    axes[0].set_ylabel('Equity floor', fontsize=11)

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label('Delivered − floor (pp)', fontsize=10)

    fig.suptitle(
        f'Equity concentration vs floor — Loft {loft_val}\n'
        f'Cells: % delivered to high-equity group; colour: deviation from floor',
        fontsize=12, fontweight='bold',
    )

    path = os.path.join(output_dir, f'05_equity_heatmaps_loft_{loft_val}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {os.path.basename(path)}")


# ----------------------------------------------------------------------------
# Plot B — Spend share vs abatement share scatter
# ----------------------------------------------------------------------------

def plot_equity_spend_vs_abatement(
    summaries: pd.DataFrame,
    output_dir: str,
    loft_val: float,
) -> None:
    """
    One point per (budget, equity_floor): x = high_eq_spend_pct,
    y = high_eq_abatement_pct. Colour = budget, marker size = equity
    floor. Reveals whether the equity-targeted spend is buying carbon
    proportional to its share — points above the diagonal would mean
    the high-equity group delivers more abatement per pound than its
    spend share, below means less. (We don't draw the diagonal here
    by request; the visual relationship is still legible.)
    """
    needed = ('high_eq_spend_pct', 'high_eq_abatement_pct',
              'budget', 'equity_floor_pct')
    if summaries.empty or not all(c in summaries.columns for c in needed):
        print("  Skipping equity spend-vs-abatement scatter: columns missing.")
        return

    df = _solved(summaries).copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    budgets = sorted(df['budget'].unique())
    
    
    colors = _budget_colors(budgets)

    for budget in budgets:
        sub = df[df['budget'] == budget]
        if sub.empty:
            continue
        sizes = 30 + (sub['equity_floor_pct'] / 100.0) * 200
        ax.scatter(
            sub['high_eq_spend_pct'], sub['high_eq_abatement_pct'],
            s=sizes, color=colors[budget], alpha=0.75,
            edgecolor='white', linewidth=0.6,
            label=f'£{budget/1e6:.0f}M',
        )

    ax.set_xlabel('High-equity share of spend (%)', fontsize=11)
    ax.set_ylabel('High-equity share of abatement (%)', fontsize=11)
    ax.set_title(
        f'Equity spend vs abatement — Loft {loft_val}\n'
        f'Marker size = equity floor; colour = budget',
        fontsize=12, fontweight='bold',
    )
    ax.legend(title='Budget', fontsize=9, loc='best')
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, f'06_equity_spend_vs_abatement_loft_{loft_val}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {os.path.basename(path)}")


# ----------------------------------------------------------------------------
# Plot C — Persona composition stacked bars
# ----------------------------------------------------------------------------

def plot_persona_composition(
    persona_df: pd.DataFrame,
    output_dir: str,
    loft_val: float,
) -> None:
    """
    Stacked bars showing persona composition of the selected portfolio
    for every (budget, equity_floor) cell. Bars are grouped by budget
    along the x-axis with equity_floor as the inner grouping, so the
    reader can see how composition shifts as the floor tightens within
    each budget. Uses the *_pct columns produced by load_equity_shares.
    """
    if persona_df.empty:
        print("  Skipping persona composition plot: no persona data.")
        return

    persona_cols = [
        ('high_risk_pct',   '#762a83'),
        ('med_risk_pct',    '#af8dc3'),
        ('middle_risk_pct', '#d9d9d9'),
        ('low_risk_pct',    '#7fbf7b'),
        ('v_low_risk_pct',  '#1b7837'),
    ]
    available = [(c, col) for c, col in persona_cols if c in persona_df.columns]
    if not available:
        print("  Skipping persona composition plot: persona columns missing.")
        return

    df = persona_df.sort_values(['budget', 'equity_floor_pct']).copy()

    # Build a categorical x position: one tick per (budget, floor) pair.
    df['scenario'] = (
        df['budget'].apply(lambda b: f'£{b/1e6:g}M')
        + '\n' + df['equity_floor_pct'].astype(int).astype(str) + '%'
    )

    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(df)), 6))
    x = np.arange(len(df))
    bottom = np.zeros(len(df))

    for col, color in available:
        vals = df[col].fillna(0).to_numpy() * 100  # to %
        label = col.replace('_pct', '').replace('_', ' ')
        ax.bar(x, vals, bottom=bottom, color=color,
               edgecolor='white', linewidth=0.4, label=label)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(df['scenario'], fontsize=8)
    ax.set_xlabel('Budget / equity floor', fontsize=11)
    ax.set_ylabel('Share of selected buildings (%)', fontsize=11)
    ax.set_ylim(0, 100)

    # Visual separators between budget groups so the eye can chunk them.
    boundaries = df['budget'].ne(df['budget'].shift()).to_numpy()
    for i, is_new in enumerate(boundaries):
        if i > 0 and is_new:
            ax.axvline(i - 0.5, color='black', linewidth=0.4, alpha=0.3)

    ax.set_title(
        f'Portfolio persona composition across scenarios — Loft {loft_val}',
        fontsize=12, fontweight='bold',
    )
    ax.legend(title='Persona', fontsize=9, loc='upper right',
              bbox_to_anchor=(1.0, -0.12), ncol=len(available))
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, f'07_persona_composition_loft_{loft_val}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {os.path.basename(path)}")


 
 
def plot_high_equity_vs_total_counts(
    summaries: pd.DataFrame,
    output_dir: str,
    loft_val: float,
) -> None:
    """
    Stacked bars of n_high_equity (bottom) and the remaining buildings
    (top) per (budget, equity_floor) scenario.

    Bars are ordered by budget then equity_floor with thin separators
    between budget groups, matching the persona composition plot. The
    high-equity count is annotated on each bar so the absolute number
    is readable even when bars are short.
    """
    needed = ('n_retrofitted', 'n_high_equity', 'budget', 'equity_floor_pct')
    if summaries.empty or not all(c in summaries.columns for c in needed):
        print("  Skipping high-equity vs total counts plot: "
              "n_high_equity / n_retrofitted columns missing.")
        return

    # Reuse the existing solved-status filter convention.
    df = summaries.copy()
    if 'status' in df.columns:
        df = df[df['status'].isin(('Optimal', 'Not Solved'))]
    df = df.dropna(subset=['n_retrofitted', 'n_high_equity'])
    if df.empty:
        return

    df = df.sort_values(['budget', 'equity_floor_pct']).reset_index(drop=True)
    df['n_other'] = (df['n_retrofitted'] - df['n_high_equity']).clip(lower=0)
    df['scenario'] = (
        df['budget'].apply(lambda b: f'£{b/1e6:g}M')
        + '\n' + df['equity_floor_pct'].astype(int).astype(str) + '%'
    )

    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(df)), 6))
    x = np.arange(len(df))

    high = df['n_high_equity'].to_numpy()
    other = df['n_other'].to_numpy()

    ax.bar(x, high, color='#762a83', edgecolor='white', linewidth=0.4,
           label='High equity')
    ax.bar(x, other, bottom=high, color='#bababa',
           edgecolor='white', linewidth=0.4, label='Other')

    # Annotate the high-equity count inside its segment where there's
    # room, otherwise above the whole bar so it stays legible.
    totals = high + other
    ymax = totals.max() if len(totals) else 1
    for i, (h, t) in enumerate(zip(high, totals)):
        if h <= 0:
            continue
        if h >= 0.08 * ymax:
            ax.text(i, h / 2, f'{int(h)}', ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')
        else:
            ax.text(i, t + 0.01 * ymax, f'{int(h)}', ha='center',
                    va='bottom', fontsize=8, color='#762a83')

    ax.set_xticks(x)
    ax.set_xticklabels(df['scenario'], fontsize=8)
    ax.set_xlabel('Budget / equity floor', fontsize=11)
    ax.set_ylabel('Buildings retrofitted', fontsize=11)

    # Visual separators between budget groups (same trick as plot C).
    boundaries = df['budget'].ne(df['budget'].shift()).to_numpy()
    for i, is_new in enumerate(boundaries):
        if i > 0 and is_new:
            ax.axvline(i - 0.5, color='black', linewidth=0.4, alpha=0.3)

    ax.set_title(
        f'High-equity vs other buildings retrofitted — Loft {loft_val}\n'
        f'Bar height = total portfolio size; numbers = high-equity count',
        fontsize=12, fontweight='bold',
    )
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, f'08_high_equity_counts_loft_{loft_val}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {os.path.basename(path)}")


# - 

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
    RHO: Optional[float] = None,  # accepted for backwards compat; ignored
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
    if RHO is not None:
        print(f"  (note: RHO={RHO} ignored — superseded by upstream "
              f"per-run epistemic propagation)")

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # ── Load summaries ────────────────────────────────────────────────────
    summaries = load_pareto_summaries(
        budgets=BUDGETS, loft_val=LOFT_VALUE,
        base_path=BASE_PATH, mip_gap=MIP_GAP,
    )
    if summaries.empty:
        print("Critical: no pareto_summary.csv files found — exiting.")
        return
    print(summaries.columns.tolist() ) 
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

 
    # New: equity / persona breakdowns across the (budget, equity_floor) grid.
    plot_equity_concentration_heatmaps(summaries, OUTPUT_PATH, LOFT_VALUE)
    plot_equity_spend_vs_abatement(summaries, OUTPUT_PATH, LOFT_VALUE)
    plot_persona_composition(persona_df, OUTPUT_PATH, LOFT_VALUE)
    plot_high_equity_vs_total_counts(summaries, OUTPUT_PATH, LOFT_VALUE)
    print("  Plotting complete.")


 