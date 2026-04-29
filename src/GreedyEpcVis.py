"""
src/GreedyEpcVis.py
===================

Comparison plots: optimiser ("Opt.T") vs EPC random selection.

Aligned to the new column schema produced by the upstream pipeline:
  - selected_projects_eq{N}.csv has mean + aleatoric_std + epistemic_std
    columns for total_co2_saved, total_capex, capex_per_net_ton.
  - pareto_summary.csv carries portfolio-level cpex_per_ton plus the
    per-run percentile envelope (cpex_per_ton_p16/median/p84).

Error bars: combined std = sqrt(aleatoric^2 + epistemic^2), propagated
across rows assuming independence (matches the upstream convention;
conservative for epistemic, which is correlated, but consistent).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# CONSTANTS
# ============================================================================

method_name = 'Opt.T'

METHOD_COLORS = {
    method_name: '#56B4E9',
    'EPC':       '#CC79A7',
}

# Persona palette aligned with the personas actually present in the data.
# Matches the purple→green ramp used by PostPareto.py.
PERSONA_COLORS = {
    'high_risk':   '#762a83',
    'med_risk':    '#af8dc3',
    'middle_risk': '#d9d9d9',
    'low_risk':    '#7fbf7b',
    'v_low_risk':  '#1b7837',
}

# --- Selected-projects schema -----------------------------------------------
CO2_MEAN_COL    = 'mean_total_co2_saved'
CO2_ALE_COL     = 'aleatoric_std_total_co2_saved'
CO2_EPI_COL     = 'epistemic_std_total_co2_saved'

CAPEX_MEAN_COL  = 'mean_total_capex'
CAPEX_ALE_COL   = 'aleatoric_std_total_capex'
CAPEX_EPI_COL   = 'epistemic_std_total_capex'

CPT_MEAN_COL    = 'mean_capex_per_net_ton'
CPT_ALE_COL     = 'aleatoric_std_capex_per_net_ton'
CPT_EPI_COL     = 'epistemic_std_capex_per_net_ton'

# --- pareto_summary.csv schema ----------------------------------------------
SUMMARY_CPEX_MEAN     = 'cpex_per_ton'
SUMMARY_CPEX_MEDIAN   = 'cpex_per_ton_median'
SUMMARY_CPEX_P16      = 'cpex_per_ton_p16'
SUMMARY_CPEX_P84      = 'cpex_per_ton_p84'
SUMMARY_TOTAL_COST    = 'total_cost'
SUMMARY_TOTAL_ABATE   = 'total_abatement'
SUMMARY_COST_ALE      = 'total_cost_aleatoric_std'
SUMMARY_COST_EPI      = 'total_cost_epistemic_std'
SUMMARY_ABATE_ALE     = 'total_abatement_aleatoric_std'
SUMMARY_ABATE_EPI     = 'total_abatement_epistemic_std'


# ============================================================================
# UNCERTAINTY HELPERS
# ============================================================================

def _row_combined_std(df: pd.DataFrame, ale_col: str, epi_col: str) -> np.ndarray:
    """Per-row combined std = sqrt(aleatoric^2 + epistemic^2)."""
    ale = df[ale_col].fillna(0).to_numpy() if ale_col in df.columns else 0.0
    epi = df[epi_col].fillna(0).to_numpy() if epi_col in df.columns else 0.0
    return np.sqrt(np.asarray(ale) ** 2 + np.asarray(epi) ** 2)


def _propagate_sum_std(df: pd.DataFrame, ale_col: str, epi_col: str) -> float:
    """Propagate combined std for a sum across rows (independence assumed)."""
    ale_var = (df[ale_col].fillna(0) ** 2).sum() if ale_col in df.columns else 0.0
    epi_var = (df[epi_col].fillna(0) ** 2).sum() if epi_col in df.columns else 0.0
    return float(np.sqrt(ale_var + epi_var))


def _grouped_sum_with_std(
    df: pd.DataFrame, group_col: str,
    mean_col: str, ale_col: str, epi_col: str,
) -> pd.DataFrame:
    """Group sums of mean and combined-std, one row per group."""
    if df.empty or group_col not in df.columns:
        return pd.DataFrame(columns=['total_mean', 'total_std'])
    g = df.groupby(group_col).apply(
        lambda s: pd.Series({
            'total_mean': s[mean_col].sum() if mean_col in s.columns else 0.0,
            'total_std':  np.sqrt(
                ((s[ale_col].fillna(0) ** 2).sum() if ale_col in s.columns else 0.0)
                + ((s[epi_col].fillna(0) ** 2).sum() if epi_col in s.columns else 0.0)
            ),
        })
    )
    return g.sort_index()


def _capex_scaling(column_name: str) -> tuple[float, str, str]:
    """Return (scale_factor, unit_label, fmt) for capex-like columns."""
    is_capex = 'capex' in column_name.lower()
    return (1e6, ' (£M)', '{:,.1f}') if is_capex else (1.0, '', '{:,.0f}')


# ============================================================================
# ENTRY POINT
# ============================================================================

def run_epc_vis(
    pareto_runs_folder,
    base_dir_outputs,
    million_budget,
    prob_loft,
    equity_floor,
    mip_gap=0.01,
):
    """
    Load Pareto-selected and EPC-selected results and generate comparison
    plots. The new pareto_summary.csv is also loaded so portfolio-level
    cpex_per_ton (with per-run percentile envelope) can be plotted.

    Parameters
    ----------
    pareto_runs_folder : str
        Bucket-level folder containing per-budget run directories.
    base_dir_outputs : str
        Root folder for saving visualisation outputs.
    million_budget : float
        Budget in millions (e.g. 50 for £50M).
    prob_loft : float
        Loft probability used in this run.
    equity_floor : int or float
        Which equity floor's Pareto result to compare against EPC.
    mip_gap : float
        MIP gap used in the run; required to reconstruct the folder name.
    """
    million_budget_str = (
        str(int(million_budget))
        if million_budget == int(million_budget)
        else f'{million_budget:g}'
    )
    eq_label = f"{int(equity_floor)}"

    output_dir = os.path.join(
        pareto_runs_folder,
        f'budget_{million_budget_str}M__loft_{prob_loft}__mip_{mip_gap}',
    )

    selected_path = os.path.join(output_dir, f'selected_projects_eq{eq_label}.csv')
    epc_path = os.path.join(output_dir, 'epc_random_selection.csv')
    summary_path = os.path.join(output_dir, 'pareto_summary.csv')

    print(f'  trying paths:\n    {selected_path}\n    {epc_path}\n    {summary_path}')

    if not os.path.isfile(selected_path):
        print(f'  ⚠️  selected file missing — skipping ({selected_path})')
        return
    if not os.path.isfile(epc_path):
        print(f'  ⚠️  EPC file missing — skipping ({epc_path})')
        return

    df = pd.read_csv(selected_path)
    epc = pd.read_csv(epc_path)
    print(f'  Loaded Pareto (eq={eq_label}): {len(df)} rows')
    print(f'  Loaded EPC: {len(epc)} rows')

    summary_row = _load_summary_row(summary_path, equity_floor)

    vis_output_dir = os.path.join(
        base_dir_outputs,
        'epc_comparisons',
        f'budget_{million_budget_str}M__loft_{prob_loft}__eq_{eq_label}',
    )

    generate_all_aggregation_plots(
        df, epc,
        output_dir=vis_output_dir,
        save=True,
        summary_row=summary_row,
        budget_million=million_budget,
        loft=prob_loft,
        equity_floor=equity_floor,
    )


def _load_summary_row(summary_path: str, equity_floor: float) -> pd.Series | None:
    """Pull the row of pareto_summary.csv matching this equity floor."""
    if not os.path.isfile(summary_path):
        print(f'  (no pareto_summary.csv at {summary_path}; '
              f'cpex envelope plot will use raw selection instead)')
        return None
    try:
        s = pd.read_csv(summary_path)
    except Exception as e:
        print(f'  (failed to read pareto_summary: {e})')
        return None
    if 'equity_floor_pct' not in s.columns:
        return None
    match = s[s['equity_floor_pct'].astype(float) == float(equity_floor)]
    if match.empty:
        print(f'  (no summary row for equity_floor={equity_floor})')
        return None
    return match.iloc[0]


# ============================================================================
# TOTAL COMPARISON  (CO2 or CAPEX)
# ============================================================================

def plot_total_comparison(
    df1, df2,
    mean_col, ale_col, epi_col,
    output_dir=None, save=False,
):
    """
    Bar comparison of summed `mean_col` between Opt.T and EPC, with
    combined-std error bars propagated across rows.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    scale_factor, unit_label, fmt_str = _capex_scaling(mean_col)

    total_mean_df1 = df1[mean_col].sum() / scale_factor
    total_mean_df2 = df2[mean_col].sum() / scale_factor

    total_std_df1 = _propagate_sum_std(df1, ale_col, epi_col) / scale_factor
    total_std_df2 = _propagate_sum_std(df2, ale_col, epi_col) / scale_factor

    x_positions = [0, 1]
    means = [total_mean_df1, total_mean_df2]
    stds = [total_std_df1, total_std_df2]
    labels = [method_name, 'EPC']
    colors = [METHOD_COLORS[method_name], METHOD_COLORS['EPC']]

    bars = ax.bar(x_positions, means,
                  yerr=stds, capsize=8,
                  color=colors, alpha=0.7, edgecolor='black', linewidth=2,
                  error_kw={'linewidth': 2, 'capthick': 2})

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2.,
                mean + std + (mean * 0.02),
                f'{fmt_str.format(mean)} ± {fmt_str.format(std)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ylabel_text = mean_col.replace('_', ' ').title() + unit_label
    ax.set_ylabel(ylabel_text, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    diff = total_mean_df2 - total_mean_df1
    diff_pct = (diff / total_mean_df1) * 100 if total_mean_df1 != 0 else 0
    ax.text(0.75, 0.85,
            f'Difference: {fmt_str.format(diff)} ({diff_pct:+.1f}%)',
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)

    plt.tight_layout()

    if save and output_dir:
        filepath = Path(output_dir) / f'{mean_col}_total_comparison.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filepath.name}")
        plt.close()
    else:
        plt.show()


# ============================================================================
# GROUP COMPARISON
# ============================================================================

def plot_by_group(
    df1, df2,
    mean_col, ale_col, epi_col,
    group_col, group_label,
    output_dir=None, save=False,
):
    """Sums of `mean_col` split by `group_col`, with combined-std bars."""
    fig, ax = plt.subplots(figsize=(12, 8))

    scale_factor, unit_label, fmt_str = _capex_scaling(mean_col)

    df1_agg = _grouped_sum_with_std(df1, group_col, mean_col, ale_col, epi_col)
    df2_agg = _grouped_sum_with_std(df2, group_col, mean_col, ale_col, epi_col)
    df1_agg['total_mean'] /= scale_factor
    df1_agg['total_std']  /= scale_factor
    df2_agg['total_mean'] /= scale_factor
    df2_agg['total_std']  /= scale_factor

    all_groups = sorted(set(df1_agg.index) | set(df2_agg.index))

    df1_means = [df1_agg.loc[g, 'total_mean'] if g in df1_agg.index else 0
                 for g in all_groups]
    df1_stds  = [df1_agg.loc[g, 'total_std']  if g in df1_agg.index else 0
                 for g in all_groups]
    df2_means = [df2_agg.loc[g, 'total_mean'] if g in df2_agg.index else 0
                 for g in all_groups]
    df2_stds  = [df2_agg.loc[g, 'total_std']  if g in df2_agg.index else 0
                 for g in all_groups]

    x = np.arange(len(all_groups))
    width = 0.35

    bars1 = ax.bar(x - width / 2, df1_means, width,
                   yerr=df1_stds, capsize=4,
                   label=method_name,
                   color=METHOD_COLORS[method_name],
                   alpha=0.7, edgecolor='black',
                   error_kw={'linewidth': 1.5, 'capthick': 1.5})

    bars2 = ax.bar(x + width / 2, df2_means, width,
                   yerr=df2_stds, capsize=4,
                   label='EPC',
                   color=METHOD_COLORS['EPC'],
                   alpha=0.7, edgecolor='black',
                   error_kw={'linewidth': 1.5, 'capthick': 1.5})

    for bars, stds in [(bars1, df1_stds), (bars2, df2_stds)]:
        for bar, std in zip(bars, stds):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2.,
                        height + std + (height * 0.02),
                        fmt_str.format(height),
                        ha='center', va='bottom', fontsize=9)

    ax.set_xlabel(group_label, fontsize=12, fontweight='bold')
    ax.set_ylabel(mean_col.replace('_', ' ').title() + unit_label, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(all_groups, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save and output_dir:
        filepath = Path(output_dir) / f'{mean_col}_by_{group_col}.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filepath.name}")
        plt.close()
    else:
        plt.show()


# ============================================================================
# HEATMAP  (no error bars — colour-by-sum only)
# ============================================================================

def plot_heatmap_comparison(df1, df2, column, output_dir=None, save=False):
    """Heatmap of summed `column` by socio persona × current energy rating."""
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))

    scale_factor, unit_label, _ = _capex_scaling(column)
    fmt_str = '.1f' if scale_factor != 1 else '.0f'

    pivot_df1 = df1.pivot_table(
        values=column, index='meta_socio_persona',
        columns='CURRENT_ENERGY_RATING', aggfunc='sum', fill_value=0,
    ) / scale_factor

    pivot_df2 = df2.pivot_table(
        values=column, index='meta_socio_persona',
        columns='CURRENT_ENERGY_RATING', aggfunc='sum', fill_value=0,
    ) / scale_factor

    sns.heatmap(pivot_df1, annot=True, fmt=fmt_str, cmap='YlOrRd',
                ax=axes[0], cbar_kws={
                    'label': column.replace('_', ' ').title() + unit_label})
    axes[0].set_title(method_name, fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Energy Rating', fontsize=11)
    axes[0].set_ylabel('Socio Persona', fontsize=11)

    sns.heatmap(pivot_df2, annot=True, fmt=fmt_str, cmap='YlOrRd',
                ax=axes[1], cbar_kws={
                    'label': column.replace('_', ' ').title() + unit_label})
    axes[1].set_title('EPC', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Energy Rating', fontsize=11)
    axes[1].set_ylabel('Socio Persona', fontsize=11)

    plt.tight_layout()

    if save and output_dir:
        filepath = Path(output_dir) / f'{column}_heatmap_comparison.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filepath.name}")
        plt.close()
    else:
        plt.show()


# ============================================================================
# BUILDING COUNTS
# ============================================================================

def _plot_grouped_counts(
    df1, df2, group_col, xlabel, filename,
    output_dir=None, save=False, rotation=0,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    df1_counts = df1[group_col].value_counts().sort_index()
    df2_counts = df2[group_col].value_counts().sort_index()

    all_keys = sorted(set(df1_counts.index) | set(df2_counts.index))
    df1_values = [df1_counts.get(k, 0) for k in all_keys]
    df2_values = [df2_counts.get(k, 0) for k in all_keys]

    x = np.arange(len(all_keys))
    width = 0.35

    bars1 = ax.bar(x - width / 2, df1_values, width, label=method_name,
                   color=METHOD_COLORS[method_name],
                   alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width / 2, df2_values, width, label='EPC',
                   color=METHOD_COLORS['EPC'],
                   alpha=0.7, edgecolor='black')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel('Building Count', fontsize=12)
    ax.set_xticks(x)
    label_strs = ([f'{int(k)}' for k in all_keys]
                  if all(isinstance(k, (int, float, np.integer, np.floating))
                         and not pd.isna(k) for k in all_keys)
                  else [str(k) for k in all_keys])
    ax.set_xticklabels(label_strs, rotation=rotation, ha='right' if rotation else 'center')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save and output_dir:
        filepath = Path(output_dir) / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filepath.name}")
        plt.close()
    else:
        plt.show()


def plot_building_counts_by_percentile(df1, df2, output_dir=None, save=False):
    _plot_grouped_counts(df1, df2, 'avg_gas_percentile',
                         'Gas Percentile', 'building_count_by_percentile.png',
                         output_dir, save, rotation=0)


def plot_building_counts_by_persona(df1, df2, output_dir=None, save=False):
    _plot_grouped_counts(df1, df2, 'meta_socio_persona',
                         'Socio Persona', 'building_count_by_persona.png',
                         output_dir, save, rotation=45)


def plot_building_counts_by_energy_rating(df1, df2, output_dir=None, save=False):
    _plot_grouped_counts(df1, df2, 'CURRENT_ENERGY_RATING',
                         'Current Energy Rating',
                         'building_count_by_energy_rating.png',
                         output_dir, save, rotation=0)


# ============================================================================
# INTERVENTION PLOTS
# ============================================================================

def _plot_interventions_stacked(
    df1, df2, group_col, xlabel, filename,
    output_dir=None, save=False, rotation=0,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    df1_crosstab = pd.crosstab(df1[group_col], df1['intervention'])
    df2_crosstab = pd.crosstab(df2[group_col], df2['intervention'])

    all_interventions = sorted(set(df1_crosstab.columns) | set(df2_crosstab.columns))
    df1_crosstab = df1_crosstab.reindex(columns=all_interventions, fill_value=0)
    df2_crosstab = df2_crosstab.reindex(columns=all_interventions, fill_value=0)

    df1_crosstab.plot(kind='bar', stacked=True, ax=axes[0],
                      colormap='tab10', edgecolor='black',
                      linewidth=0.5, legend=False)
    axes[0].set_title(method_name + ' Targeting', fontsize=13, fontweight='bold')
    axes[0].set_xlabel(xlabel, fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].tick_params(axis='x', rotation=rotation)

    df2_crosstab.plot(kind='bar', stacked=True, ax=axes[1],
                      colormap='tab10', edgecolor='black',
                      linewidth=0.5, legend=False)
    axes[1].set_title('EPC Targeting', fontsize=13, fontweight='bold')
    axes[1].set_xlabel(xlabel, fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=rotation)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Intervention',
               loc='upper right', frameon=True)
    plt.tight_layout(rect=[0, 0, 0.95, 1])

    if save and output_dir:
        filepath = Path(output_dir) / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filepath.name}")
        plt.close()
    else:
        plt.show()


def plot_interventions_by_percentile(df1, df2, output_dir=None, save=False):
    _plot_interventions_stacked(df1, df2, 'avg_gas_percentile',
                                'Gas Percentile',
                                'interventions_by_percentile.png',
                                output_dir, save, rotation=0)


def plot_interventions_by_persona(df1, df2, output_dir=None, save=False):
    _plot_interventions_stacked(df1, df2, 'meta_socio_persona',
                                'Socio Persona',
                                'interventions_by_persona.png',
                                output_dir, save, rotation=45)


def plot_interventions_by_energy_rating(df1, df2, output_dir=None, save=False):
    _plot_interventions_stacked(df1, df2, 'CURRENT_ENERGY_RATING',
                                'Energy Rating',
                                'interventions_by_energy_rating.png',
                                output_dir, save, rotation=0)


# ============================================================================
# CO2 / CAPEX BY INTERVENTION  (with combined error bars)
# ============================================================================

def plot_co2_by_intervention(df1, df2, output_dir=None, save=False):
    fig, ax = plt.subplots(figsize=(10, 6))

    df1_agg = _grouped_sum_with_std(df1, 'intervention',
                                    CO2_MEAN_COL, CO2_ALE_COL, CO2_EPI_COL)
    df2_agg = _grouped_sum_with_std(df2, 'intervention',
                                    CO2_MEAN_COL, CO2_ALE_COL, CO2_EPI_COL)

    all_interventions = sorted(set(df1_agg.index) | set(df2_agg.index))

    df1_means = [df1_agg.loc[i, 'total_mean'] if i in df1_agg.index else 0
                 for i in all_interventions]
    df1_stds  = [df1_agg.loc[i, 'total_std']  if i in df1_agg.index else 0
                 for i in all_interventions]
    df2_means = [df2_agg.loc[i, 'total_mean'] if i in df2_agg.index else 0
                 for i in all_interventions]
    df2_stds  = [df2_agg.loc[i, 'total_std']  if i in df2_agg.index else 0
                 for i in all_interventions]

    x = np.arange(len(all_interventions))
    width = 0.35

    bars1 = ax.bar(x - width / 2, df1_means, width,
                   yerr=df1_stds, capsize=4,
                   label=method_name,
                   color=METHOD_COLORS[method_name],
                   alpha=0.7, edgecolor='black',
                   error_kw={'linewidth': 1.5, 'capthick': 1.5})
    bars2 = ax.bar(x + width / 2, df2_means, width,
                   yerr=df2_stds, capsize=4,
                   label='EPC',
                   color=METHOD_COLORS['EPC'],
                   alpha=0.7, edgecolor='black',
                   error_kw={'linewidth': 1.5, 'capthick': 1.5})

    for bars, stds in [(bars1, df1_stds), (bars2, df2_stds)]:
        for bar, std in zip(bars, stds):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2.,
                        height + std + (height * 0.02),
                        f'{height:,.0f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Intervention Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total CO2 Saved (Tons/5yr)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(all_interventions, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save and output_dir:
        filepath = Path(output_dir) / 'co2_by_intervention.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filepath.name}")
        plt.close()
    else:
        plt.show()


def plot_capex_by_intervention(df1, df2, output_dir=None, save=False):
    if CAPEX_MEAN_COL not in df1.columns or CAPEX_MEAN_COL not in df2.columns:
        print(f"  Warning: {CAPEX_MEAN_COL} missing — skipping capex_by_intervention")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    df1_agg = _grouped_sum_with_std(df1, 'intervention',
                                    CAPEX_MEAN_COL, CAPEX_ALE_COL, CAPEX_EPI_COL)
    df2_agg = _grouped_sum_with_std(df2, 'intervention',
                                    CAPEX_MEAN_COL, CAPEX_ALE_COL, CAPEX_EPI_COL)
    df1_agg /= 1e6
    df2_agg /= 1e6

    all_interventions = sorted(set(df1_agg.index) | set(df2_agg.index))

    df1_means = [df1_agg.loc[i, 'total_mean'] if i in df1_agg.index else 0
                 for i in all_interventions]
    df1_stds  = [df1_agg.loc[i, 'total_std']  if i in df1_agg.index else 0
                 for i in all_interventions]
    df2_means = [df2_agg.loc[i, 'total_mean'] if i in df2_agg.index else 0
                 for i in all_interventions]
    df2_stds  = [df2_agg.loc[i, 'total_std']  if i in df2_agg.index else 0
                 for i in all_interventions]

    x = np.arange(len(all_interventions))
    width = 0.35

    bars1 = ax.bar(x - width / 2, df1_means, width,
                   yerr=df1_stds, capsize=4,
                   label=method_name,
                   color=METHOD_COLORS[method_name],
                   alpha=0.7, edgecolor='black',
                   error_kw={'linewidth': 1.5, 'capthick': 1.5})
    bars2 = ax.bar(x + width / 2, df2_means, width,
                   yerr=df2_stds, capsize=4,
                   label='EPC',
                   color=METHOD_COLORS['EPC'],
                   alpha=0.7, edgecolor='black',
                   error_kw={'linewidth': 1.5, 'capthick': 1.5})

    for bars, stds in [(bars1, df1_stds), (bars2, df2_stds)]:
        for bar, std in zip(bars, stds):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2.,
                        height + std + (height * 0.02),
                        f'{height:,.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Intervention Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total CAPEX (£M)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_interventions, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save and output_dir:
        filepath = Path(output_dir) / f'{CAPEX_MEAN_COL}_by_intervention.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filepath.name}")
        plt.close()
    else:
        plt.show()


# ============================================================================
# PORTFOLIO £/tCO2  — uses pareto_summary.csv when available
# ============================================================================

def plot_portfolio_cpex_per_ton(
    df1, df2, summary_row,
    output_dir=None, save=False,
):
    """
    Portfolio £/tCO2 = sum(mean_total_capex) / sum(mean_total_co2_saved).

    For Opt.T: if `summary_row` carries the per-run percentile envelope
    (cpex_per_ton_p16/median/p84) we use that as the error indicator,
    since it sidesteps the ratio-of-Gaussians issue.
    For EPC: a portfolio ratio with no envelope (no per-run info available
    for the random selection), so a single bar with no error.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    def _portfolio_ratio(df):
        cap = df[CAPEX_MEAN_COL].sum() if CAPEX_MEAN_COL in df.columns else np.nan
        co2 = df[CO2_MEAN_COL].sum()   if CO2_MEAN_COL  in df.columns else np.nan
        if not co2 or np.isnan(co2) or co2 == 0:
            return np.nan
        return cap / co2

    opt_mean = _portfolio_ratio(df1)
    epc_mean = _portfolio_ratio(df2)

    rom = _portfolio_ratio(df1)        # ratio of means
    mor = float(summary_row[SUMMARY_CPEX_MEDIAN])  # median of ratios
    print(f"  Opt.T: ratio-of-means={rom:.0f}, median-of-ratios={mor:.0f}, "
        f"gap={100*(mor-rom)/rom:+.1f}%")
    
    # Try to use the per-run percentile envelope from pareto_summary.
    opt_low = opt_high = None
    have_envelope = (
        summary_row is not None
        and SUMMARY_CPEX_MEDIAN in summary_row.index
        and pd.notna(summary_row.get(SUMMARY_CPEX_MEDIAN))
    )
    if have_envelope:
        opt_mean = float(summary_row[SUMMARY_CPEX_MEDIAN])
        opt_low  = float(summary_row.get(SUMMARY_CPEX_P16, opt_mean))
        opt_high = float(summary_row.get(SUMMARY_CPEX_P84, opt_mean))

    x_positions = [0, 1]
    means = [opt_mean, epc_mean]
    labels = [method_name, 'EPC']
    colors = [METHOD_COLORS[method_name], METHOD_COLORS['EPC']]

    # Asymmetric error bars only on Opt.T if we have the envelope.
    yerr = None
    if have_envelope:
        lower = opt_mean - opt_low
        upper = opt_high - opt_mean
        yerr = np.array([[lower, 0.0], [upper, 0.0]])

    bars = ax.bar(x_positions, means,
                  yerr=yerr, capsize=8,
                  color=colors, alpha=0.7, edgecolor='black', linewidth=2,
                  error_kw={'linewidth': 2, 'capthick': 2})

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)

    # Annotate
    for i, (bar, mean) in enumerate(zip(bars, means)):
        if pd.isna(mean):
            continue
        if i == 0 and have_envelope:
            label = (f'{mean:,.0f}\n[P16 {opt_low:,.0f}, '
                     f'P84 {opt_high:,.0f}]')
            ax.text(bar.get_x() + bar.get_width() / 2.,
                    (opt_high if opt_high is not None else mean) * 1.02,
                    label, ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width() / 2., mean,
                    f'{mean:,.0f}', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')

    ax.set_ylabel('Portfolio £/tCO₂', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    title = 'Portfolio cost-effectiveness'
    if have_envelope:
        title += '  (Opt.T: median + P16–P84 across epistemic runs)'
    ax.set_title(title, fontsize=12, fontweight='bold')

    if not pd.isna(opt_mean) and not pd.isna(epc_mean) and opt_mean != 0:
        diff = epc_mean - opt_mean
        diff_pct = diff / opt_mean * 100
        ax.text(0.5, 0.95,
                f'EPC − Opt.T: {diff:+,.0f} £/tCO₂ ({diff_pct:+.1f}%)',
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)

    plt.tight_layout()

    if save and output_dir:
        filepath = Path(output_dir) / 'portfolio_cpex_per_ton.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filepath.name}")
        plt.close()
    else:
        plt.show()


# ============================================================================
# META FUNCTION
# ============================================================================

def generate_all_aggregation_plots(
    df1, df2,
    output_dir='./plots', save=True,
    summary_row=None,
    budget_million=None, loft=None, equity_floor=None,
):
    """Generate all aggregation/summation comparison plots."""
    if save:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"  Output directory: {output_path}")

    print(f"  Generating aggregation plots "
          f"(budget={budget_million}M, loft={loft}, eq={equity_floor}%)...")

    # --- Total comparisons (CO2 + CAPEX) ---
    plot_total_comparison(df1, df2,
                          mean_col=CO2_MEAN_COL,
                          ale_col=CO2_ALE_COL,
                          epi_col=CO2_EPI_COL,
                          output_dir=output_dir, save=save)
    plot_total_comparison(df1, df2,
                          mean_col=CAPEX_MEAN_COL,
                          ale_col=CAPEX_ALE_COL,
                          epi_col=CAPEX_EPI_COL,
                          output_dir=output_dir, save=save)

    # --- CO2 by group ---
    for group_col, label in [
        ('meta_socio_persona',    'Persona'),
        ('avg_gas_percentile',    'Gas Consumption Decile'),
        ('CURRENT_ENERGY_RATING', 'Energy Rating'),
    ]:
        plot_by_group(df1, df2,
                      mean_col=CO2_MEAN_COL,
                      ale_col=CO2_ALE_COL,
                      epi_col=CO2_EPI_COL,
                      group_col=group_col, group_label=label,
                      output_dir=output_dir, save=save)

    # --- CAPEX by group ---
    for group_col, label in [
        ('meta_socio_persona',    'Persona'),
        ('avg_gas_percentile',    'Gas Consumption Decile'),
        ('CURRENT_ENERGY_RATING', 'Energy Rating'),
    ]:
        plot_by_group(df1, df2,
                      mean_col=CAPEX_MEAN_COL,
                      ale_col=CAPEX_ALE_COL,
                      epi_col=CAPEX_EPI_COL,
                      group_col=group_col, group_label=label,
                      output_dir=output_dir, save=save)

    # --- Heatmaps ---
    plot_heatmap_comparison(df1, df2, CO2_MEAN_COL, output_dir, save)
    plot_heatmap_comparison(df1, df2, CAPEX_MEAN_COL, output_dir, save)

    # --- Building counts ---
    plot_building_counts_by_percentile(df1, df2, output_dir, save)
    plot_building_counts_by_persona(df1, df2, output_dir, save)
    plot_building_counts_by_energy_rating(df1, df2, output_dir, save)

    # --- Intervention analysis ---
    plot_interventions_by_percentile(df1, df2, output_dir, save)
    plot_interventions_by_persona(df1, df2, output_dir, save)
    plot_interventions_by_energy_rating(df1, df2, output_dir, save)
    plot_co2_by_intervention(df1, df2, output_dir, save)
    plot_capex_by_intervention(df1, df2, output_dir, save)

    # --- Portfolio £/tCO2 (uses pareto_summary if available) ---
    plot_portfolio_cpex_per_ton(df1, df2, summary_row, output_dir, save)

    print("  All aggregation plots generated.")