"""
portfolio_epistemic_analysis.py
================================
Portfolio-level uncertainty analysis for two-stage Monte Carlo retrofit data.

Data flow
---------
  raw CSVs (one row per building per epistemic run)
      └─► pool_epistemic_runs_decomposed()        [imported]
              collapses n=50 runs → one row per building with:
                  {base}_mean, {base}_aleatoric_std, {base}_epistemic_std
          └─► GroupedStatsAccumulator.update()
                  stores triples (mean, ale_std, epi_std) per building per group
              └─► compute_group_stats()
                      per-building average within group + law of total variance
                      at the group level:
                        ale_group  = sqrt( mean(ale_std²) / n )
                        epi_group  = sqrt( mean(epi_std²) / n )
                        combined   = sqrt( ale_group² + epi_group² )
                  └─► plots + per-breakout CSVs + single master summary CSV

METRICS_MAP (must match the imported function)
----------------------------------------------
  'capex_per_net_ton'  →  '{sc}_capex_per_net_ton_co2_{sc}_{stat}'
  'co2'                →  '{sc}_total_energy_abs_co2_ton_samples_{sc}_{stat}'
  'capex'              →  '{sc}_cost_{sc}_{stat}'

Plot style
----------
  Bar              = group mean
  Inner blue band  = ±1σ aleatoric   (irreducible)
  Outer red band   = ±1σ combined    (aleatoric + epistemic in quadrature)
  Black whiskers   = ±1σ epistemic   (model-parameter uncertainty only)
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import glob
import gc
import csv

from src.utils import is_running_on_hpc


from run_pre_greedy import pool_epistemic_runs_decomposed , METRICS_MAP

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')

# ==============================================================================
# 1. CONFIGURATION & PATHS
# ==============================================================================

SCENARIOS = [
    'joint_heat_loft_decay',
    'joint_heat_wall_decay',
    'wall_installation',
    'join_heat_ins_decay',
    'heat_pump_only',
    'loft_installation',
]


METRIC_LABELS = {
    'capex_per_net_ton': 'Capex per Net Tonne CO\u2082 (\u00a3/t)',
    'co2':               'Net CO\u2082 Removal (Tonnes / 5 yr)',
    'capex':             'Installation Capex (\u00a3)',
}

OUTPUT_BASE = '3_stock_results_2001_v2/epistemic_full'
os.makedirs(OUTPUT_BASE, exist_ok=True)

is_hpc = is_running_on_hpc()
if is_hpc:
    LOG_DIR        = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/0_intermediate_data_2D/retrofit_scenario/v10/NE/*csv'
    REFERENCE_FILE = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/0_intermediate_data_2D/retrofit_scenario/v10/NE/130_log_file.csv'
else:
    LOG_DIR        = '/Volumes/T9/2025_10_RetrofitModel/1_data_runs/NE/*.csv'
    REFERENCE_FILE = None

GROUP_COLS = ['avg_gas_percentile', 'premise_type', 'inferred_insulation_type']

TYPOLOGIES = [
    'Small low terraces',
    '3-4 storey and smaller flats',
    'Tall terraces 3-4 storeys',
    'Large semi detached',
    'Standard size detached',
    'Standard size semi detached',
    '2 storeys terraces with t rear extension',
    'Semi type house in multiples',
    'Large detached',
    'Planned balanced mixed estates',
    'Linked and step linked premises',
]

SCENARIO_DISPLAY_NAMES = {
    'loft_installation':     'Loft Installation',
    'wall_installation':     'Wall Insulation',
    'joint_heat_loft_decay': 'HP + Loft (Decay)',
    'joint_heat_wall_decay': 'HP + Wall (Decay)',
    'join_heat_ins_decay':   'HP + All Insulation (Decay)',
    'heat_pump_only':        'Heat Pump Only',
}

# Uncertainty band colours
COLOUR_ALE      = '#4393c3'   # blue   – aleatoric inner band
COLOUR_COMBINED = '#d6604d'   # red    – combined outer band
ALPHA_INNER     = 0.35
ALPHA_OUTER     = 0.18


# ==============================================================================
# 2. DATA ACCUMULATOR  (post-pooling: one row per building, runs already collapsed)
# ==============================================================================

class GroupedStatsAccumulator:
    """
    Accumulates per-building triples: (mean_val, ale_std, epi_std).

    Expected input columns produced by pool_epistemic_runs_decomposed:
        {base}_mean            – E_runs[ E_inner[Y] ]
        {base}_aleatoric_std   – sqrt( E_runs[ Var_inner[Y] ] )
        {base}_epistemic_std   – sqrt( Var_runs[ E_inner[Y] ] )

    where base = f"{scenario}_{metric_key}".
    """

    def __init__(self, scenario: str, metric_key: str):
        self.scenario   = scenario
        self.metric_key = metric_key
        self.base       = f"{scenario}_{metric_key}"
        # group_tuple → list of (mean_val, ale_std, epi_std, upn)
        self.data: dict = {}

    @property
    def mean_col(self):  return f"{self.base}_mean"
    @property
    def ale_col(self):   return f"{self.base}_aleatoric_std"
    @property
    def epi_col(self):   return f"{self.base}_epistemic_std"

    def update(self, df: pd.DataFrame):
        """Ingest a pooled (per-building) DataFrame slice."""
        if self.mean_col not in df.columns:
            return

        if 'upn' not in df.columns:
            df = df.copy()
            df['upn'] = df.index

        # Graceful degradation if uncertainty cols are missing
        for col in (self.ale_col, self.epi_col):
            if col not in df.columns:
                df = df.copy()
                df[col] = 0.0
        
        needed = GROUP_COLS + [self.mean_col, self.ale_col, self.epi_col, 'upn']
        df_sub = df.dropna(subset=GROUP_COLS + [self.mean_col])
        if df_sub.empty:
            return
        df_sub = df_sub[needed]
        
        for group_key, grp in df_sub.groupby(GROUP_COLS):
            triples = list(zip(
                grp[self.mean_col].tolist(),
                grp[self.ale_col].fillna(0).tolist(),
                grp[self.epi_col].fillna(0).tolist(),
                grp['upn'].tolist(),
            ))
            if group_key not in self.data:
                self.data[group_key] = []
            self.data[group_key].extend(triples)

    def get_raw_data(self) -> dict:
        return self.data


# ==============================================================================
# 3. GROUP-LEVEL STATS  (law of total variance applied at the group level)
# ==============================================================================

def compute_group_stats(
    raw_data_dict: dict,
    group_indices: list,
    col_names: list,
) -> pd.DataFrame:
    """
    Aggregate pre-pooled per-building triples into group-level stats.

    For a group of n buildings the per-building-average metric has:

        ale_group  = sqrt( mean_b(ale_std_b²) / n )
            SE of the mean under independence — irreducible aleatoric noise.

        epi_group  = sqrt( mean_b(epi_std_b²) / n )
            Propagated epistemic uncertainty on the group mean.
            Treats buildings' epistemic uncertainties as independent
            (conservative for global-factor drivers).

        combined   = sqrt( ale_group² + epi_group² )   [law of total variance]

    Normal-approximation CI bands (used in plots):
        mean ± ale_group      → inner blue band
        mean ± epi_group      → epistemic whiskers
        mean ± combined       → outer red band

    Parameters
    ----------
    raw_data_dict : {group_tuple: [(mean_val, ale_std, epi_std, upn), ...]}
    group_indices : which indices of the 3-element GROUP_COLS key to use
    col_names     : column names for the selected group dimensions

    Returns
    -------
    DataFrame: [*col_names, mean_val, ale_std, epi_std, combined_std,
                epi_lo, epi_hi, n_buildings]
    """
    # --- 1. Re-key by requested sub-grouping ---
    merged: dict = {}
    for key_tuple, triples in raw_data_dict.items():
        new_key = tuple(key_tuple[i] for i in group_indices)
        if new_key not in merged:
            merged[new_key] = []
        merged[new_key].extend(triples)

    rows = []
    for key, triples in merged.items():
        if not triples:
            continue

        df_tmp = pd.DataFrame(triples, columns=['mean_val', 'ale_std', 'epi_std', 'upn'])

        # Deduplicate: one row per building (average if same upn appears across files)
        df_bld = df_tmp.groupby('upn').agg(
            mean_val=('mean_val', 'mean'),
            ale_std =('ale_std',  'mean'),
            epi_std =('epi_std',  'mean'),
        )

        n = len(df_bld)
        if n == 0:
            continue

        means    = df_bld['mean_val'].to_numpy()
        ale_stds = df_bld['ale_std'].to_numpy()
        epi_stds = df_bld['epi_std'].to_numpy()

        group_mean = float(np.mean(means))
        ale_group  = float(np.sqrt(np.mean(ale_stds ** 2) / n))
        epi_group  = float(np.sqrt(np.mean(epi_stds ** 2) / n))
        comb_group = float(np.sqrt(ale_group ** 2 + epi_group ** 2))

        row = dict(zip(col_names, key))
        row.update({
            'mean_val':     group_mean,
            'ale_std':      ale_group,
            'epi_std':      epi_group,
            'combined_std': comb_group,
            'epi_lo':       group_mean - epi_group,
            'epi_hi':       group_mean + epi_group,
            'n_buildings':  n,
        })
        rows.append(row)

    return pd.DataFrame(rows)


# ==============================================================================
# 4. PLOTTING
# ==============================================================================

def _draw_uncertainty_bars(ax, x_positions, df_plot, offset, bar_width, color):
    """
    Draw bars + stacked uncertainty bands for one series.

    Visual layers (back → front):
      Outer red fill    = mean ± combined_std   (aleatoric + epistemic)
      Inner blue fill   = mean ± ale_std        (aleatoric only)
      Black whiskers    = mean ± epi_std        (epistemic only)
      Bar               = group mean
    """
    means = df_plot['mean_val'].to_numpy(dtype=float)
    ale   = df_plot['ale_std'].to_numpy(dtype=float)
    epi   = df_plot['epi_std'].to_numpy(dtype=float)
    comb  = df_plot['combined_std'].to_numpy(dtype=float)
    xc    = x_positions + offset
    half  = bar_width / 2
    cap   = bar_width * 0.28

    ax.bar(xc, means, width=bar_width, color=color, alpha=0.78, zorder=3)

    for xi, m, a, e, c in zip(xc, means, ale, epi, comb):
        # Outer band: combined uncertainty
        ax.fill_between(
            [xi - half, xi + half], [m - c, m - c], [m + c, m + c],
            color=COLOUR_COMBINED, alpha=ALPHA_OUTER, zorder=2,
        )
        # Inner band: aleatoric only
        ax.fill_between(
            [xi - half, xi + half], [m - a, m - a], [m + a, m + a],
            color=COLOUR_ALE, alpha=ALPHA_INNER, zorder=2,
        )
        # Epistemic whisker caps (±1σ epi)
        for sign in (+1, -1):
            y = m + sign * e
            ax.plot([xi - cap, xi + cap], [y, y],
                    color='#222222', lw=1.3, zorder=4)
        ax.plot([xi, xi], [m - e, m + e],
                color='#222222', lw=0.8, ls='--', alpha=0.55, zorder=4)


def _uncertainty_legend_patches():
    return [
        mpatches.Patch(color=COLOUR_ALE,      alpha=ALPHA_INNER + 0.2,
                       label='\u00b11\u03c3 aleatoric (irreducible)'),
        mpatches.Patch(color=COLOUR_COMBINED,  alpha=ALPHA_OUTER + 0.2,
                       label='\u00b11\u03c3 combined (ale \u2295 epi)'),
        plt.Line2D([0], [0], color='#222222', lw=1.3,
                   label='\u00b11\u03c3 epistemic (model uncertainty)'),
    ]


def _finalise_plot(ax, x_pos, x_labels, xlabel, ylabel, title,
                   wall_types, colors, rotate=False):
    wall_patches = [
        mpatches.Patch(color=colors[i], alpha=0.78, label=w)
        for i, w in enumerate(wall_types)
    ]
    ax.legend(handles=wall_patches + _uncertainty_legend_patches(),
              fontsize=8, ncol=2, loc='upper left')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45 if rotate else 0,
                       ha='right' if rotate else 'center')
    ax.margins(y=0.20)
    ax.grid(axis='y', alpha=0.3)


def plot_stats_by_decile(df, scenario_name, metric_name, y_label, output_path):
    if df.empty:
        return
    df = df.copy()
    df['decile_numeric'] = pd.to_numeric(df['avg_gas_percentile'], errors='coerce')
    df = df.sort_values(['inferred_insulation_type', 'decile_numeric'])

    wall_types = sorted(df['inferred_insulation_type'].dropna().unique())
    deciles    = sorted(df['decile_numeric'].dropna().unique())
    n_types    = len(wall_types)
    if not n_types:
        return

    fig, ax = plt.subplots(figsize=(13, 7))
    bar_width = 0.6 if n_types == 1 else 0.8 / n_types
    x_pos  = np.arange(len(deciles))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_types))

    for i, wt in enumerate(wall_types):
        sub = (
            df[df['inferred_insulation_type'] == wt]
            .set_index('decile_numeric').reindex(deciles).reset_index()
        )
        offset = 0.0 if n_types == 1 else (i - n_types / 2 + 0.5) * bar_width
        _draw_uncertainty_bars(ax, x_pos, sub, offset, bar_width, colors[i])

    _finalise_plot(
        ax, x_pos, [int(d) for d in deciles],
        xlabel='Gas Usage Decile',
        ylabel=f'Mean {y_label} (per building)',
        title=(f"{SCENARIO_DISPLAY_NAMES.get(scenario_name, scenario_name)}"
               f" \u2014 {y_label} by Decile"),
        wall_types=wall_types, colors=colors,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"  Saved decile plot: {output_path}")


def plot_stats_by_premise_type(df, scenario_name, metric_name, y_label, output_path):
    if df.empty:
        return
    df = df.copy()
    df['premise_type'] = pd.Categorical(
        df['premise_type'], categories=TYPOLOGIES, ordered=True
    )
    df = df.sort_values(['inferred_insulation_type', 'premise_type'])

    present    = [t for t in TYPOLOGIES if t in df['premise_type'].unique()]
    wall_types = sorted(df['inferred_insulation_type'].dropna().unique())
    n_types    = len(wall_types)
    if not n_types:
        return

    fig, ax = plt.subplots(figsize=(17, 9))
    bar_width = 0.6 if n_types == 1 else 0.8 / n_types
    x_pos  = np.arange(len(present))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_types))

    for i, wt in enumerate(wall_types):
        sub = (
            df[df['inferred_insulation_type'] == wt]
            .set_index('premise_type').reindex(present).reset_index()
        )
        offset = 0.0 if n_types == 1 else (i - n_types / 2 + 0.5) * bar_width
        _draw_uncertainty_bars(ax, x_pos, sub, offset, bar_width, colors[i])

    _finalise_plot(
        ax, x_pos, present,
        xlabel='Premise Type',
        ylabel=f'Mean {y_label} (per building)',
        title=(f"{SCENARIO_DISPLAY_NAMES.get(scenario_name, scenario_name)}"
               f" \u2014 {y_label} by Typology"),
        wall_types=wall_types, colors=colors, rotate=True,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"  Saved typology plot: {output_path}")


# ==============================================================================
# 5. HELPERS
# ==============================================================================

def safe_load(filepath, headers=None) -> pd.DataFrame:
    try:
        return (
            pd.read_csv(filepath, names=headers, header=0)
            if headers else pd.read_csv(filepath)
        )
    except Exception as e:
        logging.warning(f"Failed to load {filepath}: {e}")
        return pd.DataFrame()


def _collect_summary_rows(df, scenario, metric_key, group_type, label_col) -> list:
    rows = []
    for _, row in df.iterrows():
        rows.append({
            'scenario':     scenario,
            'metric':       metric_key,
            'group_type':   group_type,
            'group_label':  str(row.get(label_col, '')),
            'wall_type':    row.get('inferred_insulation_type', 'All'),
            'mean_val':     row.get('mean_val'),
            'ale_std':      row.get('ale_std'),
            'epi_std':      row.get('epi_std'),
            'combined_std': row.get('combined_std'),
            'epi_lo':       row.get('epi_lo'),
            'epi_hi':       row.get('epi_hi'),
            'n_buildings':  row.get('n_buildings'),
        })
    return rows


# ==============================================================================
# 6. MAIN PIPELINE
# ==============================================================================


# ==============================================================================
# 4a. TOTALS PLOT — horizontal dot-and-range, one metric per figure
# ==============================================================================

def plot_totals(
    raw_data_dict: dict,
    scenario_name: str,
    metric_key: str,
    y_label: str,
    output_path: str,
):
    """
    Aggregate ALL buildings (ignore group columns) into a single portfolio
    estimate, then draw a horizontal dot-and-range plot.

    Visual encoding
    ---------------
      Thick line   = ±1σ aleatoric  (irreducible noise)
      Thin line    = ±1σ combined   (aleatoric ⊕ epistemic)
      Dot          = group mean
      Annotation   = epistemic share (%) printed to the right
    """
    if not raw_data_dict:
        return

    # Flatten all groups into one pool
    all_triples = [t for triples in raw_data_dict.values() for t in triples]
    df_tmp = pd.DataFrame(all_triples, columns=['mean_val', 'ale_std', 'epi_std', 'upn'])
    df_bld = df_tmp.groupby('upn').agg(
        mean_val=('mean_val', 'mean'),
        ale_std =('ale_std',  'mean'),
        epi_std =('epi_std',  'mean'),
    )
    n = len(df_bld)
    if n == 0:
        return

    means    = df_bld['mean_val'].to_numpy()
    ale_stds = df_bld['ale_std'].to_numpy()
    epi_stds = df_bld['epi_std'].to_numpy()

    mean_val  = float(np.mean(means))
    ale_group = float(np.sqrt(np.mean(ale_stds ** 2) / n))
    epi_group = float(np.sqrt(np.mean(epi_stds ** 2) / n))
    comb      = float(np.sqrt(ale_group ** 2 + epi_group ** 2))
    epi_share = 100 * epi_group ** 2 / (ale_group ** 2 + epi_group ** 2) if comb > 0 else 0

    display = SCENARIO_DISPLAY_NAMES.get(scenario_name, scenario_name)

    fig, ax = plt.subplots(figsize=(9, 3))

    y = 0  # single row

    # Combined range (thin)
    ax.plot(
        [mean_val - comb, mean_val + comb], [y, y],
        color='#d6604d', lw=2.0, solid_capstyle='round', zorder=2,
        label=f'±1σ combined  ({comb:.2g})',
    )
    # Aleatoric range (thick)
    ax.plot(
        [mean_val - ale_group, mean_val + ale_group], [y, y],
        color='#4393c3', lw=6.0, solid_capstyle='round', zorder=3,
        label=f'±1σ aleatoric ({ale_group:.2g})',
    )
    # Epistemic caps
    for sign in (+1, -1):
        ax.plot(
            [mean_val + sign * epi_group, mean_val + sign * epi_group],
            [y - 0.18, y + 0.18],
            color='#333333', lw=1.8, zorder=4,
        )
    # Central dot
    ax.scatter([mean_val], [y], color='white', edgecolors='#222222',
               s=80, zorder=5, label=f'Mean = {mean_val:.3g}')

    # Annotation: epistemic share
    ax.text(
        mean_val + comb * 1.05, y,
        f'{epi_share:.0f}% epistemic',
        va='center', ha='left', fontsize=10, color='#555555',
    )

    ax.set_yticks([])
    ax.set_xlabel(y_label, fontsize=11)
    ax.set_title(
        f'{display}  —  {y_label}  (n={n:,} buildings)',
        fontsize=12, fontweight='bold',
    )
    ax.legend(fontsize=9, loc='upper left', framealpha=0.6)
    ax.margins(x=0.18, y=0.6)
    ax.grid(axis='x', alpha=0.3)
    ax.spines[['top', 'right', 'left']].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"  Saved totals plot: {output_path}")


# ==============================================================================
# 4b. DECOMPOSITION HEATMAP — epistemic share as colour, CV as annotation
# ==============================================================================

def plot_epistemic_heatmap(
    df: pd.DataFrame,
    group_col: str,          # 'avg_gas_percentile' or 'premise_type'
    scenario_name: str,
    metric_key: str,
    y_label: str,
    output_path: str,
):
    """
    Heatmap: rows = group labels, columns = inferred_insulation_type.
    Colour   = epistemic share (0–1)  →  0 = aleatoric-dominated (blue),
                                         1 = epistemic-dominated  (red)
    Cell text = CV (combined_std / |mean|) as a percentage, showing
                overall uncertainty magnitude.

    If only one wall type exists the column dimension collapses gracefully.
    """
    if df.empty:
        return

    df = df.copy()
    df['epi_share'] = (
        df['epi_std'] ** 2 / (df['ale_std'] ** 2 + df['epi_std'] ** 2)
    ).where((df['ale_std'] ** 2 + df['epi_std'] ** 2) > 0)
    df['cv'] = (df['combined_std'] / df['mean_val'].abs()).where(df['mean_val'] != 0)

    wall_types = sorted(df['inferred_insulation_type'].dropna().unique())

    # Ordered group labels
    if group_col == 'avg_gas_percentile':
        group_order = sorted(df[group_col].dropna().unique())
    else:
        group_order = [t for t in TYPOLOGIES if t in df[group_col].unique()]

    n_rows = len(group_order)
    n_cols = len(wall_types)
    if n_rows == 0 or n_cols == 0:
        return

    epi_matrix = np.full((n_rows, n_cols), np.nan)
    cv_matrix  = np.full((n_rows, n_cols), np.nan)

    for ci, wt in enumerate(wall_types):
        sub = df[df['inferred_insulation_type'] == wt].set_index(group_col)
        for ri, g in enumerate(group_order):
            if g in sub.index:
                epi_matrix[ri, ci] = sub.loc[g, 'epi_share']
                cv_matrix[ri, ci]  = sub.loc[g, 'cv']

    fig_h = max(4, n_rows * 0.55 + 2)
    fig_w = max(5, n_cols * 2.2 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        epi_matrix,
        aspect='auto',
        cmap='RdBu_r',   # red=epistemic-dominated, blue=aleatoric-dominated
        vmin=0, vmax=1,
        interpolation='nearest',
    )

    # Annotate cells with CV (%)
    for ri in range(n_rows):
        for ci in range(n_cols):
            cv_val = cv_matrix[ri, ci]
            epi_val = epi_matrix[ri, ci]
            if not np.isnan(cv_val):
                # White text on dark cells, dark on light
                text_color = 'white' if (epi_val > 0.65 or epi_val < 0.35) else '#222222'
                ax.text(
                    ci, ri,
                    f'{cv_val * 100:.0f}%',
                    ha='center', va='center',
                    fontsize=8.5, color=text_color, fontweight='bold',
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Epistemic share of variance', fontsize=10)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0% (all aleatoric)', '25%', '50%', '75%', '100% (all epistemic)'])

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(wall_types, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(
        [str(g) for g in group_order],
        fontsize=8.5 if group_col == 'premise_type' else 9,
    )
    xlabel = 'Insulation Type'
    ylabel = 'Gas Decile' if group_col == 'avg_gas_percentile' else 'Typology'
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    display = SCENARIO_DISPLAY_NAMES.get(scenario_name, scenario_name)
    ax.set_title(
        f'{display}  —  {y_label}\n'
        f'Colour = epistemic share   |   Cell text = CV (combined σ / |mean|)',
        fontsize=11, fontweight='bold',
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"  Saved heatmap: {output_path}")


#main pipeline 

def run_pipeline():
    logging.info(f"Scanning: {LOG_DIR}")
    files = glob.glob(LOG_DIR)
    files=files[0:5]
    logging.info(f"Found {len(files)} files.")

    headers = None
    if is_hpc and REFERENCE_FILE:
        try:
            with open(REFERENCE_FILE, 'r') as f:
                headers = next(csv.reader(f))
        except Exception:
            logging.warning("Could not read reference headers.")

    # ------------------------------------------------------------------
    # Initialise one accumulator per (scenario, metric) pair
    # ------------------------------------------------------------------
    accumulators: dict = {
        (scn, mk): GroupedStatsAccumulator(scn, mk)
        for scn in SCENARIOS
        for mk in METRICS_MAP
    }

    # ------------------------------------------------------------------
    # Load → pool → accumulate
    # ------------------------------------------------------------------
    for i, fp in enumerate(files):
        if i % 10 == 0:
            logging.info(f"  Loading file {i}/{len(files)} ...")

        df_raw = safe_load(fp, headers)
        if df_raw.empty:
            logging.warning('df raw empty') 
            continue
        else :
            logging.info('raw loaded') 
    
      
 
        if not set(GROUP_COLS).issubset(df_raw.columns):
            continue

        df_raw = df_raw[df_raw['premise_type'].isin(TYPOLOGIES)]
 
        
        # ── PREPROCESSING: collapse n epistemic runs → one row per building ──
        df_pooled = pool_epistemic_runs_decomposed(df_raw, SCENARIOS)
        
     
        
        # pool_epistemic_runs_decomposed re-attaches metadata via COLS_KEEP.
        # If GROUP_COLS are not in COLS_KEEP, merge them back from df_raw here:
        #if not set(GROUP_COLS).issubset(df_pooled.columns):
        meta = df_raw[['upn'] + ['inferred_insulation_type'] ].drop_duplicates()
        df_pooled = df_pooled.merge(meta, on='upn', how='left')

        if df_pooled.empty:
            print('df_pooled empty') 
            continue

        for acc in accumulators.values():
            acc.update(df_pooled)

        del df_raw, df_pooled
        gc.collect()

    logging.info("Accumulation complete. Computing stats and generating plots...")

    # ------------------------------------------------------------------
    # Compute stats, plot, collect summary rows
    # ------------------------------------------------------------------
    summary_rows: list = []

    for (scn_name, metric_key), acc in accumulators.items():
        raw_data = acc.get_raw_data()
        if not raw_data:
            logging.info(f"  [skip] {scn_name} – {metric_key}: no data")
            continue

        logging.info(f"  Processing: {scn_name} [{metric_key}]")
        is_wall = 'wall' in scn_name
        y_lbl   = METRIC_LABELS.get(metric_key, metric_key)

        # ---- A. Decile breakout ----------------------------------------
        idx_d = [0, 2] if is_wall else [0]
        col_d = (['avg_gas_percentile', 'inferred_insulation_type']
                 if is_wall else ['avg_gas_percentile'])

        df_dec = compute_group_stats(raw_data, idx_d, col_d)
        if not is_wall:
            df_dec['inferred_insulation_type'] = 'All'

        df_dec.to_csv(
            os.path.join(OUTPUT_BASE, f'{scn_name}_{metric_key}_stats_decile.csv'),
            index=False,
        )
        plot_stats_by_decile(
            df_dec, scn_name, metric_key, y_lbl,
            os.path.join(OUTPUT_BASE, f'{scn_name}_{metric_key}_decile.png'),
        )
        summary_rows.extend(
            _collect_summary_rows(df_dec, scn_name, metric_key,
                                  'decile', 'avg_gas_percentile')
        )

        # ---- B. Typology breakout --------------------------------------
        idx_p = [1, 2] if is_wall else [1]
        col_p = (['premise_type', 'inferred_insulation_type']
                 if is_wall else ['premise_type'])

        df_prm = compute_group_stats(raw_data, idx_p, col_p)
        if not is_wall:
            df_prm['inferred_insulation_type'] = 'All'

        df_prm.to_csv(
            os.path.join(OUTPUT_BASE, f'{scn_name}_{metric_key}_stats_premise.csv'),
            index=False,
        )
        plot_stats_by_premise_type(
            df_prm, scn_name, metric_key, y_lbl,
            os.path.join(OUTPUT_BASE, f'{scn_name}_{metric_key}_premise.png'),
        )
        summary_rows.extend(
            _collect_summary_rows(df_prm, scn_name, metric_key,
                                  'typology', 'premise_type')
        )
    
        # ---- TOTALS (new) -----------------------------------------------
        plot_totals(
            raw_data,
            scn_name, metric_key, y_lbl,
            os.path.join(OUTPUT_BASE, f'{scn_name}_{metric_key}_totals.png'),
        )
    
        # ---- A. Decile breakout (heatmap replaces bar chart) -------------
        # ... (keep df_dec computation unchanged) ...
        plot_epistemic_heatmap(
            df_dec, 'avg_gas_percentile',
            scn_name, metric_key, y_lbl,
            os.path.join(OUTPUT_BASE, f'{scn_name}_{metric_key}_decile_heatmap.png'),
        )
    
        # ---- B. Typology breakout (heatmap) -----------------------------
        # ... (keep df_prm computation unchanged) ...
        plot_epistemic_heatmap(
            df_prm, 'premise_type',
            scn_name, metric_key, y_lbl,
            os.path.join(OUTPUT_BASE, f'{scn_name}_{metric_key}_premise_heatmap.png'),
        )
        
    # ------------------------------------------------------------------
    # Write master summary CSV
    # ------------------------------------------------------------------
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        df_sum['scenario_display'] = df_sum['scenario'].map(SCENARIO_DISPLAY_NAMES)
        df_sum['metric_label']     = df_sum['metric'].map(METRIC_LABELS)

        denom = df_sum['ale_std'] ** 2 + df_sum['epi_std'] ** 2
        df_sum['epistemic_share'] = (
            df_sum['epi_std'] ** 2 / denom
        ).where(denom > 0)
        df_sum['cv_combined'] = (
            df_sum['combined_std'] / df_sum['mean_val'].abs()
        ).where(df_sum['mean_val'] != 0)

        summary_path = os.path.join(OUTPUT_BASE, 'SUMMARY_all_scenarios_metrics.csv')
        df_sum.to_csv(summary_path, index=False)
        logging.info(f"Summary CSV: {summary_path}  ({len(df_sum)} rows)")
    else:
        logging.warning("No summary rows — check data paths and column names.")

    logging.info(f"Done. All outputs saved to: {OUTPUT_BASE}")


if __name__ == '__main__':
    run_pipeline()