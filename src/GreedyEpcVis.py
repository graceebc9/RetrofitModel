import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import os 

PERSONA_COLORS = {
    'low_deprived': '#009E73',
    'med_deprived': '#E69F00',
    'high_deprived': '#D55E00'
}

method_name = 'Opt.T'

METHOD_COLORS = {
    method_name: '#56B4E9',
    'EPC': '#CC79A7'
}

# Column names (pass-through from solver input)
total_co2_saved_col = 'mean_total_co2_saved' 
total_co2_saved_col_std = 'std_total_co2_saved'

capex_per_net_ton_mean_col = 'mean_capex_per_net_ton'
capex_per_net_ton_std_col = 'std_capex_per_net_ton'


def run_epc_vis(pareto_runs_folder, base_dir_outputs, million_budget, prob_loft, equity_floor): 
    """
    Load Pareto-selected and EPC-selected results and generate comparison plots.
    
    Parameters
    ----------
    pareto_runs_folder : str
        Root folder containing per-budget/loft run directories.
    base_dir_outputs : str
        Root folder for saving visualisation outputs.
    million_budget : float
        Budget in millions (e.g. 50 for £50M).
    prob_loft : float
        Loft probability used in this run.
    equity_floor : int or float
        Which equity floor's Pareto result to compare against EPC.
    """
    million_budget_str = str(int(million_budget)) if million_budget == int(million_budget) else str(million_budget)
    eq_label = f"{int(equity_floor)}"
    
    output_dir = os.path.join(
        pareto_runs_folder, 
        f'budget_{million_budget_str}M__loft_{prob_loft}'
    )
    
    
    selected_path = os.path.join(output_dir, f'selected_projects_eq{eq_label}.csv')
    epc_path = os.path.join(output_dir, 'epc_random_selection.csv')
    
    print(f'trying paths: {selected_path} \n and. {epc_path}  ')
    df = pd.read_csv(selected_path) 
    epc = pd.read_csv(epc_path) 
    print(f'Loaded Pareto (eq={eq_label}): {len(df)} rows from {selected_path}')
    print(f'Loaded EPC: {len(epc)} rows from {epc_path}')
    
    vis_output_dir = os.path.join(
        base_dir_outputs, 
        'epc_comparisons',
        f'budget_{million_budget_str}M__loft_{prob_loft}__eq_{eq_label}'
    )
    
    generate_all_aggregation_plots(df, epc, output_dir=vis_output_dir, save=True)


# ---------------------------------------------------------------------------
# TOTAL COMPARISON
# ---------------------------------------------------------------------------

def plot_total_comparison(df1, df2, column_mean, column_std, output_dir=None, save=False):
    """
    Plot overall total comparison for a single column with error bars.
    Handles unit conversion for Capex to Millions (£M).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    is_capex = 'capex' in column_mean.lower()
    scale_factor = 1e6 if is_capex else 1.0
    unit_label = ' (£M)' if is_capex else ''
    fmt_str = '{:,.1f}' if is_capex else '{:,.0f}'
    
    total_mean_df1 = df1[column_mean].sum() / scale_factor
    total_mean_df2 = df2[column_mean].sum() / scale_factor
    
    total_std_df1 = np.sqrt((df1[column_std]**2).sum()) / scale_factor
    total_std_df2 = np.sqrt((df2[column_std]**2).sum()) / scale_factor
    
    x_positions = [0, 1]
    means = [total_mean_df1, total_mean_df2]
    stds = [total_std_df1, total_std_df2]
    labels = [method_name, 'EPC']
    colors = [METHOD_COLORS[method_name], METHOD_COLORS['EPC']]
    
    bars = ax.bar(x_positions, means,
                  yerr=stds,
                  capsize=8,
                  color=colors,
                  alpha=0.7,
                  edgecolor='black',
                  linewidth=2,
                  error_kw={'linewidth': 2, 'capthick': 2})
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2., mean + std + (mean * 0.02),
                f'{fmt_str.format(mean)} ± {fmt_str.format(std)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ylabel_text = column_mean.replace('_', ' ').title() + unit_label
    ax.set_ylabel(ylabel_text, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    diff = total_mean_df2 - total_mean_df1
    diff_pct = (diff / total_mean_df1) * 100 if total_mean_df1 != 0 else 0
    ax.text(0.75, 0.85, f'Difference: {fmt_str.format(diff)} ({diff_pct:+.1f}%)', 
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    plt.tight_layout()
    
    if save and output_dir:
        filepath = Path(output_dir) / f'{column_mean}_total_comparison.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# GROUP COMPARISON
# ---------------------------------------------------------------------------

def plot_by_group(df1, df2, column_mean, column_std, group_col, group_label, output_dir=None, save=False):
    """
    Plot totals split by a grouping column with error bars.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    is_capex = 'capex' in column_mean.lower()
    scale_factor = 1e6 if is_capex else 1.0
    unit_label = ' (£M)' if is_capex else ''
    fmt_str = '{:,.1f}' if is_capex else '{:,.0f}'
    
    def aggregate_with_std(df):
        grouped = df.groupby(group_col).agg(
            total_mean=(column_mean, 'sum'),
            total_std=(column_std, lambda x: np.sqrt((x**2).sum()))
        )
        grouped['total_mean'] /= scale_factor
        grouped['total_std'] /= scale_factor
        return grouped.sort_index()
    
    df1_agg = aggregate_with_std(df1)
    df2_agg = aggregate_with_std(df2)
    
    all_groups = sorted(set(df1_agg.index) | set(df2_agg.index))
    
    df1_means = [df1_agg.loc[g, 'total_mean'] if g in df1_agg.index else 0 for g in all_groups]
    df1_stds = [df1_agg.loc[g, 'total_std'] if g in df1_agg.index else 0 for g in all_groups]
    df2_means = [df2_agg.loc[g, 'total_mean'] if g in df2_agg.index else 0 for g in all_groups]
    df2_stds = [df2_agg.loc[g, 'total_std'] if g in df2_agg.index else 0 for g in all_groups]
    
    x = np.arange(len(all_groups))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df1_means, width, 
                   yerr=df1_stds, capsize=4,
                   label=method_name, 
                   color=METHOD_COLORS[method_name], 
                   alpha=0.7, edgecolor='black',
                   error_kw={'linewidth': 1.5, 'capthick': 1.5})
    
    bars2 = ax.bar(x + width/2, df2_means, width,
                   yerr=df2_stds, capsize=4,
                   label='EPC', 
                   color=METHOD_COLORS['EPC'], 
                   alpha=0.7, edgecolor='black',
                   error_kw={'linewidth': 1.5, 'capthick': 1.5})
    
    for bars, stds in [(bars1, df1_stds), (bars2, df2_stds)]:
        for bar, std in zip(bars, stds):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + std + (height * 0.02),
                        fmt_str.format(height),
                        ha='center', va='bottom', fontsize=9, rotation=0)
    
    ax.set_xlabel(group_label, fontsize=12, fontweight='bold')
    ax.set_ylabel(column_mean.replace('_', ' ').title() + unit_label, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(all_groups, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save and output_dir:
        filepath = Path(output_dir) / f'{column_mean}_by_{group_col}.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# HEATMAP
# ---------------------------------------------------------------------------

def plot_heatmap_comparison(df1, df2, column, output_dir=None, save=False):
    """Plot heatmap showing totals by socio persona and energy rating."""
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    
    is_capex = 'capex' in column.lower()
    scale_factor = 1e6 if is_capex else 1.0
    unit_label = ' (£M)' if is_capex else ''
    fmt_str = '.1f' if is_capex else '.0f'
    
    pivot_df1 = df1.pivot_table(
        values=column, index='meta_socio_persona', 
        columns='CURRENT_ENERGY_RATING', aggfunc='sum', fill_value=0
    ) / scale_factor
    
    pivot_df2 = df2.pivot_table(
        values=column, index='meta_socio_persona', 
        columns='CURRENT_ENERGY_RATING', aggfunc='sum', fill_value=0
    ) / scale_factor
    
    sns.heatmap(pivot_df1, annot=True, fmt=fmt_str, cmap='YlOrRd', 
                ax=axes[0], cbar_kws={'label': column.replace('_', ' ').title() + unit_label})
    axes[0].set_title(method_name, fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Energy Rating', fontsize=11)
    axes[0].set_ylabel('Socio Persona', fontsize=11)
    
    sns.heatmap(pivot_df2, annot=True, fmt=fmt_str, cmap='YlOrRd', 
                ax=axes[1], cbar_kws={'label': column.replace('_', ' ').title() + unit_label})
    axes[1].set_title('EPC', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Energy Rating', fontsize=11)
    axes[1].set_ylabel('Socio Persona', fontsize=11)
    
    plt.tight_layout()
    
    if save and output_dir:
        filepath = Path(output_dir) / f'{column}_heatmap_comparison.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# BUILDING COUNTS
# ---------------------------------------------------------------------------

def plot_building_counts_by_percentile(df1, df2, output_dir=None, save=False):
    """Plot count of buildings by gas percentile."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df1_counts = df1['avg_gas_percentile'].value_counts().sort_index()
    df2_counts = df2['avg_gas_percentile'].value_counts().sort_index()
    
    all_percentiles = sorted(set(df1_counts.index) | set(df2_counts.index))
    
    df1_values = [df1_counts.get(p, 0) for p in all_percentiles]
    df2_values = [df2_counts.get(p, 0) for p in all_percentiles]
    
    x = np.arange(len(all_percentiles))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df1_values, width, label=method_name, 
                   color=METHOD_COLORS[method_name], alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, df2_values, width, label='EPC', 
                   color=METHOD_COLORS['EPC'], alpha=0.7, edgecolor='black')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Gas Percentile', fontsize=12, fontweight='bold')
    ax.set_ylabel('Building Count', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(p)}' for p in all_percentiles])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save and output_dir:
        filepath = Path(output_dir) / 'building_count_by_percentile.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
    else:
        plt.show()


def plot_building_counts_by_persona(df1, df2, output_dir=None, save=False):
    """Plot count of buildings by socio persona."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df1_counts = df1['meta_socio_persona'].value_counts().sort_index()
    df2_counts = df2['meta_socio_persona'].value_counts().sort_index()
    
    all_personas = sorted(set(df1_counts.index) | set(df2_counts.index))
    
    df1_values = [df1_counts.get(p, 0) for p in all_personas]
    df2_values = [df2_counts.get(p, 0) for p in all_personas]
    
    x = np.arange(len(all_personas))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df1_values, width, label=method_name, 
                   color=METHOD_COLORS[method_name], alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, df2_values, width, label='EPC', 
                   color=METHOD_COLORS['EPC'], alpha=0.7, edgecolor='black')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Socio Persona', fontsize=12, fontweight='bold')
    ax.set_ylabel('Building Count', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(all_personas, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save and output_dir:
        filepath = Path(output_dir) / 'building_count_by_persona.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
    else:
        plt.show()


def plot_building_counts_by_energy_rating(df1, df2, output_dir=None, save=False):
    """Plot count of buildings by energy rating."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df1_counts = df1['CURRENT_ENERGY_RATING'].value_counts().sort_index()
    df2_counts = df2['CURRENT_ENERGY_RATING'].value_counts().sort_index()
    
    all_ratings = sorted(set(df1_counts.index) | set(df2_counts.index))
    
    df1_values = [df1_counts.get(r, 0) for r in all_ratings]
    df2_values = [df2_counts.get(r, 0) for r in all_ratings]
    
    x = np.arange(len(all_ratings))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df1_values, width, label=method_name, 
                   color=METHOD_COLORS[method_name], alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, df2_values, width, label='EPC', 
                   color=METHOD_COLORS['EPC'], alpha=0.7, edgecolor='black')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Current Energy Rating', fontsize=12, fontweight='bold')
    ax.set_ylabel('Building Count', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(all_ratings)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save and output_dir:
        filepath = Path(output_dir) / 'building_count_by_energy_rating.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# INTERVENTION PLOTS
# ---------------------------------------------------------------------------

def plot_interventions_by_percentile(df1, df2, output_dir=None, save=False):
    """Plot stacked bar of intervention counts by gas percentile."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    
    df1_crosstab = pd.crosstab(df1['avg_gas_percentile'], df1['intervention'])
    df2_crosstab = pd.crosstab(df2['avg_gas_percentile'], df2['intervention'])
    
    all_interventions = sorted(set(df1_crosstab.columns) | set(df2_crosstab.columns))
    
    df1_crosstab = df1_crosstab.reindex(columns=all_interventions, fill_value=0)
    df2_crosstab = df2_crosstab.reindex(columns=all_interventions, fill_value=0)
    
    df1_crosstab.plot(kind='bar', stacked=True, ax=axes[0], 
                      colormap='tab10', edgecolor='black', linewidth=0.5, legend=False)
    axes[0].set_title('Consumption Targeting', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Gas Percentile', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].tick_params(axis='x', rotation=0)
    
    df2_crosstab.plot(kind='bar', stacked=True, ax=axes[1], 
                      colormap='tab10', edgecolor='black', linewidth=0.5, legend=False)
    axes[1].set_title('EPC Targeting', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Gas Percentile', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=0)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Intervention', loc='upper right', frameon=True)
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    
    if save and output_dir:
        filepath = Path(output_dir) / 'interventions_by_percentile.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
    else:
        plt.show()


def plot_interventions_by_persona(df1, df2, output_dir=None, save=False):
    """Plot stacked bar of intervention counts by socio persona."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    
    df1_crosstab = pd.crosstab(df1['meta_socio_persona'], df1['intervention'])
    df2_crosstab = pd.crosstab(df2['meta_socio_persona'], df2['intervention'])
    
    all_interventions = sorted(set(df1_crosstab.columns) | set(df2_crosstab.columns))
    
    df1_crosstab = df1_crosstab.reindex(columns=all_interventions, fill_value=0)
    df2_crosstab = df2_crosstab.reindex(columns=all_interventions, fill_value=0)
    
    df1_crosstab.plot(kind='bar', stacked=True, ax=axes[0], 
                      colormap='tab10', edgecolor='black', linewidth=0.5, legend=False)
    axes[0].set_title('Consumption Targeting', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Socio Persona', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].tick_params(axis='x', rotation=45)
    
    df2_crosstab.plot(kind='bar', stacked=True, ax=axes[1], 
                      colormap='tab10', edgecolor='black', linewidth=0.5, legend=False)
    axes[1].set_title('EPC Targeting', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Socio Persona', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=45)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Intervention', loc='center right', 
               bbox_to_anchor=(0.6, 0.8), frameon=True)
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    
    if save and output_dir:
        filepath = Path(output_dir) / 'interventions_by_persona.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
    else:
        plt.show()


def plot_interventions_by_energy_rating(df1, df2, output_dir=None, save=False):
    """Plot stacked bar of intervention counts by energy rating."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    
    df1_crosstab = pd.crosstab(df1['CURRENT_ENERGY_RATING'], df1['intervention'])
    df2_crosstab = pd.crosstab(df2['CURRENT_ENERGY_RATING'], df2['intervention'])
    
    all_interventions = sorted(set(df1_crosstab.columns) | set(df2_crosstab.columns))
    
    df1_crosstab = df1_crosstab.reindex(columns=all_interventions, fill_value=0)
    df2_crosstab = df2_crosstab.reindex(columns=all_interventions, fill_value=0)
    
    df1_crosstab.plot(kind='bar', stacked=True, ax=axes[0], 
                      colormap='tab10', edgecolor='black', linewidth=0.5, legend=False)
    axes[0].set_title('Consumption Targeting', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Energy Rating', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].tick_params(axis='x', rotation=0)
    
    df2_crosstab.plot(kind='bar', stacked=True, ax=axes[1], 
                      colormap='tab10', edgecolor='black', linewidth=0.5, legend=False)
    axes[1].set_title('EPC Targeting', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Energy Rating', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=0)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Intervention', loc='center left', 
               bbox_to_anchor=(0.75, 0.85), frameon=True)
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    
    if save and output_dir:
        filepath = Path(output_dir) / 'interventions_by_energy_rating.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CO2 / CAPEX BY INTERVENTION
# ---------------------------------------------------------------------------

def plot_co2_by_intervention(df1, df2, output_dir=None, save=False):
    """Plot total CO2 saved split by intervention type with error bars."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def aggregate_with_std(df):
        grouped = df.groupby('intervention').agg(
            total_mean=(total_co2_saved_col, 'sum'),
            total_std=(total_co2_saved_col_std, lambda x: np.sqrt((x**2).sum()))
        )
        return grouped.sort_values('total_mean', ascending=False)
    
    df1_agg = aggregate_with_std(df1)
    df2_agg = aggregate_with_std(df2)
    
    all_interventions = sorted(set(df1_agg.index) | set(df2_agg.index))
    
    df1_means = [df1_agg.loc[i, 'total_mean'] if i in df1_agg.index else 0 for i in all_interventions]
    df1_stds = [df1_agg.loc[i, 'total_std'] if i in df1_agg.index else 0 for i in all_interventions]
    df2_means = [df2_agg.loc[i, 'total_mean'] if i in df2_agg.index else 0 for i in all_interventions]
    df2_stds = [df2_agg.loc[i, 'total_std'] if i in df2_agg.index else 0 for i in all_interventions]
    
    x = np.arange(len(all_interventions))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df1_means, width,
                   yerr=df1_stds, capsize=4,
                   label=method_name, 
                   color=METHOD_COLORS[method_name], alpha=0.7, edgecolor='black',
                   error_kw={'linewidth': 1.5, 'capthick': 1.5})
    
    bars2 = ax.bar(x + width/2, df2_means, width,
                   yerr=df2_stds, capsize=4,
                   label='EPC', 
                   color=METHOD_COLORS['EPC'], alpha=0.7, edgecolor='black',
                   error_kw={'linewidth': 1.5, 'capthick': 1.5})
    
    for bars, stds in [(bars1, df1_stds), (bars2, df2_stds)]:
        for bar, std in zip(bars, stds):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + std + (height * 0.02),
                        f'{height:,.0f}', ha='center', va='bottom', fontsize=8, rotation=0)
    
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
        print(f"Saved: {filepath}")
        plt.close()
    else:
        plt.show()


def plot_capex_by_intervention(df1, df2, output_dir=None, save=False):
    """Plot total CAPEX split by intervention type."""
    capex_col = None
    for col in ['mean_total_capex', 'capex', 'total_capex', 'CAPEX']:
        if col in df1.columns and col in df2.columns:
            capex_col = col
            break
    
    if capex_col is None:
        print("Warning: No CAPEX column found, skipping capex_by_intervention plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df1_agg = df1.groupby('intervention')[capex_col].sum().sort_values(ascending=False) / 1e6
    df2_agg = df2.groupby('intervention')[capex_col].sum().sort_values(ascending=False) / 1e6
    
    all_interventions = sorted(set(df1_agg.index) | set(df2_agg.index))
    
    df1_values = [df1_agg.get(i, 0) for i in all_interventions]
    df2_values = [df2_agg.get(i, 0) for i in all_interventions]
    
    x = np.arange(len(all_interventions))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df1_values, width, label=method_name, 
                   color=METHOD_COLORS[method_name], alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, df2_values, width, label='EPC', 
                   color=METHOD_COLORS['EPC'], alpha=0.7, edgecolor='black')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:,.1f}', ha='center', va='bottom', fontsize=8, rotation=0)
    
    ax.set_xlabel('Intervention Type', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Total {capex_col.replace("_", " ").title()} (£M)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_interventions, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save and output_dir:
        filepath = Path(output_dir) / f'{capex_col}_by_intervention.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# MEAN CAPEX PER TON
# ---------------------------------------------------------------------------

def plot_mean_capex_per_ton(df1, df2, output_dir=None, save=False):
    """Plot comparison of mean capex per ton with error bars."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    combined_mean_df1 = df1[capex_per_net_ton_mean_col].mean()
    combined_mean_df2 = df2[capex_per_net_ton_mean_col].mean()
    
    def combined_std(df):
        means = df[capex_per_net_ton_mean_col].values
        stds = df[capex_per_net_ton_std_col].values
        avg_variance = np.mean(stds**2)
        between_variance = np.var(means)
        return np.sqrt(avg_variance + between_variance)
    
    combined_std_df1 = combined_std(df1)
    combined_std_df2 = combined_std(df2)
    
    x_positions = [0, 1]
    means = [combined_mean_df1, combined_mean_df2]
    stds = [combined_std_df1, combined_std_df2]
    labels = [method_name, 'EPC']
    colors = [METHOD_COLORS[method_name], METHOD_COLORS['EPC']]
    
    bars = ax.bar(x_positions, means,
                  yerr=stds, capsize=8,
                  color=colors, alpha=0.7, edgecolor='black', linewidth=2,
                  error_kw={'linewidth': 2, 'capthick': 2})
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    
    for bar, std in zip(bars, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 5,
                f'{height:,.0f} ± {std:,.0f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Capex per Net Ton (£/ton)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    diff = combined_mean_df2 - combined_mean_df1
    diff_pct = (diff / combined_mean_df1) * 100 if combined_mean_df1 != 0 else 0
    ax.text(0.22, 0.95, f'Difference: {diff:,.2f} ({diff_pct:+.1f}%)', 
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    plt.tight_layout()
    
    if save and output_dir:
        filepath = Path(output_dir) / 'capex_per_ton_mean_comparison.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
    else:
        plt.show()


def plot_mean_capex(df1, df2, output_dir=None, save=False):
    """Plot comparison of mean CAPEX."""
    capex_col = None
    for col in ['mean_total_capex', 'capex', 'total_capex', 'CAPEX']:
        if col in df1.columns and col in df2.columns:
            capex_col = col
            break
    
    if capex_col is None:
        print("Warning: No CAPEX column found, skipping mean capex plot")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    mean_df1 = df1[capex_col].mean()
    mean_df2 = df2[capex_col].mean()
    
    bars = ax.bar([method_name, 'EPC'], [mean_df1, mean_df2], 
                   color=[METHOD_COLORS[method_name], METHOD_COLORS['EPC']], 
                   alpha=0.7, edgecolor='black', linewidth=2)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel(f'{capex_col.replace("_", " ").title()} (£)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    diff = mean_df2 - mean_df1
    diff_pct = (diff / mean_df1) * 100 if mean_df1 != 0 else 0
    ax.text(0.22, 0.95, f'Difference: {diff:,.2f} ({diff_pct:+.1f}%)', 
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    median_df1 = df1[capex_col].median()
    median_df2 = df2[capex_col].median()
    ax.text(0.22, 0.87, f'Median: {method_name}={median_df1:,.2f}, EPC={median_df2:,.2f}', 
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
            fontsize=9)
    
    plt.tight_layout()
    
    if save and output_dir:
        filepath = Path(output_dir) / f'{capex_col}_mean_comparison.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# META FUNCTION
# ---------------------------------------------------------------------------

def generate_all_aggregation_plots(df1, df2, output_dir='./plots', save=True):
    """Generate all aggregation/summation comparison plots."""
    if save:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_path.absolute()}\n")
    
    print("Generating aggregation plots...\n")
    
    # Total CO2 comparison
    print("Processing: Total CO2 comparison")
    plot_total_comparison(df1, df2, 
                          column_mean=total_co2_saved_col, 
                          column_std=total_co2_saved_col_std, 
                          output_dir=output_dir, save=save)
    
    # CO2 by group
    print("Processing: CO2 by group breakdowns")
    plot_by_group(df1, df2, total_co2_saved_col, total_co2_saved_col_std, 
                  'meta_socio_persona', 'Persona', output_dir=output_dir, save=save)
    plot_by_group(df1, df2, total_co2_saved_col, total_co2_saved_col_std, 
                  'avg_gas_percentile', 'Gas Consumption Decile', output_dir=output_dir, save=save)
    plot_by_group(df1, df2, total_co2_saved_col, total_co2_saved_col_std, 
                  'CURRENT_ENERGY_RATING', 'Energy Rating', output_dir=output_dir, save=save)
    
    # Heatmap
    plot_heatmap_comparison(df1, df2, total_co2_saved_col, output_dir, save)
    print("Completed: CO2 breakdowns\n")
    
    # Building counts
    print("Processing: Building Counts")
    plot_building_counts_by_percentile(df1, df2, output_dir, save)
    plot_building_counts_by_persona(df1, df2, output_dir, save)
    plot_building_counts_by_energy_rating(df1, df2, output_dir, save)
    print("Completed: Building Counts\n")
    
    # Intervention analysis
    print("Processing: Intervention Analysis")
    plot_interventions_by_percentile(df1, df2, output_dir, save)
    plot_interventions_by_persona(df1, df2, output_dir, save)
    plot_interventions_by_energy_rating(df1, df2, output_dir, save)
    plot_co2_by_intervention(df1, df2, output_dir, save)
    plot_capex_by_intervention(df1, df2, output_dir, save)
    print("Completed: Intervention Analysis\n")
    
    # Mean comparisons
    print("Processing: Mean Value Comparisons")
    plot_mean_capex_per_ton(df1, df2, output_dir, save)
    plot_mean_capex(df1, df2, output_dir, save)
    print("Completed: Mean Value Comparisons\n")
    
    print("All aggregation plots generated!")
    if save:
        print(f"All figures saved to: {Path(output_dir).absolute()}")