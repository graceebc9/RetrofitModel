import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import os 

PERSONA_COLORS = {
    'low_deprived': '#009E73',  # Green
    'med_deprived': '#E69F00',  # Orange
    'high_deprived': '#D55E00'  # Red
}

method_name = 'Opt.T'

METHOD_COLORS = {
    method_name: '#56B4E9',  # Sky Blue (for targeted/consumption targeting)
    'EPC': '#CC79A7'  # Purple/Magenta (for EPC targeting)
}


def run_epc_vis(greedy_runs_folder, base_dir_outputs, million_budget, prob_loft, equity_factor): 
    output_dir = os.path.join(greedy_runs_folder, f'budget_{int(million_budget)}M__loft_{prob_loft}__equity_{equity_factor}')
    selected_path = os.path.join(output_dir, f'selected_projects.csv')
    epc_random_path = os.path.join(output_dir, f'epc_random_selection.csv')
    
    df = pd.read_csv(selected_path) 
    epc = pd.read_csv(epc_random_path) 
    print('df and epc loaded')
    print(selected_path)
    print(df.head())
    print(epc_random_path)
    
    generate_all_aggregation_plots(df, epc, output_dir=f'{base_dir_outputs}/epc_comaprisons/budget_{million_budget}M__loft_{prob_loft}__equity_{equity_factor}', save=True)


def plot_total_comparison(df1, df2, column, output_dir=None, save=False):
    """
    Plot overall total comparison for a single column.
    Handles unit conversion for Capex to Millions (£M).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Check for Capex to handle units
    is_capex = 'capex' in column.lower()
    scale_factor = 1e6 if is_capex else 1.0
    unit_label = ' (£M)' if is_capex else ''
    fmt_str = '{:,.1f}' if is_capex else '{:,.0f}'
    
    # Calculate totals
    total_df1 = df1[column].sum() / scale_factor
    total_df2 = df2[column].sum() / scale_factor
    
    # Create bar chart
    bars = ax.bar([method_name, 'EPC'], [total_df1, total_df2], 
                   color=[METHOD_COLORS[method_name], METHOD_COLORS['EPC']], alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                fmt_str.format(height),
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ylabel_text = column.replace('_', ' ').title() + unit_label
    ax.set_ylabel(ylabel_text, fontsize=12)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add difference annotation
    diff = total_df2 - total_df1
    diff_pct = (diff / total_df1) * 100 if total_df1 != 0 else 0
    ax.text(0.75, 0.85, f'Difference: {diff:,.1f} ({diff_pct:+.1f}%)', 
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    plt.tight_layout()
    
    if save and output_dir:
        filepath = Path(output_dir) / f'{column}_total_comparison.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
    else:
        plt.show()


def plot_by_socio_persona(df1, df2, column, output_dir=None, save=False):
    """
    Plot totals split by meta_socio_persona.
    Handles unit conversion for Capex to Millions (£M).
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Check for Capex to handle units
    is_capex = 'capex' in column.lower()
    scale_factor = 1e6 if is_capex else 1.0
    unit_label = ' (£M)' if is_capex else ''
    fmt_str = '{:,.1f}' if is_capex else '{:,.0f}'
    
    # Aggregate by socio persona
    df1_agg = df1.groupby('meta_socio_persona')[column].sum().sort_index() / scale_factor
    df2_agg = df2.groupby('meta_socio_persona')[column].sum().sort_index() / scale_factor
    
    # Get all unique personas
    all_personas = sorted(set(df1_agg.index) | set(df2_agg.index))
    
    # Prepare data for plotting
    df1_values = [df1_agg.get(p, 0) for p in all_personas]
    df2_values = [df2_agg.get(p, 0) for p in all_personas]
    
    x = np.arange(len(all_personas))
    width = 0.35
    
    # Create grouped bar chart
    bars1 = ax.bar(x - width/2, df1_values, width, label=method_name, 
                   color=METHOD_COLORS[method_name], alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, df2_values, width, label='EPC', 
                   color=METHOD_COLORS['EPC'], alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        fmt_str.format(height),
                        ha='center', va='bottom', fontsize=9, rotation=0)
    
    ax.set_xlabel('Socio Persona', fontsize=12, fontweight='bold')
    ax.set_ylabel(column.replace('_', ' ').title() + unit_label, fontsize=12)
    
    ax.set_xticks(x)
    ax.set_xticklabels(all_personas, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save and output_dir:
        filepath = Path(output_dir) / f'{column}_by_socio_persona.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
    else:
        plt.show()


def plot_by_gas_percentile(df1, df2, column, output_dir=None, save=False):
    """
    Plot totals split by avg_gas_percentile.
    Handles unit conversion for Capex to Millions (£M).
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Check for Capex to handle units
    is_capex = 'capex' in column.lower()
    scale_factor = 1e6 if is_capex else 1.0
    unit_label = ' (£M)' if is_capex else ''
    fmt_str = '{:,.1f}' if is_capex else '{:,.0f}'
    
    # Aggregate by gas percentile
    df1_agg = df1.groupby('avg_gas_percentile')[column].sum().sort_index() / scale_factor
    df2_agg = df2.groupby('avg_gas_percentile')[column].sum().sort_index() / scale_factor
    
    # Get all unique percentiles
    all_percentiles = sorted(set(df1_agg.index) | set(df2_agg.index))
    
    # Prepare data for plotting
    df1_values = [df1_agg.get(p, 0) for p in all_percentiles]
    df2_values = [df2_agg.get(p, 0) for p in all_percentiles]
    
    x = np.arange(len(all_percentiles))
    width = 0.35
    
    # Create grouped bar chart
    bars1 = ax.bar(x - width/2, df1_values, width, label=method_name, 
                   color=METHOD_COLORS[method_name], alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, df2_values, width, label='EPC', 
                   color=METHOD_COLORS['EPC'], alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        fmt_str.format(height),
                        ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Gas Percentile', fontsize=12, fontweight='bold')
    ax.set_ylabel(column.replace('_', ' ').title() + unit_label, fontsize=12)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(p)}' for p in all_percentiles])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save and output_dir:
        filepath = Path(output_dir) / f'{column}_by_gas_percentile.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
    else:
        plt.show()


def plot_by_energy_rating(df1, df2, column, output_dir=None, save=False):
    """
    Plot totals split by CURRENT_ENERGY_RATING.
    Handles unit conversion for Capex to Millions (£M).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Check for Capex to handle units
    is_capex = 'capex' in column.lower()
    scale_factor = 1e6 if is_capex else 1.0
    unit_label = ' (£M)' if is_capex else ''
    fmt_str = '{:,.1f}' if is_capex else '{:,.0f}'
    
    # Aggregate by energy rating
    df1_agg = df1.groupby('CURRENT_ENERGY_RATING')[column].sum().sort_index() / scale_factor
    df2_agg = df2.groupby('CURRENT_ENERGY_RATING')[column].sum().sort_index() / scale_factor
    
    # Get all unique ratings (typically A, B, C, D, E, F, G)
    all_ratings = sorted(set(df1_agg.index) | set(df2_agg.index))
    
    # Prepare data for plotting
    df1_values = [df1_agg.get(r, 0) for r in all_ratings]
    df2_values = [df2_agg.get(r, 0) for r in all_ratings]
    
    x = np.arange(len(all_ratings))
    width = 0.35
    
    # Create grouped bar chart
    bars1 = ax.bar(x - width/2, df1_values, width, label=method_name, 
                   color=METHOD_COLORS[method_name], alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, df2_values, width, label='EPC', 
                   color=METHOD_COLORS['EPC'], alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        fmt_str.format(height),
                        ha='center', va='bottom', fontsize=9, rotation=0)
    
    ax.set_xlabel('Current Energy Rating', fontsize=12, fontweight='bold')
    ax.set_ylabel(column.replace('_', ' ').title() + unit_label, fontsize=12)
    
    ax.set_xticks(x)
    ax.set_xticklabels(all_ratings)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save and output_dir:
        filepath = Path(output_dir) / f'{column}_by_energy_rating.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
    else:
        plt.show()


def plot_heatmap_comparison(df1, df2, column, output_dir=None, save=False):
    """
    Plot heatmap showing totals by socio persona and energy rating.
    Handles unit conversion for Capex to Millions (£M).
    """
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    
    # Check for Capex to handle units
    is_capex = 'capex' in column.lower()
    scale_factor = 1e6 if is_capex else 1.0
    unit_label = ' (£M)' if is_capex else ''
    fmt_str = '.1f' if is_capex else '.0f'
    
    # Create pivot tables and scale
    pivot_df1 = df1.pivot_table(
        values=column, 
        index='meta_socio_persona', 
        columns='CURRENT_ENERGY_RATING', 
        aggfunc='sum', 
        fill_value=0
    ) / scale_factor
    
    pivot_df2 = df2.pivot_table(
        values=column, 
        index='meta_socio_persona', 
        columns='CURRENT_ENERGY_RATING', 
        aggfunc='sum', 
        fill_value=0
    ) / scale_factor
    
    # Plot heatmaps
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


# ============= BUILDING COUNT PLOTS =============

def plot_building_counts_by_percentile(df1, df2, output_dir=None, save=False):
    """Plot count of buildings by gas percentile."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Count buildings by percentile
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
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=9)
    
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
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=9)
    
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
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=9)
    
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


# ============= INTERVENTION TYPE PLOTS =============

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
    fig.legend(handles, labels, title='Intervention', loc='upper right' 
               , frameon=True)
    
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
    fig.legend(handles, labels, title='Intervention', loc='center right', bbox_to_anchor=(0.6, 0.8) , 
                 frameon=True)
    
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


def plot_co2_by_intervention(df1, df2, output_dir=None, save=False):
    """Plot total CO2 saved split by intervention type."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df1_agg = df1.groupby('intervention')['total_co2_saved'].sum().sort_values(ascending=False)
    df2_agg = df2.groupby('intervention')['total_co2_saved'].sum().sort_values(ascending=False)
    
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
                        f'{height:,.0f}',
                        ha='center', va='bottom', fontsize=8, rotation=0)
    
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
    """
    Plot total CAPEX split by intervention type.
    Includes (£M) conversion and labelling.
    """
    # Determine which capex column to use
    capex_col = None
    for col in ['capex', 'total_capex', 'CAPEX']:
        if col in df1.columns and col in df2.columns:
            capex_col = col
            break
    
    if capex_col is None:
        print("Warning: No CAPEX column found, skipping capex_by_intervention plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Aggregate CAPEX by intervention and convert to millions
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
                        f'{height:,.1f}',
                        ha='center', va='bottom', fontsize=8, rotation=0)
    
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


def plot_mean_capex_per_ton(df1, df2, output_dir=None, save=False):
    """
    Plot comparison of mean capex per ton.
    Includes (£/ton) labelling.
    """
    capex_per_ton_cols = []
    if 'weighted_capex_per_net_ton' in df1.columns and 'weighted_capex_per_net_ton' in df2.columns:
        capex_per_ton_cols.append('weighted_capex_per_net_ton')
    if 'capex_per_net_ton' in df1.columns and 'capex_per_net_ton' in df2.columns:
        capex_per_ton_cols.append('capex_per_net_ton')
    
    if not capex_per_ton_cols:
        print("Warning: No capex per ton columns found, skipping mean capex per ton plot")
        return
    
    for col in capex_per_ton_cols:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        mean_df1 = df1[col].mean()
        mean_df2 = df2[col].mean()
        
        bars = ax.bar([method_name, 'EPC'], [mean_df1, mean_df2], 
                       color=[METHOD_COLORS[method_name], METHOD_COLORS['EPC']], alpha=0.7, edgecolor='black', linewidth=2)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:,.0f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel(f'{col.replace("_", " ").title()} (£/ton)', fontsize=12)
        
        ax.grid(True, alpha=0.3, axis='y')
        
        diff = mean_df2 - mean_df1
        diff_pct = (diff / mean_df1) * 100 if mean_df1 != 0 else 0
        ax.text(0.22, 0.95, f'Difference: {diff:,.2f} ({diff_pct:+.1f}%)', 
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
        
        median_df1 = df1[col].median()
        median_df2 = df2[col].median()
        ax.text(0.22, 0.87, f'Median: DF={median_df1:,.2f}, EPC={median_df2:,.2f}', 
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
                fontsize=9)
        
        plt.tight_layout()
        
        if save and output_dir:
            filepath = Path(output_dir) / f'{col}_mean_comparison.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
            plt.close()
        else:
            plt.show()


def plot_mean_capex(df1, df2, output_dir=None, save=False):
    """
    Plot comparison of mean CAPEX.
    Includes (£) labelling.
    """
    capex_col = None
    for col in ['capex', 'total_capex', 'CAPEX']:
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
                   color=[METHOD_COLORS[method_name], METHOD_COLORS['EPC']], alpha=0.7, edgecolor='black', linewidth=2)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,.0f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
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


def generate_all_aggregation_plots(df1, df2, output_dir='./plots', save=True):
    """
    Meta function to generate all aggregation/summation comparison plots.
    """
    if save:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_path.absolute()}\n")
    
    capex_col = None
    for col in ['capex', 'total_capex', 'CAPEX']:
        if col in df1.columns and col in df2.columns:
            capex_col = col
            break
    
    columns_to_compare = ['total_co2_saved']
    if capex_col:
        columns_to_compare.append(capex_col)
    
    print("Generating aggregation plots...\n")
    
    for column in columns_to_compare:
        print(f"Processing: {column}")
        plot_total_comparison(df1, df2, column, output_dir, save)
        plot_by_socio_persona(df1, df2, column, output_dir, save)
        plot_by_gas_percentile(df1, df2, column, output_dir, save)
        plot_by_energy_rating(df1, df2, column, output_dir, save)
        plot_heatmap_comparison(df1, df2, column, output_dir, save)
        print(f"Completed: {column}\n")
    
    print("Processing: Building Counts")
    plot_building_counts_by_percentile(df1, df2, output_dir, save)
    plot_building_counts_by_persona(df1, df2, output_dir, save)
    plot_building_counts_by_energy_rating(df1, df2, output_dir, save)
    print("Completed: Building Counts\n")
    
    print("Processing: Intervention Analysis")
    plot_interventions_by_percentile(df1, df2, output_dir, save)
    plot_interventions_by_persona(df1, df2, output_dir, save)
    plot_interventions_by_energy_rating(df1, df2, output_dir, save)
    plot_co2_by_intervention(df1, df2, output_dir, save)
    if capex_col:
        plot_capex_by_intervention(df1, df2, output_dir, save)
    print("Completed: Intervention Analysis\n")
    
    print("Processing: Mean Value Comparisons")
    plot_mean_capex_per_ton(df1, df2, output_dir, save)
    plot_mean_capex(df1, df2, output_dir, save)
    print("Completed: Mean Value Comparisons\n")
    
    print("All aggregation plots generated!")
    if save:
        print(f"All figures saved to: {output_path.absolute()}")