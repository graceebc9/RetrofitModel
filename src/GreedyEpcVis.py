import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import os 

PERSONA_COLORS = {
    'low_deprived': '#009E73',  # Blue
    'med_deprived': '#E69F00',  # Orange
    'high_deprived': '#D55E00'  # Red
}

def run_epc_vis(greedy_runs_folder, base_dir_outputs, million_budget  , prob_loft , equity_factor    ): 
    output_dir = os.path.join(greedy_runs_folder, f'budget_{int(million_budget)}M__loft_{prob_loft}__equity_{equity_factor}'  )
    # baseline_path = os.path.join(output_dir, f'baseline_selection.csv')
    selected_path = os.path.join(output_dir, f'selected_projects.csv')
    epc_random_path = os.path.join(output_dir, f'epc_random_selection.csv')
    df = pd.read_csv(selected_path) 
    epc= pd.read_csv(epc_random_path) 
    print('df adn epc laoded')
    print(selected_path)
    print(df.head() )
    print(epc_random_path)
    
    generate_all_aggregation_plots(df, epc, output_dir=f'{base_dir_outputs}/epc_comaprisons/budget_{million_budget}M__loft_{prob_loft}__equity_{equity_factor}', save=True)


def plot_total_comparison(df1, df2, column, output_dir=None, save=False):
    """
    Plot overall total comparison for a single column.
    
    Parameters:
    -----------
    df1, df2 : pandas.DataFrame
        DataFrames to compare (df and epc)
    column : str
        Column name to sum and compare
    output_dir : str or Path, optional
        Directory to save figures
    save : bool
        If True, save figure. If False, display it.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate totals
    total_df1 = df1[column].sum()
    total_df2 = df2[column].sum()
    
    # Create bar chart
    bars = ax.bar(['DF', 'EPC'], [total_df1, total_df2], 
                   color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,.0f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel(column.replace('_', ' ').title(), fontsize=12)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add difference annotation
    diff = total_df2 - total_df1
    diff_pct = (diff / total_df1) * 100 if total_df1 != 0 else 0
    ax.text(0.5, 0.95, f'Difference: {diff:,.0f} ({diff_pct:+.1f}%)', 
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
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Aggregate by socio persona
    df1_agg = df1.groupby('meta_socio_persona')[column].sum().sort_index()
    df2_agg = df2.groupby('meta_socio_persona')[column].sum().sort_index()
    
    # Get all unique personas
    all_personas = sorted(set(df1_agg.index) | set(df2_agg.index))
    
    # Prepare data for plotting
    df1_values = [df1_agg.get(p, 0) for p in all_personas]
    df2_values = [df2_agg.get(p, 0) for p in all_personas]
    
    x = np.arange(len(all_personas))
    width = 0.35
    
    # Create grouped bar chart
    bars1 = ax.bar(x - width/2, df1_values, width, label='DF', 
                   color='#3498db', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, df2_values, width, label='EPC', 
                   color='#e74c3c', alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:,.0f}',
                        ha='center', va='bottom', fontsize=9, rotation=0)
    
    ax.set_xlabel('Socio Persona', fontsize=12, fontweight='bold')
    ax.set_ylabel(column.replace('_', ' ').title(), fontsize=12)
    
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
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Aggregate by gas percentile
    df1_agg = df1.groupby('avg_gas_percentile')[column].sum().sort_index()
    df2_agg = df2.groupby('avg_gas_percentile')[column].sum().sort_index()
    
    # Get all unique percentiles
    all_percentiles = sorted(set(df1_agg.index) | set(df2_agg.index))
    
    # Prepare data for plotting
    df1_values = [df1_agg.get(p, 0) for p in all_percentiles]
    df2_values = [df2_agg.get(p, 0) for p in all_percentiles]
    
    x = np.arange(len(all_percentiles))
    width = 0.35
    
    # Create grouped bar chart
    bars1 = ax.bar(x - width/2, df1_values, width, label='DF', 
                   color='#3498db', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, df2_values, width, label='EPC', 
                   color='#e74c3c', alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:,.0f}',
                        ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Gas Percentile', fontsize=12, fontweight='bold')
    ax.set_ylabel(column.replace('_', ' ').title(), fontsize=12)
    
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
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Aggregate by energy rating
    df1_agg = df1.groupby('CURRENT_ENERGY_RATING')[column].sum().sort_index()
    df2_agg = df2.groupby('CURRENT_ENERGY_RATING')[column].sum().sort_index()
    
    # Get all unique ratings (typically A, B, C, D, E, F, G)
    all_ratings = sorted(set(df1_agg.index) | set(df2_agg.index))
    
    # Prepare data for plotting
    df1_values = [df1_agg.get(r, 0) for r in all_ratings]
    df2_values = [df2_agg.get(r, 0) for r in all_ratings]
    
    x = np.arange(len(all_ratings))
    width = 0.35
    
    # Create grouped bar chart
    bars1 = ax.bar(x - width/2, df1_values, width, label='DF', 
                   color='#3498db', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, df2_values, width, label='EPC', 
                   color='#e74c3c', alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:,.0f}',
                        ha='center', va='bottom', fontsize=9, rotation=0)
    
    ax.set_xlabel('Current Energy Rating', fontsize=12, fontweight='bold')
    ax.set_ylabel(column.replace('_', ' ').title(), fontsize=12)
    
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
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Create pivot tables
    pivot_df1 = df1.pivot_table(
        values=column, 
        index='meta_socio_persona', 
        columns='CURRENT_ENERGY_RATING', 
        aggfunc='sum', 
        fill_value=0
    )
    
    pivot_df2 = df2.pivot_table(
        values=column, 
        index='meta_socio_persona', 
        columns='CURRENT_ENERGY_RATING', 
        aggfunc='sum', 
        fill_value=0
    )
    
    # Plot heatmaps
    sns.heatmap(pivot_df1, annot=True, fmt='.0f', cmap='YlOrRd', 
                ax=axes[0], cbar_kws={'label': column.replace('_', ' ').title()})
    axes[0].set_title('DF', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Energy Rating', fontsize=11)
    axes[0].set_ylabel('Socio Persona', fontsize=11)
    
    sns.heatmap(pivot_df2, annot=True, fmt='.0f', cmap='YlOrRd', 
                ax=axes[1], cbar_kws={'label': column.replace('_', ' ').title()})
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


# ============= NEW: BUILDING COUNT PLOTS =============

def plot_building_counts_by_percentile(df1, df2, output_dir=None, save=False):
    """
    Plot count of buildings by gas percentile.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Count buildings by percentile
    df1_counts = df1['avg_gas_percentile'].value_counts().sort_index()
    df2_counts = df2['avg_gas_percentile'].value_counts().sort_index()
    
    # Get all unique percentiles
    all_percentiles = sorted(set(df1_counts.index) | set(df2_counts.index))
    
    # Prepare data
    df1_values = [df1_counts.get(p, 0) for p in all_percentiles]
    df2_values = [df2_counts.get(p, 0) for p in all_percentiles]
    
    x = np.arange(len(all_percentiles))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df1_values, width, label='DF', 
                   color='#3498db', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, df2_values, width, label='EPC', 
                   color='#e74c3c', alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Gas Percentile', fontsize=12, fontweight='bold')
    ax.set_ylabel('Building Count', fontsize=12)
    ax.set_title('Building Count by Gas Percentile', fontsize=14, fontweight='bold')
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
    """
    Plot count of buildings by socio persona.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Count buildings by persona
    df1_counts = df1['meta_socio_persona'].value_counts().sort_index()
    df2_counts = df2['meta_socio_persona'].value_counts().sort_index()
    
    # Get all unique personas
    all_personas = sorted(set(df1_counts.index) | set(df2_counts.index))
    
    # Prepare data
    df1_values = [df1_counts.get(p, 0) for p in all_personas]
    df2_values = [df2_counts.get(p, 0) for p in all_personas]
    
    x = np.arange(len(all_personas))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df1_values, width, label='DF', 
                   color='#3498db', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, df2_values, width, label='EPC', 
                   color='#e74c3c', alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Socio Persona', fontsize=12, fontweight='bold')
    ax.set_ylabel('Building Count', fontsize=12)
    ax.set_title('Building Count by Socio Persona', fontsize=14, fontweight='bold')
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
    """
    Plot count of buildings by energy rating.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Count buildings by energy rating
    df1_counts = df1['CURRENT_ENERGY_RATING'].value_counts().sort_index()
    df2_counts = df2['CURRENT_ENERGY_RATING'].value_counts().sort_index()
    
    # Get all unique ratings
    all_ratings = sorted(set(df1_counts.index) | set(df2_counts.index))
    
    # Prepare data
    df1_values = [df1_counts.get(r, 0) for r in all_ratings]
    df2_values = [df2_counts.get(r, 0) for r in all_ratings]
    
    x = np.arange(len(all_ratings))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df1_values, width, label='DF', 
                   color='#3498db', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, df2_values, width, label='EPC', 
                   color='#e74c3c', alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Current Energy Rating', fontsize=12, fontweight='bold')
    ax.set_ylabel('Building Count', fontsize=12)
    ax.set_title('Building Count by Energy Rating', fontsize=14, fontweight='bold')
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


# ============= NEW: INTERVENTION TYPE PLOTS =============

def plot_interventions_by_percentile(df1, df2, output_dir=None, save=False):
    """
    Plot stacked bar of intervention counts by gas percentile.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    
    # Create crosstab for each df
    df1_crosstab = pd.crosstab(df1['avg_gas_percentile'], df1['intervention'])
    df2_crosstab = pd.crosstab(df2['avg_gas_percentile'], df2['intervention'])
    
    # Get all unique interventions across both dfs to ensure consistent columns
    all_interventions = sorted(set(df1_crosstab.columns) | set(df2_crosstab.columns))
    
    # Reindex both crosstabs to have the same columns
    df1_crosstab = df1_crosstab.reindex(columns=all_interventions, fill_value=0)
    df2_crosstab = df2_crosstab.reindex(columns=all_interventions, fill_value=0)
    
    # Plot stacked bars
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
    
    # Create shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Intervention', loc='center left', 
               bbox_to_anchor=(1.0, 0.5), frameon=True)
    
    
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    
    if save and output_dir:
        filepath = Path(output_dir) / 'interventions_by_percentile.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
    else:
        plt.show()


def plot_interventions_by_persona(df1, df2, output_dir=None, save=False):
    """
    Plot stacked bar of intervention counts by socio persona.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    
    # Create crosstab for each df
    df1_crosstab = pd.crosstab(df1['meta_socio_persona'], df1['intervention'])
    df2_crosstab = pd.crosstab(df2['meta_socio_persona'], df2['intervention'])
    
    # Get all unique interventions across both dfs to ensure consistent columns
    all_interventions = sorted(set(df1_crosstab.columns) | set(df2_crosstab.columns))
    
    # Reindex both crosstabs to have the same columns
    df1_crosstab = df1_crosstab.reindex(columns=all_interventions, fill_value=0)
    df2_crosstab = df2_crosstab.reindex(columns=all_interventions, fill_value=0)
    
    # Plot stacked bars
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
    
    # Create shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Intervention', loc='center left', 
               bbox_to_anchor=(1.0, 0.5), frameon=True)
    
    
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    
    if save and output_dir:
        filepath = Path(output_dir) / 'interventions_by_persona.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
    else:
        plt.show()


def plot_interventions_by_energy_rating(df1, df2, output_dir=None, save=False):
    """
    Plot stacked bar of intervention counts by energy rating.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    
    # Create crosstab for each df
    df1_crosstab = pd.crosstab(df1['CURRENT_ENERGY_RATING'], df1['intervention'])
    df2_crosstab = pd.crosstab(df2['CURRENT_ENERGY_RATING'], df2['intervention'])
    
    # Get all unique interventions across both dfs to ensure consistent columns
    all_interventions = sorted(set(df1_crosstab.columns) | set(df2_crosstab.columns))
    
    # Reindex both crosstabs to have the same columns
    df1_crosstab = df1_crosstab.reindex(columns=all_interventions, fill_value=0)
    df2_crosstab = df2_crosstab.reindex(columns=all_interventions, fill_value=0)
    
    # Plot stacked bars
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
    
    # Create shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Intervention', loc='center left', 
               bbox_to_anchor=(1.0, 0.5), frameon=True)
    
    
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    
    if save and output_dir:
        filepath = Path(output_dir) / 'interventions_by_energy_rating.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
    else:
        plt.show()


def plot_co2_by_intervention(df1, df2, output_dir=None, save=False):
    """
    Plot total CO2 saved split by intervention type.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Aggregate CO2 by intervention
    df1_agg = df1.groupby('intervention')['total_co2_saved'].sum().sort_values(ascending=False)
    df2_agg = df2.groupby('intervention')['total_co2_saved'].sum().sort_values(ascending=False)
    
    # Get all unique interventions
    all_interventions = sorted(set(df1_agg.index) | set(df2_agg.index))
    
    # Prepare data
    df1_values = [df1_agg.get(i, 0) for i in all_interventions]
    df2_values = [df2_agg.get(i, 0) for i in all_interventions]
    
    x = np.arange(len(all_interventions))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df1_values, width, label='DF', 
                   color='#3498db', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, df2_values, width, label='EPC', 
                   color='#e74c3c', alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:,.0f}',
                        ha='center', va='bottom', fontsize=8, rotation=90)
    
    ax.set_xlabel('Intervention Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total CO2 Saved', fontsize=12)
    # ax.set_title('Total CO2 Saved by Intervention Type', fontsize=14, fontweight='bold')
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
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Aggregate CAPEX by intervention
    df1_agg = df1.groupby('intervention')[capex_col].sum().sort_values(ascending=False)
    df2_agg = df2.groupby('intervention')[capex_col].sum().sort_values(ascending=False)
    
    # Get all unique interventions
    all_interventions = sorted(set(df1_agg.index) | set(df2_agg.index))
    
    # Prepare data
    df1_values = [df1_agg.get(i, 0) for i in all_interventions]
    df2_values = [df2_agg.get(i, 0) for i in all_interventions]
    
    x = np.arange(len(all_interventions))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df1_values, width, label='DF', 
                   color='#3498db', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, df2_values, width, label='EPC', 
                   color='#e74c3c', alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:,.0f}',
                        ha='center', va='bottom', fontsize=8, rotation=90)
    
    ax.set_xlabel('Intervention Type', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Total {capex_col.upper()}', fontsize=12)
    # ax.set_title(f'Total {capex_col.upper()} by Intervention Type', fontsize=14, fontweight='bold')
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
    Plot comparison of mean capex per ton between dataframes.
    Creates separate plots for both weighted and unweighted versions if available.
    """
    # Check which capex per ton columns exist
    capex_per_ton_cols = []
    if 'weighted_capex_per_net_ton' in df1.columns and 'weighted_capex_per_net_ton' in df2.columns:
        capex_per_ton_cols.append('weighted_capex_per_net_ton')
    if 'capex_per_net_ton' in df1.columns and 'capex_per_net_ton' in df2.columns:
        capex_per_ton_cols.append('capex_per_net_ton')
    
    if not capex_per_ton_cols:
        print("Warning: No capex per ton columns found, skipping mean capex per ton plot")
        return
    
    # Create a plot for each available column
    for col in capex_per_ton_cols:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Calculate means
        mean_df1 = df1[col].mean()
        mean_df2 = df2[col].mean()
        
        # Create bar chart
        bars = ax.bar(['DF', 'EPC'], [mean_df1, mean_df2], 
                       color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=2)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:,.2f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel(col.replace('_', ' ').title(), fontsize=12)
        # ax.set_title(f'Mean {col.replace("_", " ").title()} Comparison',  fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add difference annotation
        diff = mean_df2 - mean_df1
        diff_pct = (diff / mean_df1) * 100 if mean_df1 != 0 else 0
        ax.text(0.5, 0.95, f'Difference: {diff:,.2f} ({diff_pct:+.1f}%)', 
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
        
        # Add median comparison as text
        median_df1 = df1[col].median()
        median_df2 = df2[col].median()
        ax.text(0.5, 0.87, f'Median - DF: {median_df1:,.2f}, EPC: {median_df2:,.2f}', 
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
    Plot comparison of mean CAPEX between dataframes.
    """
    # Determine which capex column to use
    capex_col = None
    for col in ['capex', 'total_capex', 'CAPEX']:
        if col in df1.columns and col in df2.columns:
            capex_col = col
            break
    
    if capex_col is None:
        print("Warning: No CAPEX column found, skipping mean capex plot")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate means
    mean_df1 = df1[capex_col].mean()
    mean_df2 = df2[capex_col].mean()
    
    # Create bar chart
    bars = ax.bar(['DF', 'EPC'], [mean_df1, mean_df2], 
                   color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel(capex_col.replace('_', ' ').title(), fontsize=12)
    # ax.set_title(f'Mean {capex_col.replace("_", " ").title()} Comparison',  fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add difference annotation
    diff = mean_df2 - mean_df1
    diff_pct = (diff / mean_df1) * 100 if mean_df1 != 0 else 0
    ax.text(0.5, 0.95, f'Difference: {diff:,.2f} ({diff_pct:+.1f}%)', 
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    # Add median comparison as text
    median_df1 = df1[capex_col].median()
    median_df2 = df2[capex_col].median()
    ax.text(0.5, 0.87, f'Median - DF: {median_df1:,.2f}, EPC: {median_df2:,.2f}', 
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
    
    Parameters:
    -----------
    df1, df2 : pandas.DataFrame
        DataFrames to compare (df and epc)
    output_dir : str or Path
        Directory to save all figures
    save : bool
        If True, save figures. If False, display them.
    """
    # Create output directory if saving
    if save:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_path.absolute()}\n")
    
    # Determine which capex column exists
    capex_col = None
    for col in ['capex', 'total_capex', 'CAPEX']:
        if col in df1.columns and col in df2.columns:
            capex_col = col
            break
    
    # Columns to compare (only total_co2_saved and capex, NOT per-ton metrics)
    columns_to_compare = ['total_co2_saved']
    if capex_col:
        columns_to_compare.append(capex_col)
    
    print("Generating aggregation plots...\n")
    
    # Generate plots for each metric column
    for column in columns_to_compare:
        print(f"Processing: {column}")
        
        # Total comparison
        plot_total_comparison(df1, df2, column, output_dir, save)
        
        # Split by categories
        plot_by_socio_persona(df1, df2, column, output_dir, save)
        plot_by_gas_percentile(df1, df2, column, output_dir, save)
        plot_by_energy_rating(df1, df2, column, output_dir, save)
        
        # Heatmap
        plot_heatmap_comparison(df1, df2, column, output_dir, save)
        
        print(f"Completed: {column}\n")
    
    # Building count plots
    print("Processing: Building Counts")
    plot_building_counts_by_percentile(df1, df2, output_dir, save)
    plot_building_counts_by_persona(df1, df2, output_dir, save)
    plot_building_counts_by_energy_rating(df1, df2, output_dir, save)
    print("Completed: Building Counts\n")
    
    # Intervention analysis plots
    print("Processing: Intervention Analysis")
    plot_interventions_by_percentile(df1, df2, output_dir, save)
    plot_interventions_by_persona(df1, df2, output_dir, save)
    plot_interventions_by_energy_rating(df1, df2, output_dir, save)
    plot_co2_by_intervention(df1, df2, output_dir, save)
    if capex_col:
        plot_capex_by_intervention(df1, df2, output_dir, save)
    print("Completed: Intervention Analysis\n")
    
    # Mean value comparisons
    print("Processing: Mean Value Comparisons")
    plot_mean_capex_per_ton(df1, df2, output_dir, save)
    plot_mean_capex(df1, df2, output_dir, save)
    print("Completed: Mean Value Comparisons\n")
    
    print("All aggregation plots generated!")
    if save:
        print(f"All figures saved to: {output_path.absolute()}")

 