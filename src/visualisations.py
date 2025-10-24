import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os
import matplotlib.pyplot as plt

def run_vis_new(res_df, scenario, op_base):
    # Create output directory if it doesn't exist
    os.makedirs(op_base, exist_ok=True)
    
    pl = res_df[res_df['premise_type'] != 'Domestic outbuilding'].copy() 

    # costs per type
    fig = plot_col_reduction_by_decile_epistemic(pl,
                                    mean_col=f'{scenario}_cost_{scenario}_mean', 
                                    std_col=f'{scenario}_cost_{scenario}_std',
                                    ylabel='Installation Costs (£)',
                                    costs=True, 
                                    percentage=False,
                                    groupby_col='premise_type',
                                    groupby_label='Premise Type',
                                    rot=True,
                                    )
    fig.savefig(os.path.join(op_base, f'{scenario}_costs_by_premise_type.png'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # cost per decile 
    fig = plot_col_reduction_by_decile_epistemic(pl,
                                    mean_col=f'{scenario}_cost_{scenario}_mean', 
                                    std_col=f'{scenario}_cost_{scenario}_std',
                                    ylabel='Installation Costs (£)',
                                    costs=True, 
                                    percentage=False,
                                    rot=True,
                                    )
    fig.savefig(os.path.join(op_base, f'{scenario}_costs_by_decile.png'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Gas reduction by decile
    fig = plot_col_reduction_by_decile_epistemic(pl,
                                    mean_col=f'{scenario}_{scenario}_gas_mean', 
                                    std_col=f'{scenario}_{scenario}_gas_std',
                                    ylabel='Gas Reduction (%)',
                                    costs=False, 
                                    percentage=True
                                    )
    fig.savefig(os.path.join(op_base, f'{scenario}_gas_reduction_by_decile.png'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Gas reduction by premise type
    fig = plot_col_reduction_by_decile_epistemic(pl,
                                    mean_col=f'{scenario}_{scenario}_gas_mean', 
                                    std_col=f'{scenario}_{scenario}_gas_std',
                                    ylabel='Gas Reduction (%)',
                                    costs=False, 
                                    percentage=True,
                                    groupby_col='premise_type',
                                    groupby_label='Premise Type',
                                    rot=True ,
                                    )
    fig.savefig(os.path.join(op_base, f'{scenario}_gas_reduction_by_premise_type.png'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)


 
    if scenario == 'wall_installation':
        fig = plot_total_cost_by_decile_epistemic_stacked(pl,
                                    cost_col=f'{scenario}_cost_{scenario}_mean', 
                                    cost_std_col=f'{scenario}_cost_{scenario}_std',
                                    
                                    stack_by_col='inferred_insulation_type',
                                    
                                    groupby_col='avg_gas_percentile',
                                    groupby_label='Gas Usage Decile',
                                    ylabel='Costs of Installation',
                                ) 
        fig.savefig(os.path.join(op_base, f'{scenario}_wall_type_costssby_decile.png'), 
                dpi=300, bbox_inches='tight')
        plt.close(fig)

        fig = plot_total_cost_by_decile_epistemic_stacked(pl,
                                    cost_col=f'{scenario}_cost_{scenario}_mean', 
                                    cost_std_col=f'{scenario}_cost_{scenario}_std',
                                    
                                    stack_by_col='inferred_insulation_type',
                                    
                                    groupby_col='premise_type',
                                    groupby_label='Premise Type',
                                    ylabel='Costs of Installation',
                                ) 
        fig.savefig(os.path.join(op_base, f'{scenario}_wall_type_costssby_premise_type.png'), 
                dpi=300, bbox_inches='tight')
        plt.close(fig)

    
    print(f"All figures saved to: {op_base}")
                                    
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_total_cost_by_decile_epistemic_stacked(res_df,
                                cost_col,
                                cost_std_col, 
                                stack_by_col, 
                                epistemic_run_id=None,
                                groupby_col='avg_gas_percentile',
                                groupby_label='Gas Usage Decile',
                                ylabel='Cost (£)',
                                title=None,
                                figsize=(16, 6),
                               
                                rot=False):
    
    df = res_df.copy()
    
    # ==========================================================
    # ===== LEFT PLOT: Mean Cost per Building =====
    # ==========================================================
    if epistemic_run_id is None:
        epistemic_run_id = df['epistemic_run_id'].iloc[0]
        print(f"No epistemic_run_id specified, using: {epistemic_run_id}")
    
    df_single = df[df['epistemic_run_id'] == epistemic_run_id].copy()
    
    # Use cost_std_col for aleatoric uncertainty
    df_single[f'{cost_std_col}_2'] = (df_single[cost_std_col] ** 2)
    
    grouped_single = df_single.groupby(groupby_col).agg({
        cost_col: ['mean', 'count'],
        f'{cost_std_col}_2': ['sum']
    }).reset_index()
    grouped_single.columns = ['decile', 'mean_cost', 'n_buildings', 'std_squared_summed']
    grouped_single['pooled_sd'] = np.sqrt(grouped_single['std_squared_summed'] / grouped_single['n_buildings'])
    
    # Calculate bounds for single run (LEFT)
    grouped_single['se'] = grouped_single['pooled_sd'] / np.sqrt(grouped_single['n_buildings'])
    grouped_single['sd_lower'] = grouped_single['mean_cost'] - 2 * grouped_single['pooled_sd']
    grouped_single['sd_upper'] = grouped_single['mean_cost'] + 2 * grouped_single['pooled_sd']
    
    
    # =================================================================
    # ===== RIGHT PLOT: Mean Total Cost & Proportions =====
    # =================================================================
    
    epistemic_subgroup_totals = []
    epistemic_decile_totals = []
    
    for run_id in df['epistemic_run_id'].unique():
        df_run = df[df['epistemic_run_id'] == run_id].copy()
        
        # --- NEW: Get SUM of costs for each subgroup ---
        grouped_run_sub = df_run.groupby([groupby_col, stack_by_col]).agg(
            subgroup_total_cost=(cost_col, 'sum')
        ).reset_index()
        grouped_run_sub['epistemic_run_id'] = run_id
        epistemic_subgroup_totals.append(grouped_run_sub)
        
        # --- NEW: Get SUM of costs for the whole decile ---
        grouped_run_total = df_run.groupby(groupby_col).agg(
            decile_total_cost=(cost_col, 'sum')
        ).reset_index()
        grouped_run_total['epistemic_run_id'] = run_id
        epistemic_decile_totals.append(grouped_run_total)
    
    # --- NEW: Aggregate subgroup total costs ---
    epistemic_df_sub = pd.concat(epistemic_subgroup_totals, ignore_index=True)
    grouped_epistemic_sub = epistemic_df_sub.groupby([groupby_col, stack_by_col]).agg({
        'subgroup_total_cost': ['mean'] # Mean of the subgroup's total
    }).reset_index()
    grouped_epistemic_sub.columns = ['decile', stack_by_col, 'mean_subgroup_total_cost']
    
    # --- NEW: Aggregate decile total costs (for error bars) ---
    epistemic_df_total = pd.concat(epistemic_decile_totals, ignore_index=True)
    grouped_epistemic_total = epistemic_df_total.groupby(groupby_col).agg({
        'decile_total_cost': ['mean', 'std', 'count']
    }).reset_index()
    grouped_epistemic_total.columns = ['decile', 'total_mean_cost', 'epistemic_std', 'n_epistemic_runs']
    grouped_epistemic_total['epistemic_se'] = grouped_epistemic_total['epistemic_std'] / np.sqrt(grouped_epistemic_total['n_epistemic_runs'])
    
    # Calculate bounds for epistemic (RIGHT)
    grouped_epistemic_total['ci_error'] = 2 * grouped_epistemic_total['epistemic_se']
    
    
    # ========================================================
    # ===== CREATE PLOTS =====
    # ========================================================
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # --- LEFT PLOT: Mean Cost per Building ---
    ax1.plot(grouped_single['decile'], grouped_single['mean_cost'],
             'o-', linewidth=2, markersize=8, color='steelblue')
    ax1.fill_between(grouped_single['decile'], grouped_single['sd_lower'], grouped_single['sd_upper'],
                     alpha=0.3, label='Mean ± 2σ (aleatoric)', color='steelblue')
    ax1.set_xlabel(groupby_label, fontsize=11)
    ax1.set_ylabel(f"Mean {ylabel} per Building", fontsize=11)
    ax1.set_title(f'Mean Cost & Aleatoric Uncertainty\n(Run: {epistemic_run_id})', fontsize=13)
    ax1.set_xticks(grouped_single['decile'].unique())
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0) # Costs are always positive
    ax1.axhline(0, color='r', linestyle='--', alpha=0.5)
    if rot:
        ax1.tick_params(axis='x', rotation=90)
    
    # --- RIGHT PLOT: Mean Total Cost (Stacked) ---
    
    # Pivot the mean subgroup totals for stacked plotting
    pivot_data = grouped_epistemic_sub.pivot(
        index='decile', 
        columns=stack_by_col, 
        values='mean_subgroup_total_cost'
    )
    
    # Plot the stacked bar
    pivot_data.plot(kind='bar', stacked=True, ax=ax2, alpha=0.8, width=0.8)
    
    # Get x-axis positions (0, 1, 2...) for placing error bars
    x_positions = np.arange(len(grouped_epistemic_total['decile']))
    
    # Plot the total error bar over the stacked bars
    ax2.errorbar(x=x_positions, 
                 y=grouped_epistemic_total['total_mean_cost'],
                 yerr=grouped_epistemic_total['ci_error'], fmt='none',
                 color='black', capsize=5, capthick=2)
    
    ax2.set_xlabel(groupby_label, fontsize=11)
    ax2.set_ylabel(f"Mean Total {ylabel} per Decile", fontsize=11)
    
    n_runs = int(grouped_epistemic_total['n_epistemic_runs'].iloc[0])
    ax2.set_title(f'Mean Total Cost & Epistemic Uncertainty\n(Across {n_runs} runs)', fontsize=13)
    
    # --- Combine legends ---
    handles, labels = ax2.get_legend_handles_labels()
    error_bar_legend = Line2D([0], [0], color='black', marker='_', markersize=5, 
                              label='Total Mean Cost ± 2 SE')
    handles.append(error_bar_legend)
    ax2.legend(handles=handles, title=stack_by_col)
    
    ax2.grid(alpha=0.3, axis='y')
    ax2.axhline(0, color='r', linestyle='--', alpha=0.5)
    if rot:
        # Set ticks and labels for bar chart
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(grouped_epistemic_total['decile'], rotation=90)
    else:
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(grouped_epistemic_total['decile'])

    
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    # ========================================================
    # ===== Print summaries =====
    # ========================================================
    
    print("\n=== Single Run Summary (Mean Cost per Building) ===")
    print(grouped_single[['decile', 'mean_cost', 'pooled_sd', 'n_buildings']].to_string(index=False))
    
    print(f"\n=== Epistemic Summary (Mean Total Cost Contributions by {stack_by_col}) ===")
    print(pivot_data.to_string())
    
    print("\n=== Epistemic Summary (Mean Total Cost per Decile) ===")
    print(grouped_epistemic_total[['decile', 'total_mean_cost', 'epistemic_std', 'epistemic_se', 'n_epistemic_runs']].to_string(index=False))
 
        
    return fig


def plot_col_reduction_by_decile_epistemic(res_df,
                                mean_col,
                                std_col, 
                                epistemic_run_id=None,
                                groupby_col='avg_gas_percentile',
                                groupby_label='Gas Usage Decile',
                                ylabel='Energy Reduction (%)',
                                title=None,
                                figsize=(16, 6),
                               
                                costs=True,
                                rot=False,
                                percentage=True):
    
    df = res_df.copy()
    
    # ===== LEFT PLOT: Single epistemic run =====
    if epistemic_run_id is None:
        epistemic_run_id = df['epistemic_run_id'].iloc[0]
        print(f"No epistemic_run_id specified, using: {epistemic_run_id}")
    
    df_single = df[df['epistemic_run_id'] == epistemic_run_id].copy()
    df_single[f'{std_col}_2'] = (df_single[std_col] ** 2)
    
    grouped_single = df_single.groupby(groupby_col).agg({
        mean_col: ['mean', 'count'],
        f'{std_col}_2': ['sum']
    }).reset_index()
    grouped_single.columns = ['decile', 'mean_reduction', 'n_buildings', 'std_squared_summed']
    grouped_single['pooled_sd'] = np.sqrt(grouped_single['std_squared_summed'] / grouped_single['n_buildings'])
    
    # ===== RIGHT PLOT: Across all epistemic runs =====
    # Calculate mean for each epistemic run
    epistemic_means = []
    for run_id in df['epistemic_run_id'].unique():
        df_run = df[df['epistemic_run_id'] == run_id].copy()
        
        # NOTE: Removed unused line `df_run[f'{std_col}_2'] = ...` here
        
        grouped_run = df_run.groupby(groupby_col).agg({
            mean_col: ['mean', 'count']
        }).reset_index()
        grouped_run.columns = ['decile', 'mean_reduction', 'n_buildings']
        grouped_run['epistemic_run_id'] = run_id
        epistemic_means.append(grouped_run)
    
    epistemic_df = pd.concat(epistemic_means, ignore_index=True)
    
    # Aggregate across epistemic runs
    grouped_epistemic = epistemic_df.groupby('decile').agg({
        'mean_reduction': ['mean', 'std', 'count']
    }).reset_index()
    grouped_epistemic.columns = ['decile', 'mean_reduction', 'epistemic_std', 'n_epistemic_runs']
    grouped_epistemic['epistemic_se'] = grouped_epistemic['epistemic_std'] / np.sqrt(grouped_epistemic['n_epistemic_runs'])

    # Apply percentage scaling
    if percentage:
        grouped_single['mean_reduction'] *= 100
        grouped_single['pooled_sd'] *= 100
        grouped_epistemic['mean_reduction'] *= 100
        grouped_epistemic['epistemic_std'] *= 100
        grouped_epistemic['epistemic_se'] *= 100
    
    # Calculate bounds for single run (LEFT)
    grouped_single['se'] = grouped_single['pooled_sd'] / np.sqrt(grouped_single['n_buildings'])
    grouped_single['sd_lower'] = grouped_single['mean_reduction'] - 2 * grouped_single['pooled_sd']
    grouped_single['sd_upper'] = grouped_single['mean_reduction'] + 2 * grouped_single['pooled_sd']
    
    # Calculate bounds for epistemic (RIGHT)
    grouped_epistemic['epistemic_lower'] = grouped_epistemic['mean_reduction'] - 2 * grouped_epistemic['epistemic_se']
    grouped_epistemic['epistemic_upper'] = grouped_epistemic['mean_reduction'] + 2 * grouped_epistemic['epistemic_se']
    grouped_epistemic['ci_error'] = 2 * grouped_epistemic['epistemic_se']
    
    # ===== CREATE PLOTS =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # LEFT: Individual building variability within one epistemic run
    ax1.plot(grouped_single['decile'], grouped_single['mean_reduction'],
             'o-', linewidth=2, markersize=8, color='steelblue')
    ax1.fill_between(grouped_single['decile'], grouped_single['sd_lower'], grouped_single['sd_upper'],
                     alpha=0.3, label='Mean ± 2σ (within run)', color='steelblue')
    ax1.set_xlabel(groupby_label, fontsize=11)
    ax1.set_ylabel(ylabel, fontsize=11)
    ax1.set_title(f'Within-Run Variability\n(Run: {epistemic_run_id})', fontsize=13)
    ax1.set_xticks(grouped_single['decile'].unique())
    ax1.legend()
    ax1.grid(alpha=0.3)
    if costs:
        ax1.set_ylim(0)
    if percentage:
        ax1.axhline(0, color='r', linestyle='--', alpha=0.5)
    if rot:
        ax1.tick_params(axis='x', rotation=90)
    
    # RIGHT: Epistemic uncertainty across runs
    ax2.bar(grouped_epistemic['decile'], grouped_epistemic['mean_reduction'],
            color='darkgreen', alpha=0.7, width=0.8)
    ax2.errorbar(grouped_epistemic['decile'], grouped_epistemic['mean_reduction'],
                 yerr=grouped_epistemic['ci_error'], fmt='none',
                 color='black', capsize=5, capthick=2,
                 label='Mean ± 2 SE (epistemic)')
    ax2.set_xlabel(groupby_label, fontsize=11)
    ax2.set_ylabel(ylabel, fontsize=11)
    
    n_runs = int(grouped_epistemic['n_epistemic_runs'].iloc[0])
    ax2.set_title(f'Epistemic Uncertainty\n(Across {n_runs} runs)', fontsize=13)
    ax2.set_xticks(grouped_epistemic['decile'].unique())
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')
    if percentage:
        ax2.axhline(0, color='r', linestyle='--', alpha=0.5)
    if rot:
        ax2.tick_params(axis='x', rotation=90)
    
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    # Print summaries
    print("\n=== Single Run Summary ===")
    print(grouped_single[['decile', 'mean_reduction', 'pooled_sd', 'n_buildings']].to_string(index=False))
    
    print("\n=== Epistemic Uncertainty Summary ===")
    print(grouped_epistemic[['decile', 'mean_reduction', 'epistemic_std', 'epistemic_se', 'n_epistemic_runs']].to_string(index=False))
    
        
    return fig

def plot_col_reduction_by_decile_conservation(res_df,
                                              
                                groupby_col='avg_gas_percentile',
                                mean_col='loft_installation_energy_loft_percentile_gas_mean',
                                conservation_col='conservation_area_bool',
                                groupby_label='Gas Usage Decile',
                                ylabel='Energy Reduction (%)',
                                title=None,
                                figsize=(12, 6),
                                rot=False, 
                                percentage=True,):
                                
    
    df = res_df.copy()

    # Group by both decile AND conservation area
    grouped = df.groupby([groupby_col, conservation_col]).agg({
        mean_col: ['sum', 'count']
    }).reset_index()
    grouped.columns = ['decile', 'conservation_area', 'total_reduction', 'n_buildings']

    if percentage:
        grouped['total_reduction'] *= 100

    # Pivot to get conservation areas as columns
    pivot_data = grouped.pivot(index='decile', 
                               columns='conservation_area', 
                               values='total_reduction').fillna(0)
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get conservation area labels
    conservation_labels = {True: 'Conservation Area', False: 'Non-Conservation Area'}
    colors = {True: 'coral', False: 'steelblue'}
    
    bottom = None
    for cons_area in pivot_data.columns:
        label = conservation_labels.get(cons_area, f'Conservation: {cons_area}')
        color = colors.get(cons_area, 'gray')
        
        ax.bar(pivot_data.index, pivot_data[cons_area], 
               bottom=bottom, 
               label=label,
               color=color, 
               alpha=0.8, 
               width=0.8)
        
        if bottom is None:
            bottom = pivot_data[cons_area]
        else:
            bottom += pivot_data[cons_area]
    
    ax.set_xlabel(groupby_label, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(pivot_data.index)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    if percentage: 
        ax.axhline(0, color='r', linestyle='--', alpha=0.5)
    if rot: 
        ax.tick_params(axis='x', rotation=90)
    if title:
        ax.set_title(title, fontsize=13)
    
    plt.tight_layout()

    # Print summary
    print("\nSummary Statistics by Decile and Conservation Area:")
    print(grouped.pivot_table(index='decile', 
                             columns='conservation_area', 
                             values=['total_reduction', 'n_buildings']))
    
    return fig

def plot_col_reduction_by_decile(res_df,
                                      
                                groupby_col='avg_gas_percentile',
                                mean_col='loft_installation_energy_loft_percentile_gas_mean',
                                std_col='loft_installation_energy_loft_percentile_gas_std',
                                groupby_label='Gas Usage Decile',
                                ylabel='Energy Reduction (%)',
                                title=None ,
                                figsize=(16, 6),
                                show_plot=True,
                                costs=True,
                                rot=False, 
                                
                                percentage=True ,):
                                
    
    df = res_df.copy( )
    df[f'{std_col}_2'] =(df[std_col] **2 )


    grouped = df.groupby(groupby_col).agg({ mean_col: ['mean', 'count'],  f'{std_col}_2': ['sum'] }).reset_index()
    grouped.columns = ['decile', 'mean_reduction', 'n_buildings', 'std_squared_summed']

    grouped['pooled_sd'] = np.sqrt(grouped['std_squared_summed'] / grouped['n_buildings'] ) 

    if percentage:

        
        grouped['mean_reduction'] *= 100
        grouped['pooled_sd'] *= 100

    grouped['se'] = grouped['pooled_sd'] / np.sqrt( grouped['n_buildings'] )


    # SD bounds (individual variability)
    grouped['sd_lower'] = grouped['mean_reduction'] - 2 * grouped['pooled_sd']
    grouped['sd_upper'] = grouped['mean_reduction'] + 2 * grouped['pooled_sd']

    # SE bounds (confidence in mean)
    grouped['se_lower'] = grouped['mean_reduction'] -   grouped['se']
    grouped['se_upper'] = grouped['mean_reduction'] +  grouped['se']

    # Calculate error bar size (from mean to upper/lower bound)
    grouped['ci_error'] = 2 * grouped['se']
 

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Standard Deviation (individual outcomes)
    ax1.plot(grouped['decile'], grouped['mean_reduction'], 
                'o-', linewidth=2, markersize=8, color='steelblue')
    ax1.fill_between(grouped['decile'], grouped['sd_lower'], grouped['sd_upper'], 
                        alpha=0.3, label='Mean ± 2σ', color='steelblue')
    ax1.set_xlabel(groupby_label, fontsize=11)
    ax1.set_ylabel(ylabel, fontsize=11)
    ax1.set_title('Individual Building Variability', fontsize=13)
    ax1.set_xticks(grouped['decile'].unique())
    ax1.legend()
    ax1.grid(alpha=0.3)
    if costs:
        ax1.set_ylim(0)
    if percentage: 
        ax1.axhline(0, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(0, color='r', linestyle='--', alpha=0.5)
    if rot: 
        ax1.set_xlabel(groupby_label, fontsize=11)
        ax1.tick_params(axis='x', rotation=90)

    # Plot 2: Bar chart with error bars (confidence in mean)
    ax2.bar(grouped['decile'], grouped['mean_reduction'], 
            color='darkgreen', alpha=0.7, width=0.8)
    ax2.errorbar(grouped['decile'], grouped['mean_reduction'], 
                    yerr=grouped['ci_error'], fmt='none', 
                    color='black', capsize=5, capthick=2, 
                    label='Mean ± 2 SE (95% CI)')
    ax2.set_xlabel(groupby_label, fontsize=11)
    ax2.set_ylabel(ylabel, fontsize=11)
    ax2.set_title('Confidence in Mean Estimate', fontsize=13)
    ax2.set_xticks(grouped['decile'].unique())
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')
    if rot: 
        ax2.set_xlabel(groupby_label, fontsize=11)
        ax2.tick_params(axis='x', rotation=90)

    if title:
        fig.suptitle(title)
    plt.tight_layout()

    

    # Print summary
    print("\nSummary Statistics:")
    print(grouped[['decile', 'mean_reduction', 'pooled_sd', 'se', 'n_buildings']])
    return fig

def plot_col_reduction_by_decile_sum(res_df,
                                      
                                groupby_col='avg_gas_percentile',
                                mean_col='loft_installation_energy_loft_percentile_gas_mean',
                                std_col='loft_installation_energy_loft_percentile_gas_std',
                                groupby_label='Gas Usage Decile',
                                ylabel='Energy Reduction (%)',
                                title=None,
                                figsize=(16, 6),
                                show_plot=True,
                                costs=True,
                                rot=False, 
                                
                                percentage=True,):
                                
    
    df = res_df.copy()
    df[f'{std_col}_2'] = (df[std_col] ** 2)


    # Changed 'mean' to 'sum' here
    grouped = df.groupby(groupby_col).agg({
        mean_col: ['sum', 'count'],  # ← CHANGED FROM 'mean' TO 'sum'
        f'{std_col}_2': ['sum']
    }).reset_index()
    grouped.columns = ['decile', 'total_reduction', 'n_buildings', 'std_squared_summed']

    # For sum: uncertainty is sqrt(sum of variances)
    grouped['pooled_sd'] = np.sqrt(grouped['std_squared_summed'])

    if percentage:
        grouped['total_reduction'] *= 100
        grouped['pooled_sd'] *= 100

    # For sums, SE doesn't apply the same way, but showing uncertainty bounds
    grouped['se'] = grouped['pooled_sd'] / np.sqrt(grouped['n_buildings'])

    # SD bounds (total variability)
    grouped['sd_lower'] = grouped['total_reduction'] - 2 * grouped['pooled_sd']
    grouped['sd_upper'] = grouped['total_reduction'] + 2 * grouped['pooled_sd']

    # SE bounds
    grouped['se_lower'] = grouped['total_reduction'] - grouped['se']
    grouped['se_upper'] = grouped['total_reduction'] + grouped['se']

    # Calculate error bar size
    grouped['ci_error'] = 2 * grouped['se']
 

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Standard Deviation (total variability)
    ax1.plot(grouped['decile'], grouped['total_reduction'], 
                'o-', linewidth=2, markersize=8, color='steelblue')
    ax1.fill_between(grouped['decile'], grouped['sd_lower'], grouped['sd_upper'], 
                        alpha=0.3, label='Total ± 2σ', color='steelblue')
    ax1.set_xlabel(groupby_label, fontsize=11)
    ax1.set_ylabel(ylabel, fontsize=11)
    ax1.set_title('Total Reduction Variability', fontsize=13)
    ax1.set_xticks(grouped['decile'].unique())
    ax1.legend()
    ax1.grid(alpha=0.3)
    if costs:
        ax1.set_ylim(0)
    if percentage: 
        ax1.axhline(0, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(0, color='r', linestyle='--', alpha=0.5)
    if rot: 
        ax1.set_xlabel(groupby_label, fontsize=11)
        ax1.tick_params(axis='x', rotation=90)

    # Plot 2: Bar chart with error bars
    ax2.bar(grouped['decile'], grouped['total_reduction'], 
            color='darkgreen', alpha=0.7, width=0.8)
    ax2.errorbar(grouped['decile'], grouped['total_reduction'], 
                    yerr=grouped['ci_error'], fmt='none', 
                    color='black', capsize=5, capthick=2, 
                    label='Total ± 2 SE')
    ax2.set_xlabel(groupby_label, fontsize=11)
    ax2.set_ylabel(ylabel, fontsize=11)
    ax2.set_title('Total Reduction Estimate', fontsize=13)
    ax2.set_xticks(grouped['decile'].unique())
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')
    if rot: 
        ax2.set_xlabel(groupby_label, fontsize=11)
        ax2.tick_params(axis='x', rotation=90)

    if title:
        fig.suptitle(title)
    plt.tight_layout()

    # Print summary
    print("\nSummary Statistics:")
    print(grouped[['decile', 'total_reduction', 'pooled_sd', 'se', 'n_buildings']])
    return fig



def plot_energy_reduction_by_decile(
    df,
    groupby_col='avg_gas_percentile',
    mean_col='loft_installation_energy_loft_percentile_gas_mean',
    std_col='loft_installation_energy_loft_percentile_gas_std',
    groupby_label='Gas Usage Decile',
    ylabel='Energy Reduction (%)',
    title=None ,
    figsize=(16, 6),
    show_plot=True,
    return_data=False,
    percentage=True ,
):
    """
    Plot energy reduction statistics grouped by deciles with SD and SE bounds.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing the data
    groupby_col : str
        Column name to group by (e.g., 'avg_gas_percentile')
    mean_col : str
        Column name containing mean values
    std_col : str
        Column name containing standard deviation values
    groupby_label : str
        Label for x-axis
    ylabel : str
        Label for y-axis
    figsize : tuple
        Figure size (width, height)
    show_plot : bool
        Whether to display the plot
    return_data : bool
        Whether to return the grouped dataframe
    
    Returns:
    --------
    pd.DataFrame (optional)
        Grouped statistics if return_data=True
    """
    
    # Group by decile
    grouped = df.groupby(groupby_col).agg({
        mean_col: ['mean', 'count'],
        std_col: 'mean'
    }).reset_index()
     

    
    grouped.columns = ['decile', 'mean_reduction', 'n_buildings', 'std_reduction']
    
    # Convert to percentages
    if percentage:
        grouped['mean_reduction'] *= 100
        grouped['std_reduction'] *= 100
    
 
   
    # Calculate both bounds
    grouped['se'] = grouped['std_reduction'] / np.sqrt(grouped['n_buildings'])
    
    # SD bounds (individual variability)
    grouped['sd_lower'] = grouped['mean_reduction'] - 2 * grouped['std_reduction']
    grouped['sd_upper'] = grouped['mean_reduction'] + 2 * grouped['std_reduction']
    
    # SE bounds (confidence in mean)
    grouped['se_lower'] = grouped['mean_reduction'] - 2 * grouped['se']
    grouped['se_upper'] = grouped['mean_reduction'] + 2 * grouped['se']
    
    # Calculate error bar size (from mean to upper/lower bound)
    grouped['ci_error'] = 2 * grouped['se']
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Standard Deviation (individual outcomes)
    ax1.plot(grouped['decile'], grouped['mean_reduction'], 
             'o-', linewidth=2, markersize=8, color='steelblue')
    ax1.fill_between(grouped['decile'], grouped['sd_lower'], grouped['sd_upper'], 
                     alpha=0.3, label='Mean ± 2σ', color='steelblue')
    ax1.set_xlabel(groupby_label, fontsize=11)
    ax1.set_ylabel(ylabel, fontsize=11)
    ax1.set_title('Individual Building Variability', fontsize=13)
    ax1.set_xticks(grouped['decile'].unique())
    ax1.legend()
    ax1.grid(alpha=0.3)
    if percentage: 
        ax1.axhline(0, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(0, color='r', linestyle='--', alpha=0.5)
    
    # Plot 2: Bar chart with error bars (confidence in mean)
    ax2.bar(grouped['decile'], grouped['mean_reduction'], 
            color='darkgreen', alpha=0.7, width=0.8)
    ax2.errorbar(grouped['decile'], grouped['mean_reduction'], 
                 yerr=grouped['ci_error'], fmt='none', 
                 color='black', capsize=5, capthick=2, 
                 label='Mean ± 2 SE (95% CI)')
    ax2.set_xlabel(groupby_label, fontsize=11)
    ax2.set_ylabel(ylabel, fontsize=11)
    ax2.set_title('Confidence in Mean Estimate', fontsize=13)
    ax2.set_xticks(grouped['decile'].unique())
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')
    
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    
    # Print summary
    print("\nSummary Statistics:")
    print(grouped[['decile', 'mean_reduction', 'std_reduction', 'se', 'n_buildings']])
    
    if return_data:
        return grouped


def plot_building_counts_by_age_band(
    df,
    groupby_cols=['avg_gas_percentile', 'premise_age_band'],
    cavity_col='absolute_reduction_cavity',
    solid_internal_col='absolute_reduction_solid_internal',
    solid_external_col='absolute_reduction_solid_external',
    decile_label='Gas Usage Decile',
    age_label='premise_age_band',
    title='Building Counts by Age Band',
    age_band_order=None,
    figsize=(18, 10),
    show_plot=True,
    return_data=False
):
    """
    Plot stacked bar charts showing building counts for each age band.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing the data
    groupby_cols : list
        Columns to group by [decile_col, age_band_col]
    cavity_col : str
        Column to check for cavity wall eligibility
    solid_internal_col : str
        Column to check for solid internal wall eligibility
    solid_external_col : str
        Column to check for solid external wall eligibility
    decile_label : str
        Label for x-axis
    age_label : str
        Name of age band column
    title : str
        Overall figure title
    age_band_order : list, optional
        Custom order for age bands. If None, uses sorted order
    figsize : tuple
        Figure size (width, height)
    show_plot : bool
        Whether to display the plot
    return_data : bool
        Whether to return the grouped dataframes
    
    Returns:
    --------
    dict of pd.DataFrame (optional)
        Dictionary with age bands as keys if return_data=True
    """
    
    if age_band_order is not None:
        age_bands = [ab for ab in age_band_order if ab in df[age_label].values]
    else:
        age_bands = sorted(df[age_label].dropna().unique())
    n_age_bands = len(age_bands)
    
    # Calculate grid dimensions
    n_cols = min(3, n_age_bands)
    n_rows = int(np.ceil(n_age_bands / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False, sharey=True)
    axes = axes.flatten()
    
    all_counts = {}
    
    for idx, age_band in enumerate(age_bands):
        ax = axes[idx]
        
        # Filter data for this age band
        df_age = df[df[age_label] == age_band].copy()
        
        # Create eligibility flags
        df_age['cavity_eligible'] = df_age[cavity_col].notna() & (df_age[cavity_col] != 0)
        df_age['solid_internal_eligible'] = df_age[solid_internal_col].notna() & (df_age[solid_internal_col] != 0)
        df_age['solid_external_eligible'] = df_age[solid_external_col].notna() & (df_age[solid_external_col] != 0)
        
        # Group by decile and count
        counts = df_age.groupby(groupby_cols[0]).agg({
            'cavity_eligible': 'sum',
            'solid_internal_eligible': 'sum',
            'solid_external_eligible': 'sum'
        }).reset_index()
        
        counts.columns = ['decile', 'cavity_count', 'solid_internal_count', 'solid_external_count']
        counts['total_count'] = counts['cavity_count'] + counts['solid_internal_count'] + counts['solid_external_count']
        
        if len(counts) == 0:
            ax.axis('off')
            continue
        
        # Create stacked bars
        x = counts['decile']
        ax.bar(x, counts['cavity_count'], label='Cavity Wall', 
               color='steelblue', alpha=0.8)
        ax.bar(x, counts['solid_internal_count'], bottom=counts['cavity_count'],
               label='Solid Internal Wall', color='coral', alpha=0.8)
        ax.bar(x, counts['solid_external_count'], 
               bottom=counts['cavity_count'] + counts['solid_internal_count'],
               label='Solid External Wall', color='mediumseagreen', alpha=0.8)
        
        # Add total labels
        for i, decile in enumerate(counts['decile']):
            total = counts['total_count'].iloc[i]
            if total > 0:
                ax.text(decile, total, f"{int(total)}", 
                       ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel(decile_label, fontsize=10)
        ax.set_ylabel('Number of Buildings', fontsize=10)
        ax.set_title(f'{age_band}\nTotal: {counts["total_count"].sum():.0f}', 
                    fontsize=11, fontweight='bold')
        ax.set_xticks(counts['decile'])
        
        if idx == 0:
            ax.legend(fontsize=9, loc='upper right')
        
        ax.grid(alpha=0.3, axis='y')
        
        all_counts[age_band] = counts
    
    # Hide unused subplots
    for idx in range(n_age_bands, len(axes)):
        axes[idx].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    
    # Print summary
    for age_band, counts in all_counts.items():
        print(f"\n{age_band}:")
        print(counts[['decile', 'cavity_count', 'solid_internal_count', 'solid_external_count', 'total_count']])
    
    if return_data:
        return all_counts
    return fig 
    


def plot_building_counts_by_conservation_area(
    df,
    groupby_cols=['avg_gas_percentile', 'conservation_area_bool'],
    cavity_col='absolute_reduction_cavity',
    solid_internal_col='absolute_reduction_solid_internal',
    solid_external_col='absolute_reduction_solid_external',
    decile_label='Gas Usage Decile',
    conservation_label='conservation_area_bool',
    title='Building Counts by Conservation Area',
    figsize=(14, 8),
    show_plot=True,
    return_data=False
):
    """
    Plot grouped stacked bar chart showing counts of buildings eligible for 
    cavity, solid internal, and solid external wall interventions in each decile,
    split by conservation area (True/False).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing the data
    groupby_cols : list
        Columns to group by [decile_col, conservation_col]
    cavity_col : str
        Column to check for cavity wall eligibility (non-null/non-zero)
    solid_internal_col : str
        Column to check for solid internal wall eligibility
    solid_external_col : str
        Column to check for solid external wall eligibility
    decile_label : str
        Label for x-axis
    conservation_label : str
        Name of conservation area column
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    show_plot : bool
        Whether to display the plot
    return_data : bool
        Whether to return the grouped dataframe
    
    Returns:
    --------
    pd.DataFrame (optional)
        Count statistics if return_data=True
    """
    
    # Create eligibility flags (non-null and non-zero)
    df_temp = df.copy()
    df_temp['cavity_eligible'] = df_temp[cavity_col].notna() & (df_temp[cavity_col] != 0)
    df_temp['solid_internal_eligible'] = df_temp[solid_internal_col].notna() & (df_temp[solid_internal_col] != 0)
    df_temp['solid_external_eligible'] = df_temp[solid_external_col].notna() & (df_temp[solid_external_col] != 0)
    
    # Group by decile and conservation area
    counts = df_temp.groupby(groupby_cols).agg({
        'cavity_eligible': 'sum',
        'solid_internal_eligible': 'sum',
        'solid_external_eligible': 'sum'
    }).reset_index()
    
    counts.columns = ['decile', 'conservation_area', 'cavity_count', 'solid_internal_count', 'solid_external_count']
    counts['total_count'] = counts['cavity_count'] + counts['solid_internal_count'] + counts['solid_external_count']
    
    # Pivot to get separate columns for conservation area True/False
    counts_pivot = counts.pivot(index='decile', columns='conservation_area', 
                                 values=['cavity_count', 'solid_internal_count', 'solid_external_count', 'total_count'])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique deciles
    deciles = counts['decile'].unique()
    x = np.arange(len(deciles))
    width = 0.35  # Width of each bar
    
    # Colors for each intervention type
    colors = {
        'cavity': 'steelblue',
        'solid_internal': 'coral',
        'solid_external': 'mediumseagreen'
    }
    
    # Plot for Conservation Area = False (left bars)
    if False in counts['conservation_area'].values:
        cavity_false = counts_pivot.get(('cavity_count', False), pd.Series(0, index=deciles)).fillna(0)
        solid_internal_false = counts_pivot.get(('solid_internal_count', False), pd.Series(0, index=deciles)).fillna(0)
        solid_external_false = counts_pivot.get(('solid_external_count', False), pd.Series(0, index=deciles)).fillna(0)
        
        p1 = ax.bar(x - width/2, cavity_false, width, 
                   label='Cavity (Non-CA)', color=colors['cavity'], alpha=0.7)
        p2 = ax.bar(x - width/2, solid_internal_false, width, bottom=cavity_false,
                   label='Solid Internal (Non-CA)', color=colors['solid_internal'], alpha=0.7)
        p3 = ax.bar(x - width/2, solid_external_false, width, 
                   bottom=cavity_false + solid_internal_false,
                   label='Solid External (Non-CA)', color=colors['solid_external'], alpha=0.7)
        
        # Add total labels on top of Non-CA bars
        total_false = counts_pivot.get(('total_count', False), pd.Series(0, index=deciles)).fillna(0)
        for i, (dec, total) in enumerate(zip(deciles, total_false)):
            if total > 0:
                ax.text(i - width/2, total + max(counts['total_count'])*0.01, 
                       f"{int(total)}", ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Plot for Conservation Area = True (right bars)
    if True in counts['conservation_area'].values:
        cavity_true = counts_pivot.get(('cavity_count', True), pd.Series(0, index=deciles)).fillna(0)
        solid_internal_true = counts_pivot.get(('solid_internal_count', True), pd.Series(0, index=deciles)).fillna(0)
        solid_external_true = counts_pivot.get(('solid_external_count', True), pd.Series(0, index=deciles)).fillna(0)
        
        p4 = ax.bar(x + width/2, cavity_true, width, 
                   label='Cavity (CA)', color=colors['cavity'], alpha=1.0, edgecolor='black', linewidth=1.5)
        p5 = ax.bar(x + width/2, solid_internal_true, width, bottom=cavity_true,
                   label='Solid Internal (CA)', color=colors['solid_internal'], alpha=1.0, edgecolor='black', linewidth=1.5)
        p6 = ax.bar(x + width/2, solid_external_true, width, 
                   bottom=cavity_true + solid_internal_true,
                   label='Solid External (CA)', color=colors['solid_external'], alpha=1.0, edgecolor='black', linewidth=1.5)
        
        # Add total labels on top of CA bars
        total_true = counts_pivot.get(('total_count', True), pd.Series(0, index=deciles)).fillna(0)
        for i, (dec, total) in enumerate(zip(deciles, total_true)):
            if total > 0:
                ax.text(i + width/2, total + max(counts['total_count'])*0.01, 
                       f"{int(total)}", ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel(decile_label, fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Buildings', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(deciles)
    ax.legend(loc='upper left', fontsize=10, ncol=2)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    
    # Print summary
    print("\nBuilding Counts by Decile, Conservation Area, and Intervention Type:")
    print(counts[['decile', 'conservation_area', 'cavity_count', 'solid_internal_count', 
                  'solid_external_count', 'total_count']])
    
    print("\n" + "="*80)
    print("SUMMARY BY CONSERVATION AREA:")
    print("="*80)
    
    for conservation_val in [False, True]:
        subset = counts[counts['conservation_area'] == conservation_val]
        if len(subset) > 0:
            print(f"\nConservation Area = {conservation_val}:")
            print(f"  Total Cavity Eligible: {subset['cavity_count'].sum()}")
            print(f"  Total Solid Internal Eligible: {subset['solid_internal_count'].sum()}")
            print(f"  Total Solid External Eligible: {subset['solid_external_count'].sum()}")
            print(f"  Grand Total: {subset['total_count'].sum()}")
    
    if return_data:
        return counts


