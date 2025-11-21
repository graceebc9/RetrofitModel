import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os 
import gc 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from .visualisations import plot_building_counts_by_age_band , plot_building_counts_by_conservation_area 

scenario_list_clean = {
    'joint_heat_loft_decay': 'Heat Pump + Loft Ins.',
 'joint_heat_wall_decay': 'Heat Pump + Wall Ins.',
 'wall_installation' : 'Wall Insulation',
 'join_heat_ins_decay': 'Heat Pump+ Wall+ Loft',
 'heat_pump_only': 'Heat Pump',
 'loft_installation': 'Loft Insulation ' 
} 


def create_age_buckets(df):
    df['premise_age_bucketed'] = np.where(df['premise_age'].isin(['Pre 1837', '1837-1869', '1870-1918']), 'Pre 1919', df['premise_age'])
    return df 




def run_meta(output_dir, args, proc_df):
    
    # meta costs 
    meta_op = f'{output_dir}/meta_summary'
    os.makedirs(meta_op, exist_ok=True )
    
    fig, axes = plot_co2_savings_and_costs(
        proc_df, 
        args.scenarios, 
        style="white"  
    )
    fig.savefig(f'{meta_op}/total_cost_compairsons.png')

    # stock summary - one run 
    pl_data= proc_df[proc_df['epistemic_run_id']==1].copy() 
    pl_data=create_age_buckets(pl_data )

    pl_data = pd.concat([pl_data[['avg_gas_percentile', 'premise_age_bucketed', 'conservation_area_bool', 'premise_type' ]], pd.get_dummies(pl_data['inferred_insulation_type'])], axis=1 ) 
    

    # Plot building counts by age band
    fig ,age_band_counts = plot_building_counts_by_age_band(
        pl_data,
        groupby_cols=['avg_gas_percentile', 'premise_age_bucketed'],
        cavity_col='cavity_wall_insulation',
        solid_internal_col='internal_wall_insulation',
        solid_external_col='external_wall_insulation',
        decile_label='Gas Usage Decile',
        age_label='premise_age_bucketed',
        title=None, 
        age_band_order= ['Pre 1919','1919-1944',  '1945-1959', '1960-1979', '1980-1989', 
        '1990-1999', 'Post 1999',  'Unknown date'],
        figsize=(18, 10),
        show_plot=False,
        return_data=False
    )
    fig.savefig(f'{meta_op}/building_counts_by_age_bands.png')
    # pd.DataFrame(age_band_counts).to_csv(f'{meta_op}/age_band_counts.csv', index=False )

    # from src.visualisations import 

    # Usage example:
    fig , conservation_counts = plot_building_counts_by_conservation_area(
        pl_data,
        groupby_cols=['avg_gas_percentile', 'conservation_area_bool'],
        cavity_col='cavity_wall_insulation',
        solid_internal_col='internal_wall_insulation',
        solid_external_col='external_wall_insulation',
        decile_label='Gas Usage Decile',
        conservation_label='conservation_area_bool',
        title= None, 
        figsize=(14, 8),
        show_plot=False,
        return_data=True
    )
    fig.savefig(f'{meta_op}/building_counts_by_conservation.png')
    # pd.DataFrame(conservation_counts).to_csv(f'{meta_op}/conservation_counts.csv', index=False )

     
    fig , property_type_counts = plot_building_counts_by_age_band(
        pl_data,
        groupby_cols=['avg_gas_percentile', 'premise_type'],
        cavity_col='cavity_wall_insulation',
        solid_internal_col='internal_wall_insulation',
        solid_external_col='external_wall_insulation',
        decile_label='Gas Usage Decile',
        age_label='premise_type',
        title=None, 
        age_band_order= None, 
        figsize=(18, 10),
        show_plot=False ,
        return_data=False
    )
    fig.savefig(f'{meta_op}/property_type_counts_by_decile.png')
    # pd.DataFrame(property_type_counts).to_csv(f'{meta_op}/property_type_counts.csv', index=False )

    
    del pl_data


def plot_co2_savings_and_costs(proc_df, scenario_list, 
                                figsize=(14, 7), style="whitegrid"):
    """
    Create a grouped bar chart showing CO2 savings and costs by scenario.
    
    Parameters:
    -----------
    proc_df : pd.DataFrame
        Processed dataframe containing the metrics
    scenario_list : list
        List of scenario identifiers
    scenario_list_clean : dict
        Dictionary mapping scenario identifiers to clean display names
    figsize : tuple, optional
        Figure size (width, height). Default is (14, 7)
    style : str, optional
        Seaborn style. Default is "whitegrid"
    
    Returns:
    --------
    fig, (ax1, ax2) : tuple
        Figure and axes objects
    """
    
    # Set seaborn style and color palette
    sns.set_theme(style=style)
    sns.set_palette("husl")
    
    # Function to process data for a given metric
    def get_metric_data(proc_df, scenario_list, column_pattern):
        ep_runs = proc_df.groupby('epistemic_run_id')[
            [column_pattern.format(x=x) for x in scenario_list]
        ].sum()
        
        means = ep_runs.mean().values
        stds = ep_runs.std().values
        
        return means, stds
    
    # Get CO2 data (convert to megatons)
    total_means, total_stds = get_metric_data(proc_df, scenario_list, 'total_tonne_co2_saved_{x}_5yr_mean')
    total_means = (total_means * -1) / 1_000_000
    total_stds = total_stds / 1_000_000
    
    gas_means, gas_stds = get_metric_data(proc_df, scenario_list, 'gas_total_tonne_co2_saved_{x}_5yr_mean')
    gas_means = (gas_means * -1) / 1_000_000
    gas_stds = gas_stds / 1_000_000
    
    # Get cost data (convert to millions)
    cost_means, cost_stds = get_metric_data(proc_df, scenario_list, '{x}_cost_{x}_mean')
    cost_means = cost_means / 1_000_000
    cost_stds = cost_stds / 1_000_000
    
    # Sort by total CO2 savings
    sort_idx = np.argsort(total_means)
    labels = [scenario_list_clean[scenario_list[i]] for i in sort_idx]
    total_means = total_means[sort_idx]
    total_stds = total_stds[sort_idx]
    gas_means = gas_means[sort_idx]
    gas_stds = gas_stds[sort_idx]
    cost_means = cost_means[sort_idx]
    cost_stds = cost_stds[sort_idx]
    
    # Create grouped bar chart with secondary axis
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Get seaborn colors for bars
    colors = sns.color_palette("husl", 2)
    
    # Define darker blue for cost line
    dark_blue = '#0066cc'
    
    # Plot CO2 savings on primary axis
    bars1 = ax1.bar(x - width/2, total_means, width, yerr=total_stds, 
                    label='Total CO2 Saved', capsize=5, alpha=0.85, 
                    color=colors[0], edgecolor='white', linewidth=1.2)
    bars2 = ax1.bar(x + width/2, gas_means, width, yerr=gas_stds, 
                    label='Gas CO2 Saved', capsize=5, alpha=0.85,
                    color=colors[1], edgecolor='white', linewidth=1.2)
    
    ax1.set_xlabel('Scenario', fontsize=12, fontweight='bold')
    ax1.set_ylabel('CO2 Saved (MegaTON)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=90, ha='right')
    ax1.legend(loc='upper left', frameon=True, shadow=True)
    
    # Create secondary y-axis for costs
    ax2 = ax1.twinx()
    line = ax2.plot(x, cost_means, 'o-', linewidth=2.5, markersize=8, 
                    label='Total Cost', color=dark_blue, zorder=5)
    ax2.errorbar(x, cost_means, yerr=cost_stds, fmt='none', 
                 ecolor=dark_blue, capsize=5, alpha=0.5, zorder=4)
    
    ax2.set_ylabel('Total Cost (£M)', fontsize=12, fontweight='bold', color=dark_blue)
    ax2.tick_params(axis='y', labelcolor=dark_blue)
    ax2.legend(loc='center left', frameon=True, shadow=True)
    
    # plt.title('CO2 Savings and Costs by Scenario', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    sns.despine(right=False)  # Keep right spine for secondary axis
    
    return fig, (ax1, ax2)