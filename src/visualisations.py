import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
    energy=True ,
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
    if energy:
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
    if energy: 
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
