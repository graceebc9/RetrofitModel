import pandas as pd
import matplotlib.pyplot as plt
import numpy as np




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

    

    # Print summary
    print("\nSummary Statistics:")
    print(grouped[['decile', 'mean_reduction', 'pooled_sd', 'se', 'n_buildings']])
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


