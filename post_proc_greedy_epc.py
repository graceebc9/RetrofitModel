"""
Pareto Analysis: Calculate difference between optimization methods and generate visualizations.

Unified notation:
- Loft probability markers: ● (circle) for 0.95, ▲ (triangle) for 0.65
- Budget: £M on axes, integer values (1, 10, 50, 80, 100)
- CO2: kilotonnes (kt) with comma formatting
- Uncertainty: Three levels - uncorrelated, partial (ρ), and fully correlated
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# =============================================================================
# GLOBAL STYLE CONFIGURATION
# =============================================================================
MARKER_LOFT_095 = 'o'      # Circle for loft probability 0.95
MARKER_LOFT_065 = '^'      # Triangle for loft probability 0.65
LINESTYLE_LOFT_095 = '-'   # Solid line for 0.95
LINESTYLE_LOFT_065 = '--'  # Dashed line for 0.65

BUDGET_LABEL = 'Budget (£M)'
CO2_LABEL_KT = 'CO2 Improvement (kt)'
CO2_LABEL_PCT = 'Improvement (%)'
EQUITY_LABEL = 'Equity Factor'


def get_loft_style(loft_prob):
    """Return consistent marker and linestyle for a given loft probability."""
    if loft_prob == 0.95:
        return MARKER_LOFT_095, LINESTYLE_LOFT_095
    else:
        return MARKER_LOFT_065, LINESTYLE_LOFT_065


def format_budget_ticks(budget_m_values):
    """Format budget values as integers for tick labels."""
    return [f'{int(b)}' for b in budget_m_values]


def sum_portfolio_stats(df, prefix='', rho=0.5):
    """
    Aggregates a list of buildings into a portfolio total using Method 2 logic.
    Returns Mean, Uncorrelated Std, Partially Correlated Std (rho), and Fully Correlated Std for Carbon.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Building-level data with mean and std columns
    prefix : str
        Prefix for output column names (e.g., 'opt_' or 'epc_')
    rho : float, default=0.5
        Correlation parameter for partial correlation calculation
    """
    if df.empty:
        return {
            f'{prefix}mean_co2': 0,
            f'{prefix}std_co2_uncorr': 0,
            f'{prefix}std_co2_partial': 0,
            f'{prefix}std_co2_corr': 0
        }

    # 1. Total Mean (Simple Sum)
    total_mean = df['mean_total_co2_saved'].sum()

    # 2. Uncorrelated Std (Square -> Sum -> Root)
    # Assumes independence between buildings
    var_sum = (df['std_total_co2_saved'] ** 2).sum()
    std_uncorr = np.sqrt(var_sum)

    # 3. Fully Correlated Std (Sum -> Sum)
    # Assumes worst-case perfect correlation (systemic risk)
    std_corr = df['std_total_co2_saved'].sum()
    
    # 4. Partially Correlated Std (Interpolation between uncorr and corr)
    # Formula: σ_partial = √[(1-ρ)·σ_uncorr² + ρ·σ_corr²]
    std_partial = np.sqrt(
        (1 - rho) * std_uncorr**2 + rho * std_corr**2
    )

    return {
        f'{prefix}mean_co2': total_mean,
        f'{prefix}std_co2_uncorr': std_uncorr,
        f'{prefix}std_co2_partial': std_partial,
        f'{prefix}std_co2_corr': std_corr
    }


def calculate_robust_difference(stats_opt, stats_epc, rho=0.5):
    """
    Calculates the Mean Difference and the Standard Deviation of that Difference.
    Returns the Diff, the Uncertainty of the Diff (3 levels), and Z-Scores (Significance).
    
    Key insight: We apply ρ to EACH portfolio's internal correlation,
    then combine the two portfolios using sum-of-squares (they're independent).
    
    Parameters:
    -----------
    stats_opt : dict
        Statistics for optimized portfolio
    stats_epc : dict
        Statistics for random EPC portfolio
    rho : float, default=0.5
        Correlation parameter for partial correlation calculation
    """
    # 1. Mean Difference (The Gain)
    diff_mean = stats_opt['opt_mean_co2'] - stats_epc['epc_mean_co2']
    
    # 2. Uncertainty of Difference (Square, Sum, Root)
    # Formula: Sigma_Diff = Sqrt( Sigma_Opt^2 + Sigma_Epc^2 )
    # Note: We use sum-of-squares because opt and epc are independent realizations
    
    # ... Uncorrelated Path (Optimistic)
    diff_std_uncorr = np.sqrt(
        stats_opt['opt_std_co2_uncorr']**2 + stats_epc['epc_std_co2_uncorr']**2
    )
    
    # ... Partially Correlated Path (Realistic, using rho)
    diff_std_partial = np.sqrt(
        stats_opt['opt_std_co2_partial']**2 + stats_epc['epc_std_co2_partial']**2
    )
    
    # ... Fully Correlated Path (Pessimistic)
    diff_std_corr = np.sqrt(
        stats_opt['opt_std_co2_corr']**2 + stats_epc['epc_std_co2_corr']**2
    )
    
    # 3. Z-Scores (Signal to Noise)
    # Avoid divide by zero
    z_uncorr = diff_mean / diff_std_uncorr if diff_std_uncorr > 0 else 0
    z_partial = diff_mean / diff_std_partial if diff_std_partial > 0 else 0
    z_corr = diff_mean / diff_std_corr if diff_std_corr > 0 else 0
    
    return {
        'diff_mean_co2': diff_mean,
        'diff_std_uncorr': diff_std_uncorr,
        'diff_std_partial': diff_std_partial,
        'diff_std_corr': diff_std_corr,
        'z_score_uncorr': z_uncorr,
        'z_score_partial': z_partial,
        'z_score_corr': z_corr
    }


def aggregate_scenario_results(greedy_runs_folder, budgets, loft_probs, equity_factors,
                                million_factor=1_000_000, rho=0.5):
    """
    Loop through file structure, aggregate totals with uncertainty propagation, 
    and return a Meta DataFrame.
    
    Parameters:
    -----------
    greedy_runs_folder : str
        Base folder containing all scenario runs
    budgets : list
        List of budget values to analyze
    loft_probs : list
        List of loft probabilities to analyze
    equity_factors : list
        List of equity factors to analyze
    million_factor : int, default=1_000_000
        Factor to convert budgets to millions
    rho : float, default=0.5
        Correlation parameter for partial correlation calculation
    """
    print(f'Starting robust aggregation with significance testing (ρ={rho})...')
    meta_results = []

    for prob_loft in loft_probs:
        print(f'Processing loft probability: {prob_loft}')
        for budget in budgets:
            million_budget = budget / million_factor
            for equity_factor in equity_factors:
                
                folder_name = f'budget_{int(million_budget)}M__loft_{prob_loft}__equity_{equity_factor}'
                output_dir = os.path.join(greedy_runs_folder, folder_name)

                selected_path = os.path.join(output_dir, 'selected_projects.csv')
                epc_random_path = os.path.join(output_dir, 'epc_random_selection.csv')

                if os.path.exists(selected_path) and os.path.exists(epc_random_path):
                    try:
                        df_opt = pd.read_csv(selected_path)
                        df_epc = pd.read_csv(epc_random_path)

                        # A. Get Totals with all three uncertainty levels
                        stats_opt = sum_portfolio_stats(df_opt, prefix='opt_', rho=rho)
                        stats_epc = sum_portfolio_stats(df_epc, prefix='epc_', rho=rho)

                        row = {
                            'budget_raw': budget,
                            'budget_m': million_budget,
                            'loft_prob': prob_loft,
                            'equity_factor': equity_factor,
                            'scenario_id': folder_name,
                            'rho': rho
                        }
                        row.update(stats_opt)
                        row.update(stats_epc)

                        # B. Calculate Robust Difference & Significance
                        diff_stats = calculate_robust_difference(stats_opt, stats_epc, rho=rho)
                        row.update(diff_stats)
                        
                        # C. Simple % Gain for reference
                        if row['epc_mean_co2'] != 0:
                            row['diff_pct'] = (row['diff_mean_co2'] / row['epc_mean_co2']) * 100
                        else:
                            row['diff_pct'] = 0

                        meta_results.append(row)

                    except Exception as e:
                        print(f'Error processing {folder_name}: {e}')

    if meta_results:
        meta_df = pd.DataFrame(meta_results)
        
        # Order columns logically for final report
        cols = [
            'budget_m', 'loft_prob', 'equity_factor', 'rho',
            'diff_pct', 
            'diff_mean_co2', 
            'diff_std_uncorr', 'diff_std_partial', 'diff_std_corr',
            'z_score_uncorr', 'z_score_partial', 'z_score_corr',
            'opt_mean_co2', 'epc_mean_co2',
            'scenario_id'
        ]
        # Safety check for missing columns
        cols = [c for c in cols if c in meta_df.columns]
        
        output_filename = f'diff_epc_opt_robust_significance_rho_{rho}.csv'
        meta_df = meta_df[cols]
        meta_df.to_csv(output_filename, index=False)
        print(f'Results saved to {output_filename}')
        return meta_df
    else:
        print("No results found.")
        return pd.DataFrame()


def create_all_configs_plot(df, output_dir):
    """Create Pareto plot with all configurations colored by equity factor."""
    fig, ax = plt.subplots(figsize=(14, 9))

    equity_factors = sorted(df['equity_factor'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(equity_factors)))

    for idx, eq in enumerate(equity_factors):
        for loft in sorted(df['loft_prob'].unique()):
            subset = df[(df['loft_prob'] == loft) & (df['equity_factor'] == eq)].sort_values('budget_m')
            marker, linestyle = get_loft_style(loft)
            label = f'Equity {eq}, Loft {loft}'

            ax.plot(subset['budget_m'], subset['improvement_pct'],
                    marker=marker, linestyle=linestyle, linewidth=2.5, markersize=8,
                    color=colors[idx], alpha=0.8, label=label)

    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=2)
    ax.set_xlabel(BUDGET_LABEL, fontsize=14, fontweight='bold')
    ax.set_ylabel(CO2_LABEL_PCT, fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, ncol=2, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'pareto_all_configs_colored.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filepath}")


def create_heatmap_pct(df, loft_prob, pct_vmin, pct_vmax, output_dir):
    """Create percentage improvement heatmap for a specific loft probability."""
    fig, ax = plt.subplots(figsize=(10, 6))

    pivot = df[df['loft_prob'] == loft_prob].pivot(
        index='equity_factor', columns='budget_m', values='improvement_pct'
    )
    # Rename columns to integer format
    pivot.columns = [int(c) for c in pivot.columns]

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn',
                vmin=pct_vmin, vmax=pct_vmax, center=(pct_vmin + pct_vmax) / 2,
                ax=ax, cbar_kws={'label': CO2_LABEL_PCT},
                linewidths=0.5, linecolor='gray')

    ax.set_xlabel(BUDGET_LABEL, fontsize=13)
    ax.set_ylabel(EQUITY_LABEL, fontsize=13)

    plt.tight_layout()
    loft_str = str(loft_prob).replace('.', '')
    filepath = os.path.join(output_dir, f'heatmap_pct_loft{loft_str}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filepath}")


def create_heatmap_abs(df, loft_prob, abs_vmin, abs_vmax, output_dir):
    """Create absolute CO2 improvement heatmap (in kilotonnes) for a specific loft probability."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert to kilotonnes
    df_kt = df.copy()
    df_kt['co2_improvement_kt'] = df_kt['co2_improvement'] / 1000

    pivot = df_kt[df_kt['loft_prob'] == loft_prob].pivot(
        index='equity_factor', columns='budget_m', values='co2_improvement_kt'
    )
    # Rename columns to integer format
    pivot.columns = [int(c) for c in pivot.columns]

    # Convert min/max to kilotonnes
    kt_vmin = abs_vmin / 1000
    kt_vmax = abs_vmax / 1000

    sns.heatmap(pivot, annot=True, fmt=',.1f', cmap='RdYlGn',
                vmin=kt_vmin, vmax=kt_vmax, center=(kt_vmin + kt_vmax) / 2,
                ax=ax, cbar_kws={'label': CO2_LABEL_KT},
                linewidths=0.5, linecolor='gray')

    ax.set_xlabel(BUDGET_LABEL, fontsize=13)
    ax.set_ylabel(EQUITY_LABEL, fontsize=13)

    plt.tight_layout()
    loft_str = str(loft_prob).replace('.', '')
    filepath = os.path.join(output_dir, f'heatmap_abs_loft{loft_str}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filepath}")


def create_boxplot(df, output_dir):
    """Create box plot of improvements by budget."""
    fig, ax = plt.subplots(figsize=(10, 6))

    df_box = df.copy()
    df_box['budget_m_int'] = df_box['budget_m'].astype(int)
    budget_order = sorted(df_box['budget_m_int'].unique())

    sns.boxplot(data=df_box, x='budget_m_int', y='improvement_pct',
                order=budget_order, ax=ax, palette='Set2', hue='budget_m_int',
                )

    ax.set_xlabel(BUDGET_LABEL, fontsize=14, fontweight='bold')
    ax.set_ylabel(CO2_LABEL_PCT, fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'boxplot_improvements.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filepath}")


def create_loft_comparison(df, output_dir):
    """Create loft probability comparison plot with unified symbols in legend."""
    fig, ax = plt.subplots(figsize=(12, 8))

    budgets = sorted(df['budget_m'].unique())
    colors = plt.cm.viridis(np.linspace(0.2, 0.95, len(budgets)))

    # Store handles for custom legend
    budget_handles = []
    budget_labels = []

    for idx, budget in enumerate(budgets):
        budget_int = int(budget)

        # Loft 0.95
        loft_95 = df[(df['budget_m'] == budget) & (df['loft_prob'] == 0.95)].sort_values('equity_factor')
        marker_95, ls_95 = get_loft_style(0.95)
        ax.plot(loft_95['equity_factor'], loft_95['improvement_pct'],
                marker=marker_95, linewidth=2.5, markersize=8,
                color=colors[idx], linestyle=ls_95, alpha=0.9)

        # Loft 0.65
        loft_65 = df[(df['budget_m'] == budget) & (df['loft_prob'] == 0.65)].sort_values('equity_factor')
        marker_65, ls_65 = get_loft_style(0.65)
        ax.plot(loft_65['equity_factor'], loft_65['improvement_pct'],
                marker=marker_65, linewidth=2.5, markersize=8,
                color=colors[idx], linestyle=ls_65, alpha=0.9)

        # Create dummy line without marker for legend (color only)
        budget_line, = ax.plot([], [], linestyle='-', linewidth=2.5,
                               color=colors[idx], marker='')
        budget_handles.append(budget_line)
        budget_labels.append(f'£{budget_int}M')

    # Create symbol legend entries (using dummy plots)
    symbol_handle_95, = ax.plot([], [], marker=MARKER_LOFT_095, linestyle=LINESTYLE_LOFT_095,
                                 color='gray', linewidth=2.5, markersize=8, label='')
    symbol_handle_65, = ax.plot([], [], marker=MARKER_LOFT_065, linestyle=LINESTYLE_LOFT_065,
                                 color='gray', linewidth=2.5, markersize=8, label='')

    # Build combined legend: budgets first, then symbols
    all_handles = budget_handles + [symbol_handle_95, symbol_handle_65]
    all_labels = budget_labels + ['Loft 0.95', 'Loft 0.65']

    ax.set_xlabel(EQUITY_LABEL, fontsize=14, fontweight='bold')
    ax.set_ylabel(CO2_LABEL_PCT, fontsize=14, fontweight='bold')
    ax.legend(all_handles, all_labels, fontsize=10, ncol=2, loc='center left', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=2)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'loft_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filepath}")


def create_pareto_frontier_robust(df, output_dir, annotated=True, error_type='partial'):
    """
    Create Pareto frontier plot using Robust Mean difference with selectable error bars.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Results dataframe
    output_dir : str
        Output directory for plots
    annotated : bool, default=True
        Whether to add text annotations
    error_type : str, default='partial'
        Which error to use: 'uncorr', 'partial', or 'corr'
    """
    # Configuration for axes labels
    X_LABEL = 'Budget (£ Millions)'
    Y_LABEL = 'Carbon Saved vs Random (Tonnes CO2e)'
    
    # Map error type to column name
    error_col_map = {
        'uncorr': 'diff_std_uncorr',
        'partial': 'diff_std_partial',
        'corr': 'diff_std_corr'
    }
    
    error_col = error_col_map.get(error_type, 'diff_std_partial')
    
    fig, ax = plt.subplots(figsize=(12, 8))

    # 1. Plot all configurations as context (background, gray)
    for loft in sorted(df['loft_prob'].unique()):
        for eq in sorted(df['equity_factor'].unique()):
            subset = df[(df['loft_prob'] == loft) & (df['equity_factor'] == eq)].sort_values('budget_m')
            
            # Simple line plot for context (no error bars to avoid clutter)
            ax.plot(subset['budget_m'], subset['diff_mean_co2'],
                    alpha=0.15, linewidth=1, color='gray', zorder=1)

    # 2. Identify and Plot Pareto Frontier (Best Mean Outcome per Budget)
    pareto_points = df.loc[df.groupby('budget_m')['diff_mean_co2'].idxmax()].sort_values('budget_m')
    
    # Generate colors for the frontier points
    colors = plt.cm.viridis(np.linspace(0, 1, len(pareto_points)))

    for idx, (i, row) in enumerate(pareto_points.iterrows()):
        x = row['budget_m']
        y = row['diff_mean_co2']
        error = row[error_col]
        
        # A. Plot the Error Bar
        ax.errorbar(x, y, yerr=error, 
                    fmt='none', ecolor='black', elinewidth=2, capsize=5, zorder=50)

        # B. Plot the Point
        ax.scatter(x, y,
                   s=300, c=[colors[idx]], alpha=0.9,
                   edgecolors='black', linewidth=2, zorder=100)

        # C. Annotation (Optional)
        if annotated:
            z_score_col = f'z_score_{error_type}'
            z_score = row[z_score_col] if z_score_col in row else (y / error if error > 0 else 0)
            
            label_text = (f"Equity: {row['equity_factor']}\n"
                          f"Loft: {row['loft_prob']}\n"
                          f"σ: ±{error:.1f}\n"
                          f"Z: {z_score:.2f}")

            ax.annotate(label_text,
                        xy=(x, y),
                        xytext=(0, 25), textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  alpha=0.85, edgecolor='gray', linewidth=1))

    # 3. Final Styling
    ax.set_xlabel(X_LABEL, fontsize=12, fontweight='bold')
    ax.set_ylabel(Y_LABEL, fontsize=12, fontweight='bold')
    
    # Add a horizontal line at 0
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Random Baseline')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=10)
    
    # Title with error type information
    error_type_label = {
        'uncorr': 'Uncorrelated (Optimistic)',
        'partial': f'Partially Correlated (ρ={df["rho"].iloc[0]:.2f})',
        'corr': 'Fully Correlated (Pessimistic)'
    }
    
    plt.title(f'Pareto Frontier: Optimization Gain vs. Random Selection\n'
              f'Error bars: {error_type_label.get(error_type, error_type)}',
              fontsize=14)

    plt.tight_layout()
    
    # Save
    suffix = 'annotated' if annotated else 'clean'
    filepath = os.path.join(output_dir, f'pareto_frontier_robust_{error_type}_{suffix}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filepath}")


def create_cost_benefit_analysis(df, output_dir):
    """Create cost-benefit analysis scatter plot with CO2 in kilotonnes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert to kilotonnes
    df_kt = df.copy()
    df_kt['co2_improvement_kt'] = df_kt['co2_improvement'] / 1000

    colors_rg = ['red', 'yellow', 'green']
    cmap_rg = LinearSegmentedColormap.from_list('red_green', colors_rg, N=100)

    # Create offsets for each loft_prob to improve readability
    unique_lofts = sorted(df['loft_prob'].unique())
    n_lofts = len(unique_lofts)
    offset_range = (df['budget_m'].max() - df['budget_m'].min()) * 0.02  # 2% of x-range
    offsets = {loft: (i - (n_lofts - 1) / 2) * offset_range for i, loft in enumerate(unique_lofts)}

    for loft in unique_lofts:
        subset = df_kt[df_kt['loft_prob'] == loft].copy()
        marker, _ = get_loft_style(loft)
        label = f'Loft Prob: {loft}'

        equity_norm = (subset['equity_factor'] - df['equity_factor'].min()) / \
                      (df['equity_factor'].max() - df['equity_factor'].min())

        # Apply horizontal offset
        x_values = subset['budget_m'] + offsets[loft]

        scatter = ax.scatter(x_values, subset['co2_improvement_kt'],
                             c=equity_norm, s=200, alpha=0.6,
                             cmap=cmap_rg, marker=marker, edgecolors='black',
                             linewidth=1,
                             label=label, vmin=0, vmax=1)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(f'{EQUITY_LABEL}', fontsize=12, fontweight='bold')
    equity_values = sorted(df['equity_factor'].unique())
    equity_norm_ticks = (np.array(equity_values) - df['equity_factor'].min()) / \
                        (df['equity_factor'].max() - df['equity_factor'].min())
    cbar.set_ticks(equity_norm_ticks)
    cbar.set_ticklabels([f'{v:.1f}' for v in equity_values])
    cbar.ax.tick_params(labelsize=11)

    ax.set_xlabel(BUDGET_LABEL, fontsize=14, fontweight='bold')
    ax.set_ylabel(f'Total {CO2_LABEL_KT}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    # Format x-axis as integers
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'cost_benefit_analysis.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filepath}")


def create_cost_benefit_analysis_pct(df, output_dir):
    """Create cost-benefit analysis scatter plot with improvement percentage."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors_rg = ['red', 'yellow', 'green']
    cmap_rg = LinearSegmentedColormap.from_list('red_green', colors_rg, N=100)

    # Create offsets for each loft_prob to improve readability
    unique_lofts = sorted(df['loft_prob'].unique())
    n_lofts = len(unique_lofts)
    offset_range = (df['budget_m'].max() - df['budget_m'].min()) * 0.02  # 2% of x-range
    offsets = {loft: (i - (n_lofts - 1) / 2) * offset_range for i, loft in enumerate(unique_lofts)}

    for loft in unique_lofts:
        subset = df[df['loft_prob'] == loft].copy()
        marker, _ = get_loft_style(loft)
        label = f'Loft Prob: {loft}'

        equity_norm = (subset['equity_factor'] - df['equity_factor'].min()) / \
                      (df['equity_factor'].max() - df['equity_factor'].min())

        # Apply horizontal offset
        x_values = subset['budget_m'] + offsets[loft]

        scatter = ax.scatter(x_values, subset['improvement_pct'],
                             c=equity_norm, s=200, alpha=0.6,
                             cmap=cmap_rg, marker=marker, edgecolors='black',
                             linewidth=1,
                             label=label, vmin=0, vmax=1)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(f'{EQUITY_LABEL}', fontsize=12, fontweight='bold')
    equity_values = sorted(df['equity_factor'].unique())
    equity_norm_ticks = (np.array(equity_values) - df['equity_factor'].min()) / \
                        (df['equity_factor'].max() - df['equity_factor'].min())
    cbar.set_ticks(equity_norm_ticks)
    cbar.set_ticklabels([f'{v:.1f}' for v in equity_values])
    cbar.ax.tick_params(labelsize=11)
    ax.axhline(y=0, color='white', linestyle='-', alpha=0, linewidth=0.1)
    ax.set_xlabel(BUDGET_LABEL, fontsize=14, fontweight='bold')
    ax.set_ylabel('CO₂ Improvement (%)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    # Format x-axis as integers
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'cost_benefit_analysis_pct.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {filepath}")


def main():
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    SETTING_NAME = 'lcoal'
    RHO = 0.45  # Correlation parameter for partial correlation
    
    BASE_DIR = f'/Volumes/T9/2025_10_RetrofitModel/11_finaL_sub/5_greedy_results_epc/NE/all_domestic'
    OUTPUT_DIR = f'/Volumes/T9/2025_10_RetrofitModel/11_finaL_sub/5_greedy_results_epc/meta_sumary_v2'

    greedy_runs_folder = os.path.join(BASE_DIR, 'greedy_runs', SETTING_NAME)
    budgets = [1_000_000, 25_000_000, 50_000_000, 100_000_000, 200_000_000, 500_000_000]
    loft_probs = [0.65, 0.95]
    equity_factors = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # AGGREGATE DATA WITH RHO
    # =========================================================================
    df = aggregate_scenario_results(greedy_runs_folder, budgets, loft_probs, 
                                      equity_factors, rho=RHO)

    # Create improvement metrics (positive = optimization is better)
    df['co2_improvement'] = df['diff_mean_co2']
    df['improvement_pct'] = df['diff_pct']

    # Calculate global min/max for consistent color scales
    pct_vmin = df['improvement_pct'].min()
    pct_vmax = df['improvement_pct'].max()
    abs_vmin = df['co2_improvement'].min()
    abs_vmax = df['co2_improvement'].max()

    # =========================================================================
    # GENERATE ALL FIGURES
    # =========================================================================
    print("\n" + "=" * 80)
    print(f"GENERATING VISUALIZATIONS (ρ = {RHO})")
    print("=" * 80 + "\n")

    # Figure 0: All configurations colored
    create_all_configs_plot(df, OUTPUT_DIR)

    # Figures 1-2: Percentage heatmaps
    for loft in loft_probs:
        create_heatmap_pct(df, loft, pct_vmin, pct_vmax, OUTPUT_DIR)

    # Figures 3-4: Absolute CO2 heatmaps (in kilotonnes)
    for loft in loft_probs:
        create_heatmap_abs(df, loft, abs_vmin, abs_vmax, OUTPUT_DIR)

    # Figure 5: Box plot
    create_boxplot(df, OUTPUT_DIR)

    # Figure 6: Loft comparison
    create_loft_comparison(df, OUTPUT_DIR)

    # Figures 7-12: Pareto frontiers with all three error types
    for error_type in ['uncorr', 'partial', 'corr']:
        create_pareto_frontier_robust(df, OUTPUT_DIR, annotated=True, error_type=error_type)
        create_pareto_frontier_robust(df, OUTPUT_DIR, annotated=False, error_type=error_type)

    # Figures 13-14: Cost-benefit analysis
    create_cost_benefit_analysis(df, OUTPUT_DIR)
    create_cost_benefit_analysis_pct(df, OUTPUT_DIR)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("=" * 80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Correlation parameter (ρ): {RHO}")
    print("\nFiles saved:")
    print("  0. pareto_all_configs_colored.png")
    print("  1. heatmap_pct_loft095.png")
    print("  3. heatmap_abs_loft095.png")
    print("  5. boxplot_improvements.png")
    print("  6. loft_comparison.png")
    print("  7-8. pareto_frontier_robust_uncorr_[annotated/clean].png")
    print("  9-10. pareto_frontier_robust_partial_[annotated/clean].png")
    print("  11-12. pareto_frontier_robust_corr_[annotated/clean].png")
    print("  13. cost_benefit_analysis.png")
    print("  14. cost_benefit_analysis_pct.png")
    print(f"\nData saved to: diff_epc_opt_robust_significance_rho_{RHO}.csv")


if __name__ == '__main__':
    main()