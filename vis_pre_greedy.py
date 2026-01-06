import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================================
# VISUAL THRESHOLD DETERMINATION TOOL
# ============================================================================

def create_cv_distribution_plots(spread_metrics_dict, output_dir='threshold_plots'):
    """
    Create comprehensive visualizations for each metric to help
    visually determine appropriate CV thresholds
    
    Args:
        spread_metrics_dict: {metric_name: spread_df}
        output_dir: Directory to save plots
    """
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    n_metrics = len(spread_metrics_dict)
    
    # ========================================================================
    # PLOT 1: CV Distribution Histograms (Individual)
    # ========================================================================
    for metric_name, spread_df in spread_metrics_dict.items():
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Stability Analysis: {metric_name}', fontsize=16, fontweight='bold')
        
        cv_values = spread_df['cv'].values
        
        # --- Subplot 1: CV Histogram with percentiles ---
        ax = axes[0, 0]
        ax.hist(cv_values, bins=100, edgecolor='black', alpha=0.7)
        
        # Add percentile lines
        percentiles = [50, 75, 85, 90, 95, 99]
        colors = ['green', 'blue', 'orange', 'red', 'purple', 'brown']
        for p, color in zip(percentiles, colors):
            val = np.percentile(cv_values, p)
            ax.axvline(val, color=color, linestyle='--', linewidth=2, 
                      label=f'p{p}: {val:.3f}')
        
        ax.set_xlabel('Coefficient of Variation (CV)', fontsize=12)
        ax.set_ylabel('Number of Buildings', fontsize=12)
        ax.set_title('CV Distribution with Percentiles', fontsize=13)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # --- Subplot 2: CV Cumulative Distribution ---
        ax = axes[0, 1]
        sorted_cv = np.sort(cv_values)
        cumulative = np.arange(1, len(sorted_cv) + 1) / len(sorted_cv) * 100
        ax.plot(sorted_cv, cumulative, linewidth=2)
        
        # Add threshold reference lines
        threshold_options = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
        for thresh in threshold_options:
            if thresh <= sorted_cv.max():
                pct = (sorted_cv <= thresh).sum() / len(sorted_cv) * 100
                ax.axvline(thresh, color='red', linestyle=':', alpha=0.5)
                ax.text(thresh, pct, f'{pct:.1f}%', fontsize=9, 
                       rotation=90, va='bottom')
        
        ax.set_xlabel('CV Threshold', fontsize=12)
        ax.set_ylabel('% Buildings Retained', fontsize=12)
        ax.set_title('Retention Rate vs CV Threshold', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, min(3.0, sorted_cv.max()))
        
        # --- Subplot 3: CV vs Mean Cost (scatter) ---
        ax = axes[1, 0]
        scatter = ax.scatter(spread_df['mean'], spread_df['cv'], 
                           alpha=0.3, s=10)
        ax.set_xlabel('Mean Cost (£)', fontsize=12)
        ax.set_ylabel('CV', fontsize=12)
        ax.set_title('CV vs Mean Cost', fontsize=13)
        ax.grid(True, alpha=0.3)
        
        # Add reference line
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='CV=0.5')
        ax.legend()
        
        # --- Subplot 4: Summary Statistics Table ---
        ax = axes[1, 1]
        ax.axis('off')
        
        stats_data = [
            ['Total Buildings', f'{len(spread_df):,}'],
            ['', ''],
            ['CV Statistics:', ''],
            ['  Mean', f'{cv_values.mean():.3f}'],
            ['  Median', f'{np.median(cv_values):.3f}'],
            ['  Std Dev', f'{cv_values.std():.3f}'],
            ['  Min', f'{cv_values.min():.3f}'],
            ['  Max', f'{cv_values.max():.3f}'],
            ['', ''],
            ['Percentiles:', ''],
            ['  p50', f'{np.percentile(cv_values, 50):.3f}'],
            ['  p75', f'{np.percentile(cv_values, 75):.3f}'],
            ['  p85', f'{np.percentile(cv_values, 85):.3f}'],
            ['  p90', f'{np.percentile(cv_values, 90):.3f}'],
            ['  p95', f'{np.percentile(cv_values, 95):.3f}'],
            ['  p99', f'{np.percentile(cv_values, 99):.3f}'],
        ]
        
        table = ax.table(cellText=stats_data, cellLoc='left',
                        loc='center', colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{metric_name}_analysis.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir}/{metric_name}_analysis.png")
        plt.close()
    
    # ========================================================================
    # PLOT 2: Comparative Overview (All Metrics)
    # ========================================================================
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('CV Distribution Comparison Across All Metrics', 
                 fontsize=16, fontweight='bold')
    
    # --- Subplot 1: Box plots ---
    ax = axes[0]
    data_for_box = [spread_df['cv'].values for spread_df in spread_metrics_dict.values()]
    labels = list(spread_metrics_dict.keys())
    
    bp = ax.boxplot(data_for_box, labels=labels, patch_artist=True,
                    showfliers=False)  # Hide outliers for clarity
    
    # Color boxes
    colors = plt.cm.Set3(range(len(labels)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Coefficient of Variation', fontsize=12)
    ax.set_title('CV Distribution by Metric (Box Plot)', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)
    
    # Add reference lines
    ax.axhline(0.3, color='green', linestyle='--', alpha=0.5, label='CV=0.3')
    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='CV=0.5')
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='CV=1.0')
    ax.legend(loc='upper right')
    
    # --- Subplot 2: Retention curves ---
    ax = axes[1]
    
    for metric_name, spread_df in spread_metrics_dict.items():
        cv_values = spread_df['cv'].values
        sorted_cv = np.sort(cv_values)
        cumulative = np.arange(1, len(sorted_cv) + 1) / len(sorted_cv) * 100
        ax.plot(sorted_cv, cumulative, linewidth=2, label=metric_name)
    
    ax.set_xlabel('CV Threshold', fontsize=12)
    ax.set_ylabel('% Buildings Retained', fontsize=12)
    ax.set_title('Retention Rate vs CV Threshold (All Metrics)', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    ax.set_xlim(0, 2.0)  # Focus on reasonable range
    
    # Add vertical reference lines
    for thresh in [0.3, 0.5, 1.0]:
        ax.axvline(thresh, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparative_overview.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/comparative_overview.png")
    plt.close()
    
    # ========================================================================
    # PLOT 3: Threshold Decision Matrix
    # ========================================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('Threshold Decision Matrix', fontsize=16, fontweight='bold')
    
    threshold_options = [0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0]
    matrix_data = []
    
    for metric_name, spread_df in spread_metrics_dict.items():
        cv_values = spread_df['cv'].values
        row = [metric_name]
        for thresh in threshold_options:
            pct_retained = (cv_values <= thresh).sum() / len(cv_values) * 100
            row.append(f'{pct_retained:.1f}%')
        matrix_data.append(row)
    
    # Create table
    ax.axis('tight')
    ax.axis('off')
    
    col_labels = ['Metric'] + [f'CV≤{t}' for t in threshold_options]
    table = ax.table(cellText=matrix_data, colLabels=col_labels,
                    cellLoc='center', loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Color code cells
    for i in range(1, len(matrix_data) + 1):
        for j in range(1, len(threshold_options) + 1):
            cell = table[(i, j)]
            pct = float(cell.get_text().get_text().rstrip('%'))
            
            # Color gradient: green (high %) to red (low %)
            if pct >= 90:
                cell.set_facecolor('#90EE90')  # Light green
            elif pct >= 75:
                cell.set_facecolor('#FFFFE0')  # Light yellow
            elif pct >= 50:
                cell.set_facecolor('#FFD700')  # Gold
            else:
                cell.set_facecolor('#FFA07A')  # Light red
    
    plt.savefig(f'{output_dir}/threshold_decision_matrix.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/threshold_decision_matrix.png")
    plt.close()
    
    # ========================================================================
    # Print Summary to Console
    # ========================================================================
    print(f"\n{'='*70}")
    print("VISUAL THRESHOLD DETERMINATION - SUMMARY")
    print(f"{'='*70}")
    
    for metric_name, spread_df in spread_metrics_dict.items():
        cv_values = spread_df['cv'].values
        print(f"\n{metric_name}:")
        print(f"  Median CV: {np.median(cv_values):.3f}")
        print(f"  p75 CV: {np.percentile(cv_values, 75):.3f}")
        print(f"  p90 CV: {np.percentile(cv_values, 90):.3f}")
        print(f"  p95 CV: {np.percentile(cv_values, 95):.3f}")
        print(f"  Suggested thresholds to review:")
        for thresh in [0.3, 0.5, 1.0]:
            pct = (cv_values <= thresh).sum() / len(cv_values) * 100
            print(f"    CV ≤ {thresh:.1f} → {pct:.1f}% retention")


# ============================================================================
# Load and visualize
# ============================================================================

def load_spread_metrics(metric_names):
    """Load spread metric CSVs into dictionary"""
    spread_dict = {}
    for metric_name in metric_names:
        filepath = f'pass1_spread_metrics_{metric_name}.csv'
        if Path(filepath).exists():
            spread_dict[metric_name] = pd.read_csv(filepath)
            print(f"Loaded: {filepath}")
        else:
            print(f"Warning: {filepath} not found")
    return spread_dict


if __name__ == "__main__":
    
    # Define your metrics
    METRIC_NAMES = ['hp_only' , 
            'heat_wall', 
            'wall', 
            'heat_ins',
            'loft',
            'join_heat_ins_decay'
        ]
    
    # Load spread metrics
    spread_metrics = load_spread_metrics(METRIC_NAMES)
    
    # Create visualizations
    create_cv_distribution_plots(spread_metrics, output_dir='threshold_plots')
    
    print(f"\n{'='*70}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*70}")
    print("Check the 'threshold_plots/' directory for:")
    print("  1. Individual metric analysis plots (*_analysis.png)")
    print("  2. Comparative overview (comparative_overview.png)")
    print("  3. Threshold decision matrix (threshold_decision_matrix.png)")
    print("\nUse these to visually determine appropriate CV thresholds per metric.")