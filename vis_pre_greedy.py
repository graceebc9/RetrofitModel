import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from src.RetrofitUtils import filter_typology

# ==========================================
# CONFIGURATION
# ==========================================
SAMPLE_NAMES = [8,9, 110, 111, 112, 113, 114]  # 5 sample log files
BASE_PATH = '/home/gb669/rds/hpc-work/energy_map/RetrofitModel/0_intermediate_data_2D/retrofit_scenario/v9/NE'
SIGMA = 1
SCENARIO = 'wall_installation'

# ==========================================
# PROCESSING - LOOP OVER ALL SAMPLES
# ==========================================
all_plot_data = []

for name in SAMPLE_NAMES:
    RAW_FILE_PATH = f'{BASE_PATH}/{name}_log_file.csv'
    print(f"Loading {RAW_FILE_PATH}...")
    
    try:
        df = pd.read_csv(RAW_FILE_PATH, on_bad_lines='skip', low_memory=False)
    except Exception as e:
        print(f"  Error loading {name}: {e}")
        continue
    
    df = filter_typology(df)
    
    col_cost_ton = f'{SCENARIO}_capex_per_net_ton_co2_{SCENARIO}_p50'
    col_abs_co2 = f'{SCENARIO}_total_energy_abs_co2_ton_samples_{SCENARIO}_p50'
    
    grouped = df.groupby('upn')[[col_cost_ton, col_abs_co2]].agg(['mean', 'std'])
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    
    data = pd.DataFrame()
    data['mean_cost_ton'] = grouped[f'{col_cost_ton}_mean']
    data['std_cost_ton'] = grouped[f'{col_cost_ton}_std']
    data['mean_abs_co2'] = grouped[f'{col_abs_co2}_mean']
    data['robust_cost_ton'] = data['mean_cost_ton'] + (SIGMA * data['std_cost_ton'])
    
    plot_data = data[
        (data['mean_abs_co2'] > 0) & 
        (data['mean_abs_co2'] < 5.0) &
        (data['mean_cost_ton'] < 20000)
    ].copy()
    
    all_plot_data.append(plot_data)
    print(f"  Loaded {len(plot_data)} rows from sample {name}")

# Combine all samples
combined_data = pd.concat(all_plot_data, ignore_index=True)
print(f"\nTotal combined rows: {len(combined_data)}")

# ==========================================
# PLOTTING
# ==========================================
plt.style.use('seaborn-v0_8-paper')
fig, ax = plt.subplots(figsize=(9, 6))

# Comma formatter for axis ticks
comma_fmt = FuncFormatter(lambda x, p: format(int(x), ','))

# 1. Plot the "Shift" (Arrows from Mean -> Robust)
ax.quiver(
    combined_data['mean_abs_co2'],
    combined_data['mean_cost_ton'],
    np.zeros(len(combined_data)),
    combined_data['robust_cost_ton'] - combined_data['mean_cost_ton'],
    color='gray',
    alpha=0.3,
    angles='xy', scale_units='xy', scale=1,
    width=0.002,
    headwidth=3
)

# 2. Plot the "Robust" Endpoints - colored by std
scatter = ax.scatter(
    combined_data['mean_abs_co2'], 
    combined_data['robust_cost_ton'], 
    c=combined_data['std_cost_ton'], 
    cmap='magma_r',
    s=25, 
    label='Robust Value (~P85)',
    zorder=3
)

# 3. Highlight the "Trap"
ax.axvspan(0, 1.0, color='red', alpha=0.1, label='Small Impact Zone')

# Grid background - horizontal lines only
ax.yaxis.grid(True, linestyle='-', alpha=0.7, zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)
# Match spines to grid style
for spine in ax.spines.values():
    spine.set_linestyle('-')
    spine.set_alpha(0.7)
    spine.set_color('gray')
 

# Title and labels with bold axis labels
ax.set_xlabel('Absolute Carbon Savings (Tons CO2/5yr)', fontsize=12, fontweight='bold')
ax.set_ylabel('Capex Efficiency (£/Ton)', fontsize=12, fontweight='bold')

# Apply comma formatting to axes
ax.xaxis.set_major_formatter(comma_fmt)
ax.yaxis.set_major_formatter(comma_fmt)

# Colorbar with comma formatting
cbar = plt.colorbar(scatter)
cbar.set_label('STD across runs', rotation=270, labelpad=15 )
cbar.ax.yaxis.set_major_formatter(comma_fmt)

ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('combined_capex_scatter.png', dpi=150)
plt.show()

print("\nPlot saved as 'combined_capex_scatter.png'")