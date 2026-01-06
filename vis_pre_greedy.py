import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.RetrofitUtils import filter_typology
# ==========================================
# CONFIGURATION
# ==========================================
name ='110'
RAW_FILE_PATH = f'/home/gb669/rds/hpc-work/energy_map/RetrofitModel/0_intermediate_data_2D/retrofit_scenario/v9/NE/{name}_log_file.csv' 
SIGMA = 1
SCENARIO = 'wall_installation' 

# ==========================================
# PROCESSING
# ==========================================
print(f"Loading {RAW_FILE_PATH}...")
try:
    df = pd.read_csv(RAW_FILE_PATH, on_bad_lines='skip', low_memory=False)
except:
    pass

df = filter_typology(df)

col_cost_ton = f'{SCENARIO}_capex_per_net_ton_co2_{SCENARIO}_p50'
col_abs_co2  = f'{SCENARIO}_total_energy_abs_co2_ton_samples_{SCENARIO}_p50'

grouped = df.groupby('upn')[[col_cost_ton, col_abs_co2]].agg(['mean', 'std'])
grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]

data = pd.DataFrame()
data['mean_cost_ton'] = grouped[f'{col_cost_ton}_mean']
data['std_cost_ton']  = grouped[f'{col_cost_ton}_std']
data['mean_abs_co2']  = grouped[f'{col_abs_co2}_mean']
data['robust_cost_ton'] = data['mean_cost_ton'] + (SIGMA * data['std_cost_ton'])

plot_data = data[
    (data['mean_abs_co2'] > 0) & 
    (data['mean_abs_co2'] < 5.0) &
    (data['mean_cost_ton'] < 20000)
].copy()

# ==========================================
# PLOTTING
# ==========================================
plt.style.use('seaborn-v0_8-paper')
fig, ax = plt.subplots(figsize=(9, 6))

# Comma formatter for axis ticks
comma_fmt = FuncFormatter(lambda x, p: format(int(x), ','))

# 1. Plot the "Shift" (Arrows from Mean -> Robust)
ax.quiver(
    plot_data['mean_abs_co2'],
    plot_data['mean_cost_ton'],
    np.zeros(len(plot_data)),
    plot_data['robust_cost_ton'] - plot_data['mean_cost_ton'],
    color='gray',
    alpha=0.3,
    angles='xy', scale_units='xy', scale=1,
    width=0.002,
    headwidth=3
)

# 2. Plot the "Robust" Endpoints
scatter = ax.scatter(
    plot_data['mean_abs_co2'], 
    plot_data['robust_cost_ton'], 
    c=plot_data['std_cost_ton'], 
    cmap='magma_r',
    s=25, 
    label='Robust Value (~P85)',
    zorder=3
)

# 3. Highlight the "Trap"
ax.axvspan(0, 1.0, color='red', alpha=0.1, label='Small Impact Zone')

# Grid background
ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
ax.set_axisbelow(True)

# Title and labels with bold axis labels

ax.set_xlabel('Absolute Carbon Savings (Tons CO2/5yr)', fontsize=12, fontweight='bold')
ax.set_ylabel('Capex Efficiency (£/Ton)', fontsize=12, fontweight='bold')

# Apply comma formatting to axes
ax.xaxis.set_major_formatter(comma_fmt)
ax.yaxis.set_major_formatter(comma_fmt)

# Colorbar with comma formatting
cbar = plt.colorbar(scatter)
cbar.set_label('STD across runs', rotation=270, labelpad=15, fontweight='bold')
cbar.ax.yaxis.set_major_formatter(comma_fmt)

ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f'{name}_capex_scatter.png', dpi=150)
plt.show()