#!/bin/bash
#SBATCH -A CULLEN-SL3-CPU
#SBATCH -p icelake
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=NONE
#SBATCH --mem=250G
#SBATCH --array=0  # 4 budgets × 2 lofts × 1 equity = 10 combinations
#SBATCH --output=logs_greedy_epc/rmodel_%A_%a.out
#SBATCH --error=logs_greedy_epc/rmodel_%A_%a.err
 
# Load required modules
. /etc/profile.d/modules.sh
module purge
module load rhel7/default-ccl

# Initialize conda
CONDA_BASE=/usr/local/software/archive/linux-scientific7-x86_64/gcc-9/miniconda3-4.7.12.1-rmuek6r3f6p3v6fdj7o2klyzta3qhslh
source $CONDA_BASE/etc/profile.d/conda.sh

# Activate the nebula environment
conda activate /home/gb669/.conda/envs/nebula

# Set environment variables
export SLURM_SUBMIT_DIR='/home/gb669/rds/hpc-work/energy_map/RetrofitModel'
cd $SLURM_SUBMIT_DIR

export BASE_DIR='/rds/user/gb669/hpc-work/energy_map/RetrofitModel/retrofit_scenario_analysis/2_greedy/combo_epc'
export INPUT_FILES_PATH='/home/gb669/rds/hpc-work/energy_map/RetrofitModel/intermediate_data_2D/v5_logs_with_epc/*.csv' 
export run_g_yn='Y'

# Create logs directory
mkdir -p logs_greedy_epc

# Define parameter arrays
BUDGET_SETTINGS=(1 2 3 4 5)
LOFT_SETTINGS=(1 2)
EQUITY_FACTORS=(0.8 1)

# Calculate which combination to use based on SLURM_ARRAY_TASK_ID
# Formula: task_id = budget_idx * (n_loft * n_equity) + loft_idx * n_equity + equity_idx

n_loft=2
n_equity=2

budget_idx=$((SLURM_ARRAY_TASK_ID / (n_loft * n_equity)))
remainder=$((SLURM_ARRAY_TASK_ID % (n_loft * n_equity)))
loft_idx=$((remainder / n_equity))
equity_idx=$((remainder % n_equity))

# Set the actual parameter values
export BUDGET_SETTING=${BUDGET_SETTINGS[$budget_idx]}
export loft_setting=${LOFT_SETTINGS[$loft_idx]}
export equity_factor=${EQUITY_FACTORS[$equity_idx]}

# Log job info
echo "Job started at: $(date)"
echo "Running on node: $HOSTNAME"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "BUDGET_SETTING: $BUDGET_SETTING"
echo "loft_setting: $loft_setting"
echo "equity_factor: $equity_factor"

# Run the processing script
python run_greedy_pc.py