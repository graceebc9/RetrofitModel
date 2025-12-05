#!/bin/bash
#SBATCH -A CULLEN-SL3-CPU
#SBATCH -p icelake
#SBATCH --time=02:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=NONE
#SBATCH --mem=150G
#SBATCH --output=logs_greedy/rmodel_%A_%a.out
#SBATCH --error=logs_greedy/rmodel_%A_%a.err
 
# Load required modules
. /etc/profile.d/modules.sh
module purge
module load rhel7/default-ccl

# Initialize conda
CONDA_BASE=/usr/local/software/archive/linux-scientific7-x86_64/gcc-9/miniconda3-4.7.12.1-rmuek6r3f6p3v6fdj7o2klyzta3qhslh
source $CONDA_BASE/etc/profile.d/conda.sh

# Activate the nebula environment
conda activate /home/gb669/.conda/envs/ag3

# Set environment variables
export SLURM_SUBMIT_DIR='/home/gb669/rds/hpc-work/energy_map/RetrofitModel'
cd $SLURM_SUBMIT_DIR
export EPC_YN='Y'


export RUN_GREEDY_RUNS_YN='Y'

# Create logs directory
mkdir -p logs_greedy

 
 
# Log job info
echo "Job started at: $(date)"
echo "Running on node: $HOSTNAME"

# Run the processing script
python run_greedy.py