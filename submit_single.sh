#!/bin/bash
#SBATCH -A CULLEN-SL2-GPU
#SBATCH -p ampere
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=NONE
#SBATCH --mem=12G
#SBATCH --output=logs_slurm/rmodel_%A_%a.out
#SBATCH --error=logs_slurm/rmodel_%A_%a.err
#SBATCH --gres=gpu:1

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

# Create logs directory
mkdir -p logs

# Get the batch path for this array job from batch_paths.txt
BATCH_PATHS_FILE="${SLURM_SUBMIT_DIR}/batch_paths.txt"
batch_path=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$BATCH_PATHS_FILE")
batch_path="/home/gb669/rds/hpc-work/energy_map/RetrofitModel/batches/NE/batch_9.txt"
# Validate that we got a path
if [ -z "$batch_path" ]; then
    echo "Error: Could not read batch path for array task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Validate that the batch file exists
if [ ! -f "${batch_path}" ]; then
    echo "Error: Batch file not found: ${batch_path}"
    exit 1
fi
 
  
# Log job info
echo "Job started at: $(date)"
echo "Running on node: $HOSTNAME"
echo "Processing batch path: $batch_path"

# Run the processing script
python main2D.py "$batch_path"