#!/bin/bash

# Path to the directory containing batch_paths.txt
SLURM_SUBMIT_DIR='/home/gb669/rds/hpc-work/energy_map/RetrofitModel'
BATCH_PATHS_FILE="${SLURM_SUBMIT_DIR}/batch_paths.txt"
SLURM_SCRIPT_PATH='/home/gb669/rds/hpc-work/energy_map/RetrofitModel/submit_single.sh'

# Check if directory exists
if [ ! -d "$SLURM_SUBMIT_DIR" ]; then
    echo "Error: Directory not found: $SLURM_SUBMIT_DIR"
    exit 1
fi

# Check if batch_paths.txt exists and is readable
if [ ! -f "$BATCH_PATHS_FILE" ]; then
    echo "Error: batch_paths.txt not found in $SLURM_SUBMIT_DIR"
    exit 1
fi

if [ ! -r "$BATCH_PATHS_FILE" ]; then
    echo "Error: Cannot read batch_paths.txt"
    exit 1
fi

# Check if file is empty
if [ ! -s "$BATCH_PATHS_FILE" ]; then
    echo "Error: batch_paths.txt is empty"
    exit 1
fi

# Check if SLURM script exists and is executable
if [ ! -x "$SLURM_SCRIPT_PATH" ]; then
    echo "Error: SLURM script not found or not executable: $SLURM_SCRIPT_PATH"
    exit 1
fi

# Check if main.py exists
MAIN_PY_PATH="${SLURM_SUBMIT_DIR}/main.py"
if [ ! -f "$MAIN_PY_PATH" ]; then
    echo "Error: main.py not found at: $MAIN_PY_PATH"
    exit 1
fi

# Calculate the number of jobs needed by counting lines in batch_paths.txt
num_jobs=$(wc -l < "$BATCH_PATHS_FILE")

echo "Verifying batch paths file contents..."
echo "First few batch paths:"
head -n 3 "$BATCH_PATHS_FILE"
echo "..."
echo "Last few batch paths:"
tail -n 3 "$BATCH_PATHS_FILE"
echo "Total number of batch paths: $num_jobs"
echo ""
echo "Each job will run: python main.py <batch_path>"

# Prompt for confirmation
read -p "Do you want to submit $num_jobs array jobs? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Submission cancelled"
    exit 1
fi

# Submit the job array with the calculated range
# Add 1 to account for 1-based line numbering
job_id=$(sbatch --array=1-${num_jobs} "$SLURM_SCRIPT_PATH" | awk '{print $4}')
submit_status=$?

if [ $submit_status -eq 0 ]; then
    echo "Successfully submitted $num_jobs jobs"
    echo "Job array ID: $job_id"
    echo "Array task IDs: 1-${num_jobs}"
    echo ""
    echo "Monitor with: squeue -j $job_id"
    echo "Cancel with: scancel $job_id"
    echo "Check logs in: ${SLURM_SUBMIT_DIR}/logs/"
else
    echo "Error: Job submission failed"
    exit 1
fi