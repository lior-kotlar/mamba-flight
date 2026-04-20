#!/bin/bash
#SBATCH --job-name=mamba_flight_training
#SBATCH -o flight_dynamics/logs/%x_%J.out
#SBATCH -e flight_dynamics/logs/%x_%J.err
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=salmon
#SBATCH --mail-user=lior.kotlar@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL

# Capture the flag type (--config or --resume_dir) and the path provided
FLAG=$1
PATH_ARG=$2

# Automatically grab the job name provided via the -J flag
EXPERIMENT_NAME=$SLURM_JOB_NAME

# Safety checks and routing logic
if [ "$FLAG" == "--config" ] && [ -n "$PATH_ARG" ]; then
  RUN_MODE="FRESH"
  PYTHON_ARGS="--config $PATH_ARG --name $EXPERIMENT_NAME"
elif [ "$FLAG" == "--resume_dir" ] && [ -n "$PATH_ARG" ]; then
  RUN_MODE="RESUME"
  PYTHON_ARGS="--resume_dir $PATH_ARG"
else
  echo "Error: Invalid or missing arguments."
  echo "Usage for a NEW run: sbatch -J <EXPERIMENT_NAME> sbatch_configurable.sh --config path/to/config.json"
  echo "Usage to RESUME:     sbatch -J <EXPERIMENT_NAME> sbatch_configurable.sh --resume_dir path/to/run_directory"
  exit 1
fi

echo "Job started on $(hostname) at $(date)"
echo "Run Mode: $RUN_MODE"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Experiment Name: $EXPERIMENT_NAME"
echo "Executing Python script with args: $PYTHON_ARGS"

# Navigate to the correct workspace
cd /cs/labs/tsevi/lior.kotlar/mamba-flight
source .env/bin/activate

# Execute the python script with the dynamically mapped flags
python flight_dynamics/main_flight.py $PYTHON_ARGS

echo "Finished working at $(date)"