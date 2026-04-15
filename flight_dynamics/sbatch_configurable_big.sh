#!/bin/bash
#SBATCH --job-name=s4_flight_training
#SBATCH -o flight_dynamics/logs/%x_%J.out
#SBATCH -e flight_dynamics/logs/%x_%J.err
#SBATCH --mem=128g
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=catfish-[01-05],salmon-[01-10]
#SBATCH --mail-user=lior.kotlar@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL

# Usage: sbatch -J <EXPERIMENT_NAME> sbatch_configurable.sh <CONFIG_PATH>
CONFIG_PATH=$1

# Safety checks
if [ -z "$CONFIG_PATH" ]; then
  echo "Error: No config file path provided."
  echo "Usage: sbatch -J <EXPERIMENT_NAME> sbatch_configurable.sh path/to/config.json"
  exit 1
fi

# Automatically grab the job name provided via the -J flag
EXPERIMENT_NAME=$SLURM_JOB_NAME

echo "started"

# Navigate to the correct S4 workspace
cd /cs/labs/tsevi/lior.kotlar/mamba-flight
source .env/bin/activate

echo "Job started on $(hostname)"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Config File: $CONFIG_PATH"
echo "Experiment Name: $EXPERIMENT_NAME"

# Execute the python script with the mapped flags
# Adjust "flight_map/train.py" to your exact python script filename if different
python flight_dynamics/main_flight.py --config "$CONFIG_PATH" --name "$EXPERIMENT_NAME"

echo "finished working"