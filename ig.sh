#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --account=<account_name>
#SBATCH --gpus-per-node=v100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --mail-user=<email_address>
#SBATCH --mail-type=ALL
#SBATCH --output=./slurm/%A_%a.out
#SBATCH --array=0-30  # (60802 instances / 2000 instances per job = 31 jobs, 0-indexed)

# Load necessary modules and set up environment
module load StdEnv/2023 python/3.11.5 arrow/15.0.1
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ../cedar/requirements.txt
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Calculate i_min and i_max for the current array task
TASK_ID=$SLURM_ARRAY_TASK_ID
BATCH_SIZE=2000
i_min=$(( TASK_ID * BATCH_SIZE ))
i_max=$(( i_min + BATCH_SIZE ))

# Ensure i_max does not exceed the dataset size
if [ $i_max -gt 60802 ]; then
    i_max=60802
fi

# Run the script with the calculated indices
python ig.py $i_min $i_max