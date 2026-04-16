#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --account=<account_name>
#SBATCH --gpus-per-node=v100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --mail-user=<email_address>
#SBATCH --mail-type=ALL
#SBATCH --output=./slurm/%A/%a.out
#SBATCH --array=35,39,40,41,46,49,51,53,57,59,60

module load StdEnv/2023 python/3.11.5 arrow/15.0.1
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ../cedar/requirements.txt
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

start_index=$((SLURM_ARRAY_TASK_ID * 1000 ))


# Run the Python script with the selected dataset and logit
python calculate_aopc_separate.py $start_index