#!/bin/bash
#SBATCH --account=def-dlafre
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=50G
#SBATCH --time=3-00:00
#SBATCH --job-name W33_d2
#SBATCH --output=/home/adb/scratch/sbatch_outputs/out_sbatch_%j.txt
#SBATCH --mail-type=FAIL

source ~/.venvs/starships_env_39/bin/activate

echo "Starting python code..."
python retrieval_WASP-33b_JR_guillot_chem_eq_copy.py yaml_file=retrieval_inputs_example_wasp33.yaml

# Burnin (use with >>> #SBATCH --array=1-10 for example)
# echo "Starting python code with SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
# python retrieval_WASP-33b_JR_guillot_chem_eq_copy.py idx_file=$SLURM_ARRAY_TASK_ID
