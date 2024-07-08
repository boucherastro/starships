#!/bin/bash
#SBATCH --job-name=
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --account=def-rdoyon
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=olivia.pereira@mail.mcgill.ca
#SBATCH --array=0-4

# Define a name for this reduction run, used for file naming in the pipeline
run_name="05072024"

# Define the directory where run_pipe.py is located, should be where you cloned starships
PIPE_DIR=/home/opereira/starships/cc_scripts

# Identify which targets you want to run
# For each target, you need a config file and a model file of the following file name format:
# config-<name>.yaml and model-<name>.yaml
names=("HD189733b" "WASP69b" "WASP80b" "WASP107b" "WASP127b")

# activate your virtual environment where starships is installed
source /home/opereira/.venvs/starships-env/bin/activate

# Execute run_pipe.py with the inputs config.yaml and model_config.yaml
# Save all outputs to a .txt file called output.txt
python run_pipe.py config.yaml model_config.yaml > output.txt

# Use SLURM_ARRAY_TASK_ID to select the target
target=${names[$SLURM_ARRAY_TASK_ID]}

# Execute run_pipe.py with the inputs config-<name>.yaml and model-<name>.yaml
# Save all outputs to a .txt file called output_<name>.txt
python run_pipe.py config-${target}.yaml model-${target}.yaml > output_${target}.txt