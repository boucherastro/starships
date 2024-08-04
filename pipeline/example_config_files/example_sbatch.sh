#!/bin/bash
#SBATCH --account=def-rdoyon
#SBATCH --mem=120G
#SBATCH --time=16:00:00
#SBATCH --job-name=starships_pipeline
#SBATCH --output=out_sbatch_%A_%a.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=olivia.pereira@mail.mcgill.ca
#SBATCH --array=1-5

# Name of this reduction run, will be used for file naming only
run_name="14072024"

# Define the directory where run_pipe.py is located, should be where you cloned starships
PIPE_DIR=/home/opereira/starships/pipeline

echo "Starting run_sbatch with ID $SLURM_JOB_ID"

# Set the name of the input file containing the list of yaml files to process in the pipeline
export INPUT_FILE=input_files.txt

# activate your virtual environment where starships is installed
source /home/opereira/.venvs/starships-env/bin/activate

echo "Starting python code with SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"

# Read the specific line from the input_file.txt, given by the SLURM_ARRAY_TASK_ID
# where input_files.txt is a file containing the addresses of the yaml files
# for each independent reduction.
yaml_file=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $INPUT_FILE)

# Print the yaml file that will be used for this job
echo "Using yaml file: $yaml_file"

# Execute run_pipe.py with the input config-<name>.yaml
# Save all outputs to a .txt file called output_<name>.txt
python $PIPE_DIR/run_pipe.py $yaml_file $run_name > output_${target}.txt

# Generate a summary of the output

