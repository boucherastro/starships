#!/bin/bash
#SBATCH --account=def-dlafre
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=50G
#SBATCH --time=3-00:00
#SBATCH --job-name W33_d2
#SBATCH --output=/home/adb/scratch/sbatch_outputs/out_sbatch_%A_%a.txt
#SBATCH --mail-type=FAIL

# For burnin, use slurms array option
# Ex: #SBATCH --array=1-10

# When using the slurm array option, it is preferable to use the followin output name:
# #SBATCH --output=/home/adb/scratch/sbatch_outputs/out_sbatch_%A_%a.txt
# If not, it is preferable to use the following output name:
# #SBATCH --output=/home/adb/scratch/sbatch_outputs/out_sbatch_%j.txt
# This will be easier to identify the output file corresponding to the job ID.

# Note regarding slurm array:
#    The default behaviour of run_starships_retrieval is to assume that if slurm array is used,
#    the retrieval is in its burnin phase. Therefore, the retrieval is launched as many times
#    as the number of jobs in the array (1 to 10 in this example, so 10 times).
#    This is useful to explore a wider parameter space, using 10 times more walker in this case.
#    This behaviour is controlled by the flag `slurm_array_behaviour` in the yaml file.

source ~/.venvs/starships_env_39/bin/activate

echo "Starting python code..."
run_starships_retrieval yaml_file=retrieval_inputs_example_wasp33.yaml