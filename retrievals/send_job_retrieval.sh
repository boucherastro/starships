#!/bin/bash
# send_job_retrieval.sh

# Description: This script sends a job to the retrieval system.
# It prepares a temporary YAML file with a unique timestamp and
# forwards it along with any additional arguments to the run_retrieval.sh script.
#
# Usage: ./send_job_retrieval.sh [mode] [input_yaml_file] [other_arguments...]
#   - mode: The mode in which to run the retrieval, e.g., 'burnin'
#   - input_yaml_file: Path to the input YAML file to be processed
#   - other_arguments: Additional arguments to be passed to run_retrieval.sh. These
#     can be used to override values in the input YAML file. The arguments should
#     be in the form key=value separated by spaces.
#
# Example:
#   ./send_job_retrieval.sh burnin config.yaml walker_file_in='burnin.h5' init_mode='from_burnin'
#
# Note: This script assumes the existence of a $SCRATCH environment variable
# pointing to a directory used for temporary files.

# Print the first lines of this script as a usage message if the number of arguments is less than 2
if [ $# -lt 2 ]; then
    head -n 30 $0
    exit 1
else
    echo "Running $0"
fi

# ********** EDIT THIS SECTION **********
# You need to specify the sbatch arguments here

# Put all sbatch arguments here
# (except for arguments that depend on burnin or sampling, see following lines)
# Put them in the right format to be passed to sbatch
# (note that you could add more sbatch arguments here if needed)
sbatch_args=(
    "--account=def-dlafre"
    "--nodes=1"
    "--cpus-per-task=28"
    "--mem=60G"
    "--job-name=W33_JR_s"
    "--mail-type=FAIL"
)

# *** Some arguments depend on burnin or sampling ***
# First argument is the either 'burnin' or 'sampling'.
# Set some of the sbatch arguments based on sampling or burnin
# and raise an error if it is not valid.
if [ $1 == "burnin" ]; then
    echo "Running in 'burnin' mode, so using sbatch array"
    sbatch_args+=("--time=1-00:00")
    sbatch_args+=("--array=1-10")
    sbatch_args+=("--output=/home/adb/scratch/sbatch_outputs/out_sbatch_%A_%a.txt")
elif [ $1 == "sampling" ]; then
    echo "Running in 'sampling' mode."
    sbatch_args+=("--time=3-00:00")
    sbatch_args+=("--output=/home/adb/scratch/sbatch_outputs/out_sbatch_%j.txt")
else
    echo "First argument must be either 'burnin' or 'sampling'"
    exit 1
fi


# ********** END OF EDIT SECTION **********


############################################
# The rest of the code should not be changed

# Generate a timestamp
timestamp=$(date +%s)

# Original filename passed as argument
input_yaml_file=$2
echo "Input yaml file is: ${input_yaml_file}"

# Show other arguments if they exist
echo "Other arguments (if any):"
echo "${@:3}"

# Temporary directory
temp_folder="${SCRATCH}/temp_yaml_files"

# Create the dictionary if it doesn't exist already
mkdir -p $temp_folder

# New filename with timestamp (replace .yaml with _timestamp.yaml)
input_yaml_filename=$(basename "$input_yaml_file")
input_yaml_file_temp="${temp_folder}/${input_yaml_filename%.yaml}_${timestamp}.yaml"

# Use $new_filename as the filename for operations
echo "Copying ${input_yaml_file} to ${input_yaml_file_temp}"
cp $input_yaml_file $input_yaml_file_temp

# Combine array elements into a string separated by spaces
sbatch_args_string="${sbatch_args[@]}"

# Pass the temp yaml file + all other arguments (excluding $1) to the shell script that launches the retrieval
echo "Sending the job with sbatch using the command:"
echo "sbatch ${sbatch_args_string} run_retrieval.sh ${input_yaml_file_temp} ${@:3}"
sbatch $sbatch_args_string run_retrieval.sh "${input_yaml_file_temp}" "${@:3}"
