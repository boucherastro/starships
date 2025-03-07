#!/bin/bash
# run_retrieval.sh

# Activate the virtual environment
source ~/.venvs/starships_env_39/bin/activate

# Run the python code (now any number of arguments can be passed)
echo "yaml file: $1"
echo "Other arguments: ${@:2}"
echo "Running the python code..."
run_starships_retrieval yaml_file=$1 ${@:2}
