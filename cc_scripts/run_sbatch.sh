#!/bin/bash
#SBATCH --job-name=
#SBATCH --output=wasp-127b.out
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --account=def-dlafre
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=olivia.pereira@mail.mcgill.ca

main_path = PATH/TO/ALL/REDUCTIONS

# Execute run_pipe.py with the inputs config.yaml and model_config.yaml
# Save all outputs to a .txt file called output.txt
python run_pipe.py config.yaml model_config.yaml > output.txt

# Get the name of the output directory from output.txt
out_dir=$(grep -oP '(?<=Output directory: ).*' output.txt)

# If the output directory was found, copy the input yaml files and output.txt to it
if [[ -n "$out_dir" ]]; then
    cp config.yaml model_config.yaml output.txt "$out_dir"
else
    echo "Output directory not found"
fi