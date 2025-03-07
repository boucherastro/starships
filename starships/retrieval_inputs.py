"""
What is expected in the YAML file.
"""
import logging

# Set up logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.basicConfig()

# Put the expected data types.
# This is used to convert command line arguments to integers when needed.
# NOTE: It could be used to validate the YAML file as well eventually.
# For now, it does not need to be exhaustive.
params_dtypes = dict()
param_types = {
    'instrum': ["list"],  # List of strings, each representing an instrument name
    'high_res_file_stem_list': ["list"],  # List of strings, each a file stem for high-resolution data
    'spectrophotometric_data': ["null", "dict"],  # Optional; can be null or a dictionary specifying paths to spectrophotometric data
    'photometric_data': ["null", "dict"],  # Dictionary specifying paths to photometric data
    'retrieval_type': ["string"],  # String indicating the type of retrieval
    'white_light': ["bool"],  # Boolean indicating if white light is used
    'chemical_equilibrium': ["bool"],  # Boolean indicating if chemical equilibrium is considered
    'dissociation': ["bool"],  # Boolean indicating if dissociation is considered
    'kind_temp': ["string"],  # String indicating the type of temperature model
    'run_name': ["null", "string"],  # String for the name of the run
    'walker_file_out': ["string", "null"],  # String specifying the output file for walkers
    'walker_file_in': ["string", "null"],  # Optional; string specifying the input file for walkers, can be null
    'init_mode': ["string"],  # String indicating the initialization mode
    'n_cpu': ["integer", "null"],  # Integer specifying the number of CPUs to use
    'n_walkers': ["integer", "null"],  # Integer specifying the total number of walkers
    'n_walkers_per_cpu': ["integer", "null"],  # Integer specifying the number of walkers per CPU
    'line_opacities': ["list"],  # List of strings, each specifying a line opacity source
    'continuum_opacities': ["list"],  # List of strings, each specifying a continuum opacity source
    'other_species': ["list"],  # Optional; list of strings specifying other species, can be null
    'species_in_prior': ["list"],  # List of strings, each specifying a species included in the prior
    'base_dir': ["string", "null"],  # String specifying the base directory for the retrieval
    'high_res_path': ["string"],  # String specifying the path to high-resolution data
    'star_spectrum': ["string", "null"],  # Optional; string specifying the path to the star spectrum, can be null
    'slurm_array_behaviour': ["string", "null"],  # Boolean indicating if slurm array behaviour is used
    'n_steps_burnin': ["integer"],  # Integer specifying the number of burn-in steps
    'n_steps_sampling': ["integer"],  # Integer specifying the number of sampling steps
    'walker_path': ["string", "null"],  # String specifying the path to save walkers
    'params_path': ["string", "null"],  # String specifying the path for parameter files
    'params_file_out': ["string", "null"]  # String specifying the output parameter file
}

# Function to convert command line arguments to their expected types
def convert_cmd_line_to_types(parameters: dict):
    for key, value in parameters.items():
        if key in param_types:
            if "integer" in param_types[key] and value.isdigit():
                log.info(f"Converting {key} to integer")
                parameters[key] = int(value)
            elif "bool" in param_types[key]:
                log.info(f"Converting {key} to boolean")
                if value.lower() == "true":
                    parameters[key] = True
                elif value.lower() == "false":
                    parameters[key] = False
            elif "list" in param_types[key]:
                log.info(f"Converting {key} to list")
                parameters[key] = value.split(",")
            elif "null" in param_types[key]:
                if value.lower() == "null" or value.lower() == "none":
                    log.info(f"Converting {key} to None")
                    parameters[key] = None
            else:
                log.info(f"Leaving {key} as string")
        else:
            log.info(f"Expected type not found for {key} = {value}, leaving as string.")
    return parameters
