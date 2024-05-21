import numpy as np
import yaml, os
from sys import path

path.append("opereira/starships/cc_scripts/")

import cc_scripts.reduction as red
import cc_scripts.make_model as mod
import cc_scripts.correlations as corr

# unpack input parameters
config_filepath = 'config.yaml'

with open(config_filepath, 'r') as file:
    config = yaml.sale_load(file)

# Choose which exposures to use
all_exposures = np.arange(obs.n_spec)
transit_tags = np.delete(all_exposures, [20, 21, 22, 23, 31, 32, 33])  # Here we exclude the exposures [20, 21, ..., 33]

'''                                      Reducing the Data                                       '''

p, obs = ed.load_planet(pl_name, obs_dir, pl_kwargs, instrument)
list_tr = red.build_trans_spec(mask_tellu, mask_wings, n_pc, coeffs, ld_model, kind_trans, iout_all, 
                        clip_ratio, clip_ts, unberv_it, obs, planet)

save_pl_sig(list_tr, out_dir, n_pc, mask_wings, visit_name, do_tr = [1])

reduction_plots(list_tr, idx_ord = 40, path_fig)

set_save_location(pl_name, reduction, instrument) # might replace with out_dir from YAML file


'''                                     Generating the Model                                     '''

# some params from config will be unpacked into theta_dict to make the model

mod.prepare_model_high_or_low(theta_dict, config[mode], atmo_obj=None, fct_star=None,
                              species_dict=None, Raf=None, rot_ker=None)
