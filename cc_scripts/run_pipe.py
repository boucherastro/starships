import numpy as np
import yaml, os
from sys import path
from pathlib import Path
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const

path.append("opereira/starships/cc_scripts/")

import cc_scripts.reduction as red
import cc_scripts.make_model as mod
import cc_scripts.correlations as corr

from starships.correlation import quick_correl
from starships.correlation_class import Correlations
from starships.instruments import load_instrum


## running the pipeline
def run_pipe(config_filepath, model_filepath):
    # unpack input parameters into config dictionary
    with open(config_filepath, 'r') as file:
        config_dict = yaml.safe_load(file)

    # unpack the model input parameters
    with open(model_filepath, 'r') as file:
        config_model = yaml.safe_load(file)

    config_dict['obs_dir'] = Path.home() / Path(config_dict['obs_dir'])

    # creating the planet and observation objects
    planet, obs = red.load_planet(config_dict)

    out_dir, path_fig = red.set_save_location(planet.name, config_dict['reduction'], config_dict['instrument'])

    # making the model
    wave_mod, mod_spec = mod.make_model(config_model, planet, out_dir, path_fig, config_dict)

    # creating the dictionaries to store all reductions
    all_reductions = {}
    all_ccf_map = {}
    all_logl_map = {}

    # performing the reduction for each n_pc, mask_tellu, mask_wings
    for mask_tellu in config_dict['mask_tellu']:
        for mask_wings in config_dict['mask_wings']:
            for n_pc in config_dict['n_pc']:

                # reducing the data
                reduc = red.reduce_data(config_dict, n_pc, mask_tellu, mask_wings, planet, obs, out_dir, path_fig)
                all_reductions[(n_pc, mask_tellu, mask_wings)] = reduc

                # performing correlations
                ccf_map, logl_map = corr.perform_ccf(config_dict, reduc, wave_mod, mod_spec, n_pc, mask_tellu, mask_wings, out_dir)
                all_ccf_map[(n_pc, mask_tellu, mask_wings)] = ccf_map
                all_logl_map[(n_pc, mask_tellu, mask_wings)] = logl_map

            # plot all ccf maps for each n_pc
            ccf_obj, logl_obj = corr.plot_all_ccf(all_ccf_map, all_logl_map, all_reductions, config_dict, mask_tellu, 
                                                    mask_wings, id_pc0=0, order_indices=np.arange(75))

            # plotting ttest maps for each n_pc
            for n_pc in config_dict['n_pc']:
                obs = all_reductions[(n_pc, mask_tellu, mask_wings)]
                ccf_obj.ttest_map(obs, kind='logl', vrp=np.zeros_like(obs.vrp), orders=np.arange(75), 
                  kp0=0, RV_limit=75, kp_step=5, rv_step=2, RV=None, speed_limit=3, icorr=obs.iIn, equal_var=False
                  )

    return all_reductions, all_ccf_map, all_logl_map