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

def reduce_data(config_dict, n_pc, mask_tellu, mask_wings, planet, obs, out_dir, path_fig):
    # building the transit spectrum
    list_tr = red.build_trans_spec(config_dict, n_pc, mask_tellu, mask_wings, obs, planet)

    # saving the transit spectrum
    red.save_pl_sig(n_pc, mask_tellu, mask_wings, list_tr, out_dir, do_tr = [1])

    # outputting plots for reduction step
    red.reduction_plots(list_tr, n_pc, mask_tellu, mask_wings, config_dict['idx_ord'], path_fig)

    return list_tr

def make_model(config_dict, config_model, planet, out_dir, path_fig):
    # computing extra parameters needed for model making
    int_dict = mod.create_internal_dict(config_model, planet)

    # create the model
    wave_mod, mod_spec = mod.prepare_model_high_or_low(config_model, int_dict, planet,out_dir=out_dir, path_fig=path_fig)
    
    # convolving with the instrument if not already done
    # if config_model['instrument'] == None:
    #     wave_mod, mod_spec = mod.add_instrum_model(config_dict, wave_mod, mod_spec)
    
    return wave_mod, mod_spec


def perform_correlations(config_dict, transit, wave_mod, mod_spec, obs, path_fig):
    # performs standard ccf
    corr_obj = corr.classic_ccf(config_dict, transit, wave_mod, mod_spec, str(path_fig))

    # performs injected ccf - CHECK WHETHER CORRECT OBJECT IS BEING RETURNED HERE
    ccf_obj, logl_obj = corr.inj_ccf(config_dict, transit, wave_mod, mod_spec, obs, str(path_fig))

    # perform ttest
    # corr.ttest_map(ccf_obj, config_dict, transit, obs, str(path_fig))

    return ccf_obj, logl_obj


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
    wave_mod, mod_spec = make_model(config_dict, config_model, planet, out_dir, path_fig)

    # performing the reduction for each n_pc, mask_tellu, mask_wings and the corrlations
    for n_pc in config_dict['n_pc']:
        for mask_tellu in config_dict['mask_tellu']:
            for mask_wings in config_dict['mask_wings']:

                # reducing the data
                transit = reduce_data(config_dict, n_pc, mask_tellu, mask_wings, planet, obs, out_dir, path_fig)

                # performing correlations
                ccf_obj, logl_obj = perform_correlations(config_dict, transit['1'], wave_mod, mod_spec, obs, path_fig)
