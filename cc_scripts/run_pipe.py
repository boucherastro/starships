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


def perform_correlations(config_dict, transit, wave_mod, mod_spec, n_pc, path_fig):
    # performs standard ccf
    corr_obj = corr.classic_ccf(config_dict, transit, wave_mod, mod_spec, str(path_fig))

    # performs injected ccf
    ccf_obj, logl_obj = corr.inj_ccf(config_dict, transit, wave_mod, mod_spec, n_pc)

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

    # creating the dictionaries to store all reductions
    all_reductions = {}
    all_ccf_map = {}
    all_logl_map = {}

    # performing the reduction for each n_pc, mask_tellu, mask_wings
    for mask_tellu in config_dict['mask_tellu']:
        for mask_wings in config_dict['mask_wings']:
            for n_pc in config_dict['n_pc']:

                # reducing the data
                reduc = reduce_data(config_dict, n_pc, mask_tellu, mask_wings, planet, obs, out_dir, path_fig)
                all_reductions[(n_pc, mask_tellu, mask_wings)] = reduc

                # performing correlations
                ccf_map, logl_map = perform_correlations(config_dict, reduc['1'], wave_mod, mod_spec, n_pc, path_fig)
                all_ccf_map[(n_pc, mask_tellu, mask_wings)] = ccf_map
                all_logl_map[(n_pc, mask_tellu, mask_wings)] = logl_map

            # plot all ccf maps for each n_pc
            ccf_obj, logl_obj = corr.plot_all_ccf(all_ccf_map, all_logl_map, all_reductions, config_dict, mask_tellu, 
                                                    mask_wings, id_pc0=0, order_indices=np.arange(75))

            # plotting ttest maps for each n_pc
            for n_pc in config_dict['n_pc']:
                ccf_obj.ttest_map(all_reductions[(n_pc, mask_tellu, mask_wings)], str(path_fig))

    return all_reductions, all_ccf_map, all_logl_map

