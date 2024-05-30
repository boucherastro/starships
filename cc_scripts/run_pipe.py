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

# unpack input parameters into config dictionary
config_filepath = 'config.yaml'
with open(config_filepath, 'r') as file:
    config_dict = yaml.safe_load(file)

config_dict['obs_dir'] = Path.home() / Path(config_dict['obs_dir'])

# creating the planet and observation objects
planet, obs = red.load_planet(config_dict)

# # Choose which exposures to use
# all_exposures = np.arange(obs.n_spec)
# transit_tags = np.delete(all_exposures, [20, 21, 22, 23, 31, 32, 33])  # Here we exclude the exposures [20, 21, ..., 33]

'''---------------------------------------Reducing the Data--------------------------------------'''
def reduce_data():
    out_dir, path_fig = red.set_save_location(planet.name, config_dict['reduction'], config_dict['instrument']) # might replace with out_dir from YAML file

    # building the transit spectrum
    list_tr = red.build_trans_spec(config_dict, obs, planet)

    # saving the transit spectrum
    red.save_pl_sig(config_dict, list_tr, out_dir, do_tr = [1])

    # outputting plots for reduction step
    red.reduction_plots(list_tr, config_dict['idx_ord'], path_fig)

def make_model():
    # computing extra parameters needed for model making
    int_dict = mod.create_internal_dict(config_dict, planet)

    # create the model
    wave_mod, mod_spec = mod.prepare_model_high_or_low(config_dict, int_dict, planet)
    return wave_mod, mod_spec

def perform_correlations():
    # performs standard ccf
    # performs injected ccf





## running the pipeline

if __name__ == '__main__':
    # unpack input parameters into config dictionary
    config_filepath = 'config.yaml'
    with open(config_filepath, 'r') as file:
        config_dict = yaml.safe_load(file)

    config_dict['obs_dir'] = Path.home() / Path(config_dict['obs_dir'])

    # creating the planet and observation objects
    planet, obs = red.load_planet(config_dict)

    # need to iterate over n_pc, mask_tellu and mask_wings
    for n_pc in config_dict['n_pc']:
        for mask_tellu in config_dict['mask_tellu']:
            for mask_wings in config_dict['mask_wings']:

                config_dict['n_pc'] = n_pc
                config_dict['mask_tellu'] = mask_tellu
                config_dict['mask_wings'] = mask_wings

                # reducing the data
                reduce_data()

                # making the model
                make_model()

