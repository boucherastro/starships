import numpy as np
import yaml, os
from sys import path
import matplotlib.pyplot as plt

path.append("opereira/starships/cc_scripts/")

import cc_scripts.reduction as red
import cc_scripts.make_model as mod
import cc_scripts.correlations as corr

# unpack input parameters into config dictionary
config_filepath = 'config.yaml'
with open(config_filepath, 'r') as file:
    d = yaml.safe_load(file)

# unpacking variables
pl_name, instrument, visit_name, kind_trans, coeffs, ld_model, kind_res = \
    d['pl_name'], d['instrument'], d['visit_name'], d['kind_trans'], d['coeffs'], d['ld_model'], d['kind_res']

obs_dir, pl_kwargs = d['obs_dir'], d['pl_kwargs']
obs_dir = Path.home() + Path(obs_dir)

# creating the planet and observation objects
p, obs = red.load_planet(pl_name, obs_dir, pl_kwargs, instrument)

# Choose which exposures to use
all_exposures = np.arange(obs.n_spec)
transit_tags = np.delete(all_exposures, [20, 21, 22, 23, 31, 32, 33])  # Here we exclude the exposures [20, 21, ..., 33]

'''---------------------------------------Reducing the Data--------------------------------------'''

reduction, mask_tellu, mask_wings, n_pc, iout_all, clip_ratio, clip_ts, unberv_it, del_exposures = \
    d['reduction'], d['mask_tellu'], d['mask_wings'], d['n_pc'], d['iout_all'], d['clip_ratio'], \
    d['clip_ts'], d['unberv_it'], d['del_exposures']

out_dir, path_fig = red.set_save_location(pl_name, reduction, instrument) # might replace with out_dir from YAML file

# building the transit spectrum
list_tr = red.build_trans_spec(mask_tellu, mask_wings, n_pc, coeffs, ld_model, kind_trans, iout_all, 
                        clip_ratio, clip_ts, unberv_it, obs, p)

# saving the transit spectrum
red.save_pl_sig(list_tr, out_dir, n_pc, mask_wings, visit_name, do_tr = [1])

# outputting plots for reduction step
red.reduction_plots(list_tr, d['idx_ord'], path_fig)

'''-------------------------------------Generating the Model-------------------------------------'''

# some params from config will be unpacked into theta_dict to make the model

wave_mod, mod_spec = mod.prepare_model_high_or_low(d, d['mode'], atmo_obj=None, fct_star=None,
                              species_dict=None, Raf=None, rot_ker=None)

plt.plot(wave_mod, mod_spec)
plt.savefig(path_fig+'test_model.pdf')

'''--------------------------------------Cross correlations--------------------------------------'''
