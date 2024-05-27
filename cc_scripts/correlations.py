import os
import numpy as np
from pathlib import Path
from sys import path
# path.append('/home/adb/PycharmProjects/')
# path.append('/home/adb/PycharmProjects/starships_analysis/')

import matplotlib.pyplot as plt
from importlib import reload
from itertools import product
from scipy.interpolate import interp1d

import astropy.units as u
import astropy.constants as const

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

import starships.plotting_fcts as pf 
import starships.planet_obs as pl_obs
from starships import homemade as hm
from starships import correlation as corr
from starships import correlation_class as cc
from starships.planet_obs import Observations,Planet

from itertools import product
# cc=reload(cc)

# - Setting plot parameters - #
plt.rc('figure', figsize=(9, 6))


def gen_ccf(config_dict, planet, obs, wave_mod, mod_spec, corrRV0):
    
    Kp_array = np.array([obs.Kp.value]) 
    ccf_map, logl_map = corr.calc_logl_injred(
                obs,'seq', planet, Kp_array, corrRV0, [config_dict['n_pc']], wave_mod, mod_spec,  config_dict['kind_trans'])
    
    return ccf_map, logl_map

# def plot_ccf(config_dict, ccf_map, logl_map, corrRV0):
#     cf_obj, logl_obj = cc.plot_ccflogl(all_visits, 
#                                     ccf_map,
#                                     logl_map,
#                                     corrRV0, Kp_array, [1],
#                                     split_fig = [0,t1.n_spec,t1.n_spec+t2.n_spec],
#                                     orders=idx_orders
#                                    )
#     return cf_obj, logl_obj








