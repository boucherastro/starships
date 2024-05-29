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

from starships.correlation import quick_correl
from starships.correlation_class import Correlations

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

def classic_ccf(config_dict, transit, wave_mod, mod_spec):

    # 1. standard CCF
    corrRV = np.arange(config_dict['RV_range'][0], config_dict['RV_range'][1], config_dict['RV_step'])

    # Do the correlation
    ccf = quick_correl(transit.wave, transit.final, corrRV, wave_mod, mod_spec, wave_ref=None, 
                        get_logl=False, kind='BL', mod2d=False, expand_mask=0, noise=None, somme=False)

    # Create the object do make the plots and compute the different Kp

    corr_obj = Correlations(ccf, kind='BL', rv_grid=corrRV, kp_array=transit.Kp)
    corr_obj.calc_ccf(orders=None, N=transit.N_frac[None, :, None], alpha=np.ones_like(transit.alpha_frac), index=None, ccf0=None, rm_vert_mean=False,)
    corr_obj.calc_correl_snr_2d(transit, plot=False)
    corr_obj.RV_shift = np.zeros_like(transit.alpha_frac)

    # Make the plots and save them
    corr_obj.full_plot(transit, []) #, kind_max='grid')










