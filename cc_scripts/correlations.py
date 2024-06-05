import os
import numpy as np
from pathlib import Path
from sys import path

import matplotlib.pyplot as plt
from importlib import reload
from itertools import product
from scipy.interpolate import interp1d

import astropy.units as u
import astropy.constants as const

from starships.correlation import quick_correl
from starships.correlation_class import Correlations
import starships.correlation as correl

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

def classic_ccf(config_dict, transit, wave_mod, mod_spec, path_fig, corrRV = []):
    """
    Perform classic cross-correlation function (CCF) analysis.

    Parameters:
    - config_dict (dict): A dictionary containing configuration parameters.
    - transit (Transit): An object representing the transit data.
    - wave_mod (array-like): The wavelength array of the model spectrum.
    - mod_spec (array-like): The model spectrum.
    - path_fig (str): The path to save the generated figures.
    - corrRV (array-like, optional): The array of radial velocities for the CCF. 
        If not provided, it will be generated based on the RV range and step defined in config_dict.

    Returns:
    None

    """
    # 1. standard CCF
    if len(corrRV) == 0:
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
    corr_obj.full_plot(transit, [], save_fig = 'classic_ccf', path_fig = path_fig) 

    return corr_obj

def inj_ccf(config_dict, transit, wave_mod, mod_spec, obs, path_fig, corrRV = []):
    if len(corrRV) == 0:
        corrRV = np.arange(config_dict['RV_range'][0], config_dict['RV_range'][1], config_dict['RV_step'])
    
    Kp_array = np.array([transit.Kp.value])
    ccf_map, logl_map = correl.quick_calc_logl_injred_class(transit, Kp_array, corrRV, config_dict['n_pc'], 
                                                    wave_mod, np.array([mod_spec]), nolog=True, 
                                                    inj_alpha='ones', RVconst=transit.RV_const)

    # plot ccf figures
    ccf_obj, logl_obj = cc.plot_ccflogl(obs, ccf_map, logl_map, corrRV,
                                    Kp_array, config_dict['n_pc'], id_pc0=0, orders=np.arange(75), 
                                    path_fig = str(path_fig))
    
    return ccf_obj, logl_obj

def ttest_map(ccf_obj, config_dict, transit, obs, path_fig):
    # ttest map
    ccf_obj.ttest_map(transit, kind='logl', vrp=np.zeros_like(transit.vrp), orders=np.arange(75), 
                  kp0=0, RV_limit=config_dict['RV_lim'], kp_step=config_dict['Kp_step'], 
                  rv_step=config_dict['RV_step'], RV=None, speed_limit=3, icorr=obs.iIn, equal_var=False, path_fig = str(path_fig))
    






