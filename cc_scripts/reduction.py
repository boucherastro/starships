import os, warnings
# import sys
from pathlib import Path

import matplotlib.pyplot as plt
import starships.homemade as hm
from starships import spectrum
from starships.mask_tools import interp1d_masked

interp1d_masked.iprint=False

import astropy.constants as const
import astropy.units as u
import numpy as np
import starships.planet_obs as pl_obs
import starships.plotting_fcts as pf
from starships.planet_obs import Observations, instruments_drs

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

def set_save_location(pl_name, reduction, instrument):

    ''' Note: Better to use the scratch space to save the reductions.
    You have infinte space, but the files are deleted if untouched for 2 months. It allows to save 
    as many reductions as desired. Once the correct reduction parameters are set, you can either 
    move them into your home directory. '''

    # Use scratch if available, use home if not.
    try:
        out_dir = Path(os.environ['SCRATCH'])
    except KeyError:
        out_dir = Path.home()

    # Output reductions in dedicated directory
    pl_name_fname = ''.join(pl_name.split())
    out_dir /= Path(f'DataAnalysis/{instrument}/Reductions/{pl_name_fname}_{reduction}')

    # Make sure the directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Where to save figures?
    path_fig = Path(str(out_dir) + f'/Figures/')
    path_fig.mkdir(parents=True, exist_ok=True)

    return out_dir, path_fig

def convert_to_quantity(quantity_dict):
    """
    Convert a quantity dictionary to a physical quantity.

    Parameters:
    quantity_dict (dict): A dictionary containing the value and unit of the quantity.

    Returns:
    Quantity: The converted physical quantity.

    """
    value = quantity_dict['value']
    unit = quantity_dict['unit']

    # Handle custom astrophysical units
    if unit == 'R_sun': unit = const.R_sun
    elif unit == 'M_sun': unit = const.M_sun
    elif unit == 'R_jup': unit = const.R_jup
    elif unit == 'M_jup': unit = const.M_jup
    elif unit == None: unit = 1

    else: unit = u.Unit(unit)

    return value * unit

def pl_param_units(config_dict):
    """
    Convert the values in the 'pl_params' dictionary of the given 'config_dict' to appropriate units.

    Args:
        config_dict (dict): A dictionary containing configuration parameters.

    Returns:
        dict: A dictionary with the converted values.

    """
    pl_kwargs = {}

    for key, value in config_dict['pl_params'].items():
        pl_kwargs[key] = convert_to_quantity(value)

    return pl_kwargs


def load_planet(config_dict):
    
    # All the observations must be listed in files.
    # We need the e2ds, the telluric corrected and the reconstructed spectra.
    list_filenames = {'list_e2ds': 'list_e2ds',
                    'list_tcorr': 'list_tellu_corrected',
                    'list_recon': 'list_tellu_recon'}

    # check if any planet attributes were manually specified
    if bool(config_dict['pl_params']): 
        pl_kwargs = pl_param_units(config_dict)
        obs = Observations(name=config_dict['pl_name'], instrument=config_dict['instrument'], pl_kwargs=pl_kwargs)
    else: 
        obs = Observations(name=config_dict['pl_name'], instrument=config_dict['instrument'])

    p = obs.planet

    # set other planet parameters
    p.A_star = np.pi*u.rad * p.R_star**2
    surf_grav = (const.G * p.M_star / p.R_star**2).cgs
    p.logg = np.log10(surf_grav.value)
    p.gp = const.G * p.M_pl / p.R_pl**2
    p.H = (const.k_B * p.Tp / (p.mu * p.gp)).decompose()
    p.sync_equat_rot_speed = (2*np.pi*p.R_pl/p.period).to(u.km/u.s)

    # Get the data
    obs.fetch_data(config_dict['obs_dir'], **list_filenames)

    # new_mask = obs.count.mask | (obs.count < 400.)
    # obs.flux = np.ma.array(obs.flux, mask=new_mask)
    return p, obs

def build_trans_spec(config_dict, n_pc, mask_tellu, mask_wings, obs, planet):

    # Parameters for extraction
    # param_all: Reduction parameters
    # param_all = [
    #     telluric fraction to mask (usually varied between 0.2 and 0.5), 
    #     limits for the wings (usually between 0.9 and 0.98), 
    #     width of the smoothing kernel for the low pass filter (fixed at 51), 
    #     useless param, 
    #     width of the gaussian kernel for low pass filter (fixed at 5),
    #     nPC to remove (depends on the data, usually between 1 and 8),
    #     sigma clips params (fixed at 5.0)
    #     ]
    # (So I basically only change tellu frac, the wings and nPC)

    params_all=[[mask_tellu, mask_wings, 51, 41, 5, n_pc, 5.0, 5.0, 5.0, 5.0]]

    RVsys = [planet.RV_sys.value[0]]
    transit_tags = [None]

    kwargs_gen_tr = {
    'coeffs' : config_dict['coeffs'],
    'ld_model' : config_dict['ld_model'],
    'do_tr' : [1],
    'kind_trans' : config_dict['kind_trans'],
    'polynome' : [False],
    'cbp': True # correct bad pixels
    }

    kwargs_build_ts = {
    'clip_ratio' : config_dict['clip_ratio'],
    'clip_ts' : config_dict['clip_ts'],
    'unberv_it' : config_dict['unberv_it'],
    }

    # Extract the planetary signal
    list_tr = pl_obs.generate_all_transits(obs, transit_tags, RVsys, params_all, config_dict['iout_all'],
                                        **kwargs_gen_tr, **kwargs_build_ts)

    return list_tr

def save_pl_sig(n_pc, mask_tellu, mask_wings, list_tr, out_dir, do_tr = [1]):
    # Save sequence with only the info needed for a retrieval (to compute log likelihood).
    out_filename = f'retrieval_input_{n_pc}-pc_mask_wings{mask_wings*100:n}_mask_tellu{mask_tellu*100:n}'
    pl_obs.save_sequences(out_filename, list_tr, do_tr, path=out_dir) # save to projects
    # add where save_all = True to scratch

def reduction_plots(list_tr, n_pc, mask_tellu, mask_wings, idx_ord, path_fig):
    visit_list = [list_tr['1']]  # You could put multiple visits in the same figure
    pf.plot_airmass(visit_list, path_fig=str(path_fig), fig_name=f'_{n_pc}-pc_mask_wings{mask_wings*100:n}_mask_tellu{mask_tellu*100:n}')

    sequence_obj = list_tr['1']
    pf.plot_steps(sequence_obj, idx_ord, path=path_fig, fig_name = f'_{n_pc}-pc_mask_wings{mask_wings*100:n}_mask_tellu{mask_tellu*100:n}')

'''                              TEST RUN FOR WASP 127-B TRANSIT 1                               '''
if __name__ == "__main__" :
    # setting parameters that would be set in a YAML file
    obs_dir = Path.home() / Path(f'projects/def-dlafre/fgenest/nirps/WASP-127b/')
    reduction = 'TEST'
    pl_name = 'WASP-127 b'
    instrument = 'NIRPS-APERO'
    visit_name = 'tr1'
    mask_tellu = 0.2 # Identify the deep tellurics that will be masked (in fraction of abosption)
    mask_wings = 0.9 # Mask wings of these deep tellurics (in fraction of absorption)
    n_pc = 3

    # --- For emission spectra
    # coeffs = [0.532]
    # ld_model = 'linear'
    # kind_trans='emission'

    # --- For transmission spectra
    coeffs = [ 0.02703969,  1.10037972, -0.96372403,  0.28750393]  # For transits
    ld_model = 'nonlinear'
    kind_trans='transmission'

    iout_all = ['all']
    
    pl_kwargs = {
    #     'M_star': 1.89 *u.M_sun,
    #     'R_star': 1.60 *u.R_sun,
    #     'M_pl' : 3.38 * u.M_jup,
    #     'R_pl' : 1.83 * u.R_jup,
                }
    
    clip_ratio = 6
    clip_ts = 6
    unberv_it = True

    # recreating notebook outputs to compare
    out_dir, path_fig = set_save_location(pl_name, reduction, instrument)
    p, obs = load_planet(pl_name, obs_dir, pl_kwargs, instrument)

    # Print some parameters (to check if they are satisfying)
    print((f"M_pl: {p.M_pl.to('Mjup')}\n"
        f"M_star: {p.M_star.to('Msun')}\n"
        f"R_star: {p.R_star.to('Rsun')}\n"
        f"R_pl: {p.R_pl.to('Rjup')}\n"
        f"RV_sys: {p.RV_sys}"))

    list_tr = build_trans_spec(mask_tellu, mask_wings, n_pc, coeffs, ld_model, kind_trans, iout_all, 
                        clip_ratio, clip_ts, unberv_it, obs, p)
    
    save_pl_sig(list_tr, out_dir, n_pc, mask_wings, visit_name, do_tr = [1])
    # check in folder

    print('Kp =', list_tr['1'].Kp)
    reduction_plots(list_tr, idx_ord = 40, path_fig = path_fig)