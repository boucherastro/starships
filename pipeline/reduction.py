import os, warnings
# import sys
from pathlib import Path

from starships.mask_tools import interp1d_masked

interp1d_masked.iprint=False

import astropy.constants as const
import astropy.units as u
import numpy as np
import starships.planet_obs as pl_obs
from starships.planet_obs import Observations
import starships.plotting_fcts as pf

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

def set_save_location(pl_name, visit_name, reduction, instrument, out_dir = None):

    ''' Note: Better to use the scratch space to save the reductions.
    You have infinte space, but the files are deleted if untouched for 2 months. It allows to save 
    as many reductions as desired. Once the correct reduction parameters are set, you can either 
    move them into your home directory. '''

    # Set scratch directory for larger data products, use home is scratch not available

    pl_name_fname = ''.join(pl_name.split())

    try:
        scratch_dir = Path(os.environ['SCRATCH'])
    except KeyError:
        scratch_dir = Path.home()

    # Output reductions in dedicated directory
    if out_dir == None:
        out_dir = Path.home()
        #out_dir /= Path(f'projects/def-dlafre/shared/{instrument}/Reductions/{reduction}/{pl_name_fname}/{visit_name}')
        out_dir /= Path(f'projects/def-rdoyon/shared/{instrument}/Reductions/{reduction}/{pl_name_fname}/{visit_name}')


    # Output reductions in dedicated directory
    scratch_dir /= Path(f'{instrument}/Reductions/{reduction}/{pl_name_fname}')

    # Make sure the directories exists
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # Where to save figures?
    path_fig = out_dir / Path('Results')
    path_fig.mkdir(parents=True, exist_ok=True)

    path_fig = str(path_fig) + '/'

    return scratch_dir, out_dir, path_fig


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


def load_planet(config_dict, visit_name):
    
    # All the observations must be listed in files.
    # We need the e2ds, the telluric corrected and the reconstructed spectra.
    list_filenames = {'list_e2ds': f'list_e2ds_{visit_name}',
                    'list_tcorr': f'list_tcorr_{visit_name}'} #,
                    #'list_recon': f'list_tellu_recon{visit_name}'}

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
    obs.fetch_data(config_dict['obs_dir'], **list_filenames, CADC = True)

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

    RVsys = [planet.RV_sys.value]
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
    list_tr = pl_obs.generate_all_transits(obs, transit_tags, RVsys, params_all, config_dict['iout_all'], counting = False,
                                        **kwargs_gen_tr, **kwargs_build_ts)

    return list_tr


def save_pl_sig(list_tr, nametag, scratch_dir, bad_indexs=[]):
    # Save sequence with only the info needed for a retrieval (to compute log likelihood).
    out_filename = f'retrieval_input' + nametag
    pl_obs.save_single_sequences(out_filename, list_tr, path=scratch_dir, save_all=True, bad_indexs = bad_indexs)


def reduction_plots(config_dict, obs, list_tr, n_pc, path_fig, nametag): 
    visit_list = [list_tr]  # You could put multiple visits in the same figure

    if n_pc == config_dict['n_pc'][0]:
        pf.plot_night_summary_NIRPS(visit_list, obs, path_fig=str(path_fig), fig_name='')

    sequence_obj = list_tr

    # plot for specified orders
    for idx_ord in config_dict['idx_ord']:
        pf.plot_steps(sequence_obj, idx_ord, path_fig=str(path_fig), fig_name = nametag + f'_ord{idx_ord}')


def reduce_data(config_dict, planet, obs, scratch_dir, out_dir, path_fig, n_pc, mask_tellu, mask_wings, visit_name):

    nametag = f'_{visit_name}_maskwings{mask_wings*100:n}_masktellu{mask_tellu*100:n}_pc{n_pc}'
    # check if reduction already exists
    if os.path.exists(scratch_dir / f'retrieval_input{nametag}_data_trs_.npz'):
        print(f"Reduction already exists for {nametag}. Loading...")
        list_tr = pl_obs.load_single_sequences(f'retrieval_input{nametag}_data_trs_.npz', planet.name, path=scratch_dir,
                          load_all=True, filename_end='', planet=planet, plot = False)

    else: # building the transit spectrum
        list_tr = build_trans_spec(config_dict, n_pc, mask_tellu, mask_wings, obs, planet)
        list_tr = list_tr['1']

    # saving the transit spectrum

    if config_dict['bad_indexs']:
        bad_indexs = config_dict['bad_indexs']['visit_name']
        print('Masking exposure(s) at index(s) ', bad_indexs)

    else: bad_indexs = []
    
    save_pl_sig(list_tr, nametag, scratch_dir, bad_indexs)

    # outputting plots for reduction step
    reduction_plots(config_dict, obs, list_tr, n_pc, path_fig, nametag)

    return list_tr