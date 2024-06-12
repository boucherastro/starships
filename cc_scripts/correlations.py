
import numpy as np
from pathlib import Path

from starships.correlation import quick_correl
from starships.correlation_class import Correlations
import starships.correlation as correl

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

from starships import correlation as corr
from starships import correlation_class as cc

def classic_ccf(config_dict, transit, wave_mod, mod_spec, path_fig, nametag, corrRV = []): #, n_pc, mask_tellu, mask_wings, path_fig, corrRV = []):
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

    corr_obj.full_plot(transit, [], save_fig = f'classic_ccf{nametag}', path_fig = str(path_fig))

    return corr_obj


def inj_ccf(config_dict, transit, wave_mod, mod_spec, n_pc, scratch_dir, nametag, corrRV = []):
    if len(corrRV) == 0:
        corrRV = np.arange(config_dict['RV_range'][0], config_dict['RV_range'][1], config_dict['RV_step'])
    
    Kp_array = np.array([transit.Kp.value])
    ccf_map, logl_map = correl.quick_calc_logl_injred_class(transit, Kp_array, corrRV, n_pc, 
                                                    wave_mod, np.array([mod_spec]), nolog=True, 
                                                    inj_alpha='ones', RVconst=transit.RV_const)

    corr.save_logl_seq(scratch_dir / Path(f'inj_ccf_logl_seq{nametag}'), ccf_map, logl_map,
                           wave_mod, mod_spec, n_pc, Kp_array, corrRV, config_dict['kind_trans'])

    return ccf_map, logl_map


def perform_ccf(config_dict, transit, mol, wave_mod, mod_spec, n_pc, mask_tellu, mask_wings, scratch_dir, path_fig, corrRV = []):

    nametag = f'_{mol}_pc{n_pc}_maskwings{mask_wings*100:n}_masktellu{mask_tellu*100:n}'
    
    corr_obj = classic_ccf(config_dict, transit, wave_mod, mod_spec, path_fig, nametag)  
    ccf_map, logl_map = inj_ccf(config_dict, transit, wave_mod, mod_spec, [n_pc], scratch_dir, nametag) 
    
    return ccf_map, logl_map


def plot_all_ccf(all_ccf_map, all_logl_map, all_reductions, config_dict, mol, mask_tellu, mask_wings, id_pc0=None, order_indices=np.arange(75), path_fig = Path('.')):
    
    corrRV = np.arange(config_dict['RV_range'][0], config_dict['RV_range'][1], config_dict['RV_step'])
    transit = all_reductions[(config_dict['n_pc'][0], mask_tellu, mask_wings)]['1']
    Kp_array = np.array([transit.Kp.value])

    ccf_maps_in = [all_ccf_map[(n_pc, mask_tellu, mask_wings)] for n_pc in config_dict['n_pc']]
    ccf_maps_in = np.concatenate(ccf_maps_in, axis=-2)
    logl_maps_in = [all_logl_map[(n_pc, mask_tellu, mask_wings)] for n_pc in config_dict['n_pc']]
    logl_maps_in = np.concatenate(logl_maps_in, axis=-2)

    out_filename = f'_{mol}_maskwings{mask_wings*100:n}_masktellu{mask_tellu*100:n}'

    ccf_obj, logl_obj = cc.plot_ccflogl(transit, ccf_maps_in, logl_maps_in, corrRV,
                                        Kp_array, config_dict['n_pc'], id_pc0=id_pc0, orders=order_indices, 
                                        path_fig = path_fig, map = True, fig_name = out_filename)
    
    return ccf_obj, logl_obj
