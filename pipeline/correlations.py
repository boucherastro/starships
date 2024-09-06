
import numpy as np
from pathlib import Path

from starships.correlation import quick_correl
from starships.correlation_class import Correlations
import starships.correlation as correl
import starships.planet_obs as pl_obs

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
                        get_logl=False, kind='BL', mod2d=False, expand_mask=0, noise=None, somme=False, counting = False)

    # Create the object do make the plots and compute the different Kp

    corr_obj = Correlations(ccf, kind='BL', rv_grid=corrRV, kp_array=transit.Kp)
    corr_obj.calc_ccf(orders=None, N=transit.N_frac[None, :, None], alpha=np.ones_like(transit.alpha_frac), index=None, ccf0=None, rm_vert_mean=False,)
    corr_obj.calc_correl_snr_2d(transit, plot=False, counting = False)
    corr_obj.RV_shift = np.zeros_like(transit.alpha_frac)

    # Make the plots and save them

    corr_obj.full_plot(transit, [], save_fig = f'classic_ccf{nametag}', path_fig = str(path_fig) + '/')

    return corr_obj


def inj_ccf(config_dict, transit, wave_mod, mod_spec, n_pc, scratch_dir, nametag, corrRV = []):
    if len(corrRV) == 0:
        corrRV = np.arange(config_dict['RV_range'][0], config_dict['RV_range'][1], config_dict['RV_step'])
    
    Kp_array = np.array([transit.Kp.value])
    print('Computing CCF')
    ccf_map, logl_map = correl.quick_calc_logl_injred_class(transit, Kp_array, corrRV, n_pc, 
                                                    wave_mod, np.array([mod_spec]), nolog=True, 
                                                    inj_alpha='ones', RVconst=transit.RV_const, counting = False)
    print('Done!')
    corr.save_logl_seq(scratch_dir / Path(f'inj_ccf_logl_seq{nametag}'), ccf_map, logl_map,
                           wave_mod, mod_spec, n_pc, Kp_array, corrRV, config_dict['kind_trans'])

    return ccf_map, logl_map


def perform_ccf(config_dict, transit, mol, wave_mod, mod_spec, n_pc, mask_tellu, mask_wings, scratch_dir, path_fig, visit_name, corrRV = []):

    nametag = f'_{visit_name}_{mol}_maskwings{mask_wings*100:n}_masktellu{mask_tellu*100:n}_pc{n_pc}'
    
    # changed to be put directly into the pipeline
    # corr_obj = classic_ccf(config_dict, transit, wave_mod, mod_spec, path_fig, nametag) 

    try:
        # Check if ccf for individual visit was already generated
        out_filename = f'inj_ccf_logl_seq{nametag}'
        saved_values = np.load(scratch_dir / Path(out_filename).with_suffix('.npz'))
        ccf_map = saved_values['corr']
        logl_map = saved_values['logl']
        print(f'CCF already exists for {nametag}. Loading...')
    except FileNotFoundError: 
        ccf_map, logl_map = inj_ccf(config_dict, transit, wave_mod, mod_spec, [n_pc], scratch_dir, nametag) 

    return ccf_map, logl_map


def plot_all_ccf(config_dict, mol, mask_tellu, mask_wings, scratch_dir, visit_name, planet, id_pc0=None, order_indices=np.arange(75), path_fig = Path('.')):
    
    corrRV = np.arange(config_dict['RV_range'][0], config_dict['RV_range'][1], config_dict['RV_step'])

    n_pc = config_dict['n_pc'][0]
    fname = f'retrieval_input_{visit_name}_maskwings{mask_wings*100:n}_masktellu{mask_tellu*100:n}_pc{n_pc}'
    transit = pl_obs.load_single_sequences(fname, planet.name, path=scratch_dir,
                                load_all=False, filename_end='', plot=False, planet=planet)
    
    Kp_array = np.array([transit.Kp.value])

    ccf_maps_in = []
    logl_maps_in = []
    for n_pc in  config_dict['n_pc']:
        out_filename = f'inj_ccf_logl_seq_{visit_name}_{mol}_maskwings{mask_wings*100:n}_masktellu{mask_tellu*100:n}_pc{n_pc}'
        saved_values = np.load(scratch_dir / Path(out_filename).with_suffix('.npz'))
        ccf_maps_in.append(saved_values['corr'])
        logl_maps_in.append(saved_values['logl'])

    ccf_maps_in = np.concatenate(ccf_maps_in, axis=-2)
    logl_maps_in = np.concatenate(logl_maps_in, axis=-2)

    out_filename = f'_{mol}_maskwings{mask_wings*100:n}_masktellu{mask_tellu*100:n}'

    ccf_obj, logl_obj = cc.plot_ccflogl(transit, ccf_maps_in, logl_maps_in, corrRV,
                                        Kp_array, config_dict['n_pc'], id_pc0=id_pc0, orders=order_indices, 
                                        path_fig = path_fig, map = True, fig_name = out_filename)
    
    return ccf_obj, logl_obj

def plot_all_maskwings(config_dict, planet, mol, mask_tellu, n_pc, scratch_dir, visit_name, id_pc0=None, order_indices=np.arange(75), path_fig = Path('.')):

    corrRV = np.arange(config_dict['RV_range'][0], config_dict['RV_range'][1], config_dict['RV_step'])

    mask_wings = config_dict['mask_wings'][0]
    fname = f'retrieval_input_{visit_name}_maskwings{mask_wings*100:n}_masktellu{mask_tellu*100:n}_pc{n_pc}'
    transit = pl_obs.load_single_sequences(fname, planet.name, path=scratch_dir,
                                load_all=False, filename_end='', plot=False, planet=planet)
    
    Kp_array = np.array([transit.Kp.value])

    # load existing ccf maps for each mask_wings
    ccf_maps_in = []
    logl_maps_in = []

    for mask_wings in config_dict['mask_wings']:
        out_filename = f'inj_ccf_logl_seq_{visit_name}_{mol}_maskwings{mask_wings*100:n}_masktellu{mask_tellu*100:n}_pc{n_pc}'
        saved_values = np.load(scratch_dir / Path(out_filename).with_suffix('.npz'))
        ccf_maps_in.append(saved_values['corr'])
        logl_maps_in.append(saved_values['logl'])

    ccf_maps_in = np.concatenate(ccf_maps_in, axis=-1)
    logl_maps_in = np.concatenate(logl_maps_in, axis=-1)

    out_filename = f'_{mol}_masktellu{mask_tellu*100:n}_pc{n_pc}'

    # CHANGE THIS
    ccf_obj, logl_obj = cc.plot_ccflogl(transit, ccf_maps_in, logl_maps_in, corrRV,
                                        Kp_array, config_dict['mask_wings'], swapaxes=(-2, -1), 
                                        fig_name = out_filename, param = 'mask_wings', path_fig = path_fig, 
                                        plot_prf = False, id_pc0 = None)
    return ccf_obj, logl_obj

def plot_all_masktellu(config_dict, planet, mol, mask_wings, n_pc, scratch_dir, visit_name, id_pc0=None, order_indices=np.arange(75), path_fig = Path('.')):

    corrRV = np.arange(config_dict['RV_range'][0], config_dict['RV_range'][1], config_dict['RV_step'])

    # load in a transit to get planet/obs values, doesn't matter which values you use here
    mask_tellu = config_dict['mask_tellu'][0]
    fname = f'retrieval_input_{visit_name}_maskwings{mask_wings*100:n}_masktellu{mask_tellu*100:n}_pc{n_pc}'
    transit = pl_obs.load_single_sequences(fname, planet.name, path=scratch_dir,
                                load_all=False, filename_end='', plot=False, planet=planet)

    Kp_array = np.array([transit.Kp.value])

    # load existing ccf maps for each mask_tellu
    ccf_maps_in = []
    logl_maps_in = []

    for mask_tellu in config_dict['mask_tellu']:
        out_filename = f'inj_ccf_logl_seq_{visit_name}_{mol}_maskwings{mask_wings*100:n}_masktellu{mask_tellu*100:n}_pc{n_pc}'
        saved_values = np.load(scratch_dir / Path(out_filename).with_suffix('.npz'))
        ccf_maps_in.append(saved_values['corr'])
        logl_maps_in.append(saved_values['logl'])

    ccf_maps_in = np.concatenate(ccf_maps_in, axis=-1)
    logl_maps_in = np.concatenate(logl_maps_in, axis=-1)

    out_filename = f'_{mol}_maskwings{mask_wings*100:n}_pc{n_pc}'

    ccf_obj, logl_obj = cc.plot_ccflogl(transit, ccf_maps_in, logl_maps_in, corrRV,
                                        Kp_array, config_dict['mask_tellu'], swapaxes=(-2, -1), 
                                        fig_name = out_filename, param = 'mask_tellu', path_fig = path_fig, 
                                        plot_prf = False, id_pc0 = None)
    return ccf_obj, logl_obj
    
def combined_visits_ccf(planet, mol, wave_mod, mod_spec, dir_dict, config_dict, order_indices=np.arange(75)):
    mask_tellu, mask_wings, n_pc = config_dict['night_params'][:3]
    combined_ccf = []
    combined_logl = []
    combined_obs = []
    visit_dict = {}

    # getting all the reductions to combine
    filename_dict = {}
    for idx, visit_name in enumerate(config_dict['visit_name']):
        file_name = f'retrieval_input_{visit_name}_maskwings{mask_wings*100:n}_masktellu{mask_tellu*100:n}_pc{n_pc}'
        filename_dict[(str(idx+1), visit_name)] = file_name
    
    for key, fname in filename_dict.items():
        idx = key[0]
        visit_name = key[1]

        # load existing reductions for each visit
        transit = pl_obs.load_single_sequences(fname, planet.name, path=dir_dict['scratch_dir'],
                                load_all=False, filename_end='', plot=False, planet=planet)
        
        out_filename = f'inj_ccf_logl_seq_{visit_name}_{mol}_maskwings{mask_wings*100:n}_masktellu{mask_tellu*100:n}_pc{n_pc}'

        corrRV = np.arange(config_dict['RV_range'][0], config_dict['RV_range'][1], config_dict['RV_step'])
        Kp_array = np.array([transit.Kp.value])

        try:
            # Check if ccf for individual visit was already generated
            saved_values = np.load(dir_dict['scratch_dir'] / Path(out_filename).with_suffix('.npz'))
            ccf_map = saved_values['corr']
            logl_map = saved_values['logl']
        except FileNotFoundError:
            # if not, generate the ccf for each visit and save it
            print('Computing CCF')
            ccf_map, logl_map = correl.quick_calc_logl_injred_class(transit, Kp_array, corrRV, [n_pc], 
                                                    wave_mod, np.array([mod_spec]), nolog=True, 
                                                    inj_alpha='ones', RVconst=transit.RV_const, counting = False)
            print('Done!')
            corr.save_logl_seq(dir_dict['scratch_dir'] / Path(out_filename), ccf_map, logl_map,
                            wave_mod, mod_spec, n_pc, Kp_array, corrRV, config_dict['kind_trans'])

        visit_dict[idx] = transit
        combined_obs.append(transit)
        combined_ccf.append(ccf_map)
        combined_logl.append(logl_map)

    transit_tags = [np.arange(transit.n_spec) for transit in visit_dict.values()]
    all_visits = pl_obs.gen_merge_obs_sequence(transit, visit_dict, list(np.arange(len(transit_tags)) + 1), None, config_dict['coeffs'], 
                                           config_dict['ld_model'], config_dict['kind_trans'], light=True) 
    
    ccf_maps_in = np.concatenate(combined_ccf)
    logl_maps_in = np.concatenate(combined_logl)

    # do combined ccf
    out_filename = f'_combined_{mol}_maskwings{mask_wings*100:n}_masktellu{mask_tellu*100:n}_pc{n_pc}'

    split_fig = []
    height = 0
    for o in combined_obs:
        split_fig.append(height)
        height += o.n_spec
    split_fig.append(height)
    
    ccf_obj, logl_obj = cc.plot_ccflogl(all_visits, ccf_maps_in, logl_maps_in, corrRV,
                                        Kp_array, config_dict['n_pc'], orders=order_indices, 
                                        split_fig = split_fig, path_fig = dir_dict['injected_ccf_dir'], fig_name = out_filename, map = True)

    return ccf_obj, logl_obj

