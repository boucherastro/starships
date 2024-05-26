from sys import path
# path.append('/home/adb/PycharmProjects/')

import os
import sys

from pathlib  import Path
import numpy as np
import yaml

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.basicConfig()

import emcee
import starships.spectrum as spectrum
from starships.orbite import rv_theo_t
from starships.mask_tools import interp1d_masked

interp1d_masked.iprint = False
import starships.correlation as corr
from starships.analysis import bands, resamp_model
import starships.planet_obs as pl_obs
from starships.planet_obs import Observations, Planet
import starships.petitradtrans_utils as prt
from starships.homemade import unpack_kwargs_from_command_line
# from starships import retrieval_utils as ru

from starships.instruments import load_instrum

import astropy.units as u
import astropy.constants as const
from astropy.table import Table

from scipy.interpolate import interp1d

import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

import gc

# from petitRADTRANS import nat_cst as nc
try:
    from petitRADTRANS.physics import guillot_global, guillot_modif
except ModuleNotFoundError:
    from petitRADTRANS.nat_cst import guillot_global, guillot_modif

# functions
def init_model_retrieval(theta_dict, mol_species=None, kind_res='high', lbl_opacity_sampling=None,
                         wl_range=None, continuum_species=None, pressures=None, **kwargs):
    """
    Initialize some objects needed for model: atmo, species, star_fct, pressures
    :param mol_species: list of species included (without continuum opacities)
    :param kind_res: str, 'high' or 'low'
    :param lbl_opacity_sampling: ?
    :param wl_range: wavelength range (2 elements tuple or list, or None)
    :param continuum_species: list of continuum opacities, H and He excluded
    :param pressures: pressure array. Default is `default_params['pressures']`
    :param kwargs: other kwargs passed to `starships.petitradtrans_utils.select_mol_list()`
    :return: atmos, species, star_fct, pressure array
    """

    if mol_species is None:
        mol_species = theta_dict['line_opacities']

    if lbl_opacity_sampling is None:
        lbl_opacity_sampling = theta_dict['opacity_sampling']

    if continuum_species is None:
        continuum_species = theta_dict['continuum_opacities']

    # if pressures is None:
    #     pressures = default_params['pressures']

    species = prt.select_mol_list(mol_species, kind_res=kind_res, **kwargs)
    species_2_lnlst = {mol: lnlst for mol, lnlst in zip(mol_species, species)}

    if kind_res == 'high':
        mode = 'lbl'
        instrum = load_instrum('NIRPS-APERO')
        Raf = instrum['resol']
        pix_per_elem = 2
        if wl_range is None:
            wl_range = instrum['high_res_wv_lim']

    elif kind_res == 'low':
        mode = 'c-k'
        Raf = 1000
        pix_per_elem = 1
        if wl_range is None:
            wl_range = low_res_wv_lim
    else:
        raise ValueError(f'`kind_res` = {kind_res} not valid. Choose between high or low')


    atmo, _ = prt.gen_atm_all([*species.keys()], pressures, mode=mode,
                                      lbl_opacity_sampling=lbl_opacity_sampling, wl_range=wl_range,
                                      continuum_opacities=continuum_species)

    # --- downgrading the star spectrum to the wanted resolution
    if theta_dict['kind_trans'] == 'emission' and star_wv is not None:
        resamp_star = np.ma.masked_invalid(
            resamp_model(star_wv[(star_wv >= wl_range[0] - 0.1) & (star_wv <= wl_range[1] + 0.1)],
                         star_flux[(star_wv >= wl_range[0] - 0.1) & (star_wv <= wl_range[1] + 0.1)], 500000, Raf=Raf,
                         pix_per_elem=pix_per_elem))
        fct_star = interp1d(star_wv[(star_wv >= wl_range[0] - 0.1) & (star_wv <= wl_range[1] + 0.1)],
                                     resamp_star)
    else:
        fct_star = None

    return atmo, species_2_lnlst, fct_star

def unpack_theta(theta):
    """Unpack the theta array into a list of dictionnary with the parameter names as keys.
    Also add other values needed for the model.
    Return a list of dictionnary with all values needed for the model.
    Why a list of dict?  Because it can account for multiple regions
    (so different values for some parameters).
    """
    
    # Get the parameters and unpack them in a dictionnary.
    theta_dict = {key: val for key, val in zip(params_prior.keys(), theta)}
    
    # Convert from log to linear scale if needed.
    for key, prior_info in params_prior.items():
        if prior_info[0] == 'log_uniform':
            theta_dict[key] = 10 ** theta_dict[key]
    
    # Make a dictionnary for each region if needed.
    dict_list = list()
    for i_reg in region_id:
        # Create a dictionnary for each region and remove the region number from the key.
        theta_region = {key: theta_dict[key] for key in global_params}
        if n_regions > 1:
            for key in reg_params:
                key_reg = f'{key}_{i_reg}'
                try:
                    theta_region[key] = theta_dict.pop(key_reg)
                except KeyError:
                    theta_region[key] = reg_fixed_params[key_reg]
        else:
            theta_region.update(theta_dict)

        # Create a dictionnary with all values needed for the model.
        # The values are either taken from theta_region in priority or from default_params.
        combined_dict = {**default_params, **theta_region}

        # gravity depends on Rp if included in the fit
        if 'R_pl' in theta_region:
            combined_dict['gravity'] = (const.G * planet.M_pl /
                                        (theta_region['R_pl'] * const.R_jup) ** 2).cgs.value
            
        # Some values need to be set to None if not included in the fit or not in default_params.
        for key in ['wind', 'p_cloud', 'gamma_scat', 'scat_factor', 'C/O', 'Fe/H']:
            if key not in combined_dict:
                combined_dict[key] = None
            
        # --- Generating the temperature profile
        if kind_temp == "modif":
            fct_inputs = ['pressures', 'tp_delta', 'tp_gamma', 'T_int', 'T_eq', 'ptrans', 'tp_alpha']
            args = (combined_dict[key] for key in fct_inputs)
            combined_dict['temperatures'] = guillot_modif(*args)
        elif kind_temp == 'iso':
            combined_dict['temperatures'] = combined_dict['T_eq'] * np.ones_like(combined_dict['pressures'])
        elif kind_temp == 'guillot':
            fct_inputs = ['pressures', 'kappa_IR', 'tp_gamma', 'gravity', 'T_int', 'T_eq']
            args = (combined_dict[key] for key in fct_inputs)
            combined_dict['temperatures'] = guillot_global(*args)
        else:
            raise ValueError(f'`kind_temp` = {kind_temp} not valid. Choose between guillot, modif or iso')
        
        # Convert some values to cgs units if not done already
        combined_dict['R_pl'] = combined_dict['R_pl'] * const.R_jup.cgs.value
        combined_dict['R_star'] = combined_dict['R_star'] * const.R_sun.cgs.value
        
        dict_list.append(combined_dict)
    
    
    return dict_list

def prepare_abundances(theta_dict, mode=None, ref_linelists=None):
    """Use the correct linelist name associated to the species."""

    if ref_linelists is None:
        if mode is None:
            ref_linelists = theta_dict['line_opacities'].copy()
        else:
            ref_linelists = [linelist_names[mode][mol] for mol in theta_dict['line_opacities']]

    # --- Prepare the abundances (with the correct linelist name for species)
    species = {lnlst: theta_dict[mol] for lnlst, mol
               in zip(ref_linelists[mode], theta_dict['line_opacities'])}
    
    # --- Adding continuum opacities
    for mol in theta_dict['continuum_opacities']:
        species[mol] = mol
        
    # --- Adding other species
    for mol in theta_dict['other_species']:
        species[mol] = mol

    return species

def prepare_model_high_or_low(theta_dict, mode, instrument, planet, atmo_obj=None, fct_star=None,
                              species_dict=None, Raf=None, rot_ker=None):
    '''
    '''
        
    theta_dict['pressures'] = np.logspace(theta_dict['limP'][0], theta_dict['limP'][1], theta_dict['n_pts'])
    theta_dict['gravity'] =  1000 #(const.G * planet.M_pl / (planet.R_pl * const.R_jup) ** 2).cgs.value
    
    # NEED TO INTEGRATE THE DIFFERENT TEMPERATURE MODELS, FOR NOW ONLY ISOTHERMAL
    theta_dict['temperatures'] = 1200. * np.ones_like(theta_dict['pressures'])
    print(theta_dict['temperatures'])
    print(type(theta_dict['temperatures']))
    print(theta_dict['temperatures'].dtype)
    
    fct_star_global = {'high': None, 'low': None}
    
    linelist_names = {'high': None, 'low': None}
    
    if theta_dict['chemical_equilibrium'] == 'True':
        for mol in (theta_dict['line_opacities']):
            theta_dict[mol] = 10 ** (-99.0)
    
    
    if Raf is None:
        instrum = load_instrum('NIRPS-APERO')
        Raf = instrum['resol']
    
    if atmo_obj is None:
        # Use atmo object in globals parameters if it exists
        # atmo_obj = atmo_high if mode == 'high' else atmo_low
        # Initiate if not done yet
        if atmo_obj is None:
            log.info(f'Model not initialized for mode = {mode}. Starting initialization...')
            output = init_model_retrieval(theta_dict, kind_res=mode)
            log.info('Saving values in `linelist_names`.')
            atmo_obj, lnlst_names, fct_star_global[mode] = output
            # Update the values of the global variables
            # Need to use globals() otherwise an error is raised.
            if mode == 'high':
                globals()['atmo_high'] = atmo_obj
            else:
                globals()['atmo_low'] = atmo_obj
                
            # Update the line list names
            if linelist_names[mode] is None:
                linelist_names[mode] = lnlst_names
            else:
                # Keep the predefined values and complete with the new ones
                linelist_names[mode] = {**lnlst_names, **linelist_names[mode]}

    if fct_star is None:
        fct_star = fct_star_global[mode]

    # --- Prepare the abundances (with the correct name for species)
    # Note that if species is None (not specified), `linelist_names[mode]` will be used inside `prepare_abundances`.
    species = prepare_abundances(theta_dict, mode, ref_linelists = linelist_names)

    # --- Generating the model
    args = [theta_dict[key] for key in ['pressures', 'temperatures', 'gravity', 'P0', 'p_cloud']]
    args.append(planet.R_pl.cgs.value[0])
    args.append(planet.R_star.cgs.value[0])
    kwargs = dict(gamma_scat=theta_dict['gamma_scat'],
                  kappa_factor=theta_dict['scat_factor'],
                  C_to_O=theta_dict['C/O'],
                    Fe_to_H=theta_dict['Fe/H'],
                    specie_2_lnlst=linelist_names[mode],
                    kind_trans=theta_dict['kind_trans'],
                    dissociation=theta_dict['dissociation'],
                    fct_star=theta_dict['fct_star'])
    print(kwargs)
    wv_out, model_out = prt.retrieval_model_plain(atmo_obj, species, planet, *args, **kwargs)

    if mode == 'high':
        # --- Downgrading and broadening the model (if broadening is included)
        # if np.isfinite(model_out[100:-100]).all():
            # Get wind broadening parameters
            if theta_dict['wind'] is not None:
                rot_kwargs = {'rot_params': [theta_dict['R_pl'] * const.R_jup,
                                             theta_dict['M_pl'],
                                             theta_dict['T_eq'] * u.K,
                                             [theta_dict['wind']]],
                                'gauss': True, 'x0': 0,
                                'fwhm': theta_dict['wind'] * 1e3, }
            else:
                rot_kwargs = {'rot_params': None}
            
            # Downgrade the model
            wv_out, model_out = prt.prepare_model(wv_out, model_out, lbl_res, Raf=Raf,
                                                  rot_ker=rot_ker, **rot_kwargs)

    return wv_out, model_out

if __name__ == '__main__':


    # initialisation
    pl_name = 'KELT-20 b'
    # Name used for filenames
    pl_name_fname = '_'.join(pl_name.split())

    # - Chemical equilibrium?
    chemical_equilibrium = False
    # - Include dissociation in abundance profiles? (not used in chemical equilibrium)
    dissociation = True
    # -- which kind of TP profile you want? (guillot, modif, iso)
    kind_temp = 'modif'

    # -- Name of the run (will be used to save the results)
    run_name = 'citrus_day1-2_pc6_modif_H2O'

    # -- Use retrieval parameters from a yaml file?
    use_yaml_file = False
    # params_file_in = None  # if None, params_file_in = params_file_out

    # - Types of models : in emission or transmission
    kind_trans = 'emission'

    # --- Data parameters
    pl_kwargs = {'M_star': 1.89 *u.M_sun,
                'R_star': 1.60 *u.R_sun,
                'M_pl' : 3.38 * u.M_jup,
                'R_pl' : 1.83 * u.R_jup,
                }
    obs = Observations(name=pl_name, pl_kwargs=pl_kwargs)

    p = obs.planet

    # --- the planet gravity and H must be changed if you change Rp and/or Tp
    p.gp = const.G * p.M_pl / p.R_pl**2
    p.H = (const.k_B * p.Tp / (p.mu * p.gp)).decompose()

    planet = p

    # --- For emission spectra, you need a star spectrum
    # spec_star = spectrum.PHOENIXspectrum(Teff=7400, logg=4.5)
    # spec_star = np.load('/home/adb/Models/RotationSpectra/phoenix_teff_07400_logg_4.50_Z0.0_RotKer_vsini50.npz')

    star_wv = None  # spec_star['grid']
    # star_wv = (spec_star['wave']).to(u.um).value
    star_flux = None   #spec_star['flux']  # .to(u.erg / u.cm**2 /u.s /u.cm).value
    # --------------

    # - Selecting the wanted species contributing to the opacities:
    line_opacities = ['CO', 'H2O']

    # - Adding continuum opacities:
    continuum_opacities = []

    # - Other species to add to the model (ex: e- and H needed for H- opacity)):
    other_species = []

    # all_species = line_opacities + continuum_opacities + other_species  # was not used

    # Associate each species to a linelist name
    # if set to None, prt.select_mol_list will use the default name for each type (high or low res)
    linelist_names = {'high': None, 'low': None}

    ## --- HIGH RES DATA ---

    # --- IF HRR OR JR, YOU NEED TO INCLUDE HIGH RES DATA -----
    # - Data resolution and wavelength limit for the instrument
    instrum = load_instrum('NIRPS-APERO')
    high_res_wv_lim = instrum['high_res_wv_lim']

    # - Which sequences are taken
    do_tr = [1]

    # - Selecting bad exposures if wanted/needed  (not implemented yet)
    # bad_indexs = None

    ## --- Additionnal global variables
    plot = False
    nolog = True
    inj_alpha = 'ones'
    orders = np.arange(49) #np.array([14,15,30,31,46,47])
    opacity_sampling = 4  # downsampling of the petitradtrans R = 1e6, ex.: o_s = 4 -> R=250000
    lbl_res = 1e6 / opacity_sampling  # Resolution of the lbl model

    #################
    # Parameters and Priors
    #################

    # --- Default parameters needed for the retrieval, but not necessarily fitted
    default_params = dict()
    default_params['M_pl'] = p.M_pl.to(u.M_jup).value  # In jupiter mass
    default_params['R_pl'] = planet.R_pl[0].to(u.R_jup).value  # In jupiter radius
    default_params['R_star'] = p.R_star.to(u.R_sun).value  # In solar radius
    default_params['P0'] = 1e-3  # -- reference pressure fixed (for transmission spectra)
    default_params['log_f'] = 0.0  # -- scaling factor for WFC3
    default_params['spec_scale'] = 1.0  # -- scaling factor for the Fp/Fs spectrum

    # --- For the abundances
    default_params['e-'] = 10 ** (-6.0)
    default_params['H'] = 10 ** (-99.0)
    # Add species used for opacities if chemical equilibrium
    if chemical_equilibrium:
        for mol in (line_opacities + continuum_opacities):
            default_params[mol] = 10 ** (-99.0)

    # --- For both guillot profiles
    default_params['T_eq'] = 2500.  # Use for isothermal profile as well
    default_params['tp_gamma'] = 10 ** (1)  # 0.01
    default_params['T_int'] = 500.
    # --- For the basic guillot profile (guillot)
    default_params['kappa_IR'] = 10 ** (-2)
    default_params['gravity'] = p.gp.cgs.value
    # --- For the modified guillot profile (guillot_modif)
    default_params['tp_delta'] = 10 ** (-7.0)
    default_params['ptrans'] = 10 ** (-3)
    default_params['tp_alpha'] = 0.3
    # --- For the Akima Spline TP profile
    for iP in range(1, 4):
        # The pressure nodes values can be fitted or fixed
        default_params[f'P{iP}'] = 10 ** (-iP)  # bar
        # Same for the temperature nodes
        default_params[f'T{iP}'] = 1000.  # K

    # - Generating the basis of the temperature profile
    limP=(-10, 2)  # pressure log range
    n_pts=50  # pressure n_points
    default_params['pressures'] = np.logspace(*limP, n_pts)  # pressures in bars
        

    #####################################################
    # --- Define priors ---
    #####################################################
    # Default priors are uniform, log_unigform or gaussian.
    # Other prior types can be added in the prior function dictionnary.
    # NOTE: The difference between uniform and log_uniform is that the values are
    # converted to log scale inside the logl call for log_uniform.
    # Ex:
    # >>> params_prior['CO'] = ['log_uniform', -12.0, -0.5]
    # will give in the logl function:
    # >>> theta_dict['CO'] = 10 ** theta_dict['CO']

    params_prior = {}
    # - Abundances priors - #
    # By default, if chemical equilibrium is True, the molecular abundances depend on C/O and Fe/H
    if chemical_equilibrium:
        # Valid range for PRT: C/O = [0.1, 1.6] and [Fe/H] = [-2, 3]
        params_prior['C/O'] = ['uniform', 0.1, 1.6]
        params_prior['Fe/H'] = ['uniform', -2, 3]
    else:
        # Same prior for all molecules in the list of opacity species
        for mol in line_opacities:
            params_prior[mol] = ['log_uniform', -12.0, -0.5]
    # NOTE: Some molecules are not in the list of opacities, but need to be include in the fit (ex: 'e-')
    # You can add a prior for a specific molecule by adding a line like this:
    # params_prior['e-'] = ['log_uniform', -12.0, -0.5]

    # - Other parameters priors - #
    params_prior['T_eq'] = ['uniform', 100, 4500]
    # params_prior['p_cloud'] = ['log_uniform', -5.0, 2]
    # params_prior['R_pl'] = ['uniform', 1.0, 1.6]  # In jupiter radius
    params_prior['kp'] = ['uniform',100, 250]
    params_prior['rv'] = ['uniform', -40, 40]
    # params_prior['wind'] = ['uniform', 0, 20]
    # params_prior['kappa_IR'] = ['log_uniform', -3, 3]  # Guillot
    params_prior['tp_gamma'] = ['log_uniform', -2, 6]  # Guillot & Guillot_modif
    params_prior['tp_delta'] = ['log_uniform', -8, -3]  # Guillot_modif
    params_prior['ptrans'] = ['log_uniform', -8, 3]  # Guillot_modif
    params_prior['tp_alpha'] = ['uniform', -1.0, 1.0]  # Guillot_modif
    # params_prior['gamma_scat'] = ['uniform', -6, 0]
    # params_prior['scat_factor'] = ['uniform', 0, 10]   # * (5.31e-31*u.m**2/u.u).cgs.value
    # params_prior['log_f'] = ['uniform', -2, 3]  # yes, it is a uniform prior, not a log_uniform
    params_prior['phase1'] = ['uniform', 0.2 , 0.8]
    # params_prior['phase2'] = ['uniform', 0.67, 1.0]
    params_prior['rot_factor'] = ['uniform', 0.5, 4]
    # params_prior['spec_scale'] = ['log_uniform', -2, 3]

    '''                                  Handling Multiple Regions                                   '''
    # If the planet is split in multiple regions, specify which paremeters are specific to each region.
    # Each region is identified with an ID starting at 1 (not 0).
    region_id = [1, 2]  # list of region ids (usefull if not all regions are contributing to the planet flux)
    n_regions = len(region_id)
    reg_params = ['T_eq', 'tp_gamma', 'tp_delta', 'ptrans', 'tp_alpha']
    # global parameters are all other parameters not in reg_params
    global_params = [param for param in params_prior.keys() if param not in reg_params]

    # It is possible to fix some parameters in specific regions.
    reg_fixed_params = {}
    # reg_fixed_params['H2O_2'] = 1e-12  # Force water abundance to ~zero in region 2

    # Assign specific index to each regional parameter (if not already done manually)
    if n_regions > 1:
        for param in reg_params:
            # Add the parameter for each region if not already specified
            for i_reg in region_id:
                key = f'{param}_{i_reg}'
                if (key not in params_prior) and (key not in reg_fixed_params):
                    params_prior[f'{param}_{i_reg}'] = params_prior[param]
                else:
                    log.debug(f'Parameter {key} already in params_prior or reg_fixed_params. Skipping.')
            # Remove the global parameter
            params_prior.pop(param)


    ####################################################
    # hig vs low res obj
    ####################################################

    # Define low res wavelength range (might not be used)
    lower_wave_lim = high_res_wv_lim[0]
    upper_wave_lim = high_res_wv_lim[1]

    low_res_wv_lim = (lower_wave_lim, upper_wave_lim)

    log.debug(f"Low res wavelength range: {low_res_wv_lim}")

    # Dictionnaries is not the best way for multiprocessing because it is not shared between processes.
    # Better to use global variables for big arrays or atmo objects. Use dictionaries only for small objects.
    # Define global dictionaries where to put the model infos so they will be shared in functions.
    fct_star_global = {'high': None, 'low': None}

    atmo_high = None
    atmo_low = None

    # ####################################################
