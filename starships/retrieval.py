# -----------------------------------------------------------
# ##################################################################
# ############# New version of the retrieval that uses the yaml file
# ##################################################################
# -----------------------------------------------------------

# Do imports when needed

import sys

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import logging
import numpy as np

import scipy
from scipy.interpolate import interp1d

from astropy import constants as const
from astropy import units as u
from astropy.table import Table


import emcee
import starships
import starships.spectrum as spectrum
from starships.orbite import rv_theo_t
from starships.mask_tools import interp1d_masked


# %%
interp1d_masked.iprint = False
import starships.correlation as corr
from starships.analysis import bands
from starships.convolution import degrade_and_resample
import starships.planet_obs as pl_obs
from starships.planet_obs import Observations, Planet
import starships.petitradtrans_utils as prt
import starships.model_sequence as model_seq
from starships.logl_grid import _chi2_from_terms, _logl_from_chi2_terms
from starships.homemade import unpack_kwargs_from_command_line, pop_kwargs_with_message, calc_shift
from starships import retrieval_utils as ru
from starships.retrieval_inputs import convert_cmd_line_to_types

from starships.instruments import load_instrum


import astropy.units as u
import astropy.constants as const
from astropy.table import Table




from multiprocessing import Pool

import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

import gc

# from petitRADTRANS import nat_cst as nc
try:
    from petitRADTRANS.physics import guillot_global, guillot_modif
except ModuleNotFoundError:
    try:
        from petitRADTRANS.nat_cst import guillot_global, guillot_modif
    except ModuleNotFoundError:
        print('petitRADTRANS is not installed on this system')


# other newly implemented TP profiles
from starships.extra_TP_profiles import madhu_seager


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Here is a list of all the global parameters that are used in the code
# This is done to optimize the multiprocessing for two main reasons:
# - Avoid passing big arguments to functions makes a huge difference in speed
# - Avoid loading the same data multiple times.
#   Indeed, when multiprocessing in python, most variable are copied as many times
#   as there are processes (so that can become a lot of memory).
#   There is a hack with numpy arrays, if they are defined in the global space.
# Finally, there is also the reason that it becomes easier to analyse the results
# of the retrievals by setting some variables as globals. Then, the retrieval code
# can be imported inside a notebook or another code and used like an object, with
# attributes and methods to reproduce the spectra or TP profiles for example.
global pl_name
global base_dir
global high_res_path
global reduc_name
global high_res_file_stem_list
global spectrophotometric_data
global photometric_data
global retrieval_type
global white_light
global chemical_equilibrium
global dissociation
global kind_temp
global n_steps_burnin
global n_steps_sampling
global run_name
global walker_path
global walker_file_out
global walker_file_in
global init_mode
global slurm_array_behaviour
global params_path
global params_file_out
global kind_trans
global n_cpu
global n_walkers
global n_walkers_per_cpu
global opacity_sampling
global orders
global pl_params
global instrum
global line_opacities
global continuum_opacities
global other_species
global species_in_prior
global linelist_names
global fixed_params
global params_prior
global region_id
global reg_params
global reg_fixed_params
global custom_prior_file
global special_init
global get_ker_file
global rotation_kernel
global representative_phases_low
global limP
global n_pts
global star_spectrum
# new global variables to be able to remove species at either resolution
global remove_mol_high
global remove_mol_low


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

    if unit is None:
        out = value
    else:
        try:
            # Get if from astropy unit
            out = value * u.Unit(unit)
        except ValueError:
            # Or get it from astropy constant
            out = value * getattr(const, unit)
    
    return out

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
        # Do not include None values in the dictionary
        if value['value'] is not None:
            pl_kwargs[key] = convert_to_quantity(value)

    return pl_kwargs


def get_slurm_id():

    if 'SLURM_ARRAY_JOB_ID' in os.environ:
        job_id_key_list = ['SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID']
    else:
        job_id_key_list = ['SLURM_JOB_ID']

    try:
        # jobid is combining all ID
        jobid = '_'.join([os.environ[key] for key in job_id_key_list])
    except KeyError:
        jobid = None
        log.info('slurm job ID not found.')
            
    return jobid

def get_run_name(input_params):
    """Create a run name from the input parameters.
    For example, the walkers will be saved in base_dir/DataAnalysis/walker_steps/<pl_name>/walker_steps_<run_name>.h5
    The run_name will be define by joining with "_" all the following parameters
    - kind_trans
    - retrieval_type
    - "WL" (if white_light is True)
    - keys in spectrophotometric_data dictionary (ex: wfc3)
    - keys in photometric_data dictionary (ex: spitzer)
    - kind_temp
    - "disso" (if dissociation is True)
    - "chemEq" (if chemical equilibrium is True)
    - the sbatch job ID if available
    """
    
    run_name_args = [input_params[key] for key in ['kind_trans', 'retrieval_type']]
    if input_params['white_light']:
        run_name_args.append('WL')    
    run_name_args += list(input_params['spectrophotometric_data'].keys())
    run_name_args += list(input_params['photometric_data'].keys())
    run_name_args.append(input_params['kind_temp'])
    if input_params['dissociation']:
        run_name_args.append('disso')
    if input_params['chemical_equilibrium']:
        run_name_args.append('chemEq')

    # Take job ID from environment variable if available
    # jobid is combining all ID 
    jobid = get_slurm_id()
    if jobid is not None:
        run_name_args.append(jobid)
    
    run_name = '_'.join(run_name_args)
    
    return run_name


def unpack_input_parameters(input_parameters, **kwargs):
    """ Read input parameters from a yaml file or a dictionary.
    And make sure they are in the right format for the retrieval code.
    For example, put default values if some parameters are None."""
    
    # --- Read the input parameters from a yaml file or a dictionary
    if isinstance(input_parameters, dict):
        input_params = input_parameters
    else:
        with open(input_parameters, 'r') as f:
            input_params = yaml.load(f, Loader=yaml.FullLoader)
            
    # Replace the keys in input_params with the kwargs if they are in kwargs
    for key, val in kwargs.items():
        if key in input_params:
            input_params[key] = val
            log.info(f'Replacing {key} using the kwargs with value = {val}')
        else:
            raise KeyError(f'{key} is not in the expected input parameters.')
    
    ########################################
    # --- Make some checks on the inputs ---
    ########################################
        
    # --- Check for None values that should be empty dictionaries ---
    empty_dict_if_none = ['spectrophotometric_data', 'photometric_data',
                          'pl_params', 'linelist_names', 'fixed_params',
                          'reg_fixed_params', 'reg_params', 'special_init',
                          'remove_mol_high', 'remove_mol_low']
    for key in empty_dict_if_none:
        if input_params.get(key) is None:
            log.info(f'{key} is None. Setting it to an empty dictionary instead.')
            input_params[key] = {}

    # Check if some numbers are in string format and raise a warning if so.
    for key in input_params:
        if isinstance(input_params[key], str) and input_params[key].isnumeric():
            msg = f'{key} is in string format. It should be a number.'
            msg += ' Make sure the exponent format includes the decimal point.'
            msg += ' Ex: 1.0e-3 and not 1e-3'
            log.warning(msg)
            
    # Check that white_light is only used in HRR mode
    if input_params['white_light'] and input_params['retrieval_type'] != 'HRR':
        msg = f"white_light is set to True but the retrieval type is"
        msg += f" '{input_params['retrieval_type']}'. Forcing white_light to False"
        log.warning(msg)
        input_params['white_light'] = False
    
    ####################################
    # --- Check paths and file names ---
    ####################################
    
    # Check all path and file names. Put default values if None.
    # First create the planet name with the spaces replaced by '_'
    pl_fname = input_params['pl_name'].replace(' ', '_')

    # Base directory
    if input_params['base_dir'] is None:
        try:
            base_dir = os.environ['SCRATCH']
            base_dir = Path(base_dir)
        except KeyError:
            base_dir = Path.home()
        input_params['base_dir'] = base_dir
    else:
        base_dir = Path(input_params['base_dir'])  # for later use

    # Check run name
    if input_params['run_name'] is None:
        input_params['run_name'] = get_run_name(input_params)
        
    # Check walker path
    if input_params['walker_path'] is None:
        input_params['walker_path'] = base_dir / Path(f'DataAnalysis/walker_steps/{pl_fname}')
        
    # Check walker file out
    if input_params['walker_file_out'] is None:
        input_params['walker_file_out'] = f'walker_steps_{input_params["run_name"]}.h5'
        log.info(f"No walker file out name given. Setting it to {input_params['walker_file_out']}")
        
    # Check params path
    if input_params['params_path'] is None:
        input_params['params_path'] = base_dir / Path(f'DataAnalysis/retrieval_params/{pl_fname}')
    
    # Check params file out
    if input_params['params_file_out'] is None:
        input_params['params_file_out'] = f'params_{input_params["run_name"]}.yaml'
        
    # Make sure all the file paths are Path objects
    all_file_keys = ['base_dir', 'high_res_path', 'walker_path', 'walker_file_out',
                     'walker_file_in', 'params_path', 'params_file_out', 'star_spectrum',
                     'custom_prior_file']
    for key in all_file_keys:
        if input_params[key] is not None:
            # expanduser() to make sure to replace the '~' in the paths
            input_params[key] = Path(input_params[key]).expanduser()
            
    ####################################
    # --- Other parameters that need to be manipulated ---
    ####################################
    
    # --- Convert the planet parameters to Quantity
    input_params['pl_params'] = pl_param_units(input_params)
    
    # --- Number of walkers and cpus
    if input_params['n_cpu'] is None:
        # Get the number of cpus from the slurm environment variable
        try:
            input_params['n_cpu'] = int(os.environ['SLURM_CPUS_PER_TASK'])
        except KeyError:
            input_params['n_cpu'] = 1

    # If n_walkers is None, set it to n_walkers_per_cpu * n_cpu
    if input_params['n_walkers'] is None:
        input_params['n_walkers'] = input_params['n_walkers_per_cpu'] * input_params['n_cpu']

    # --- Convert orders to array if a list is given
    if input_params['orders'] is not None:
        input_params['orders'] = np.array(input_params['orders'])
        
    # --- Unpack species_in_prior ---
    species_in_prior = []
    for item in input_params['species_in_prior']:
        if item in ['line_opacities', 'continuum_opacities', 'other_species']:
            species_in_prior += input_params[item]
        else:
            species_in_prior.append(item)
    input_params['species_in_prior'] = species_in_prior
    
    # --- instrument needs the same shape as high_res_file_stem_list
    instrum = input_params['instrum']
    if len(instrum) == 1:
        instrum = instrum * len(input_params['high_res_file_stem_list'])
    input_params['instrum'] = instrum

    # --- n_pc needs the same shape as high_res_file_stem_list (B3, Chantier B): PCA
    # truncation now happens at read time instead of being baked into the reduced file, so
    # the retrieval must say explicitly which n_pc to use for each visit -- same broadcasting
    # convention as `instrum` above (one value per file, or a single value for all of them).
    # A KeyError here means the YAML predates B3 and needs an `n_pc:` key added.
    n_pc = input_params['n_pc']
    if len(n_pc) == 1:
        n_pc = n_pc * len(input_params['high_res_file_stem_list'])
    input_params['n_pc'] = n_pc

    return input_params


#  This function needs to be defined in this script to correctly assign the global variables.
def setup_retrieval(input_parameters, **kwargs):
    """ Read input parameters from a yaml file or a dictionary.
    All parameters in the yaml file will be used to update the global variables.
    This is done for efficiency when multiprocessing, to avoid passing arguments
    to functions if not needed.
    All the parameters will be accesible in the global variables, so for example,
    the `params_prior` can be accessed in any function just by typing `params_prior`.
    Also, if you import this retrieval script in another script, you can access the
    global variables in the other script.
    Ex:
    ```
    import retrieval_example as retrieval
    # Read the input parameters from a yaml file
    retrieval.setup_retrieval("input_params.yaml")
    # Access the params_prior variable which was in the input_params.yaml file
    print(retrieval.params_prior)
    ```
    This is useful for post-processing the retrieval results. For example if you
    want to reproduce the spectra for a sample of the posterior, you can import
    this script in another script and run the functions that generate the model.
    
    NOTE: kwargs will replace the value in the input_parameters if the key is the same.
    """
    
    # Unpack the input parameters
    input_params = unpack_input_parameters(input_parameters, **kwargs)

    # --- Update the global variables with the parameters from the yaml file
    for key, val in input_params.items():
        
        # --- Update the global variables with the parameters from the yaml file
        globals()[key] = val
        
        # --- Print the updated global variable
        log.info(f'{key} = {val}')

    ##########################################
    # --- Additionnal variable assignment ---
    ##########################################

    # --- Stellar spectrum ---
    global star_wv, star_flux, star_res
    if star_spectrum is None:
        star_wv = None
        star_flux = None
        star_res = None
    else:
        star_data = np.load(star_spectrum)
        try:
            star_wv = star_data['wave']
        except KeyError:
            # Old format
            star_wv = star_data['grid']
        star_flux = star_data['flux']
        try:
            star_res = star_data['sampling_res']
        except KeyError:
            star_res = 500000
            log.info(f'No sampling resolution found in the stellar spectrum. Using R = {star_res} as default.')
        star_data.close()

    # Init the star fct
    for mode in ['high', 'low']:
        globals()[f'fct_star_{mode}'] = None

    # --- Planet and observation objects ---
    global obs, planet, Kp_scale
    obs = Observations(name=pl_name, pl_kwargs=pl_params)
    planet = obs.planet
    Kp_scale = (planet.M_pl / planet.M_star).decompose().value

    # --- Setup wavelength range ---
    global instrum_param_list, wv_range_high, wv_range_low
    instrum_param_list = [load_instrum(instrum_name) for instrum_name in instrum]

    # Low-res data is loaded first (instead of after wv_range_high, as before
    # Chantier A Phase 4) because building wv_range_high below now needs to know
    # which spectrophotometric instruments, if any, opted into `opacity_mode: 'lbl'`.
    load_low_res_data()
    load_photometry()

    # --- High resolution wavelength range ---
    # Chantier A Phase 4 (c-k vs lbl per instrument, Antoine 2026-09-11): a
    # spectrophotometric instrument can opt into `opacity_mode: 'lbl'` (new
    # per-instrument YAML key under `spectrophotometric_data`, default 'c-k' --
    # unchanged behaviour for every existing config) to be modelled with full
    # lbl (line-by-line) opacities instead of the coarser c-k ones. Rather than
    # inventing a separate low-res "lbl" model (its own atmo object, its own
    # species/linelist selection, its own resolution knob), such an instrument
    # is simply treated as *another high-res instrument*: its wavelength range
    # is folded into wv_range_high below, so it gets its own (or a shared, if
    # overlapping) `atmo_high_i` object, exactly like a real high-res
    # spectrograph's range from `load_instrum`. This lets it flow through the
    # existing `assign_model_type`/`model_type == 'high'` reuse path (Phase 4,
    # earlier this session): full-resolution lbl model generated once, then
    # re-degraded to its own (coarser) resolution by `prepare_spectrophotometry`
    # downstream, same as any other low-res instrument that happens to be fully
    # covered by the high-res data. No YAML key exists for `photometric_data`:
    # whether a given photometric instrument ends up using the lbl-backed model
    # or the default c-k one simply falls out of the same coverage check every
    # other low-res instrument already goes through in `assign_model_type` --
    # lbl if its own wavelength range happens to be fully covered by the (now
    # lbl-extended) wv_range_high, c-k otherwise. `instrum` (hence
    # instrum_param_list) may be completely empty here -- a retrieval can be run
    # on lbl-flagged spectrophotometric data alone, with no real high-res
    # instrument at all (see the res_instru fallback below).
    lbl_spectrophotometric_ranges = get_lbl_spectrophotometric_ranges(spectrophotometric_data)

    wv_range_high = [instrum_param['high_res_wv_lim'] for instrum_param in instrum_param_list]
    wv_range_high += lbl_spectrophotometric_ranges
    wv_range_high = get_wv_range(wv_range_high)
    log.info(f'wavelength range for model at high res: {wv_range_high}')

    # --- Low resolution (c-k) wavelength range ---
    # NOTE: The low-res models are taking less memory, so we model the full range,
    #       even the regions in between where there is no data.
    #       lbl-flagged spectrophotometric instruments are excluded here -- they
    #       are modelled through wv_range_high/atmo_high above instead (see the
    #       comment block above), and will always be assigned `model_type ==
    #       'high'` below since their own range is now always part of wv_range_high.
    wv_range_all_low = [infos['wv_range'] for infos in spectrophotometric_data.values()
                         if infos.get('opacity_mode', 'c-k') != 'lbl']
    wv_range_all_low += [infos['wv_range'] for infos in photometric_data.values()]

    if retrieval_type != 'LRR':
        # In JR or HRR, model at least the full range of the high-res data.
        # It won't necessarily be used, but it is useful for analysis later on.
        # So add it to the list of wv_range
        wv_range_all_low.append([np.min(wv_range_high), np.max(wv_range_high)])

    # The low resolution data will model the full range.
    wv_range_low = get_wv_range_low(wv_range_all_low)
    log.info(f'wavelength range for model at low res: {wv_range_low}')

    # Assign (to each low res dataset) which kind of model (high or low)
    # will be used to compare with the data. Normally JR-only (the low-res
    # data can only reuse the high-res model when a real high-res block also
    # runs) -- but also run whenever an lbl-flagged spectrophotometric
    # instrument is present, regardless of retrieval_type: that instrument has
    # no other path to actually get its lbl model without this assignment
    # running (see the wv_range_high comment block above).
    if (retrieval_type == 'JR') or lbl_spectrophotometric_ranges:
        assign_model_type(wv_range_high)
        
    # --- Resolution of the planet model ---
    # Moved ahead of res_instru below (Chantier A Phase 4, c-k vs lbl): res_instru's
    # empty-instrum_param_list fallback needs prt_res['high'] already defined.
    global prt_res
    prt_res = {'high': int(1e6 / opacity_sampling), 'low': 1000}

    # --- Define the reference resolution for high-res models ---
    # Chantier A Phase 4: each visit's per-exposure model sequence is now degraded
    # to *its own* instrument's resolution (instrum_param_list[visit_i]['resol']),
    # not this shared value -- a multi-instrument run must not compare a
    # lower-resolution instrument's exposures to a model that was only ever
    # blurred to a finer instrument's resolution. `res_instru` (the max across
    # every high-res instrument in the run) is kept only as the resolution used
    # to generate the single, visit-independent high-res model reused by JR's
    # LOW RES block for spectrophotometric/photometric data whose wavelength
    # range is fully covered by the high-res data (`model_type == 'high'`,
    # `assign_model_type`) -- that path compares a static spectrum, not a
    # per-exposure sequence, and is itself re-degraded to the low-res
    # instrument's own (coarser) resolution by prepare_photometry/
    # prepare_spectrophotometry, so using the finest available high-res model as
    # its starting point is the right choice, not a leftover collapse.
    global res_instru
    res_instru = get_res_instru(instrum_param_list, prt_res['high'])

    # --- Initialize model objects to None ---
    # Initialize atmo objects based on the wavelength ranges (put None for now)
    for mode in ['high', 'low']:
        wv_rng = globals()[f'wv_range_{mode}']
        for i_range, _ in enumerate(wv_rng):
            globals()[f'atmo_{mode}_{i_range}'] = None

    # Same for the stellar spectra
    for mode in ['high', 'low']:
        globals()[f'fct_star_{mode}'] = None

    # --- Chantier A Phase 3f: representative phases for LOW RES multi-region ---
    # Low-res data (photometry/spectrophotometry) is usually integrated over a whole
    # visit, unlike high-res, which has one real exposure time per point (Phase 3c's
    # per-exposure region combination). So instead of a per-exposure loop, the
    # multi-region LOW RES path (`prepare_model_multi_reg_low`) evaluates the
    # region-combination kernel at a handful of representative phases and averages --
    # computed once here (ephemeris only, circular-orbit approximation -- Antoine:
    # eccentricity-aware timing elsewhere in the code, e.g. eclipse phase, is
    # considered unreliable, so deliberately not used here either) unless the YAML
    # gives an explicit override.
    global representative_phases_low
    representative_phases_low = input_params.get('representative_phases_low', None)
    if representative_phases_low is None:
        n_phases_low = input_params.get('n_phases_low', None)
        representative_phases_low = get_representative_low_res_phases(planet, kind_trans, n_phases_low)
    else:
        representative_phases_low = np.asarray(representative_phases_low, dtype=float)

    # --- Additional variables ---
    global inj_alpha, nolog
    inj_alpha = 'ones'
    nolog = True

    # --- Chantier A Phase 2: Fp/Fstar-separated model engine options ---
    # Same YAML keys/names as logl_grid.py::setup_logl_grid (`apply_alpha`,
    # `use_real_stellar_rv`) -- retrieval.py and logl_grid.py are meant to read the
    # same YAML file for a given run, so the two must not drift into different names
    # or different defaults for the same modelling choice.
    global apply_alpha, use_real_stellar_rv
    # apply_alpha: modulate the injected model by the real per-exposure
    # eclipse/transit light curve (Observations.alpha_frac) instead of assuming full
    # visibility for every exposure. Default True (closer to reality, per Antoine).
    apply_alpha = input_params.get('apply_alpha', True)
    # use_real_stellar_rv: Doppler-shift Fstar at the star's real, per-exposure reflex
    # velocity (Observations.vr, needs planet_obs.py::save_sequences from Chantier A
    # Phase 2 onward) instead of keeping it fixed. Default False: the reflex motion is
    # negligible next to the planet's orbital velocity and the BERV for essentially
    # every target (Antoine-confirmed approximation) -- so the extra realism usually
    # is not worth requiring every dataset to have been re-saved with `vr` first.
    use_real_stellar_rv = input_params.get('use_real_stellar_rv', False)

    # --- Chantier A Phase 4: HIGH RES logL combination across visits ---
    # How to group visits before taking the log of the pooled scaling-free chi2
    # terms (see model_sequence.group_visit_indices's docstring for the full
    # reasoning). Default 'per_instrument': numerically identical to the old,
    # always-pool-everything behaviour for any existing single-instrument config
    # (multiple visits of the same instrument still end up in one group), but a
    # multi-instrument run works out of the box instead of crashing (found on a
    # real SPIRou+NIRPS WASP-189b dataset -- different instruments can have a
    # different number of spectral orders, which the old unconditional pooling
    # could not handle at all).
    global logl_grouping
    logl_grouping = input_params.get('logl_grouping', 'per_instrument')

    # rotation_kernel: phase-*independent* default rotation kernel applied once per
    # theta by model_sequence.precompute_theta_model (Chantier A Phase 3) -- vsini-
    # style solid rotation broadening in emission (spectrum.SolidRotationKernel), or
    # geometric wind broadening in transmission (replaces the dead
    # RotKerTransitCloudy(gauss=True) path). Default None: no default kernel,
    # unchanged behaviour for existing configs that do not set this key. Distinct
    # from the multi-region kernel (`get_ker`/`get_ker_file` above), which is
    # phase-*dependent* and applied per exposure instead.
    global rotation_kernel
    rotation_kernel = input_params.get('rotation_kernel', None)

    # --- Add some useful parameters for model---
    pressures = np.logspace(limP[0], limP[1], n_pts)
    fixed_params['pressures'] = pressures
    fixed_params['temperatures'] = None  # Will be computed from the TP profile parameters
    fixed_params['M_pl'] = planet.M_pl.to('Mjup').value
    fixed_params['R_pl'] = planet.R_pl.to('Rjup').value
    fixed_params['R_star'] = planet.R_star.to('Rsun').value
    # Get gravity in cgs units if not already given
    if fixed_params.get('gravity', None) is None:
        fixed_params['gravity'] = planet.gp.cgs.value

    # --- Complete prior parameters ---
    global general_params, n_regions, reg_fixed_params

    # Add species that are fitted (if not included yet in priors)
    for specie in species_in_prior:
        if specie not in params_prior:
            params_prior[specie] = ['log_uniform', -12.0, -0.5]
            log.info(f'Adding {specie} to params_prior.')

    # Define the number of regions
    n_regions = len(region_id)

    # global parameters are all other parameters not in reg_params
    general_params = [param for param in params_prior.keys() if param not in reg_params]

    # Assign  specific index to each regional parameter (if not already done manually)
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

    # Prior functions
    global prior_init_func, prior_func_dict, custom_prior_file
    if custom_prior_file is None:
        prior_func_dict = ru.default_prior_func
        prior_init_func = ru.default_prior_init_func
    else:
        c_prior_func, c_prior_init = ru.load_custom_prior(custom_prior_file)
        prior_func_dict = {**ru.default_prior_func, **c_prior_func}
        prior_init_func = {**ru.default_prior_init_func, **c_prior_init}

    # Get the initialisation from prior
    global walker_init
    walker_init = ru.init_from_prior(n_walkers, prior_init_func, params_prior,
                                     special_treatment=special_init)

    # --- Rotation kernel ----
    # If `get_ker_file` is given (see `retrievals/retrieval_inputs_example_rotation.yaml`
    # for the documented contract), load the user's `get_ker` function from it, the
    # same way `custom_prior_file` is loaded above. Otherwise, fall back to a no-op
    # (no rotation kernel, `prepare_model_multi_reg` treats `None` as "instrumental
    # profile only", see `prepare_model_high_or_low`).
    global get_ker, get_ker_file
    if get_ker_file is None:
        get_ker = lambda theta_regions, phase=None, planet=None, instrum=None, \
                          model_resolution=None: [None for _ in theta_regions]
    else:
        get_ker = ru.load_custom_get_ker(get_ker_file)

    return input_params

# Once the setup_retrieval function is run, all the parameters will be accessible in the global variables.

def get_wv_range(list_of_ranges):
    """Define the most effective wavelength range given a list of wavelenght ranges.
    The list is in the format [(wv_1_min, wv_1_max), (wv_2_min, wv_2_max), ...].
    The output is a list of wavelength ranges. If all the input wavelength ranges
    were overlapping, there will be only one set of wavelength ranges. If not,
    The output will be the smallest number of wavelength ranges that cover all the
    input ranges.
    """
    if not list_of_ranges:
        return []

    # Sort the ranges by their start value
    list_of_ranges.sort(key=lambda x: x[0])

    # Initialize the result with the first range
    result = [list_of_ranges[0]]

    for current_range in list_of_ranges[1:]:
        last_range = result[-1]

        # If the current range overlaps with the last range in the result, merge them
        if current_range[0] <= last_range[1]:
            last_range[1] = max(last_range[1], current_range[1])
        else:
            # Otherwise, add the current range to the result
            result.append(list(current_range))

    return result


def get_complementary_ranges(ranges1, ranges2):
    """Get the complementary ranges between two sets of wavelength ranges.
    The output is a list of wavelength ranges that are in the first set but not in the second set.
    """
    # Sort the ranges by their start value
    ranges1.sort(key=lambda x: x[0])
    ranges2.sort(key=lambda x: x[0])

    result = []
    i, j = 0, 0

    while i < len(ranges1) and j < len(ranges2):
        # If ranges1[i] is to the left of ranges2[j]
        if ranges1[i][1] < ranges2[j][0]:
            result.append(ranges1[i])
            i += 1
        # If ranges1[i] overlaps with ranges2[j]
        elif ranges1[i][0] < ranges2[j][1]:
            if ranges1[i][0] < ranges2[j][0]:
                result.append((ranges1[i][0], ranges2[j][0]))
            if ranges1[i][1] > ranges2[j][1]:
                ranges1.insert(i + 1, (ranges2[j][1], ranges1[i][1]))
            i += 1
        # If ranges2[j] is to the left of ranges1[i]
        else:
            j += 1

    # Add the remaining ranges in ranges1 to the result
    while i < len(ranges1):
        result.append(ranges1[i])
        i += 1

    return result


def load_high_res_data():
    """This function needs to be run after ´­setup_retrieval´.
    The function reads the input data and prepare it.

    Chantier A Phase 4: `data_info_list` keeps each visit's alpha_frac/icorr/N
    separate (one dict per visit, same convention as `logl_grid.py`'s
    `data_info_list`) instead of eagerly flattening everything into one
    concatenated `data_info` dict -- the old flattening assumed every visit
    shares the same number of spectral orders, which breaks for a genuinely
    multi-instrument run (found on a real SPIRou+NIRPS WASP-189b dataset). How
    (and whether) visits get pooled before the log-likelihood's log is taken is
    now `lnprob`'s job, via `logl_grouping` (see `model_sequence.group_visit_indices`).
    """

    global data_info_list, data_visits

    data_info_list = []
    data_visits = []

    for high_res_file_stem, n_pc_i in zip(high_res_file_stem_list, n_pc):
        log.debug(f'Hires files stem: {high_res_path / high_res_file_stem}')
        log.info('Loading Hires files.')
        # B3 (Chantier B): n_pc is applied at read time (see save_reduced_sequence/
        # load_reduced_sequence), one value per file (n_pc, broadcast in unpack_input_parameters).
        # Reuse the already-built `planet` (config `pl_kwargs` overrides applied) instead of
        # a fresh ExoFile lookup by name for every visit (see load_sequences's docstring).
        data_info_i, data_visit_i = pl_obs.load_sequences(high_res_file_stem, n_pc_i,
                                                            path=high_res_path, planet=planet)
        # Add index of the exposures where we expect to see the planet signal (to be used in kernel function)
        # all_alpha_frac is the fraction of the total planet signal received during the exposure.
        data_visit_i['i_pl_signal'] = data_info_i['all_alpha_frac'] > 0.5
        data_visits.append(data_visit_i)
        data_info_list.append(data_info_i)

    return data_info_list, data_visits


def load_low_res_data(pad_n_res_elem=5):
    """Load the low resolution data.

    This function loads the low resolution data for each instrument
    specified in the `spectrophotometric_data` dictionary, which needs
    to exist in the global variables.
    The function reads the data file (found in `spectrophotometric_data`),
    extracts the wavelengths, data, uncertainties, instrument resolution, and wavelength range.

    Each instrument's entry may also set `opacity_mode: 'c-k'` (default) or
    `'lbl'` (Chantier A Phase 4). `'lbl'` models that instrument with full
    line-by-line opacities instead of the coarser c-k ones, by folding its
    wavelength range into the high-res one (`setup_retrieval`) instead of the
    low-res one -- see `setup_retrieval`'s wv_range_high comment block for why.

    Args:
    pad_n_res_elem (int, optional):
        The number of resolution elements to use for padding the wavelength range. 
        Defaults to 5.

    Returns:
        dict: The `spectrophotometric_data` dictionary containing the loaded data for each instrument.

    """
    for instru_name, infos in spectrophotometric_data.items():
        log.info(f'Loading data for instrument {instru_name}')
        
        # Read the data file (astropy table)
        low_res_path = Path(infos['file_path'])
        low_res_file = Path(infos['file_name'])
        data_table = Table.read(low_res_path / low_res_file)
        
        # Get the wavelenghts
        default_name = 'wave'
        col_name = infos.get('wv_col_name', default_name)        
        try:
            infos['wave'] = data_table[col_name].to('um').value
        except UnitConversionError:
            log.warning(f"Could not convert wavelengths for instrument {instru_name}. Assuming they are in microns.")
            infos['wave'] = data_table[col_name].value
        
        # Get the data (depends on emission or transmission)
        default_name = 'F_p/F_star' if (kind_trans == 'emission') else 'dppm'
        col_name = infos.get('data_col_name', default_name)
        infos['data'] = data_table[col_name].quantity

        # Get uncertainties
        default_name = 'err'
        col_name = infos.get('err_col_name', default_name)
        infos['err'] = data_table[col_name].quantity

        # Check units for uncertainties and data
        for key in ['data', 'err']:
            if infos[key].unit == 'percent':
                infos[key] = infos[key].value / 100.
            elif infos[key].unit == 'None':
                infos[key] = infos[key].value
            else:
                infos[key] = infos[key].decompose().value
                
        # Get instrument resolution
        infos['res'] = data_table.meta['Resolution']
        
        # Get wavelength range
        if 'wv_range' in data_table.meta:
            infos['wv_range'] = data_table.meta['wv_range']
        else:
            # Define a padding based on the resolution (R = lambda / d_lambda)
            wv = np.sort(infos['wave'])
            dwv = wv[[0, -1]] / infos['res']
            wv_min = wv[0] - pad_n_res_elem * dwv[0]
            wv_max = wv[-1] + pad_n_res_elem * dwv[-1]
            infos['wv_range'] = [wv_min, wv_max]

    return spectrophotometric_data


def read_response(f_name, f_path, fmt):
    """Read the response fonction for photometric bands."""
    
    f_name = Path(f_name)
    f_path = Path(f_path)

    log.debug(f"Reading {f_path}/{f_name} with format='{fmt}'")
    
    # Read transmission function

    response = Table.read(f_path / f_name, format=fmt)
    
    x_rsp, y_rsp = response['col1'].value, response['col2'].value
    
    return x_rsp, y_rsp



def get_wv_band_coverage(x_rsp, y_rsp, coverage_percent=99.9):
    """
    Compute the limits in wavelengths of each photometric bands
    based on a specified coverage percentage of y values.
    """

    # Normalize y_rsp
    y_rsp_normalized = y_rsp / np.max(y_rsp)
    
    # Calculate the cumulative sum of the normalized y_rsp
    cumulative_sum = np.cumsum(y_rsp_normalized)
    cumulative_sum_normalized = cumulative_sum / np.max(cumulative_sum)
    
    # Calculate the lower and upper bounds for the specified coverage percentage
    lower_bound = (100 - coverage_percent) / 2 / 100
    upper_bound = 1 - lower_bound
    
    # Find the x values corresponding to the calculated bounds of the cumulative sum
    lower_index = np.where(cumulative_sum_normalized > lower_bound)[0][0]
    upper_index = np.where(cumulative_sum_normalized < upper_bound)[0][-1]
    
    # The band limits are the x values at the lower and upper indices
    band_limits = (x_rsp[lower_index], x_rsp[upper_index])
    
    return band_limits


def load_photometry(model_res=250, pad_n_res_elem=5):

    for instru_name, infos in photometric_data.items():
        log.info(f'Loading data for instrument {instru_name}')
    
        # Read the data file (astropy table)
        data_path = Path(infos['file_path'])
        data_file = Path(infos['file_name'])
        data_table = Table.read(data_path / data_file)
        
        # Get the wavelenghts
        default_name = 'wave'
        col_name = infos.get('wv_col_name', default_name)
        infos['wave'] = data_table[col_name].to('um').value
        
        # Get the data (depends on emission or transmission)
        default_name = 'F_p/F_star' if (kind_trans == 'emission') else 'dppm'
        col_name = infos.get('data_col_name', default_name)
        infos['data'] = data_table[col_name].quantity

        # Get uncertainties
        default_name = 'err'
        col_name = infos.get('err_col_name', default_name)
        infos['err'] = data_table[col_name].quantity

        # Check units for uncertainties and data
        for key in ['data', 'err']:
            if infos[key].unit == 'percent':
                infos[key] = infos[key].value * 100.
            elif infos[key].unit == 'None':
                infos[key] = infos[key].value
            else:
                infos[key] = infos[key].decompose().value
                
        # --- Response function
        # if the path for the response function is not available, use the data path
        response_path = data_table.meta.get('response_path', data_path)
        response_format = data_table.meta.get('response_format', 'ascii')
        # Get the transmission function and wavelenght grid for each filters
        fcts, wv_grids, wv_coverages = [], [], []
        for f_name in data_table.meta['response_files']:
            x_rsp, y_rsp = read_response(f_name, response_path, response_format)
            # Transmission function
            fct_band = interp1d(x_rsp, y_rsp, kind='cubic', bounds_error=False, fill_value=0.)
            fcts.append(fct_band)
            wv_grids.append(x_rsp)  # Used later
            # Save intervals for plotting purposes
            wv_cov = get_wv_band_coverage(x_rsp, y_rsp, coverage_percent=99.9)
            wv_coverages.append(wv_cov)
        # Save
        infos['response_fcts'] = fcts
        infos['wv_coverages'] = wv_coverages
        
        # Get spectral resolution that will be used to downgrade the model
        # before applying the transmission function.
        # This is done to insure a smooth spectrum before integrating
        # with the response functions.
        infos['res'] = data_table.meta.get('model_resolution', model_res)
        
        # Get wavelength range
        if 'wv_range' in data_table.meta:
            infos['wv_range'] = data_table.meta['wv_range']
        else:
            # Use the grid range + a padding based on a given resolution
            wv_grids = np.concatenate(wv_grids)
            wv = np.array([np.min(wv_grids), np.max(wv_grids)])
            dwv = wv[[0, -1]] / infos['res']
            wv_min = wv[0] - pad_n_res_elem * dwv[0]
            wv_max = wv[-1] + pad_n_res_elem * dwv[-1]
            infos['wv_range'] = [wv_min, wv_max]

    return photometric_data


def get_lbl_spectrophotometric_ranges(spectrophotometric_data_dict: Dict[str, dict]) -> List[list]:
    """Wavelength ranges of spectrophotometric instruments using `opacity_mode: 'lbl'`.

    Chantier A Phase 4 (c-k vs lbl per instrument): pure helper factored out of
    `setup_retrieval` so the folding-into-`wv_range_high` decision (see that
    function's wv_range_high comment block) is unit-testable without needing to
    run the whole, heavily side-effectful `setup_retrieval`.

    Parameters
    ----------
    spectrophotometric_data_dict : dict
        `spectrophotometric_data`, keyed by instrument name. Each value is
        expected to already have a `'wv_range'` key (set by `load_low_res_data`)
        and may optionally have an `'opacity_mode'` key (`'c-k'` or `'lbl'`,
        defaults to `'c-k'` if absent -- unchanged behaviour for existing configs).

    Returns
    -------
    list of [float, float]
        `wv_range` of every instrument with `opacity_mode == 'lbl'`, in
        dictionary iteration order. Empty if none (the common case today).
    """
    return [infos['wv_range'] for infos in spectrophotometric_data_dict.values()
            if infos.get('opacity_mode', 'c-k') == 'lbl']


def get_res_instru(instrum_param_list: List[dict], native_high_res_resolution: float) -> float:
    """Reference resolution used to generate the high-res model reused by the LOW
    RES block for spectrophotometric/photometric data (`res_instru` in
    `setup_retrieval`, see that function's comment block for the full rationale).

    Chantier A Phase 4 (c-k vs lbl per instrument): pure helper factored out of
    `setup_retrieval` for the same reason as `get_lbl_spectrophotometric_ranges` --
    `instrum_param_list` can now be empty (a retrieval running entirely on
    lbl-flagged low-res data, no real high-res instrument at all), a case with no
    real HR-instrument data available on any dataset used so far this session, so
    worth covering with a direct unit test rather than only by reasoning.

    Parameters
    ----------
    instrum_param_list : list of dict
        One `load_instrum(...)` dict per real high-res instrument in `instrum`.
        May be empty.
    native_high_res_resolution : float
        Fallback resolution when `instrum_param_list` is empty (`prt_res['high']`
        in `setup_retrieval` -- the model's own native lbl sampling resolution,
        so no degradation happens at this stage; `prepare_spectrophotometry`/
        `prepare_photometry` re-degrade to each instrument's own, coarser,
        resolution downstream anyway).

    Returns
    -------
    float
        `max(p['resol'] for p in instrum_param_list)` if non-empty, else
        `native_high_res_resolution`.
    """
    if instrum_param_list:
        return max(p_list['resol'] for p_list in instrum_param_list)
    return native_high_res_resolution


def get_wv_range_low(wv_range_all_low: List[list]) -> List[list]:
    """The single wavelength range the low-res (c-k) model will be generated over
    (`wv_range_low` in `setup_retrieval`), spanning the min/max of every input range.

    Chantier A Phase 4 (c-k vs lbl per instrument): pure helper factored out of
    `setup_retrieval`, mainly to guard a real crash found on real data (not just a
    hypothetical edge case): `wv_range_all_low` can be genuinely empty when every
    low-res instrument is lbl-flagged (folded into `wv_range_high` instead) and
    there is no photometric data either, in a pure LRR run (no high-res-range
    padding fallback) -- `np.min`/`np.max` raise `ValueError` on an empty array.

    Parameters
    ----------
    wv_range_all_low : list of [float, float]
        Every c-k low-res instrument's `wv_range` (plus, outside LRR, the full
        high-res range -- see `setup_retrieval`), not yet merged/reduced.

    Returns
    -------
    list of [float, float]
        `[[min, max]]` over every input range, or `[]` if `wv_range_all_low` is
        itself empty (no dedicated low-res model is needed at all in that case).
    """
    if wv_range_all_low:
        return [[np.min(wv_range_all_low), np.max(wv_range_all_low)]]
    return []


def get_low_res_dv_shift(theta_dict: dict) -> float:
    """Systemic-velocity Doppler shift (km/s) applied to a low-res model spectrum
    -- one with no genuine per-exposure timing of its own to track BERV/orbital
    motion with (unlike high-res, where `data_visit['RV_const']` already bakes in
    RV_sys, BERV and the mean orbital velocity at mid-transit, see `norv_sequence`
    in planet_obs.py). Only the (fixed) systemic velocity matters, plus the same
    `rv` residual parameter used in high-res, so a Joint Retrieval fits a single
    shared RV offset for both resolutions.

    Shared by every low-res-model caller: `prepare_model_high_or_low`'s `mode ==
    'low'` branch, `prepare_model_multi_reg_low` (any `mode`), and `lnprob`'s LOW
    RES block (Chantier A Phase 4's `model_type == 'high'` reuse path, single-region
    case -- the multi-region case gets this for free through
    `prepare_model_multi_reg_low`, called with `mode='high'` there).

    Parameters
    ----------
    theta_dict : dict
        Any one region's parameter dict (as produced by `unpack_theta`) with an
        `'rv'` key -- `rv` is a shared, not per-region, parameter, so any region
        gives the same value.

    Returns
    -------
    float
        `planet.RV_sys` (km/s) + `theta_dict['rv']` (defaults to 0.0 if absent).
    """
    return planet.RV_sys.to(u.km / u.s).value + theta_dict.get('rv', 0.0)


def assign_model_type(wv_rng_list_high: List[list]) -> None:
    """Assign the kind of model (high res or low res) that will be used
    to create synthetic data. The input is the list of wavelength ranges
    that are covered by the high res models. If one of these ranges covers
    entirely the data of a specific instrument, then the high res model is used.
    The low res model is used otherwise.

    Chantier A Phase 4 (c-k vs lbl per instrument): a spectrophotometric
    instrument with `opacity_mode: 'lbl'` has its own wavelength range folded
    into `wv_rng_list_high` by the caller (`setup_retrieval`), so it is always
    assigned 'high' here -- that is how it gets its lbl model. A photometric
    instrument (no `opacity_mode` key) is assigned 'high' the same way any
    other low-res instrument would be: only if it happens to be fully covered
    by `wv_rng_list_high`, lbl-extended or not."""
    
    for data_dict in [spectrophotometric_data, photometric_data]:
        for infos in data_dict.values():
            model_type = 'low'
            for wv_rng in wv_rng_list_high:
                wv_min, wv_max = infos['wv_range']
                if (wv_min >= wv_rng[0]) and (wv_max <= wv_rng[-1]):
                    model_type = 'high'
                    
            infos['model_type'] = model_type
        
    return
        
# Here are other functions that need to stay in the retrieval script

def init_model_retrieval(mol_species=None, kind_res='high', lbl_opacity_sampling=None,
                         wl_range=None, continuum_species=None, pressures=None, **kwargs):
    """
    Initialize some objects needed for modelization: atmo, species, fct_star, pressures
    :param mol_species: list of species included (without continuum opacities)
    :param kind_res: str, 'high' or 'low'
    :param lbl_opacity_sampling: ?
    :param wl_range: wavelength range (2 elements tuple or list, or None)
    :param continuum_species: list of continuum opacities, H and He excluded
    :param pressures: pressure array. Default is `fixed_params['pressures']`
    :param kwargs: other kwargs passed to `starships.petitradtrans_utils.select_mol_list()`
    :return: atmos, species, fct_star, pressure array
    """

    if mol_species is None:
        mol_species = line_opacities

    if lbl_opacity_sampling is None:
        lbl_opacity_sampling = opacity_sampling

    if continuum_species is None:
        continuum_species = continuum_opacities

    if pressures is None:
        pressures = fixed_params['pressures']

    species = prt.select_mol_list(mol_species, kind_res=kind_res, **kwargs)
    species_2_lnlst = {mol: lnlst for mol, lnlst in zip(mol_species, species)}

    if kind_res == 'high':
        mode = 'lbl'
        if wl_range is None:
            wl_range = wv_range_high[0]

    elif kind_res == 'low':
        mode = 'c-k'
        if wl_range is None:
            wl_range = wv_range_low[0]
    else:
        raise ValueError(f'`kind_res` = {kind_res} not valid. Choose between high or low')


    atmo, _ = prt.gen_atm_all([*species.keys()], pressures, mode=mode,
                                      lbl_opacity_sampling=lbl_opacity_sampling, wl_range=wl_range,
                                      continuum_opacities=continuum_species)

    return atmo, species_2_lnlst


def init_atmo_if_not_done(mode):

    wv_range = globals()[f'wv_range_{mode}']
    for i_range, wv_rng in enumerate(wv_range):
        # Use atmo object in globals parameters if it exists
        # atmo_obj = atmo_high if mode == 'high' else atmo_low
        atmo_obj_name = f'atmo_{mode}_{i_range}'
        atmo_obj = globals()[atmo_obj_name]
        # Initiate if not done yet
        if atmo_obj is None:
            log.info(f'Model not initialized for mode = {mode} and range {wv_rng}. Starting initialization...')
            output = init_model_retrieval(kind_res=mode, wl_range=wv_rng)
            log.info('Saving values in `linelist_names`.')
            atmo_obj, lnlst_names = output
            # Update the values of the global variables
            # Need to use globals() otherwise an error is raised.
            globals()[atmo_obj_name] = atmo_obj
                
            # Update the line list names
            if linelist_names.get(mode, None) is None:
                linelist_names[mode] = lnlst_names
            else:
                # Keep the predefined values and complete with the new ones
                # TODO: should be the opposite, but need to make sure the input linelist names are properly handled
                linelist_names[mode] = {**lnlst_names, **linelist_names[mode]}
                log.info(f"final linelist_names['{mode}'] = {linelist_names[mode]}")

    return None


def init_stellar_spectrum_if_not_done(mode):
    
    # No need to make different fct for the different wv_range (as opposed to atmo object)
    fct_star_name = f'fct_star_{mode}'
    fct_star_obj = globals()[fct_star_name]
    if fct_star_obj is None:
        # Initiate if not done yet
        log.info(f'Star spectrum not initialized for mode = {mode}. Starting initialization...')
        fct_star_obj = init_stellar_spectrum(mode=mode)
        # Update the values of the global variables
        # Need to use globals() otherwise an error is raised.
        globals()[fct_star_name] = fct_star_obj

    return None
               

def init_stellar_spectrum(mode: str, wl_range: Optional[Tuple[float, float]] = None):
    """Prepare an interpolator for the stellar spectrum, degraded to the model resolution.

    Relies on the module-level globals `star_wv`/`star_flux`/`star_res` (the stellar
    model loaded at package init) and `kind_trans`/`prt_res` (retrieval config).

    Parameters
    ----------
    mode : str
        'high' or 'low', selects which PRT resolution (`prt_res[mode]`) to degrade to.
    wl_range : tuple of float, optional
        (wv_min, wv_max) to restrict the stellar spectrum to. If None, uses
        `wv_range_{mode}` (the full list of wavelength ranges for this mode).

    Returns
    -------
    scipy.interpolate.interp1d or str
        Interpolator for the degraded stellar flux as a function of wavelength, or
        the string 'blackbody' if no stellar spectrum is available (a blackbody at
        Teff is used downstream in that case).
    """
    if wl_range is None:
        wv_range_list = globals()[f'wv_range_{mode}']
    else:
        wv_range_list = [wl_range]

    # Use same resolution as the PRT model
    Raf = prt_res[mode]

    # --- Interpolate the stellar spectrum and downgrade to model resolution ---
    if kind_trans == 'emission' and star_wv is not None:
        log.info(f'Interpolating the stellar spectrum for mode = {mode}.')
        # Only interpolate over the valid wavelength ranges in the list of wavelength ranges
        is_in_range = (star_wv >= np.min(wv_range_list) - 0.1) & (star_wv <= np.max(wv_range_list) + 0.1)
        sample = star_wv[is_in_range]
        # `star_res` is the stellar model's native/physical resolution (Rbf); degrade
        # it to the model resolution `Raf` with the unified convolution engine
        # (Chantier A Phase 1 -- see convolution.py::degrade_and_resample).
        resamp_star = degrade_and_resample(sample, star_flux[is_in_range],
                                            resolution=Raf, input_resolution=star_res,
                                            sample=sample)
        resamp_star = np.ma.masked_invalid(resamp_star)
        fct_star = interp1d(sample, resamp_star)

    else:
        log.info('No stellar spectrum provided. A blackbody at Teff will be used.')
        fct_star = 'blackbody'

    return fct_star


####################################################

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
        # Check the last parameter, which tells if the parameter is in log scale
        convert_from_log10 = (prior_info[-1] == 'log10')
        if prior_info[0] == 'log_uniform' or convert_from_log10:
            log.debug(f'Converting {key} to 10**({key}).')
            theta_dict[key] = 10 ** theta_dict[key]
    
    # Make a dictionnary for each region if needed.
    dict_list = list()
    for i_reg in region_id:
        # Create a dictionnary for each region and remove the region number from the key.
        theta_region = {key: theta_dict[key] for key in general_params}
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
        # The values are either taken from theta_region in priority or from fixed_params.
        combined_dict = {**fixed_params, **theta_region}

        # gravity depends on Rp if included in the fit
        if 'R_pl' in theta_region and not 'gravity' in theta_region:
            combined_dict['gravity'] = (const.G * planet.M_pl /
                                        (theta_region['R_pl'] * const.R_jup) ** 2).cgs.value
            
        # Some values need to be set to None if not included in the fit or not in fixed_params.
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
        elif kind_temp == 'madhu':
            fct_inputs = ['pressures', 'a1', 'a2', 'log_P1', 'log_P2', 'log_P3', 'T_set', 'P_set']
            args = (combined_dict[key] for key in fct_inputs)
            combined_dict['temperatures'] = madhu_seager(*args)
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
            ref_linelists = line_opacities.copy()
        else:
            ref_linelists = [linelist_names[mode][mol] for mol in line_opacities]

    # --- Prepare the abundances (with the correct linelist name for species)
    species = {lnlst: theta_dict[mol] for lnlst, mol
               in zip(ref_linelists, line_opacities)}
    
    # --- Adding continuum opacities
    for mol in continuum_opacities:
        species[mol] = theta_dict[mol]
        
    # --- Adding other species
    for mol in other_species:
        species[mol] = theta_dict[mol]
        
    # Tentative implementation of removing species at high or low res
    # Put lists in yaml (can be empty or absent): remove_mol_high and remove_mol_low
    # abundances for mols in those lists will be set to 0 for the appropriate mode
    for lnlst in species.keys():
        for mol in globals()[f'remove_mol_{mode}']:
            if mol in lnlst:
                species[lnlst] = 0.

    return species


def prepare_model_high_or_low(theta_dict, mode, atmo_obj=None, fct_star=None,
                              species_dict=None, Raf=None, rot_ker=None):
    """Generate one theta's model spectrum, at high or low resolution.

    High res: petitRADTRANS spectrum degraded to instrument resolution (optionally
    convolved with a multi-region kernel, see `rot_ker` below). Low res:
    petitRADTRANS spectrum Doppler-shifted by the systemic velocity (see the comment
    in the ``mode == 'low'`` branch below for why no per-exposure term is needed
    there).

    Note: `lnprob`'s own HIGH RES model generation (Chantier A Phase 2-4) does not
    go through this function -- it needs Fp/Fstar kept separate (not combined into
    the ratio this function returns) so it can Doppler-shift them independently per
    exposure, and (since Phase 4) generates the native spectrum once per theta and
    degrades it once per visit instead of once per call. See
    `_prepare_fp_native_by_region`/`model_sequence.degrade_fp_fstar`, called
    directly from `lnprob` for that path. This function (and `prepare_model_multi_reg`
    below) stays the combined-ratio entry point for every other caller (LOW RES,
    `retrieval_utils.py` analysis/plotting, regression-test model generation).

    Parameters
    ----------
    theta_dict : dict
        One region's parameter dict, as produced by `unpack_theta`.
    mode : {'high', 'low'}
    atmo_obj : petitRADTRANS.Radtrans, optional
        Defaults to the module's cached object(s) for `mode` (`init_atmo_if_not_done`).
    fct_star : callable or 'blackbody' or None, optional
        Defaults to the module's cached stellar spectrum for `mode`
        (`init_stellar_spectrum_if_not_done`).
    species_dict : dict, optional
        Forwarded to `prepare_abundances`.
    Raf : float, optional
        Target (instrument) resolving power. Defaults to the highest resolution among
        the configured instruments.
    rot_ker : object, optional
        Pre-built multi-region kernel (from `get_ker`, one region's array),
        forwarded to `petitradtrans_utils.prepare_model` (`mode='high'`). `None`
        (default) takes the plain Gaussian-degradation path.

    Returns
    -------
    wv_all : np.ndarray
    model_all : np.ndarray
    """
    # Take the highest resolution among instruments
    if Raf is None:
        Raf = max([p_list['resol'] for p_list in instrum_param_list])

    if atmo_obj is None:
        init_atmo_if_not_done(mode)
        n_wv_rng = len(globals()[f'wv_range_{mode}'])
        atmo_obj_list = [globals()[f'atmo_{mode}_{i_rng}'] for i_rng in range(n_wv_rng)]
    else:
        atmo_obj_list = [atmo_obj]

    if fct_star is None:
        init_stellar_spectrum_if_not_done(mode)
        fct_star = globals()[f'fct_star_{mode}']

    # --- Prepare the abundances (with the correct name for species)
    # Note that if species is None (not specified), `linelist_names[mode]` will be used inside `prepare_abundances`.
    species = prepare_abundances(theta_dict, mode, species_dict)

    # --- Generating the model
    args = [theta_dict[key] for key in ['pressures', 'temperatures', 'gravity', 'P0', 'p_cloud', 'R_pl', 'R_star']]
    kwargs = dict(gamma_scat=theta_dict['gamma_scat'],
                  kappa_factor=theta_dict['scat_factor'],
                  C_to_O=theta_dict['C/O'],
                    Fe_to_H=theta_dict['Fe/H'],
                    specie_2_lnlst=linelist_names[mode],
                    kind_trans=kind_trans,
                    dissociation=dissociation,
                    fct_star=fct_star)
    wv_all, model_all = list(), list()
    for atmo_obj in atmo_obj_list:
        wv_out, model_out = prt.retrieval_model_plain(atmo_obj, species, planet, *args, **kwargs)

        if mode == 'high':
            # Downgrade the model. `rot_ker`, if given (from prepare_model_multi_reg's
            # get_ker call), is a pre-built multi-region kernel array applied here;
            # `None` (the common case) takes the plain Gaussian-degradation path
            # (petitradtrans_utils.prepare_model -> convolution.degrade_and_resample).
            # Wind broadening no longer has a special case here -- removed 2026-08-28
            # (see lnprob's docstring): it used to build a `RotKerTransitCloudy(gauss=True)`
            # kernel from `theta_dict['wind']`, a crude Gaussian approximation
            # confirmed physically wrong, now fully superseded by the geometric
            # `rotation_kernel: 'transmission'` mechanism (spectrum.RotKerTransit,
            # Chantier A Phase 3) used by the Fp/Fstar engine instead.
            wv_out, model_out = prt.prepare_model(wv_out, model_out, prt_res[mode], Raf=Raf,
                                                rot_ker=rot_ker)

        elif mode == 'low':
            # --- Applying the Doppler shift due to the star's systemic velocity ---
            # See get_low_res_dv_shift's docstring for why low-res never gets a
            # per-exposure BERV/orbital term the way high-res does.
            dv_shift = get_low_res_dv_shift(theta_dict)
            wv_out = wv_out * calc_shift(dv_shift, kind='rel')

        wv_all.append(wv_out)
        model_all.append(model_out)

    wv_all = np.concatenate(wv_all)
    model_all = np.concatenate(model_all)

    return wv_all, model_all


def _prepare_fp_native_by_region(theta_regions, atmo_obj_list, fct_star, mode='high'):
    """Generate every region's native (undegraded) Fp for one visit.

    Chantier A Phase 3: true multi-region wiring into the Fp/Fstar-separated
    engine (`len(theta_regions) > 1`; `theta_regions` is an arbitrary user-defined
    split of the planet -- e.g. citrus/longitude slices, the documented example, but
    nothing here assumes that specific geometry; also reused, as of Chantier A Phase
    4, for the single-region case -- `lnprob` calls this once per theta regardless of
    region count, see its HIGH RES block). Mirrors the combined-ratio path in
    `prepare_model_high_or_low` (same per-region/per-atmo_obj loop) but calls
    `model_sequence.generate_native_fp_fstar` (no degradation) instead of
    `precompute_theta_model`: the per-exposure kernel built by `get_ker`
    already bakes in the resolution degradation (see
    `model_sequence.combine_regions_with_kernel`), so degrading here too would
    degrade twice. Also returns the first region's native Fstar (region-independent
    -- stellar flux does not depend on the region -- any region gives the same
    array) for the caller to degrade once, the normal way.

    Parameters
    ----------
    theta_regions : list of dict
        One dict per region (`len(theta_regions) > 1`), as produced by
        `unpack_theta`.
    atmo_obj_list : list of petitRADTRANS.Radtrans
        One entry per wavelength range/instrument (`init_atmo_if_not_done(mode)`).
    fct_star : callable or 'blackbody' or None
        Forwarded to `retrieval_model_plain` (see `retrieval.py::init_stellar_spectrum`).
    mode : {'high', 'low'}, default 'high'
        Which resolution's species/linelist mapping to use (`linelist_names[mode]`,
        Chantier A Phase 3f: needed to reuse this function for the LOW RES
        multi-region path, `prepare_model_multi_reg_low`).

    Returns
    -------
    wave : np.ndarray
        Native wavelength grid, shared across regions (NOT edge-trimmed --
        `combine_regions_with_kernel` trims after its own per-exposure convolution).
    Fp_by_region : list of np.ndarray
        One native Fp array per region, on `wave`.
    Fstar : np.ndarray or None
        Native Fstar (region-independent), on `wave`. `None` in transmission.
    """
    wave, Fp_by_region, Fstar = None, [], None
    for theta_dict in theta_regions:
        species = prepare_abundances(theta_dict, mode)
        kwargs = dict(gamma_scat=theta_dict['gamma_scat'],
                     kappa_factor=theta_dict['scat_factor'],
                     C_to_O=theta_dict['C/O'],
                     Fe_to_H=theta_dict['Fe/H'],
                     specie_2_lnlst=linelist_names[mode],
                     dissociation=dissociation)
        wv_all, Fp_all, Fstar_all = [], [], []
        for atmo_obj in atmo_obj_list:
            wv_out, Fp_out, Fstar_out = model_seq.generate_native_fp_fstar(
                atmo_obj, species, planet, theta_dict, kind_trans,
                fct_star=fct_star, **kwargs)
            wv_all.append(wv_out)
            Fp_all.append(Fp_out)
            Fstar_all.append(Fstar_out)

        wv_all = np.concatenate(wv_all)
        Fp_all = np.concatenate(Fp_all)
        if wave is None:
            wave = wv_all
            # Fstar does not depend on the region -- keep the first region's.
            Fstar = np.concatenate(Fstar_all) if Fstar_all[0] is not None else None
        Fp_by_region.append(Fp_all)

    return wave, Fp_by_region, Fstar


def _build_multi_region_kernel(theta_regions, visit_i, mode='high', instrum=None):
    """Build the per-exposure `region_kernel` closure for multi-region combination (Phase 3).

    Generic across whatever region geometry `get_ker` implements (a citrus/
    longitude-slice split is the documented example, but nothing here assumes it).
    Returned closure matches `model_sequence.build_model_sequence`'s `region_kernel`
    contract: called once per exposure as `region_kernel(wave, Fp_by_region, phase_i)`,
    it fetches this phase's per-region kernels from `get_ker` (`retrieval.py`'s own
    global, loaded from `get_ker_file` -- see `setup_retrieval`'s "Rotation kernel"
    block) and combines them with `model_sequence.combine_regions_with_kernel`,
    weighted by each region's `spec_scale` (same weighting convention as the old
    combined-ratio multi-region path, `prepare_model_multi_reg`).

    Parameters
    ----------
    theta_regions : list of dict
        One dict per region, as produced by `unpack_theta`.
    visit_i : int
        Visit index. Selects `instrum_param_list[visit_i]` when `instrum` is not given
        (not otherwise forwarded to `get_ker` -- nothing in its documented contract uses
        the raw index, only the `phase`/`instrum` derived from it).
    mode : {'high', 'low'}, default 'high'
        Which `prt_res` entry to pass as `get_ker`'s `model_resolution` (Chantier A
        Phase 3f: needed to reuse this function for the LOW RES multi-region path,
        `prepare_model_multi_reg_low`).
    instrum : dict, optional
        Forwarded to `get_ker` as its `instrum` argument. Defaults to
        `instrum_param_list[visit_i]` (unchanged behaviour for existing HIGH RES
        callers) -- LOW RES has no per-visit `instrum_param_list` entry, so its
        caller builds and passes its own instrument-shaped dict instead.

    Returns
    -------
    callable
        `region_kernel(wave, Fp_by_region, phase_i)` -> combined spectrum for that
        phase, ready for `build_model_sequence`.
    """
    weights = [theta_dict['spec_scale'] for theta_dict in theta_regions]
    if instrum is None:
        instrum = instrum_param_list[visit_i]

    def region_kernel(wave, Fp_by_region, phase_i):
        rot_ker_list = get_ker(theta_regions, phase=phase_i, planet=planet,
                               instrum=instrum, model_resolution=prt_res[mode])
        return model_seq.combine_regions_with_kernel(wave, Fp_by_region, rot_ker_list, weights)

    return region_kernel


def prepare_model_multi_reg_high_per_exposure(theta_regions, visit_i, Raf, native=None):
    """Fp/Fstar-separated model generation for the true multi-region case (Phase 3).

    Counterpart to `lnprob`'s single-region path for `len(theta_regions) > 1`: since
    the per-region rotation kernel is phase-*dependent* in general (regions rotate
    into/out of view across a visit), the per-region combination cannot happen once
    per theta the way the single-region case degrades its one native spectrum -- it
    must happen once *per exposure*, inside `build_model_sequence`'s `region_kernel` hook
    (see `_build_multi_region_kernel`, `model_sequence.combine_regions_with_kernel`).
    This function only prepares what is needed *before* the exposure loop: each
    region's native Fp (kept separate, not yet combined) and the single shared,
    degraded Fstar.

    Parameters
    ----------
    theta_regions : list of dict
        One dict per region (`len(theta_regions) > 1`), as produced by
        `unpack_theta`.
    visit_i : int
        Visit index, forwarded to `_build_multi_region_kernel`/`get_ker`.
    Raf : float
        Target (instrument) resolving power, used to degrade the shared Fstar.
    native : tuple, optional
        Pre-computed `(wave_native, Fp_by_region, Fstar_native)` from
        `_prepare_fp_native_by_region`, reused across visits sharing the same theta
        (Chantier A Phase 4 optimization -- the native spectrum depends only on
        theta/region, never on the visit; only its degradation to `Raf` and its
        per-exposure kernel genuinely do). Computed fresh internally, as before, when
        not given.

    Returns
    -------
    wave : np.ndarray
        Wavelength grid (native, edge-trimmed by 15 points), shared by `Fstar` and
        by the spectra `region_kernel` returns.
    Fp_by_region : list of np.ndarray
        Native (undegraded, un-trimmed), per-region Fp -- see
        `model_sequence.combine_regions_with_kernel` for why they are not degraded here.
    Fstar : np.ndarray or None
        Degraded stellar flux, on `wave`. `None` in transmission.
    region_kernel : callable
        `region_kernel(wave, Fp_by_region, phase_i)` -> combined spectrum for that
        phase, ready for `build_model_sequence`.
    """
    native_res = prt_res['high']

    if native is None:
        init_atmo_if_not_done('high')
        n_wv_rng = len(globals()['wv_range_high'])
        atmo_obj_list = [globals()[f'atmo_high_{i_rng}'] for i_rng in range(n_wv_rng)]
        init_stellar_spectrum_if_not_done('high')
        fct_star = globals()['fct_star_high']
        wave_native, Fp_by_region, Fstar_native = _prepare_fp_native_by_region(
            theta_regions, atmo_obj_list, fct_star)
    else:
        wave_native, Fp_by_region, Fstar_native = native

    if Fstar_native is not None:
        # Fstar is region-independent -- degrade the shared native spectrum the
        # normal way (same convention as precompute_theta_model's Step 2), once.
        Fstar_pre = degrade_and_resample(wave_native, Fstar_native, resolution=Raf,
                                         input_resolution=native_res, sample=wave_native)
        Fstar_out = np.ma.masked_invalid(Fstar_pre)[15:-15]
    else:
        Fstar_out = None

    wave_out = wave_native[15:-15]
    region_kernel_fct = _build_multi_region_kernel(theta_regions, visit_i)

    return wave_out, Fp_by_region, Fstar_out, region_kernel_fct


def prepare_model_multi_reg(theta_regions, mode, rot_ker_list=None, atmo_obj=None, visit_i=0, Raf=None):
    """Generate and combine the model for every region in `theta_regions`.

    Calls `prepare_model_high_or_low` once per region in `theta_regions`, weights
    each region's contribution by its `spec_scale`, and sums them.

    Note: `lnprob`'s own HIGH RES model generation does not call this function --
    see `prepare_model_high_or_low`'s docstring.

    Parameters
    ----------
    theta_regions : list of dict
        One dict per region, as produced by `unpack_theta` (each with its own
        `spec_scale` weight and, potentially, different atmospheric parameters).
    mode : {'high', 'low'}
    rot_ker_list : list, optional
        Unused positional parameter kept for backward compatibility -- the actual
        list used is always the one returned by `get_ker(...)` (see below).
    atmo_obj, Raf : optional
        Forwarded to `prepare_model_high_or_low`.
    visit_i : int, default 0
        Transit/visit index. Used to select `data_visits[visit_i]` (to compute the mean
        orbital phase of the planet signal, passed to `get_ker` as `phase`) and
        `instrum_param_list[visit_i]` (this visit's instrument, passed to `get_ker` as
        `instrum`) -- not otherwise forwarded to `get_ker` itself. If there is no real
        high-res visit at `visit_i` (Chantier A Phase 4: `lnprob`'s LOW RES block calls
        this function with `visit_i=0` even for a pure LRR run on lbl-flagged low-res
        data alone, with no real high-res visit whatsoever -- `data_visits` may not
        even exist as a global in that case, since `load_high_res_data` is only called
        for JR/HRR), falls back to an ephemeris-only representative phase (same
        mid-eclipse/mid-transit convention as `get_representative_low_res_phases`) and
        `instrum=None`. Harmless when `get_ker` is the default no-op (`get_ker_file:
        null`, ignores both arguments) -- only actually changes the result if a
        phase/instrument-aware custom `get_ker` is combined with a run that has no real
        high-res visit at all (unusual: such kernels are normally used with real
        per-exposure high-res time series).

    Returns
    -------
    wv_out : np.ndarray
    model_out : np.ndarray
    """
    # Mean orbital phase of the planet signal for this visit -- computed here (not
    # inside `get_ker`) because a custom `get_ker_file` is loaded as its own module
    # and cannot see `data_visits`/`planet`, retrieval.py's own globals, just by naming
    # them (see `ru.load_custom_get_ker`'s docstring).
    data_visits_avail = globals().get('data_visits', [])
    if visit_i < len(data_visits_avail):
        all_phases = (data_visits_avail[visit_i]['t_start'] - planet.mid_tr.value) / planet.period.to('d').value % 1
        mean_phase = np.mean(all_phases[data_visits_avail[visit_i]['i_pl_signal']])
        instrum_for_ker = instrum_param_list[visit_i]
    else:
        mean_phase = 0.5 if kind_trans == 'emission' else 0.0
        instrum_for_ker = None

    # Get the list of rotation kernels (one per region)
    rot_ker_list = get_ker(theta_regions, phase=mean_phase, planet=planet,
                           instrum=instrum_for_ker, model_resolution=prt_res[mode])

    wv_list = []
    model_list = []
    for theta_dict, reg_id in zip(theta_regions, region_id):
        wv_i, model_i = prepare_model_high_or_low(theta_dict, mode,
                                                  rot_ker=rot_ker_list[reg_id - 1],
                                                  atmo_obj=atmo_obj,
                                                  Raf=Raf)
        model_i *= theta_dict['spec_scale']
        wv_list.append(wv_i)
        model_list.append(model_i)

    wv_out = wv_list[0]
    model_out = np.sum(model_list, axis=0)

    return wv_out, model_out


def get_representative_low_res_phases(planet, kind_trans, n_phases=None):
    """Representative orbital phases for the LOW RES multi-region path (Chantier A Phase 3f).

    Low-res data (photometry/spectrophotometry) is usually integrated over a whole
    visit, unlike high-res, which has one real exposure time per data point (Phase
    3c's per-exposure region combination). So instead of combining regions once per
    exposure, the LOW RES multi-region path (`prepare_model_multi_reg_low`)
    evaluates the region-combination kernel at a handful of representative phases
    and averages the result -- this function picks those phases from the planet's
    ephemeris alone.

    Circular-orbit approximation: no eccentricity/argument-of-periastron
    correction, so secondary eclipse is assumed at exactly phase 0.5. This mirrors
    the simplification already used elsewhere in this file for phase (e.g.
    `prepare_model_multi_reg`'s `mean_phase`, `(t - mid_tr) / period % 1`) and for
    eclipse timing (`planet_obs.py::where_is_the_transit`'s
    `mid_tr + 0.5 * period` shortcut) -- Antoine: the orbit/eccentricity code in
    this package is unreliable enough that circular-orbit is the safer default for
    now, a real fix is a separate, future item.

    Parameters
    ----------
    planet : starships.planet_obs.Planet
        Used for `planet.period` and `planet.trandur`.
    kind_trans : {'transmission', 'emission'}
        Selects which window the phases are centred on: transit (phase 0) for
        transmission, secondary eclipse (phase 0.5, circular-orbit approximation)
        for emission.
    n_phases : int, optional
        Number of representative phases. Defaults to 4 for transmission (spread
        across the whole transit chord) or 2 for emission (just before/after
        eclipse) -- Antoine's original idea for this design.

    Returns
    -------
    np.ndarray
        Representative phases, in [0, 1).
    """
    # `planet.trandur`/`planet.period` are astropy Quantities but not necessarily
    # true scalars -- `Planet.__init__` stores them as 1-element arrays (straight
    # from the ExoFile table query) -- `float(...)` coerces either shape to a plain
    # Python scalar so `half_width` doesn't silently broadcast into an extra axis
    # of `np.linspace` below (same array-vs-scalar gotcha other code in this file
    # works around with explicit `[0]` indexing, e.g.
    # `_build_default_rotation_kernel`'s `planet.period[0].to('s').value`).
    half_width = float((planet.trandur.to(u.d) / 2 / planet.period.to(u.d)).decompose().value)

    if kind_trans == 'emission':
        center = 0.5
        if n_phases is None:
            n_phases = 2
    else:
        center = 0.0
        if n_phases is None:
            n_phases = 4

    if n_phases == 1:
        phases = np.array([center])
    else:
        phases = np.linspace(center - half_width, center + half_width, n_phases)

    return phases % 1


def prepare_model_multi_reg_low(theta_regions, mode: str = 'low'):
    """Generate and combine a whole-visit (no genuine per-exposure timing) model for
    every region, averaged over a handful of representative phases instead of a real
    exposure sequence (Chantier A Phase 3f; `mode='high'` added in Chantier A Phase
    4 -- see below).

    LOW RES counterpart to `prepare_model_multi_reg_high_per_exposure`: reuses the
    exact same multi-region kernel machinery (`_prepare_fp_native_by_region`,
    `_build_multi_region_kernel`, `model_sequence.build_model_sequence`) rather than
    a separate low-res-specific combination scheme -- neither `get_ker`'s kernel
    contract nor `model_sequence.py`'s helpers assume anything about resolution, and
    low-res data is degraded to instrument resolution downstream anyway
    (`prepare_photometry`/`prepare_spectrophotometry`), exactly like high-res is
    degraded downstream of the per-exposure engine.

    Low-res data has no real per-exposure time series (it is typically integrated
    over a whole visit), so there is no genuine sequence of exposures to loop
    `build_model_sequence` over. Instead, a *fake* sequence is built out of
    `representative_phases_low` (computed once in `setup_retrieval`, see
    `get_representative_low_res_phases`): every "exposure" evaluates the same
    native wavelength grid, so the only thing that varies between them is which
    per-region kernel `get_ker` returns at that phase. The resulting per-phase
    spectra are then simply averaged. No per-exposure Doppler shift is needed here
    (`vrp_orb=vr_orb=0`): unlike high-res, low-res only ever applies a single,
    fixed Doppler shift (systemic velocity + `rv`), the same regardless of phase,
    applied once after the region combination -- see `get_low_res_dv_shift`.

    No explicit resolution degradation happens here (unlike the HIGH RES per-
    exposure path, which degrades `Fstar` to `Raf` once) -- `prepare_photometry`/
    `prepare_spectrophotometry` already degrade+bin downstream, once per
    instrument, at that instrument's own resolution (`infos['res']`); degrading
    here too, at some other shared resolution, would either be wrong (a single
    resolution can't fit every low-res instrument if they differ) or degrade
    twice (the same class of bug fixed by Chantier A Phase 1 -- conflating an
    already-degraded resolution with the model's true physical resolution).

    `mode='high'` (Chantier A Phase 4): `lnprob`'s LOW RES block reuses this same
    function, unchanged beyond the `mode` string, for spectrophotometric/
    photometric instruments whose data is synthesized from the high-res model
    (`model_type == 'high'` -- a real high-res spectrograph's range, or an
    `opacity_mode: 'lbl'` low-res instrument's own range folded into
    `wv_range_high`). That reuse is *also* a whole-visit-integrated comparison with
    no genuine per-exposure timing (spectrophotometric/photometric data is not a
    real high-res exposure sequence, regardless of which atmo objects generate the
    underlying spectrum) -- the same representative-phase averaging applies
    uniformly, whether or not a real high-res visit happens to exist elsewhere in
    the run (previously, this reuse path pulled a real visit's own mean phase via
    `prepare_model_multi_reg`'s `visit_i`, a mismatch: that visit's timing has
    nothing to do with the low-res data being synthesized, and simply didn't exist
    for a pure LRR run on lbl-flagged low-res data alone).

    Parameters
    ----------
    theta_regions : list of dict
        One dict per region (`len(theta_regions) > 1`), as produced by
        `unpack_theta`.
    mode : {'low', 'high'}, default 'low'
        Which resolution's atmo objects/species/stellar spectrum to use
        (`wv_range_{mode}`, `atmo_{mode}_i`, `fct_star_{mode}`, `prt_res[mode]`,
        `linelist_names[mode]` via `_prepare_fp_native_by_region`).

    Returns
    -------
    wv_out : np.ndarray
    model_out : np.ndarray
        Same signature as `prepare_model_high_or_low(theta_dict, mode)`.
    """
    init_atmo_if_not_done(mode)
    n_wv_rng = len(globals()[f'wv_range_{mode}'])
    atmo_obj_list = [globals()[f'atmo_{mode}_{i_rng}'] for i_rng in range(n_wv_rng)]
    init_stellar_spectrum_if_not_done(mode)
    fct_star = globals()[f'fct_star_{mode}']

    wave_native, Fp_by_region, Fstar_native = _prepare_fp_native_by_region(
        theta_regions, atmo_obj_list, fct_star, mode=mode)

    # `get_ker`'s documented contract reads `instrum['resol']` (see
    # `retrievals/retrieval_inputs_example_rotation.yaml`'s "Rotation kernel
    # function" block) -- low-res instrument dicts (`spectrophotometric_data`/
    # `photometric_data`) use the key `'res'` instead, and there is no single
    # low-res "instrument" the way there is a high-res visit
    # (`instrum_param_list[visit_i]`) -- `prt_res[mode]`, the model's own native
    # sampling resolution, is the only resolution genuinely defined at this stage.
    instrum_for_ker = {'resol': prt_res[mode]}
    region_kernel_fct = _build_multi_region_kernel(theta_regions, visit_i=0, mode=mode,
                                                    instrum=instrum_for_ker)

    # Edge-trim to match combine_regions_with_kernel's convolution-boundary
    # convention (same 15-point trim as prepare_model_multi_reg_high_per_exposure's
    # `wave_out`) -- Fp_by_region itself stays native/untrimmed, as
    # combine_regions_with_kernel expects.
    wave_out = wave_native[15:-15]
    Fstar_out = Fstar_native[15:-15] if Fstar_native is not None else None

    n_phases = len(representative_phases_low)
    data_wave = np.tile(wave_out[np.newaxis, np.newaxis, :], (n_phases, 1, 1))

    model_seq_fake = model_seq.build_model_sequence(
        wave_out, Fp_by_region, data_wave, vrp_orb=0.0, Fstar=Fstar_out, vr_orb=0.0,
        alpha=1.0, kind_trans=kind_trans, RV=0.0, region_kernel=region_kernel_fct,
        phase=representative_phases_low)
    injected_avg = model_seq_fake.mean(axis=0).filled(np.nan)[0]

    # `build_model_sequence` returns the per-exposure *injection* formula
    # (`1 + alpha*depth` in emission, `1 - alpha*depth` in transmission -- meant
    # for HIGH RES time series compared against PCA-detrended data around a unity
    # baseline), not the raw depth/ratio itself. LOW RES callers downstream
    # (`prepare_photometry`/`prepare_spectrophotometry`, and the single-region
    # `prepare_model_high_or_low` path they also consume) all expect the raw,
    # unwrapped quantity (`Fp/Fstar` in emission, transit depth in transmission) --
    # undo the injection formula here (`alpha=1.0`, so this is exact, not an
    # approximation) rather than changing what every other LOW RES consumer expects.
    if kind_trans == 'emission':
        model_avg = injected_avg - 1.0
    else:
        model_avg = 1.0 - injected_avg

    # Same fixed systemic-velocity shift as the single-region low-res path
    # (`get_low_res_dv_shift`) -- applied once, after averaging, since it is the
    # same for every representative phase.
    dv_shift = get_low_res_dv_shift(theta_regions[0])
    wv_out = wave_out * calc_shift(dv_shift, kind='rel')

    return wv_out, model_avg


def prepare_static_model(theta_regions, mode: str, Raf: Optional[float] = None, atmo_obj=None):
    """Generate a whole-visit (no genuine per-exposure timing) model spectrum, for
    any region count and mode -- the single entry point for this need (Chantier A
    Phase 3f/4), used both by `lnprob`'s LOW RES block (`needs_low_model`/
    `needs_high_model`) and by post-retrieval analysis
    (`retrieval_utils.get_contribution`), so both compute this the exact same way.

    Dispatches to `prepare_model_multi_reg_low` (multi-region: phase-averaged over
    `representative_phases_low`, `Raf`/`atmo_obj` unused -- see that function's
    docstring for why no explicit degradation happens there) or
    `prepare_model_high_or_low` (single-region: no combination needed) +
    `get_low_res_dv_shift` applied explicitly for `mode != 'low'`
    (`prepare_model_high_or_low`'s `mode == 'low'` branch already applies it
    internally; `prepare_model_multi_reg_low` applies it internally for every mode).

    Parameters
    ----------
    theta_regions : list of dict
        One dict per region, as produced by `unpack_theta`.
    mode : {'low', 'high'}
    Raf : float, optional
        Target resolving power for the single-region, `mode != 'low'` path only
        (forwarded to `prepare_model_high_or_low`). Defaults to `get_res_instru`
        (safe even with no real high-res instrument at all) if not given --
        `lnprob` passes its own precomputed `res_instru` instead of recomputing it
        every call; other callers (e.g. `get_contribution`, for post-retrieval
        analysis) can just omit it.
    atmo_obj : optional
        Forwarded to `prepare_model_high_or_low` (single-region only -- the
        multi-region path always uses the module's own cached atmo objects for
        `mode`, see `prepare_model_multi_reg_low`).

    Returns
    -------
    wv_out : np.ndarray
    model_out : np.ndarray
    actual_res : float
        The resolving power `model_out` is actually at: `prt_res[mode]` (native --
        multi-region, or single-region `mode == 'low'`) or the single-region
        `mode != 'low'` path's effective `Raf`. Callers that degrade further
        downstream (e.g. `prepare_photometry`/`prepare_spectrophotometry`) must use
        this, not assume a fixed resolution regardless of region count.
    """
    if len(theta_regions) > 1:
        wv_out, model_out = prepare_model_multi_reg_low(theta_regions, mode=mode)
        return wv_out, model_out, prt_res[mode]

    if mode != 'low' and Raf is None:
        Raf = get_res_instru(instrum_param_list, prt_res['high'])

    wv_out, model_out = prepare_model_high_or_low(theta_regions[0], mode, Raf=Raf, atmo_obj=atmo_obj)

    if mode == 'low':
        return wv_out, model_out, prt_res['low']

    dv_shift = get_low_res_dv_shift(theta_regions[0])
    wv_out = wv_out * calc_shift(dv_shift, kind='rel')
    return wv_out, model_out, Raf


def prepare_photometry(wv_mod: np.ndarray, spec_mod: np.ndarray, model_res: float, data_info: dict,
                        mod_sampling: Optional[float] = None, integrate_fct: str = 'simpson'):
    """Degrade a model spectrum to instrument resolution and integrate it over photometric bands.

    Parameters
    ----------
    wv_mod : np.ndarray
        Model wavelength grid.
    spec_mod : np.ndarray
        Model flux values, same shape as `wv_mod`.
    model_res : float
        Native/physical resolving power of the model spectrum (`Rbf`).
    data_info : dict
        Photometric data description, with keys 'wv_range', 'res', 'wave',
        'response_fcts' (one response function per band).
    mod_sampling : float, optional
        Unused by the degradation step itself; kept for interface consistency with
        `prepare_spectrophotometry`. Defaults to `model_res`.
    integrate_fct : str
        Name of the `scipy.integrate` function used to integrate the response
        function over each band.

    Returns
    -------
    wv_band : np.ndarray
        Central wavelength of each photometric band.
    mod_out : np.ndarray
        Model flux integrated over each band's response function.
    """
    if mod_sampling is None:
        mod_sampling = model_res

    if isinstance(integrate_fct, str):
        integrate_fct = getattr(scipy.integrate, integrate_fct)

    # Get the values needed from the data_info dictionary
    info_keys = ['wv_range', 'res', 'wave', 'response_fcts']
    wv_rng, instru_res, wv_band, fct_band = (data_info[key] for key in info_keys)

    # First downgrade to a lower resolution to make sure the spectrum is smooth
    cond = (wv_mod >= wv_rng[0]) & (wv_mod <= wv_rng[-1])
    wv_mod_sub, spec_mod_sub = wv_mod[cond], spec_mod[cond]
    # Pass the full wv_mod/spec_mod (not the wv_rng-cropped _sub arrays) so
    # degrade_and_resample has margin to pad internally without clipping wv_mod_sub's
    # edges (Chantier A Phase 1 -- see convolution.py::degrade_and_resample).
    resamp_prt = degrade_and_resample(wv_mod, spec_mod, resolution=instru_res,
                                       input_resolution=model_res, sample=wv_mod_sub)

    # Apply the response function to the spectrum
    mod_out = list()
    for fct_i in fct_band:
        response_i = fct_i(wv_mod_sub)
        norm = integrate_fct(response_i, x=wv_mod_sub)
        mod_i = integrate_fct(response_i * resamp_prt, x=wv_mod_sub) / norm
        mod_out.append(mod_i)
    mod_out = np.array(mod_out)
    

    return wv_band, mod_out


def prepare_spectrophotometry(wv_mod: np.ndarray, spec_mod: np.ndarray, model_res: float, data_info: dict,
                               mod_sampling: Optional[float] = None):
    """Degrade a model spectrum to instrument resolution and project it onto a spectrophotometric grid.

    Parameters
    ----------
    wv_mod : np.ndarray
        Model wavelength grid.
    spec_mod : np.ndarray
        Model flux values, same shape as `wv_mod`.
    model_res : float
        Native/physical resolving power of the model spectrum (`Rbf`).
    data_info : dict
        Spectrophotometric data description, with keys 'wv_range', 'res', 'wave'.
    mod_sampling : float, optional
        Sampling density (in resolving power) used for the box-binning step before
        the final interpolation. Defaults to `model_res`.

    Returns
    -------
    wv_grid : np.ndarray
        Instrument wavelength grid (from `data_info['wave']`).
    mod : np.ndarray
        Model flux projected onto `wv_grid`.
    """
    if mod_sampling is None:
        mod_sampling = model_res

    # Get the values needed from the data_info dictionary
    wv_rng, instru_res, wv_grid = (data_info[key] for key in ['wv_range', 'res', 'wave'])

    # TODO: Add the possibility to use unequal spectral bins
    # The binning function spectrum.box_binning needs to be replaced
    # because it assumes evenly spaced grid for now.
    # The function that reads the spectrophotometry should also
    # be changed to be able to read bin limits.

    # Downgrade to instrument resolution. Pass the full wv_mod/spec_mod (not the
    # wv_rng-cropped array) so degrade_and_resample has margin to pad internally
    # without clipping wv_mod[cond]'s edges (Chantier A Phase 1).
    cond = (wv_mod >= wv_rng[0]) & (wv_mod <= wv_rng[-1])
    resamp_prt = degrade_and_resample(wv_mod, spec_mod, resolution=instru_res,
                                       input_resolution=model_res, sample=wv_mod[cond])

    # Bin the spectrum and interpolate
    # TODO: replace the binning function, which is just a box convolution for now.
    binned_prt = spectrum.box_binning(resamp_prt, mod_sampling / instru_res)
    fct_prt = interp1d(wv_mod[cond], binned_prt)
    # Project into instrument wv grid
    mod = fct_prt(wv_grid)

    return wv_grid, mod


def lnprob(theta, ):
    """Log-probability (prior + logL) for one MCMC step.

    High-res block: uses the Fp/Fstar-separated engine (`model_sequence.py`, Chantier
    A Phase 2 -- fixes bug #2, star's reflex RV no longer dragged at the planet's
    orbital velocity) unconditionally. Wind broadening (transmission) has its own
    correct, unified path here too since Chantier A Phase 3
    (`rotation_kernel: 'transmission'` -> `spectrum.RotKerTransit`, computed once per
    theta in `precompute_theta_model`) -- the older combined-ratio fallback this
    docstring used to describe (`correlation.py::calc_log_likelihood_grid_retrieval`,
    a cruder `RotKerTransitCloudy(gauss=True)` kernel) was removed 2026-08-28: no
    real config ever set `wind` to trigger it, and it duplicated what Phase 3 already
    does correctly. See `apply_alpha`/`use_real_stellar_rv` (set in `setup_retrieval`,
    same YAML keys as `logl_grid.py::setup_logl_grid`) for the two accuracy/cost knobs
    of the engine.
    """
    global params_prior, retrieval_type, kind_trans, orders, white_light
    global photometric_data, spectrophotometric_data
    
    log.debug(f"In `lnprob`, input array = {theta}")
    
    # --- Prior ---
    log.debug('Commpute Prior')
    total = ru.log_prior(theta, params_prior, prior_func_dict=prior_func_dict)

    if not np.isfinite(total):
        log.debug('Prior = -inf')
        return -np.inf

    theta_regions = unpack_theta(theta)
    
    # First check if the TP profile gives negative temperatures. Discard if so.
    for theta_dict in theta_regions:
        if np.any(theta_dict['temperatures'] < 0):
            log.debug('Negative temperatures in TP profile. >>> return -np.inf')
            return -np.inf

    ####################
    # --- HIGH RES --- #
    ####################
    # High res is needed in joint retrievals or High-res retrievals
    if (retrieval_type == 'JR') or (retrieval_type == 'HRR'):

        # For the rest, just use the first region (we only need general informations)
        theta_dict = theta_regions[0]

        # Chantier A Phase 3: true multi-region (more than one entry in theta_regions
        # -- e.g. citrus/longitude slices, the documented get_ker example, but any
        # user-defined region split works the same way) needs a per-exposure kernel
        # -- regions rotate into/out of view across a visit, so they cannot be
        # combined once per theta the way the single-region fast path below does.
        # Same for every visit (depends only on theta_regions), hoisted out of the
        # per-visit loop below.
        is_multi_region = len(theta_regions) > 1

        # Chantier A Phase 4 (optimization): the native (undegraded) petitRADTRANS
        # spectrum depends only on theta/region, never on the visit -- generate it
        # once per theta here (shared across every visit, even across different
        # instruments/resolutions), instead of once per visit inside the loop below
        # (a real, avoidable petitRADTRANS call per visit -- found investigating a
        # stale "not optimal to re-compute the model for each sequence" comment;
        # `_prepare_fp_native_by_region` already generalizes to a single region, so
        # this covers both the multi-region and single-region cases identically).
        # Only the per-visit degradation to that visit's own instrument resolution
        # (`instru_res`, Phase 4) and, for multi-region, the per-exposure kernel
        # genuinely depend on the visit.
        init_atmo_if_not_done('high')
        n_wv_rng_high = len(wv_range_high)
        atmo_obj_list_high = [globals()[f'atmo_high_{i_rng}'] for i_rng in range(n_wv_rng_high)]
        init_stellar_spectrum_if_not_done('high')
        fct_star_high = globals()['fct_star_high']
        native_res = prt_res['high']
        wave_native, Fp_native_by_region, Fstar_native = _prepare_fp_native_by_region(
            theta_regions, atmo_obj_list_high, fct_star_high)

        if not all(np.isfinite(Fp_i[100:-100]).all() for Fp_i in Fp_native_by_region):
            log.warning("NaN in high res model spectrum encountered")
            return -np.inf

        # Chantier A Phase 4: one entry per visit, kept separate (not pooled into
        # one running concatenation) -- how visits get grouped before the log is
        # taken is decided once, after this loop, by `logl_grouping` (see
        # model_seq.group_visit_indices).
        visit_terms = []
        # --- Computing the logL for all sequences
        for visit_i, data_visit_i in enumerate(data_visits):

            vrp_orb = rv_theo_t(theta_dict['kp'],
                                data_visit_i['t_start'] * u.d, planet.mid_tr,
                                planet.period, plnt=True).value

            # Chantier A Phase 4: degrade to *this visit's* own instrument
            # resolution, not the shared res_instru (max across every high-res
            # instrument in the run) -- a lower-resolution instrument's exposures
            # must not be compared to a model that was only ever blurred to a finer
            # instrument's resolution.
            instru_res = instrum_param_list[visit_i]['resol']

            if is_multi_region:
                wv_high, Fp_by_region, Fstar_high, region_kernel_fct = \
                    prepare_model_multi_reg_high_per_exposure(
                        theta_regions, visit_i, instru_res,
                        native=(wave_native, Fp_native_by_region, Fstar_native))

                if not all(np.isfinite(Fp_i[100:-100]).all() for Fp_i in Fp_by_region):
                    log.warning("NaN in high res model spectrum encountered")
                    return -np.inf

                # Per-exposure orbital phase, forwarded to region_kernel (unlike
                # prepare_model_multi_reg's mean_phase, used only by the old
                # combined-ratio path -- the whole point here is per-exposure).
                phase_i = (data_visit_i['t_start'] - planet.mid_tr.value) \
                    / planet.period.to('d').value % 1

                # Reconstruct the combined ratio too (LOW RES block further down),
                # same reasoning as the single-region branch below, using the mean
                # phase across the visit as a representative combination (same
                # simplification the old combined-ratio multi-region path always
                # used, since that block has no per-exposure Doppler shift of its own).
                mean_phase = np.mean(phase_i[data_visit_i['i_pl_signal']])
                Fp_high = region_kernel_fct(wv_high, Fp_by_region, mean_phase)
                model_high = Fp_high / Fstar_high if Fstar_high is not None else Fp_high
            else:
                # Chantier A Phase 4: only the (cheap) degradation to this visit's
                # own instrument resolution happens here now -- the (expensive)
                # native spectrum was already generated once, above, outside this
                # loop (`_prepare_fp_native_by_region`, with a single region).
                wv_high, Fp_high, Fstar_high = model_seq.degrade_fp_fstar(
                    wave_native, Fp_native_by_region[0], Fstar_native,
                    resolution=instru_res, native_resolution=native_res,
                    theta_dict=theta_dict, planet=planet, rotation_kernel=rotation_kernel)

                if not np.isfinite(Fp_high[100:-100]).all():
                    log.warning("NaN in high res model spectrum encountered")
                    return -np.inf

                # Reconstruct the combined ratio too: the LOW RES block further down
                # (Joint Retrieval, spectrophotometric/photometric data whose
                # `model_type` is 'high') still needs a single `model_high` spectrum
                # to synthesize predictions from -- that step has no per-exposure
                # Doppler shift of its own, so recombining Fp/Fstar here costs
                # nothing and keeps that path working unchanged.
                model_high = Fp_high / Fstar_high if Fstar_high is not None else Fp_high

            # --- Stellar velocity: independent from the planet's (the actual bug fix) ---
            # Fixed by default (use_real_stellar_rv=False, set in setup_retrieval):
            # the reflex motion is negligible next to the planet's orbital velocity
            # and the BERV for essentially every target (Antoine-confirmed). Use the
            # real per-exposure vr (planet_obs.py::save_sequences, Chantier A Phase 2)
            # only if the run asked for it *and* the loaded data actually has it
            # (older .npz files predating this addition fall back to None).
            if use_real_stellar_rv and data_visit_i.get('vr') is not None:
                vr_orb = data_visit_i['vr'].to(u.km / u.s).value
            else:
                vr_orb = 0.0

            # --- Occultation fraction: real light curve by default (apply_alpha) ---
            # Same YAML key/global as logl_grid.py's apply_alpha/_current_apply_alpha,
            # kept consistent between the two rather than picking a different default.
            alpha_arg = (data_visit_i['alpha_frac'] if apply_alpha
                        else np.ones_like(data_visit_i['t_start']))

            # Doppler-shift Fp and Fstar independently per exposure and recombine
            # into the model sequence compared to the data (model_sequence.py).
            # `RV_const` (BERV + stellar reflex at mid-transit + RV_sys,
            # planet_obs.py::norv_sequence) is the star-rest-frame -> data-grid
            # baseline shift and must be applied to *both* Fp and Fstar (both need
            # to land on the same data wavelength grid) -- vrp_orb/vr_orb are then
            # the differential excursions on top of that shared baseline (vrp_orb
            # is the planet's velocity *relative to the star*, not to the
            # observer). Folded into `RV` here since build_model_sequence already
            # adds `RV` to both vrp_orb and vr_orb.
            if is_multi_region:
                # NOTE: unlike the single-region branch below, no extra [20:-20]
                # margin here on top of the 15-point edge trim already applied by
                # combine_regions_with_kernel/prepare_model_multi_reg_high_per_exposure --
                # wave/Fstar_high and each region's raw Fp would need to shrink by
                # a consistent amount for build_model_sequence's per-exposure
                # spline to stay aligned, and the extra margin's purpose is not
                # documented elsewhere in the codebase. Revisit together with the
                # Narval validation of this phase if it turns out to matter.
                model_seq_i = model_seq.build_model_sequence(
                    wv_high, Fp_by_region, data_visit_i['wave'], vrp_orb,
                    Fstar=Fstar_high, vr_orb=vr_orb, alpha=alpha_arg,
                    kind_trans=kind_trans, RV=theta_dict['rv'] + data_visit_i['RV_const'],
                    region_kernel=region_kernel_fct, phase=phase_i)
            else:
                model_seq_i = model_seq.build_model_sequence(
                    wv_high[20:-20], Fp_high[20:-20], data_visit_i['wave'], vrp_orb,
                    Fstar=Fstar_high[20:-20] if Fstar_high is not None else None,
                    vr_orb=vr_orb, alpha=alpha_arg, kind_trans=kind_trans,
                    RV=theta_dict['rv'] + data_visit_i['RV_const'])

            # Remove the same number of PCs that were used during reduction --
            # same post-processing step as the old gen_model_sequence_noinj path.
            n_pc = int(data_visit_i['params'][5])
            model_norm = model_seq.apply_pca_to_model(
                model_seq_i, data_visit_i['pca'], n_pca=n_pc) / data_visit_i['noise']

            # Cross/squared terms computed per order, kept separate (not yet
            # combined into chi2, not yet log-transformed) -- same terms
            # logl_grid.py's get_logl() combines (`_chi2_from_terms`/
            # `_logl_from_chi2_terms`, Chantier A Phase 2/4) instead of the older
            # correlation.py::calc_logl_BL_ord. Kept per visit here (not pooled
            # into a running concatenation across visits) so that grouping
            # (`logl_grouping`, below) can pool exactly the visits it should --
            # e.g. per instrument -- and no others.
            flux = data_visit_i['flux']
            ct_tr = np.ma.zeros((model_norm.shape[0], model_norm.shape[1]))
            st_tr = np.ma.zeros((model_norm.shape[0], model_norm.shape[1]))
            for iOrd in range(model_norm.shape[1]):
                if flux[:, iOrd].mask.all():
                    continue
                ct_tr[:, iOrd] = np.ma.sum(model_norm[:, iOrd] * flux[:, iOrd], axis=-1)
                st_tr[:, iOrd] = np.ma.sum(model_norm[:, iOrd] ** 2, axis=-1)

            if not np.isfinite(_chi2_from_terms(ct_tr, st_tr, data_visit_i['s2f'])).all():
                return -np.inf

            data_info_i = data_info_list[visit_i]
            visit_terms.append(dict(
                ct=ct_tr, st=st_tr, sf=data_visit_i['s2f'],
                N=data_info_i['all_N'], icorr=data_info_i['all_icorr'],
                alpha_frac=data_info_i['all_alpha_frac'],
                instrument=instrum_param_list[visit_i]['name'],
            ))

        # --- Chantier A Phase 4: group visits, then take the log once per group ---
        # The scaling-free logL prescription (Brogi & Line 2019, `_logl_from_chi2_terms`,
        # kind='BL') pools raw chi2 terms and N *within* a group and takes the log
        # once for that group -- see model_seq.group_visit_indices's docstring for
        # why grouping (not just concatenating everything, the old behaviour) is
        # needed for a genuinely multi-instrument run, and why combining across
        # groups afterward is just a sum of the resulting scalar logL values
        # (independent datasets -> additive log-likelihoods), nothing fancier.
        orders_sel = slice(None) if orders is None else orders
        instrument_keys = [vt['instrument'] for vt in visit_terms]
        for group in model_seq.group_visit_indices(instrument_keys, logl_grouping):
            # Concatenate only this group's visits -- icorr is a per-visit index
            # into that visit's own exposures, so it needs the same running
            # offset the old code applied globally, just scoped to the group.
            ct_list, st_list, sf_list, N_list, icorr_list, alpha_list = [], [], [], [], [], []
            offset = 0
            for visit_i in group:
                vt = visit_terms[visit_i]
                ct_list.append(vt['ct'])
                st_list.append(vt['st'])
                sf_list.append(vt['sf'])
                N_list.append(vt['N'])
                alpha_list.append(vt['alpha_frac'])
                icorr_list.append(vt['icorr'] + offset)
                offset += vt['ct'].shape[0]

            ct_g = np.ma.concatenate(ct_list, axis=0)
            st_g = np.ma.concatenate(st_list, axis=0)
            sf_g = np.ma.concatenate(sf_list, axis=0)
            N_g = np.ma.concatenate(N_list, axis=0)
            alpha_g = np.concatenate(alpha_list, axis=0)
            icorr_g = np.concatenate(icorr_list, axis=0)

            # Per-exposure light-curve weighting (apply_alpha/all_alpha_frac):
            # weighting the summed chi2 by alpha_frac is the same as weighting
            # ct/st/sf individually before summing (linear in each term, model
            # scaling alpha fixed at 1 here) -- matches the old
            # correlation.py::sum_logl behaviour exactly.
            w = alpha_g[icorr_g][:, None]
            ct_sum = np.ma.sum(ct_g[icorr_g][:, orders_sel] * w)
            st_sum = np.ma.sum(st_g[icorr_g][:, orders_sel] * w)
            sf_sum = np.ma.sum(sf_g[icorr_g][:, orders_sel] * w)
            N_sum = np.ma.sum(N_g[icorr_g][:, orders_sel])

            total += _logl_from_chi2_terms(ct_sum, st_sum, sf_sum, N_sum)

    ###################
    # --- LOW RES --- #
    ###################
    if (retrieval_type == 'JR') or (retrieval_type == 'LRR') or white_light:
        # `theta_dict` may not be `theta_regions[0]` at this point: in a pure LRR
        # run, the HIGH RES block above never runs, so `theta_dict` is whatever the
        # negative-temperature check loop (top of this function) left behind
        # (`theta_regions[-1]`). Pin it explicitly -- only matters when there is
        # more than one region, and even then only for `scale_uncert` below
        # (Chantier A Phase 3f), since the model itself is now generated from
        # `theta_regions` as a whole (`prepare_model_multi_reg_low`), not a single
        # region's `theta_dict`.
        theta_dict = theta_regions[0]

        # If at least one instrument need the low-res model, then compute it
        model_type = [infos.get('model_type', 'low') for infos
                      in list(spectrophotometric_data.values()) + list(photometric_data.values())]
        # Captured before the per-instrument loop below shadows `model_type`
        # with a single string per iteration.
        needs_low_model = 'low' in model_type
        needs_high_model = 'high' in model_type
        if needs_low_model:
            # Chantier A Phase 3f/4: whole-visit model (no genuine per-exposure
            # timing), any region count -- see `prepare_static_model`'s docstring.
            wv_low, model_low, _ = prepare_static_model(theta_regions, 'low')

            if np.sum(np.isnan(model_low)) > 0:
                log.info("NaN in low res model spectrum encountered")
                return -np.inf

        if needs_high_model:
            # Chantier A Phase 4: this path (`model_type == 'high'`,
            # `assign_model_type` -- a spectrophotometric/photometric dataset
            # whose wavelength range is fully covered by the high-res data: a
            # real high-res spectrograph's range, or an `opacity_mode: 'lbl'`
            # low-res instrument's own range folded into wv_range_high) compares
            # a single static, whole-visit spectrum, not a per-exposure sequence
            # -- exactly the same situation `needs_low_model` above is in, just
            # sourced from the high-res atmo objects instead of the dedicated
            # low-res ones (`prepare_static_model`, mode='high', regardless of
            # whether a real high-res visit happens to exist elsewhere in this
            # run -- previously, this path pulled a real visit's own mean phase
            # via `prepare_model_multi_reg`'s `visit_i`, a mismatch: that
            # visit's timing has nothing to do with the low-res data being
            # synthesized here, and simply doesn't exist at all for a pure LRR
            # run on lbl-flagged low-res data alone). `model_high_for_lowres_res`
            # (the resolution `model_high_for_lowres` is actually at -- varies
            # with region count, see `prepare_static_model`'s docstring) is
            # needed by the per-instrument loop below, which re-degrades to each
            # low-res instrument's own (coarser) resolution by
            # `prepare_photometry`/`prepare_spectrophotometry`.
            wv_high_for_lowres, model_high_for_lowres, model_high_for_lowres_res = \
                prepare_static_model(theta_regions, 'high', Raf=res_instru)

        # Iterate over all low-res spectrophotometric observations
        # NOTE: You may think that you can use the function to clean the following loop,
        #       but the problem is that passing the spectra (especially the high res one)
        #       will slow down the multiprocessing considerably.
        for low_res_data_type in ['spectrophotometric', 'photometric']:
            if low_res_data_type == 'photometric':
                prepare_fct = prepare_photometry
            else:
                prepare_fct = prepare_spectrophotometry
                
            low_res_data_dict = globals()[f'{low_res_data_type}_data']
            for instru_name, infos in low_res_data_dict.items():
                log.debug(f"Generating synthetic {low_res_data_type} data for {instru_name}")
                model_type = infos.get('model_type', 'low')
                log.debug(f"Using the {model_type}-res model to synthetize {instru_name} data.")
                if model_type == 'low':
                    args = (wv_low, model_low, prt_res['low'], infos)
                else:
                    args = (wv_high_for_lowres, model_high_for_lowres, model_high_for_lowres_res,
                             infos, prt_res['high'])
                # Generate the synthetic data
                _, synt_data = prepare_fct(*args)        
                
                # Get data measured by the instrument
                data, uncert = infos['data'], infos['err']
                
                # In white-light mode, use the mean of the data
                if white_light:
                    log.debug(f"Using white light from {instru_name}.")
                    synt_data = np.mean(synt_data)
                    data = np.mean(data)
                    uncert = np.sqrt(np.sum(uncert ** 2)) / len(uncert)
                    
                # Compute the log likelihood
                scale_uncert = theta_dict.get(f'log_f_{instru_name}', 1)
                total += corr.calc_logl_chi2_scaled(data, uncert, synt_data, scale_uncert)

        # Pre-existing latent bug fixed in passing (Chantier A Phase 4): these
        # were deleted unconditionally, but only exist when their respective
        # `needs_*_model` flag is True -- in JR, if `assign_model_type` ever
        # assigned every low-res instrument to `model_type == 'high'` (none
        # left needing the dedicated low-res model), the unconditional
        # `del wv_low, model_low` would have raised a `NameError`.
        if needs_low_model:
            del wv_low, model_low
        if needs_high_model:
            del wv_high_for_lowres, model_high_for_lowres

    if retrieval_type != 'LRR':
        # Chantier A Phase 2: the Fp/Fstar-separated engine also sets Fp_high/
        # Fstar_high (in addition to the reconstructed model_high, see the HIGH
        # RES block above) -- free those too.
        del wv_high, model_high, Fp_high, Fstar_high
        # Chantier A Phase 3: the multi-region branch also keeps each region's
        # raw Fp and a per-visit closure alive -- free those too.
        if is_multi_region:
            del Fp_by_region, region_kernel_fct

    gc.collect()

    log.debug(f'logL = {total}')

    return total




def save_yaml_file_with_version(yaml_file_in, yaml_file_out, output_dir=None, **kwargs):

    with open(yaml_file_in, 'r') as f:
            params_yaml = yaml.load(f, Loader=yaml.FullLoader)

    # Add the version of starships to the yaml file
    params_yaml['starships_version'] = starships.__version__
    
    # Edit some values to make sure it is consistent with the current run
    params_yaml['walker_file_out'] = str(globals()['walker_file_out'])
    
    # Add the JOB ID
    params_yaml['slurm_id'] = get_slurm_id()
    
    # Edit all other values that have been specify with kwargs
    for key, val in kwargs.items():
        # Make sure they are valid values for a yaml file
        if isinstance(val, Path):
            val = str(val)
        elif isinstance(val, np.ndarray):
            val = val.tolist()
        params_yaml[key] = val

    if output_dir is None:
        output_dir = Path.cwd()
    else:
        # Make sure it exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    yaml_file_out = output_dir / yaml_file_out

    log.info(f'Saving the yaml file to: {yaml_file_out}')

    with open(yaml_file_out, 'w') as f:
        yaml.dump(params_yaml, f, sort_keys=False)

    return yaml_file_out


def prepare_run(yaml_file=None, **kwargs):

    # walker_file_out needs to be specificaly defined as global
    # because a value can be assigned in the function (I think... anyway, it was raising an error if not)
    global walker_file_out
    
    # Other globals used in the function
    global n_steps_burnin, n_steps_sampling, n_walkers, n_dim
    global walker_path, n_cpu, slurm_array_behaviour, retrieval_type

    if yaml_file is not None:
        # Unpack the yaml_file and add variables to the global space
        setup_retrieval(yaml_file, **kwargs)
    elif kwargs:
        raise NotImplementedError("kwargs passed without yaml_file. Not implemented yet.")

    # Read the data and add them to the global space
    if retrieval_type == 'JR' or retrieval_type == 'HRR':
        _ = load_high_res_data()
    else:
        log.info(f"Retrieval type = {retrieval_type}. High res data is not needed.")

    ############################
    # Define additional parameters that are not in the yaml file
    ############################

    # Define n_steps based on the retrieval phase (burnin or sampling)
    n_steps = n_steps_burnin if walker_file_in is None else n_steps_sampling
    log.info(f"Number of steps: {n_steps}")

    ############################
    # Start retrieval!
    ############################

    warnings.simplefilter("ignore", FutureWarning)
    # warnings.simplefilter("ignore", RuntimeWarning)

    # --- Walkers initialisation ---
    # -- Either a random uniform initialisation (for every parameters)
    if walker_file_in is None:
        pos = walker_init
    elif init_mode == 'from_burnin':
        pos, _ = ru.init_from_burnin(n_walkers, wlkr_file=walker_file_in, wlkr_path=walker_path, n_best_min=10000)
    elif init_mode == 'continue':
        # Last step of the chain
        pos = ru.read_walkers_file(walker_path / walker_file_in, discard=0)[-1]
    else:
        raise ValueError(f"{init_mode} not valid.")

    log.info(f"(Number of walker, Number of parameters) = {pos.shape}")

    # Pre-run the log likelihood function
    log.info("Checking if log likelihood function is working.")
    good_to_go = False
    for i_walker in range(n_walkers):
        logl = lnprob(pos[i_walker])
        if np.isfinite(logl):
            good_to_go = True
            log.info("log likelihood function is indeed working! Success!")
            break
    else:
        log.warning("log likelihood function test was not successful... (sad face)")

    # Add index to the file name if slurm array is used in sbatch
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        if slurm_array_behaviour == 'burnin':
            idx_file = os.environ['SLURM_ARRAY_TASK_ID']
            walker_file_out = walker_file_out.with_stem(f'{walker_file_out.stem}_{idx_file}')
            log.info(f'Using SLURM_ARRAY_TASK_ID={idx_file} detected. This will be added to `walker_file_out`.')
        elif slurm_array_behaviour is None:
            log.info('SLURM_ARRAY_TASK_ID detected but not used. slurm_array_behaviour is None.')
        else:
            raise ValueError(f"slurm_array_behaviour = {slurm_array_behaviour} not valid.")
    
    # Make sure file does not already exist
    if init_mode != 'continue':
        file_stem = walker_file_out.stem
        for idx_file in range(100):
            if (walker_path / walker_file_out).is_file():
                log.info(f'File {walker_file_out} already exists.')
                walker_file_out = walker_file_out.with_stem(f'{file_stem}_{idx_file}')
                log.info(f'Trying {walker_file_out}')
            else:
                break
        else:
            raise ValueError('Walker File already exists.')
    log.info(f'Output walker file: {walker_file_out}')

    return n_steps, pos, walker_file_out, yaml_file, good_to_go


# Define the main function that will be called by the script
def main(yaml_file=None, **kwargs):
    
    global params_file_out, params_path

    # Read the input yaml file passed from command line
    if yaml_file is None:
        log.info("`yaml_file` not specified. Assuming the code is run from command line.")
        
        # Read command line arguments (more importantly, the yaml_file)
        log.info("Reading arguments from command line...")
        log.debug("kwargs in main() are not used. Taking the command line kw instead.")
        kwargs = unpack_kwargs_from_command_line(sys.argv)
        yaml_file = pop_kwargs_with_message('yaml_file', kwargs)

        # Now that yaml_file is removed from the kwargs from command line,
        # print all the other kwargs passed (if there are any)
        if kwargs:
            log.info(f"keys from command line will replace values in yaml file: {kwargs.keys()}")
            log.info("Converting command line arguments to expected type if needed...")
            kwargs = convert_cmd_line_to_types(kwargs)
    
    # Prepare the run
    n_steps, pos, walker_file_out, yaml_file, good_to_go = prepare_run(yaml_file=yaml_file, **kwargs)
    n_walkers, ndim = pos.shape
    
    if not good_to_go:
        raise ValueError("Retrieval not initialized correctly.")

    # Save the yaml file with the version of starships
    # Only if not in slurm array mode
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        if slurm_array_behaviour=='burnin':
            # Save only if == 1
            if os.environ['SLURM_ARRAY_TASK_ID'] == '1':
                yaml_file = save_yaml_file_with_version(yaml_file, params_file_out, output_dir=params_path)
            else:
                msg = f"SLURM_ARRAY_TASK_ID != 1 and 'slurm_array_behaviour'={slurm_array_behaviour}. "
                msg += "Not saving the yaml file."
                log.warning(msg)
        else:
            raise ValueError(f"slurm_array_behaviour = '{slurm_array_behaviour}' not valid.")    
    else:
        yaml_file = save_yaml_file_with_version(yaml_file, params_file_out, output_dir=params_path, **kwargs)

    # --- backend to track evolution ---
    # Create output directory if it does not exist
    walker_path.mkdir(parents=True, exist_ok=True)
    backend = emcee.backends.HDFBackend(walker_path / walker_file_out)

    # Run it!
    with Pool(n_cpu) as pool:
        log.info('Initialize sampler...')
        sampler = emcee.EnsembleSampler(n_walkers, ndim, lnprob,
                                        pool=pool,
                                        backend=backend, a=2)  ### step size -- > à changer
        log.info('Starting the retrieval!')
        sampler.run_mcmc(pos, n_steps, progress=False)  # , skip_initial_state_check=True)

    log.info('End of retrieval. It seems to be a success!')
    
    
if __name__ == '__main__':
    main()

# %%
