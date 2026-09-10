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
import starships.transpec as ts
import starships.homemade as hm

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

    # Sept 2024: not using path_fig anymore, as using a single out_dir is more flexible for the pipeline
    # path_fig = str(path_fig) + '/' 

    # instantiating the nested directories to be used later for saving plots
    classic_ccf_dir = out_dir / Path('Results') / Path('CCF_classic')
    classic_ccf_dir.mkdir(parents=True, exist_ok=True)

    injected_ccf_dir = out_dir / Path('Results') / Path('CCF_injected')
    injected_ccf_dir.mkdir(parents=True, exist_ok=True)

    red_steps_dir = out_dir / Path('Results') / Path('Reduction_steps')
    red_steps_dir.mkdir(parents=True, exist_ok=True)

    ttest_dir = out_dir / Path('Results') / Path('T-test')
    ttest_dir.mkdir(parents=True, exist_ok=True)

    param_dir = out_dir / Path('Results') / Path('Parameters')
    param_dir.mkdir(parents=True, exist_ok=True)

    dirs_dict = {'scratch_dir': scratch_dir, 'out_dir': out_dir, 'classic_ccf_dir': classic_ccf_dir, 
                 'injected_ccf_dir': injected_ccf_dir, 'red_steps_dir': red_steps_dir, 'param_dir': param_dir, 'ttest_dir': ttest_dir}

    return dirs_dict # all output as posix path objects


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
        # Do not include None values in the dictionary
        if value['value'] is not None:
            pl_kwargs[key] = convert_to_quantity(value)

    return pl_kwargs


def load_custom_instrument(config_dict):
    """Register a custom instrument/DRS from `config_dict['custom_instrument_file']`, if set.

    For an instrument STARSHIPS doesn't already know about (i.e. not one of
    `starships.planet_obs.instruments_drs`' built-in entries), point
    `custom_instrument_file` at a Python file defining:

    - a module-level `read_all_sp(path, file_list, **kwargs)` function (see
      `starships.planet_obs.read_all_sp_spirou_apero` for the expected
      signature/return), and
    - a module-level `INSTRUMENT_FIELDS` dict with the header keywords/patterns
      `starships.planet_obs.register_instrument` accepts (`airmass`, `telaz`,
      `adc1`, `adc2`, `mjd`, `bjd`, `exptime`, `berv`, `list_file_patterns`).

    Mirrors `retrieval_utils.load_custom_get_ker`'s pattern for user-supplied Python files
    in a config-driven pipeline. Does nothing if `custom_instrument_file` isn't set —
    `config_dict['instrument']` is then expected to already name a built-in instrument/DRS.
    """
    custom_instrument_file = config_dict.get('custom_instrument_file')
    if not custom_instrument_file:
        return

    cstm = hm.import_module_by_path('dummy_custom_instrument', custom_instrument_file)
    pl_obs.register_instrument(config_dict['instrument'], read_all_sp=cstm.read_all_sp,
                                **cstm.INSTRUMENT_FIELDS)


def load_planet(config_dict, visit_name):

    load_custom_instrument(config_dict)

    # All the observations must be listed in files.
    # We need the e2ds and the telluric corrected spectra, plus the reconstructed telluric
    # spectra when available. Without list_recon, Observations.fetch_data() falls back to a
    # flat (all-ones) telluric correction instead of the real one -- passing it here used to
    # be commented out unconditionally, which was a real bug for datasets that DO have it (see
    # Chantier B, B0 analysis). Some DRS formats provide recon as a *separate* list of files
    # (the usual case); others bundle it as an extension of the tcorr file itself, in which
    # case fetch_data() picks it up automatically without needing a separate list at all (see
    # Chantier B, B2 follow-up -- confirmed real for a NIRPS-APERO dataset with no separate
    # recon files, but a real reconstruction spectrum embedded in every tcorr file). So: pass
    # list_recon when the separate list file exists, and let fetch_data() sort out the
    # embedded-vs-missing distinction on its own otherwise.
    obs_dir = Path(config_dict['obs_dir'])
    list_filenames = {'list_e2ds': f'list_e2ds_{visit_name}',
                    'list_tcorr': f'list_tcorr_{visit_name}'}
    recon_list_path = obs_dir / f'list_recon_{visit_name}'
    if recon_list_path.exists():
        list_filenames['list_recon'] = f'list_recon_{visit_name}'
    else:
        # fetch_data's own list_recon default is the literal string 'list_tellu_recon', not
        # None -- omitting the kwarg here would still try (and fail) to open that default
        # filename, instead of actually triggering fetch_data's own "no recon" fallback.
        list_filenames['list_recon'] = None
        print(f'No separate {recon_list_path.name} found in {obs_dir} -- will use a telluric '
              'reconstruction spectrum embedded in the tcorr files if this DRS format '
              'provides one, otherwise proceed without (flat/no-op telluric correction).')

    # check if any planet attributes were manually specified
    if bool(config_dict['pl_params']):
        pl_kwargs = pl_param_units(config_dict)
        obs = Observations(name=config_dict['pl_name'], instrument=config_dict['instrument'], pl_kwargs=pl_kwargs)
    else:
        pl_kwargs = {}
        obs = Observations(name=config_dict['pl_name'], instrument=config_dict['instrument'])

    p = obs.planet
    # Manually specified planet params for *this visit* (e.g. mid_tr for a TTV/resonant
    # system, where the transit epoch genuinely differs per visit -- load_planet is called
    # once per visit, so config_dict['pl_params'] can already vary by visit at reduction
    # time). Recorded on the planet so save_reduced_sequence can persist them into the
    # reduced file (B3, Chantier B) -- otherwise this per-visit choice would be lost as soon
    # as the file is loaded again with a different (or no) override.
    p.reduction_overrides = pl_kwargs

    # set other planet parameters
    p.A_star = np.pi*u.rad * p.R_star**2
    surf_grav = (const.G * p.M_star / p.R_star**2).cgs
    p.logg = np.log10(surf_grav.value)
    p.gp = const.G * p.M_pl / p.R_pl**2
    p.H = (const.k_B * p.Tp / (p.mu * p.gp)).decompose()
    p.sync_equat_rot_speed = (2*np.pi*p.R_pl/p.period).to(u.km/u.s)

    # Get the data
    obs.fetch_data(config_dict['obs_dir'], **list_filenames)

    # Optional: mask pixels with too little signal (e.g. detector edges/orders that should
    # already have been masked upstream in the DRS reduction, but sometimes aren't).
    # config_dict['minimum_signal'] = None (default) disables this entirely.
    minimum_signal = config_dict.get('minimum_signal')
    if minimum_signal is not None:
        new_mask = obs.count.mask | (obs.count < minimum_signal)
        obs.flux = np.ma.array(obs.flux, mask=new_mask)

    return p, obs


def build_reduction_params(config_dict, mask_tellu, mask_wings, n_pc):
    """Build a `transpec.ReductionParams` for one (mask_tellu, mask_wings, n_pc) combination.

    `mask_tellu`, `mask_wings` and `n_pc` are the parameters actually swept in
    practice (see `config_dict['mask_tellu']`/`['mask_wings']`/`['n_pc']`), passed
    explicitly since a single reduction run only ever uses one value of each.
    The remaining "deep" parameters (rarely changed, see `ReductionParams`'
    docstring) fall back to `ReductionParams`' own defaults unless explicitly
    overridden under `config_dict['reduction_params']` in the config YAML.
    """
    overrides = config_dict.get('reduction_params', {}) or {}
    return ts.ReductionParams(mask_tellu=mask_tellu, mask_wings=mask_wings, n_pc=n_pc, **overrides)


def build_trans_spec(config_dict, n_pc, mask_tellu, mask_wings, obs, planet, bad_indexs=None):

    reduction_params = build_reduction_params(config_dict, mask_tellu, mask_wings, n_pc)
    params_all = [reduction_params]

    # Always use the real systemic radial velocity (not conditional on kind_trans):
    # confirmed with Antoine that the WASP-33 example notebooks' RVsys=[0.0] just reflects
    # that WASP-33's real RV_sys happens to be close to zero, not an emission-specific rule.
    RVsys = [planet.RV_sys.value]

    # Real exposure exclusion at reduction time (not just flagged post-hoc in the saved file):
    # bad_indexs, if given, is dropped from the exposures used to build the reference spectrum
    # and run the PCA, not only recorded as metadata after the fact.
    if bad_indexs:
        all_exposures = np.arange(len(obs.filenames))
        transit_tags = [np.delete(all_exposures, bad_indexs)]
    else:
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

    # Extract the planetary signal.
    # config_dict['iout_all']: which exposures build the reference spectrum.
    # 'all' (default) = every exposure (planetary signal negligible + diluted by its own motion,
    # so this improves the reference spectrum's S/N). null/None = the real out-of-transit/eclipse
    # exposures computed from the orbit.
    # config_dict['noise_npc']: fixed number of PCA components used to estimate `noise` (B3),
    # independent of whatever `n_pc` is used for the science spectrum. Defaults to 2 (see
    # `generate_all_transits`) if not set in the config.
    # `do_tr=[1]` above means `generate_all_transits` always hands back a single-key dict --
    # this always deals with one visit at a time, so unwrap it here rather than leaking the
    # dict (only needed for the multi-visit merge machinery, e.g. `pipeline/correlations.py`'s
    # combined-visit CCF, which this reduction-only entry point does not use) to callers.
    visits = pl_obs.generate_all_transits(obs, transit_tags, RVsys, params_all, config_dict['iout_all'], counting = False,
                                        noise_npc=config_dict.get('noise_npc', 2),
                                        **kwargs_gen_tr, **kwargs_build_ts)

    return visits['1']


def save_planet_signal(visit, nametag, scratch_dir, bad_indexs=[]):
    """Save the reduced sequence to a single file (B3, Chantier B).

    Before B3, this wrote two files (a heavy "diagnostic" one with every intermediate
    reduction step, and a "light" one with only what a retrieval needs) because `n_pc` was
    baked into what got saved. Now that PCA truncation happens at read time (`n_pc` is no
    longer a reduction-time axis, see `save_reduced_sequence`), there is nothing left that
    is specific to one `n_pc` to leave out of a "light" file -- so there is only one file.
    """
    out_filename = f'retrieval_input' + nametag
    pl_obs.save_reduced_sequence(out_filename, visit, path=scratch_dir, bad_indexs=bad_indexs)


def reduction_plots(config_dict, obs, visit, n_pc, path_fig, nametag):
    if n_pc == config_dict['n_pc'][0]:
        pf.plot_night_summary_NIRPS(visit, obs, path_fig=str(path_fig.parent.parent) + '/', fig_name='')

    # plot for specified orders
    for idx_ord in config_dict['idx_ord']:
        pf.plot_steps(visit, idx_ord, path_fig=str(path_fig) + '/', fig_name = nametag + f'_ord{idx_ord}')


def reduce_data(config_dict, planet, obs, scratch_dir, out_dir, n_pc, mask_tellu, mask_wings, visit_name, plot = True, saved = False):

    # No `_pc{n_pc}` in the filename anymore (B3): the PCA fit itself does not depend on
    # n_pc (only its truncation does, applied at read time), so a reduction only needs to
    # run once per (mask_tellu, mask_wings) combination. When `run_pipe.py` sweeps several
    # `n_pc` values for the same (mask_tellu, mask_wings), the first call below actually
    # reduces and saves; every later call for a different `n_pc` hits the cache-hit branch
    # and only pays for the cheap read-time PCA truncation (`load_reduced_sequence`).
    nametag = f'_{visit_name}_maskwings{mask_wings*100:n}_masktellu{mask_tellu*100:n}'

    # Exposures to exclude for this visit, if any (was `config_dict['bad_indexs']['visit_name']` —
    # the literal string 'visit_name' instead of the variable, which raised a KeyError on any real
    # config). Computed before building the transit spectrum so exclusion is applied to the
    # reduction itself (reference spectrum, PCA), not just flagged in the saved file afterwards.
    bad_indexs = config_dict['bad_indexs'].get(visit_name, []) if config_dict['bad_indexs'] else []
    if bad_indexs:
        print('Masking exposure(s) at index(s) ', bad_indexs)

    # check if reduction already exists
    if os.path.exists(scratch_dir / f'retrieval_input{nametag}_data_trs_.npz'):
        saved = True
        print(f"Reduction already exists for {nametag}. Loading with n_pc={n_pc}...")
        visit = pl_obs.load_reduced_sequence(f'retrieval_input{nametag}_data_trs_.npz', n_pc, path=scratch_dir,
                          filename_end='', planet=planet, plot = False)

    else: # building the transit spectrum
        visit = build_trans_spec(config_dict, n_pc, mask_tellu, mask_wings, obs, planet, bad_indexs=bad_indexs)

    if saved == False:
        save_planet_signal(visit, nametag, scratch_dir, bad_indexs)

    # outputting plots for reduction steps
    if plot:
        reduction_plots(config_dict, obs, visit, n_pc, out_dir, nametag)

    return visit