
# -------------------------------------------
# ##################################################################
# ############# Old retrieval example without yaml file input ######
# ##################################################################
# --------------------------------------------


# import os
# import sys

# from pathlib  import Path
# import numpy as np
# import yaml

# import logging
# log = logging.getLogger(__name__)
# log.setLevel(logging.INFO)
# logging.basicConfig()

# import emcee
# import starships.spectrum as spectrum
# from starships.orbite import rv_theo_t
# from starships.mask_tools import interp1d_masked


# # %%
# interp1d_masked.iprint = False
# import starships.correlation as corr
# from starships.analysis import bands, resamp_model
# import starships.planet_obs as pl_obs
# from starships.planet_obs import Observations, Planet
# import starships.petitradtrans_utils as prt
# from starships.homemade import unpack_kwargs_from_command_line, pop_kwargs_with_message
# from starships import retrieval_utils as ru

# from starships.instruments import load_instrum


# import astropy.units as u
# import astropy.constants as const
# from astropy.table import Table



# from scipy.interpolate import interp1d

# import warnings

# warnings.simplefilter("ignore", UserWarning)
# warnings.simplefilter("ignore", RuntimeWarning)

# import gc

# # from petitRADTRANS import nat_cst as nc
# try:
#     from petitRADTRANS.physics import guillot_global, guillot_modif
# except ModuleNotFoundError:
#     from petitRADTRANS.nat_cst import guillot_global, guillot_modif

# # initialisation

# ##############################################################################
# # os.environ['OMP_NUM_THREADS'] = '2'
# # print_var = os.environ['OMP_NUM_THREADS']
# # print(f'OMP_NUM_THREADS = {print_var}')

# from multiprocessing import Pool





# # Get kwargs from command line (if called from command line)
# if __name__ == '__main__':
#     kw_cmd_line = unpack_kwargs_from_command_line(sys.argv)
# else:
#     kw_cmd_line = dict()

# # Set the number of cpu and walkers
# n_cpu = kw_cmd_line.pop('n_cpu', 20)
# n_cpu = int(n_cpu)  # Input is a string
# n_walkers = n_cpu * 18  # use at least 2x n_cpu

# log.info(f'Using {n_cpu} cpu and {n_walkers} walkers.')


# ####################################

# # %%

# try:
#     base_dir = os.environ['SCRATCH']
#     base_dir = Path(base_dir)
# except KeyError:
#     base_dir = Path.home()

# pl_name = 'KELT-20 b'
# # Name used for filenames
# pl_name_fname = '_'.join(pl_name.split())

# # --- Data path ---
# reduc_name = 'KELT-20b_oct2022'
# high_res_path = base_dir / Path(f'DataAnalysis/SPIRou/Reductions/{reduc_name}')
# high_res_file_stem_list = [
#     'retrieval_input_6-pc_mask_wings90_day1',
#     'retrieval_input_6-pc_mask_wings90_day2',
#     ]

# wfc3_path = Path('/home/adb/projects/def-dlafre/adb/Observations/HST/WFC3')
# # wfc3_file = Path('WASP-33_WFC3_subsample.ecsv')
# wfc3_file = Path('WASP-33_WFC3_full_res.ecsv')

# spitzer_path = Path('/home/adb/projects/def-dlafre/adb/Observations/Spitzer')
# spitzer_file = Path('WASP-33_Spitzer.ecsv')

# # - Type of retrieval : JR, HRR or LRR
# retrieval_type = 'HRR'

# # In HRR mode, use white light curve from WFC3 bandpass?
# white_light = False

# # - Chemical equilibrium?
# chemical_equilibrium = False
# # - Include dissociation in abundance profiles? (not used in chemical equilibrium)
# dissociation = True
# # - Will you add Spitzer data points?
# add_spitzer = False
# # -- which kind of TP profile you want? (guillot, modif, iso)
# kind_temp = 'modif'

# # -- How many steps
# n_steps = 3000

# # -- Name of the run (will be used to save the results)
# run_name = 'citrus_day1-2_pc6_modif_H2O'

# # -- Use retrieval parameters from a yaml file?
# use_yaml_file = False
# params_file_in = None  # if None, params_file_in = params_file_out

# # --- Save walker steps here :
# walker_path = base_dir / Path(f'DataAnalysis/SPIRou/walker_steps/{pl_name_fname}')
# walker_file_out = Path(f'walker_steps_{run_name}.h5')

# log.info(f'walker_path = {walker_path}')
# log.info(f'run_name = {run_name}')

# try:
#     idx_file = kw_cmd_line['idx_file']
#     walker_file_out = walker_file_out.with_stem(f'{walker_file_out.stem}_{idx_file}')
# except KeyError:
#     log.debug('No `idx_file` found in command line arguments.')
# log.info(f'walker_file_out = {walker_file_out}')

# # Walker file used to init walkers (set to None if not used)
# # either None, Path('file_name.h5') or walker_file_out
# walker_file_in = Path('walker_steps_citrus_day1-2_pc6_modif_H2O_burnin.h5')
# init_mode = 'burnin'  # Options: "burnin", "continue"
# if walker_file_in is None:
#     log.info('No walker_file_in given.')
# else:
#     log.info(f'walker_file_in = {walker_file_in}')
#     log.info(f'init_mode = {init_mode}')
    
# # Parameter file to save the parameters of each retrival run
# params_path = base_dir / Path(f'DataAnalysis/SPIRou/retrieval_params/{pl_name_fname}')
# params_file_out = Path(f'params_{run_name}.yaml')
# log.info(f'params_path = {params_path}')
# log.info(f'Params will be saved in {params_file_out}')

# # - Types of models : in emission or transmission
# kind_trans = 'emission'

# # %%

# # --- Data parameters
# pl_kwargs = {'M_star': 1.89 *u.M_sun,
#             'R_star': 1.60 *u.R_sun,
#             'M_pl' : 3.38 * u.M_jup,
#             'R_pl' : 1.83 * u.R_jup,
#              }
# obs = Observations(name=pl_name, pl_kwargs=pl_kwargs)

# p = obs.planet

# # --- the planet gravity and H must be changed if you change Rp and/or Tp
# p.gp = const.G * p.M_pl / p.R_pl**2
# p.H = (const.k_B * p.Tp / (p.mu * p.gp)).decompose()

# planet = p

# Kp_scale = (planet.M_pl / planet.M_star).decompose().value


# # --- For emission spectra, you need a star spectrum
# # spec_star = spectrum.PHOENIXspectrum(Teff=7400, logg=4.5)
# # spec_star = np.load('/home/adb/Models/RotationSpectra/phoenix_teff_07400_logg_4.50_Z0.0_RotKer_vsini50.npz')

# star_wv = None  # spec_star['grid']
# # star_wv = (spec_star['wave']).to(u.um).value
# star_flux = None   #spec_star['flux']  # .to(u.erg / u.cm**2 /u.s /u.cm).value
# # --------------

# # - Selecting the wanted species contributing to the opacities:
# line_opacities = ['CO', 'H2O']

# # - Adding continuum opacities:
# continuum_opacities = []

# # - Other species to add to the model (ex: e- and H needed for H- opacity)):
# other_species = []

# # all_species = line_opacities + continuum_opacities + other_species  # was not used

# # Associate each species to a linelist name
# # if set to None, prt.select_mol_list will use the default name for each type (high or low res)
# linelist_names = {'high': None, 'low': None}

# ## --- HIGH RES DATA ---

# # --- IF HRR OR JR, YOU NEED TO INCLUDE HIGH RES DATA -----
# # - Data resolution and wavelength limit for the instrument
# instrum = load_instrum('spirou')
# high_res_wv_lim = instrum['high_res_wv_lim']

# # - Which sequences are taken
# do_tr = [1]

# # - Selecting bad exposures if wanted/needed  (not implemented yet)
# # bad_indexs = None

# ## --- Additionnal global variables
# plot = False
# nolog = True
# inj_alpha = 'ones'
# orders = np.arange(49) #np.array([14,15,30,31,46,47])
# opacity_sampling = 4  # downsampling of the petitradtrans R = 1e6, ex.: o_s = 4 -> R=250000
# lbl_res = 1e6 / opacity_sampling  # Resolution of the lbl model


# # -----------------------------------------------------------
# ## LOAD HIGHRES DATA
# data_info = {'trall_alpha_frac': [], 'trall_icorr': [], 'trall_N': [], 'bad_indexs': []}
# data_trs = []

# if retrieval_type == 'JR' or retrieval_type == 'HRR':
#     for high_res_file_stem in high_res_file_stem_list:
#         log.debug(f'Hires files stem: {high_res_path / high_res_file_stem}')
#         log.info('Loading Hires files.')
#         data_info_i, data_trs_i = pl_obs.load_sequences(high_res_file_stem, do_tr, path=high_res_path)
#         # Add index of the exposures where we expect to see the planet signal (to be used in kernel function)
#         # trall_alpha_frac is the fraction of the total planet signal received during the exposure.
#         data_trs_i['0']['i_pl_signal'] = data_info_i['trall_alpha_frac'] > 0.5
#         for data_tr in data_trs_i.values():
#             data_trs.append(data_tr)
#         # Patch for now data_info. Need to modify how the logl is computed to make it more clean.
#         # Would not work with different instruments
#         for key in ['trall_alpha_frac', 'trall_N', 'bad_indexs']:
#             data_info[key].append(data_info_i[key])
#         try:
#             data_info['trall_icorr'].append(data_info_i['trall_icorr'] + data_info['trall_icorr'][-1][-1] + 1)
#         except IndexError:
#             data_info['trall_icorr'].append(data_info_i['trall_icorr'])
#     for key in data_info.keys():
#         data_info[key] = np.concatenate(data_info[key], axis=0)
    
#     data_info['bad_indexs'] = None  # Leave it to None. Not implemented yet.


# ## --- LOW RES DATA ---


# #%%

# # --- SELECT WHICH SPITZER (or other photometric data points) DATA YOU HAVE
# if add_spitzer is True:
#     # --- Reading transmission functions of the broadband points
#     spit_trans_i1 = Table.read(spitzer_path / 'transmission_IRAC1.txt', format='ascii')  # 3.6 um
#     spit_trans_i2 = Table.read(spitzer_path / 'transmission_IRAC2.txt', format='ascii')  # 4.5 um

#     fct_i1 = interp1d(spit_trans_i1['col1'], spit_trans_i1['col2'] / spit_trans_i1['col2'].max())
#     fct_i2 = interp1d(spit_trans_i2['col1'], spit_trans_i2['col2'] / spit_trans_i2['col2'].max())

#     # - Select which spitzer/other broadband data points to add
#     # *** in the same order as your data will be given
#     wave_sp = [spit_trans_i1['col1'], spit_trans_i2['col1']]  # spit_trans_f2['col1'],
#     fct_sp = [fct_i1, fct_i2]  # fct_f2,

#     # --- READ YOUR SPITZER DATA
#     #     spit_tab = Table.read(data_path + 'WASP-33/Spitzer.csv', format='ascii.ecsv')
#     #     spit_wave, spit_data, spit_data_err = spit_tab['wave'].data, spit_tab['F_p/F_star'].data, spit_tab['err'].data
#     data_spit = Table.read(spitzer_path / spitzer_file)
#     spit_wave = data_spit['wave'].value
#     spit_data = data_spit['F_p/F_star'].value
#     spit_data_err = data_spit['err'].value

#     spitzer = spit_wave, spit_data, spit_data_err, wave_sp, fct_sp
# else:
#     spitzer = None

# # --- HST DATA ---

# hst = {}
# if wfc3_file is not None:
#     data_HST = Table.read(wfc3_path / wfc3_file)

#     HST_wave = data_HST['wave'].value
#     HST_data = data_HST['F_p/F_star'].value / 100
#     HST_data_err = data_HST['err'].value / 100

#     ### - Must give : wave, data, data_err, instum_resolution ; to the hst dictionary, for each instrument used
#     hst['WFC3'] = HST_wave, HST_data, HST_data_err, 75

# if white_light:
#     wl_wave = HST_wave
#     mean_wl = np.mean(HST_data)
#     mean_wl_err = np.sqrt(np.sum(HST_data_err ** 2)/HST_data_err.size)


# # --- Will you add STIS data points?
# add_stis = False

# if add_stis:
#     data_HST = Table.read(data_path + planet_path + 'HST_data_VIS.ecsv')
#     HST_wave_VIS = data_HST['wavelength'].value
#     HST_data_VIS = data_HST['data'].value
#     HST_data_err_VIS = data_HST['err'].value

#     hst['STIS'] = HST_wave_VIS, HST_data_VIS, HST_data_err_VIS, 50


# #################
# # Parameters and Priors
# #################

# # --- Default parameters needed for the retrieval, but not necessarily fitted
# fixed_params = dict()
# fixed_params['M_pl'] = p.M_pl.to(u.M_jup).value  # In jupiter mass
# fixed_params['R_pl'] = planet.R_pl[0].to(u.R_jup).value  # In jupiter radius
# fixed_params['R_star'] = p.R_star.to(u.R_sun).value  # In solar radius
# fixed_params['P0'] = 1e-3  # -- reference pressure fixed (for transmission spectra)
# fixed_params['log_f'] = 0.0  # -- scaling factor for WFC3
# fixed_params['spec_scale'] = 1.0  # -- scaling factor for the Fp/Fs spectrum

# # --- For the abundances
# fixed_params['e-'] = 10 ** (-6.0)
# fixed_params['H'] = 10 ** (-99.0)
# # Add species used for opacities if chemical equilibrium
# if chemical_equilibrium:
#     for mol in (line_opacities + continuum_opacities):
#         fixed_params[mol] = 10 ** (-99.0)

# # --- For both guillot profiles
# fixed_params['T_eq'] = 2500.  # Use for isothermal profile as well
# fixed_params['tp_gamma'] = 10 ** (1)  # 0.01
# fixed_params['T_int'] = 500.
# # --- For the basic guillot profile (guillot)
# fixed_params['kappa_IR'] = 10 ** (-2)
# fixed_params['gravity'] = p.gp.cgs.value
# # --- For the modified guillot profile (guillot_modif)
# fixed_params['tp_delta'] = 10 ** (-7.0)
# fixed_params['ptrans'] = 10 ** (-3)
# fixed_params['tp_alpha'] = 0.3
# # --- For the Akima Spline TP profile
# for iP in range(1, 4):
#     # The pressure nodes values can be fitted or fixed
#     fixed_params[f'P{iP}'] = 10 ** (-iP)  # bar
#     # Same for the temperature nodes
#     fixed_params[f'T{iP}'] = 1000.  # K

# # - Generating the basis of the temperature profile
# limP=(-10, 2)  # pressure log range
# n_pts=50  # pressure n_points
# fixed_params['pressures'] = np.logspace(*limP, n_pts)  # pressures in bars

# # --- Define rotation kernel
# # Define a function that takes as input a list of dictionnaries of
# # fixed_params and params_prior combined (theta_dict)
# # and output a list of rotation kernels (a 1D array) at the right sampling
# # and resolution as the data (so uses instrum['resol'] and lbl_res).

# from starships.spectrum import CitrusRotationKernel

# def get_ker(theta_regions, tr_i=0):
#     all_phases = (data_trs[tr_i]['t_start'] - planet.mid_tr.value) / planet.period.to('d').value % 1
#     mean_phase = np.mean(all_phases[data_trs[tr_i]['i_pl_signal']])  # Only  where there is planet signal
#     theta_dict = theta_regions[0]
#     longitudes = [theta_dict['phase1'], 0.99]
#     R_pl = theta_dict['R_pl'] * 0.01  # cgs to SI  (cm to m)
#     angular_freq = 2 * np.pi / planet.period[0].to('s').value
#     rot_factor = theta_dict['rot_factor']
    
#     ker_obj = CitrusRotationKernel(longitudes, R_pl, rot_factor * angular_freq, instrum['resol'])
    
#     rot_ker = ker_obj.resample(lbl_res, phase=mean_phase, n_os=1000)
    
#     return rot_ker
    
    


# #####################################################
# # --- Define priors ---
# #####################################################
# # Default priors are uniform, log_unigform or gaussian.
# # Other prior types can be added in the prior function dictionnary.
# # NOTE: The difference between uniform and log_uniform is that the values are
# # converted to log scale inside the logl call for log_uniform.
# # Ex:
# # >>> params_prior['CO'] = ['log_uniform', -12.0, -0.5]
# # will give in the logl function:
# # >>> theta_dict['CO'] = 10 ** theta_dict['CO']

# params_prior = {}
# # - Abundances priors - #
# # By default, if chemical equilibrium is True, the molecular abundances depend on C/O and Fe/H
# if chemical_equilibrium:
#     # Valid range for PRT: C/O = [0.1, 1.6] and [Fe/H] = [-2, 3]
#     params_prior['C/O'] = ['uniform', 0.1, 1.6]
#     params_prior['Fe/H'] = ['uniform', -2, 3]
# else:
#     # Same prior for all molecules in the list of opacity species
#     for mol in line_opacities:
#         params_prior[mol] = ['log_uniform', -12.0, -0.5]
# # NOTE: Some molecules are not in the list of opacities, but need to be include in the fit (ex: 'e-')
# # You can add a prior for a specific molecule by adding a line like this:
# # params_prior['e-'] = ['log_uniform', -12.0, -0.5]

# # - Other parameters priors - #
# params_prior['T_eq'] = ['uniform', 100, 4500]
# # params_prior['p_cloud'] = ['log_uniform', -5.0, 2]
# # params_prior['R_pl'] = ['uniform', 1.0, 1.6]  # In jupiter radius
# params_prior['kp'] = ['uniform',100, 250]
# params_prior['rv'] = ['uniform', -40, 40]
# # params_prior['wind'] = ['uniform', 0, 20]
# # params_prior['kappa_IR'] = ['log_uniform', -3, 3]  # Guillot
# params_prior['tp_gamma'] = ['log_uniform', -2, 6]  # Guillot & Guillot_modif
# params_prior['tp_delta'] = ['log_uniform', -8, -3]  # Guillot_modif
# params_prior['ptrans'] = ['log_uniform', -8, 3]  # Guillot_modif
# params_prior['tp_alpha'] = ['uniform', -1.0, 1.0]  # Guillot_modif
# # params_prior['gamma_scat'] = ['uniform', -6, 0]
# # params_prior['scat_factor'] = ['uniform', 0, 10]   # * (5.31e-31*u.m**2/u.u).cgs.value
# # params_prior['log_f'] = ['uniform', -2, 3]  # yes, it is a uniform prior, not a log_uniform
# params_prior['phase1'] = ['uniform', 0.2 , 0.8]
# # params_prior['phase2'] = ['uniform', 0.67, 1.0]
# params_prior['rot_factor'] = ['uniform', 0.5, 4]
# # params_prior['spec_scale'] = ['log_uniform', -2, 3]

# # If the planet is splitted in multiiple regions,
# # specify which paremeters are specific to each region.
# # Each  region is identified with an ID starting at 1 (not 0).
# region_id = [1, 2]  # list of region ids (usefull if not all regions are contributing to the planet flux)
# n_regions = len(region_id)
# reg_params = ['T_eq', 'tp_gamma', 'tp_delta', 'ptrans', 'tp_alpha']
# # global parameters are all other parameters not in reg_params
# global_params = [param for param in params_prior.keys() if param not in reg_params]

# # It is possible to fix some parameters in specific regions.
# reg_fixed_params = {}
# # reg_fixed_params['H2O_2'] = 1e-12  # Force water abundance to ~zero in region 2

# # Assign  specific index to each regional parameter (if not already done manually)
# if n_regions > 1:
#     for param in reg_params:
#         # Add the parameter for each region if not already specified
#         for i_reg in region_id:
#             key = f'{param}_{i_reg}'
#             if (key not in params_prior) and (key not in reg_fixed_params):
#                 params_prior[f'{param}_{i_reg}'] = params_prior[param]
#             else:
#                 log.debug(f'Parameter {key} already in params_prior or reg_fixed_params. Skipping.')
#         # Remove the global parameter
#         params_prior.pop(param)


# # --- Assign the prior names to a log_prior function using a dictionnary ---
# # The functions take the value from theta and return the log of the prior.
# prior_func_dict = ru.default_prior_func
# # You can add or replace with custom prior functions in the dictionnary.
# # Ex: prior_func_dict['your_prior'] = your_prior_func

# # --- Initialize walkers using the prior ---
# # Define the initialisation function for each type of prior.
# # If you create your own prior function, you need to add it here.
# prior_init_func = ru.default_prior_init_func
# # You can add or replace with custom prior functions in the dictionnary.
# # Ex: prior_init_func['your_prior'] = your_prior_init_func
# # Special treatment for some paramters
# special_treatment = {'kp': ['uniform', 150, 190],
#                      'rv': ['uniform', -20, 20],
#                      'phase1': ['uniform', 0.2, 0.6],
#                      'H2O_1': ['log_uniform', -6.0, -0.5],
#                      }
# walker_init = ru.init_from_prior(n_walkers, prior_init_func, params_prior,
#                                  special_treatment=special_treatment)



# # --- If parameters are given in a yaml file, load them.
# if use_yaml_file:
#     log.info(f'Loading parameters from {params_file_in}')
#     with open(params_file_in, 'r') as f:
#         yaml_params = yaml.load(f, Loader=yaml.FullLoader)
#     # --- Update the global variables with the parameters from the yaml file
#     yaml_params_list =['params_prior', 'fixed_params', 'region_id', 'reg_params',
#                        'reg_fixed_params', 'dissociation', 'kind_temp',
#                        'line_opacities', 'continuum_opacities', 'other_species', 'orders']
#     for key, val in yaml_params_list:
#         globals()[key] = val




# ####################################################
# # hig vs low res obj
# ####################################################

# # Define low res wavelength range (might not be used)
# lower_wave_lim = high_res_wv_lim[0]
# upper_wave_lim = high_res_wv_lim[1]

# if add_spitzer:
#     upper_wave_lim = 5.

# if add_stis:
#     lower_wave_lim = 0.3

# low_res_wv_lim = (lower_wave_lim, upper_wave_lim)

# log.debug(f"Low res wavelength range: {low_res_wv_lim}")

# def init_model_retrieval(mol_species=None, kind_res='high', lbl_opacity_sampling=None,
#                          wl_range=None, continuum_species=None, pressures=None, **kwargs):
#     """
#     Initialize some objects needed for modelization: atmo, species, fct_star, pressures
#     :param mol_species: list of species included (without continuum opacities)
#     :param kind_res: str, 'high' or 'low'
#     :param lbl_opacity_sampling: ?
#     :param wl_range: wavelength range (2 elements tuple or list, or None)
#     :param continuum_species: list of continuum opacities, H and He excluded
#     :param pressures: pressure array. Default is `fixed_params['pressures']`
#     :param kwargs: other kwargs passed to `starships.petitradtrans_utils.select_mol_list()`
#     :return: atmos, species, fct_star, pressure array
#     """

#     if mol_species is None:
#         mol_species = line_opacities

#     if lbl_opacity_sampling is None:
#         lbl_opacity_sampling = opacity_sampling

#     if continuum_species is None:
#         continuum_species = continuum_opacities

#     if pressures is None:
#         pressures = fixed_params['pressures']

#     species = prt.select_mol_list(mol_species, kind_res=kind_res, **kwargs)
#     species_2_lnlst = {mol: lnlst for mol, lnlst in zip(mol_species, species)}

#     if kind_res == 'high':
#         mode = 'lbl'
#         Raf = instrum['resol']
#         pix_per_elem = 2
#         if wl_range is None:
#             wl_range = high_res_wv_lim

#     elif kind_res == 'low':
#         mode = 'c-k'
#         Raf = 1000
#         pix_per_elem = 1
#         if wl_range is None:
#             wl_range = low_res_wv_lim
#     else:
#         raise ValueError(f'`kind_res` = {kind_res} not valid. Choose between high or low')


#     atmo, _ = prt.gen_atm_all([*species.keys()], pressures, mode=mode,
#                                       lbl_opacity_sampling=lbl_opacity_sampling, wl_range=wl_range,
#                                       continuum_opacities=continuum_species)

#     # --- downgrading the star spectrum to the wanted resolution
#     if kind_trans == 'emission' and star_wv is not None:
#         resamp_star = np.ma.masked_invalid(
#             resamp_model(star_wv[(star_wv >= wl_range[0] - 0.1) & (star_wv <= wl_range[1] + 0.1)],
#                          star_flux[(star_wv >= wl_range[0] - 0.1) & (star_wv <= wl_range[1] + 0.1)], 500000, Raf=Raf,
#                          pix_per_elem=pix_per_elem))
#         fct_star = interp1d(star_wv[(star_wv >= wl_range[0] - 0.1) & (star_wv <= wl_range[1] + 0.1)],
#                                      resamp_star)
#     else:
#         fct_star = None

#     return atmo, species_2_lnlst, fct_star

# # Dictionnaries is not the best way for multiprocessing because it is not shared between processes.
# # Better to use global variables for big arrays or atmo objects. Use dictionaries only for small objects.
# # Define global dictionaries where to put the model infos so they will be shared in functions.
# fct_star_global = {'high': None, 'low': None}

# atmo_high = None
# atmo_low = None


# ## logl

# # lnprob

# ####################################################

# def unpack_theta(theta):
#     """Unpack the theta array into a list of dictionnary with the parameter names as keys.
#     Also add other values needed for the model.
#     Return a list of dictionnary with all values needed for the model.
#     Why a list of dict?  Because it can account for multiple regions
#     (so different values for some parameters).
#     """
    
#     # Get the parameters and unpack them in a dictionnary.
#     theta_dict = {key: val for key, val in zip(params_prior.keys(), theta)}
    
#     # Convert from log to linear scale if needed.
#     for key, prior_info in params_prior.items():
#         if prior_info[0] == 'log_uniform':
#             theta_dict[key] = 10 ** theta_dict[key]
    
#     # Make a dictionnary for each region if needed.
#     dict_list = list()
#     for i_reg in region_id:
#         # Create a dictionnary for each region and remove the region number from the key.
#         theta_region = {key: theta_dict[key] for key in global_params}
#         if n_regions > 1:
#             for key in reg_params:
#                 key_reg = f'{key}_{i_reg}'
#                 try:
#                     theta_region[key] = theta_dict.pop(key_reg)
#                 except KeyError:
#                     theta_region[key] = reg_fixed_params[key_reg]
#         else:
#             theta_region.update(theta_dict)

#         # Create a dictionnary with all values needed for the model.
#         # The values are either taken from theta_region in priority or from fixed_params.
#         combined_dict = {**fixed_params, **theta_region}

#         # gravity depends on Rp if included in the fit
#         if 'R_pl' in theta_region:
#             combined_dict['gravity'] = (const.G * planet.M_pl /
#                                         (theta_region['R_pl'] * const.R_jup) ** 2).cgs.value
            
#         # Some values need to be set to None if not included in the fit or not in fixed_params.
#         for key in ['wind', 'p_cloud', 'gamma_scat', 'scat_factor', 'C/O', 'Fe/H']:
#             if key not in combined_dict:
#                 combined_dict[key] = None
            
#         # --- Generating the temperature profile
#         if kind_temp == "modif":
#             fct_inputs = ['pressures', 'tp_delta', 'tp_gamma', 'T_int', 'T_eq', 'ptrans', 'tp_alpha']
#             args = (combined_dict[key] for key in fct_inputs)
#             combined_dict['temperatures'] = guillot_modif(*args)
#         elif kind_temp == 'iso':
#             combined_dict['temperatures'] = combined_dict['T_eq'] * np.ones_like(combined_dict['pressures'])
#         elif kind_temp == 'guillot':
#             fct_inputs = ['pressures', 'kappa_IR', 'tp_gamma', 'gravity', 'T_int', 'T_eq']
#             args = (combined_dict[key] for key in fct_inputs)
#             combined_dict['temperatures'] = guillot_global(*args)
#         else:
#             raise ValueError(f'`kind_temp` = {kind_temp} not valid. Choose between guillot, modif or iso')
        
#         # Convert some values to cgs units if not done already
#         combined_dict['R_pl'] = combined_dict['R_pl'] * const.R_jup.cgs.value
#         combined_dict['R_star'] = combined_dict['R_star'] * const.R_sun.cgs.value
        
#         dict_list.append(combined_dict)
    
#     return dict_list


# def prepare_abundances(theta_dict, mode=None, ref_linelists=None):
#     """Use the correct linelist name associated to the species."""

#     if ref_linelists is None:
#         if mode is None:
#             ref_linelists = line_opacities.copy()
#         else:
#             ref_linelists = [linelist_names[mode][mol] for mol in line_opacities]

#     # --- Prepare the abundances (with the correct linelist name for species)
#     species = {lnlst: theta_dict[mol] for lnlst, mol
#                in zip(ref_linelists, line_opacities)}
    
#     # --- Adding continuum opacities
#     for mol in continuum_opacities:
#         species[mol] = theta_dict[mol]
        
#     # --- Adding other species
#     for mol in other_species:
#         species[mol] = theta_dict[mol]

#     return species


# def prepare_model_high_or_low(theta_dict, mode, atmo_obj=None, fct_star=None,
#                               species_dict=None, Raf=None, rot_ker=None):

#     if Raf is None:
#         Raf = instrum['resol']
    
#     if atmo_obj is None:
#         # Use atmo object in globals parameters if it exists
#         atmo_obj = atmo_high if mode == 'high' else atmo_low
#         # Initiate if not done yet
#         if atmo_obj is None:
#             log.info(f'Model not initialized for mode = {mode}. Starting initialization...')
#             output = init_model_retrieval(kind_res=mode)
#             log.info('Saving values in `linelist_names`.')
#             atmo_obj, lnlst_names, fct_star_global[mode] = output
#             # Update the values of the global variables
#             # Need to use globals() otherwise an error is raised.
#             if mode == 'high':
#                 globals()['atmo_high'] = atmo_obj
#             else:
#                 globals()['atmo_low'] = atmo_obj
                
#             # Update the line list names
#             if linelist_names[mode] is None:
#                 linelist_names[mode] = lnlst_names
#             else:
#                 # Keep the predefined values and complete with the new ones
#                 linelist_names[mode] = {**lnlst_names, **linelist_names[mode]}

#     if fct_star is None:
#         fct_star = fct_star_global[mode]

#     # --- Prepare the abundances (with the correct name for species)
#     # Note that if species is None (not specified), `linelist_names[mode]` will be used inside `prepare_abundances`.
#     species = prepare_abundances(theta_dict, mode, species_dict)

#     # --- Generating the model
#     args = [theta_dict[key] for key in ['pressures', 'temperatures', 'gravity', 'P0', 'p_cloud', 'R_pl', 'R_star']]
#     kwargs = dict(gamma_scat=theta_dict['gamma_scat'],
#                   kappa_factor=theta_dict['scat_factor'],
#                   C_to_O=theta_dict['C/O'],
#                     Fe_to_H=theta_dict['Fe/H'],
#                     specie_2_lnlst=linelist_names[mode],
#                     kind_trans=kind_trans,
#                     dissociation=dissociation,
#                     fct_star=fct_star)
#     wv_out, model_out = prt.retrieval_model_plain(atmo_obj, species, planet, *args, **kwargs)

#     if mode == 'high':
#         # --- Downgrading and broadening the model (if broadening is included)
#         # if np.isfinite(model_out[100:-100]).all():
#             # Get wind broadening parameters
#             if theta_dict['wind'] is not None:
#                 rot_kwargs = {'rot_params': [theta_dict['R_pl'] * const.R_jup,
#                                              theta_dict['M_pl'],
#                                              theta_dict['T_eq'] * u.K,
#                                              [theta_dict['wind']]],
#                                 'gauss': True, 'x0': 0,
#                                 'fwhm': theta_dict['wind'] * 1e3, }
#             else:
#                 rot_kwargs = {'rot_params': None}
            
#             # Downgrade the model
#             wv_out, model_out = prt.prepare_model(wv_out, model_out, lbl_res, Raf=Raf,
#                                                   rot_ker=rot_ker, **rot_kwargs)

#     return wv_out, model_out

# def prepare_model_multi_reg(theta_regions, mode, rot_ker_list=None, atmo_obj=None, tr_i=0):
    
#     # Get the list of rotation kernels
#     rot_ker_list = get_ker(theta_regions, tr_i=tr_i)
    
#     wv_list = []
#     model_list = []
#     for theta_dict, reg_id in zip(theta_regions, region_id):
#         wv_i, model_i = prepare_model_high_or_low(theta_dict, mode,
#                                                   rot_ker=rot_ker_list[reg_id - 1],
#                                                   atmo_obj=atmo_obj)
#         model_i *= theta_dict['spec_scale']
#         wv_list.append(wv_i)
#         model_list.append(model_i)

#     wv_out = wv_list[0]
#     model_out = np.sum(model_list, axis=0)
    
#     return wv_out, model_out


# def prepare_spitzer(wv_low, model_low):

#     # print('Spitzer')
#     spit_wave, _, _, wave_sp, fct_sp = spitzer

#     spit_mod = []
#     for wave_i, fct_i in zip(wave_sp, fct_sp):
#         # --- Computing the model broadband point
#         cond = (wv_low >= wave_i[0]) & (wv_low <= wave_i[-1])
#         spit_mod.append(np.average(model_low[cond], weights=fct_i(wv_low[cond])))

#     spit_mod = np.array(spit_mod)

#     return spit_wave, spit_mod


# def prepare_hst(wv_low, model_low, Rbf, R_sampling, instrument, wave_pad=None):

#     hst_wave, _, _, hst_res = hst[instrument]
#     log.debug('Prepare HST...')
#     log.debug(f"hst_wave: {hst_wave}")
#     log.debug(f"hst_res: {hst_res}")

#     if wave_pad is None:
#         d_wv_bin = np.diff(hst_wave)
#         wave_pad = 10 * d_wv_bin[[0, -1]]   # This is a bit arbitrary

#     cond = (wv_low >= hst_wave[0] - wave_pad[0]) & (wv_low <= hst_wave[-1] + wave_pad[-1])

#     _, resamp_prt = spectrum.resampling(wv_low[cond], model_low[cond], Raf=hst_res, Rbf=Rbf, sample=wv_low[cond])
#     binned_prt_hst = spectrum.box_binning(resamp_prt, R_sampling / hst_res)
#     fct_prt = interp1d(wv_low[cond], binned_prt_hst)
#     mod = fct_prt(hst_wave)

#     return hst_wave, mod


# def lnprob(theta, ):
    
#     log.debug(f"lnprob: {theta}")
    
#     # --- Prior ---
#     log.debug('Commpute Prior')
#     total = ru.log_prior(theta, params_prior, prior_func_dict=prior_func_dict)

#     if not np.isfinite(total):
#         log.debug('Prior = -inf')
#         return -np.inf

#     theta_regions = unpack_theta(theta)

#     ####################
#     # --- HIGH RES --- #
#     ####################
#     if (retrieval_type == 'JR') or (retrieval_type == 'HRR'):

#         # For the rest, just use the first region
#         theta_dict = theta_regions[0]

#         logl_i = []
#         # --- Computing the logL for all sequences
#         for tr_i, data_tr_i in enumerate(data_trs):
            
#             # NOTE: Not optimal to re-compute the model for each sequence.
#             # Could be done once for all regions and then the rotation kernel
#             # could be applied to the model for each region depending on the phase.
#             wv_high, model_high = prepare_model_multi_reg(theta_regions, 'high', tr_i=tr_i)
#             if not np.isfinite(model_high[100:-100]).all():
#                 log.warning("NaN in high res model spectrum encountered")
#                 return -np.inf
            
#             vrp_orb = rv_theo_t(theta_dict['kp'],
#                                 data_tr_i['t_start'] * u.d, planet.mid_tr,
#                                 planet.period, plnt=True).value

#             args = (theta_dict['rv'], data_tr_i, planet, wv_high, model_high)
#             kwargs = dict(vrp_orb=vrp_orb, vr_orb=-vrp_orb * Kp_scale, nolog=nolog,
#                           alpha=np.ones_like(data_tr_i['t_start']), kind_trans=kind_trans)
#             logl_tr = corr.calc_log_likelihood_grid_retrieval(*args, **kwargs)

#             if not np.isfinite(logl_tr).all():
#                 return -np.inf

#             logl_i.append(logl_tr)

#         total += corr.sum_logl(np.concatenate(np.array(logl_i), axis=0), data_info['trall_icorr'], orders,
#                                data_info['trall_N'], axis=0, del_idx=data_info['bad_indexs'], nolog=True,
#                                alpha=data_info['trall_alpha_frac'])

#         if (retrieval_type == "HRR") and (white_light is True):
#             log.debug("Using White Light from WFC3.")
#             # --- White light info ---
#             Rbf = instrum['resol']
#             R_sampling = int(1e6 / opacity_sampling)
#             _, mod = prepare_hst(wv_high, model_high, Rbf, R_sampling, 'WFC3')
#             mean_mod = np.mean(mod)
#             log.debug(f"White Light value: {mean_mod}")

#             total += -1 / 2 * corr.calc_chi2(mean_wl, mean_wl_err, mean_mod)

#     ###################
#     # --- LOW RES --- #
#     ###################
#     if ((retrieval_type == 'JR') and (spitzer is not None)) or (retrieval_type == 'LRR') or (atmo_low is not None):
#         #         print('Low res')

#         wv_low, model_low = prepare_model_high_or_low(theta_dict, 'low')

#         if np.sum(np.isnan(model_low)) > 0:
#             print("NaN in low res model spectrum encountered")
#             return -np.inf

#         if spitzer is not None:
#             _, spit_mod = prepare_spitzer(wv_low, model_low)

#             # --- Computing the logL
#             _, spit_data, spit_data_err, _, _ = spitzer
#             total += -1 / 2 * corr.calc_chi2(spit_data, spit_data_err, spit_mod)

#     #             print('Spitzer', spitzer_logl)

#     if (retrieval_type == 'JR') or (retrieval_type == 'LRR'):
#         #         print('HST')

#         if (retrieval_type == 'JR') and (spitzer is None):
#             # --- If no Spitzer or STIS data is included, only the high res model is generated
#             # and this is the model that will be downgraded for WFC3
#             wv_low, model_low = wv_high, model_high
#             Rbf = instrum['resol']
#             R_sampling = int(1e6 / opacity_sampling)
#         else:
#             Rbf = 1000
#             R_sampling = 1000
#         #         print(Rbf)
#         for instrument in hst.keys():

#             _, mod = prepare_hst(wv_low, model_low, Rbf, R_sampling, instrument)

#             # --- Computing the logL
#             _, hst_data, hst_data_err, _ = hst[instrument]
#             total += corr.calc_logl_chi2_scaled(hst_data, hst_data_err, mod, theta_dict['log_f'])
#             # total += -1 / 2 * corr.calc_chi2(hst_data, hst_data_err, mod)

#         del wv_low, model_low

#     if retrieval_type != 'LRR':
#         del wv_high, model_high

#     gc.collect()

#     return total
    


# # %%
# # #######################

# if __name__ == '__main__':

#     # Start retrieval!

#     import warnings

#     warnings.simplefilter("ignore", FutureWarning)
#     # warnings.simplefilter("ignore", RuntimeWarning)

#     # --- Walkers initialisation ---
#     # -- Either a random uniform initialisation (for every parameters)
#     if walker_file_in is None:
#         pos = walker_init
#     elif init_mode == 'burnin':
#         pos, _ = ru.init_from_burnin(n_walkers, wlkr_file=walker_file_in, wlkr_path=walker_path, n_best_min=10000)
#     elif init_mode == 'continue':
#         # Last step of the chain
#         pos = ru.read_walkers_file(walker_path / walker_file_in, discard=0)[-1]
#     else:
#         raise ValueError(f"{init_mode} not valid.")

#     n_walkers, ndim = pos.shape

#     log.info(f"(Number of walker, Number of parameters) = {pos.shape}")

#     # Pre-run the log liklyhood function
#     log.info("Checking if log likelyhood function is working.")
#     for i_walker in range(n_walkers):
#         logl = lnprob(pos[i_walker])
#         if np.isfinite(logl):
#             log.info("Success!")
#             break
#     else:
#         log.warning("test not successful")

#     # Make sure file does not already exist
#     if init_mode != 'continue':
#         file_stem = walker_file_out.stem
#         for idx_file in range(100):
#             if (walker_path / walker_file_out).is_file():
#                 walker_file_out = walker_file_out.with_stem(f'{file_stem}_{idx_file}')
#             else:
#                 break
#         else:
#             raise ValueError('Walker File already exists.')
        
#     # --- Save all useful parameters in a yaml file ---
#     # Create output directory if it does not exist
#     params_path.mkdir(parents=True, exist_ok=True)
#     # Save parameters in a yaml file only if it does not exist already
#     if not (params_path / params_file_out).is_file():
#         with open(params_path / params_file_out, 'w') as f:
#             log.info(f'Saving parameters in {params_path / params_file_out}')
#             variables_to_save = ['params_prior', 'fixed_params', 'region_id', 'reg_params', 
#                                  'reg_fixed_params', 'n_walkers', 'walker_file_in', 'walker_file_out',
#                                 'walker_path', 'n_cpu', 'reduc_name', 'high_res_file_stem_list',
#                                 'dissociation', 'kind_temp', 'line_opacities', 'continuum_opacities',
#                                 'other_species', 'orders']
#             yaml_dict = {key: globals()[key] for key in variables_to_save}
#             yaml.dump(yaml_dict, f)
#     else:
#         log.info(f'Parameters file {params_file_out} already exists. Skipping.')

#     # --- backend to track evolution ---
#     # Create output directory if it does not exist
#     walker_path.mkdir(parents=True, exist_ok=True)
#     backend = emcee.backends.HDFBackend(walker_path / walker_file_out)

#     # Run it!
#     with Pool(n_cpu) as pool:
#         log.info('Initialize sampler...')
#         sampler = emcee.EnsembleSampler(n_walkers, ndim, lnprob,
#                                         pool=pool,
#                                         backend=backend, a=2)  ### step size -- > à changer
#         log.info('Starting the retrieval!')
#         sampler.run_mcmc(pos, n_steps, progress=False)  # , skip_initial_state_check=True)

#     log.info('End of retrieval. It seems to be a success!')

# -------------------------------------------
# ##################################################################
# ############# yaml file input example ######
# ##################################################################
# --------------------------------------------

# # This is an example of a retrieval input file for WASP-33 b.

# # -----------------------------------------------------------
# # --- Main parameters ---
# # -----------------------------------------------------------

# # Planet name 
# # This is used to fetch information in ExoFile, and alos
# # to save the results, with the spaces replaced by '_'
# pl_name: WASP-33 b

# # Path to the main directory where outputs are saved
# # If None, use scratch (if available) or home otherwise
# base_dir: null

# # Path to the high resolution data
# # Assuming the data is in a directory with the following structure:
# # <high_res_path>/<reduc_name>
# high_res_path: ~/DataAnalysis/SPIRou/Reductions/

# # Name of the reduction used to load the high resolution data
# # Assuming the data is in a directory with the following structure:
# # <high_res_path>/<reduc_name>
# reduc_name: WASP-33b_v07232

# # List of the file stems for the high resolution data
# # The files are assumed to be in the high_res_path directory
# high_res_file_stem_list:
#   # - retrieval_input_2-pc_mask_wings90_day1
#   - retrieval_input_4-pc_mask_wings97_day2

# # Dictionary of path to spectrophotometric data (ex: JWST, HST)
# # If empty, no spectrophotometric data will be used
# spectrophotometric_data:
#   wfc3: 
#     file_path: /home/adb/projects/def-dlafre/adb/Observations/HST/WFC3
#     file_name: WASP-33_WFC3_full_res.ecsv


# # Dictionary of path to photometric data (ex: Spitzer)
# # If empty, no photometric data will be used
# photometric_data:
#   # spitzer:
#   #   file_path: /home/adb/projects/def-dlafre/adb/Observations/Spitzer
#   #   file_name: WASP-33_Spitzer.ecsv

# # Type of retrieval : JR, HRR or LRR  (This should be removed eventually)
# retrieval_type: HRR

# # In HRR mode, use white light curve from spectrophotometric bandpass?
# white_light: false

# # Chemical equilibrium?
# # Note that C/O and Fe/H need to be included in the parameters if chemical equilibrium is True
# chemical_equilibrium: false

# # Include dissociation parametrization in abundance profiles? (not used in chemical equilibrium)
# dissociation: true

# # TODO: this should be removed. The spitzer points will be used
# #       if the data is specified in the photometric_data dictionary.
# # Will you add Spitzer data points?
# add_spitzer: false

# # Which kind of TP profile you want? (guillot, modif, iso, akima)
# kind_temp: modif

# # How many steps for burnin phase
# n_steps_burnin: 300

# # How many steps for sampling phase
# n_steps_sampling: 3000

# # Name of the run (will be used to save the results)
# # For example, the walkers will be saved in base_dir/DataAnalysis/walker_steps/<pl_name>/walker_steps_<run_name>.h5
# # If null, the run_name will be define by joining with "_" all the following parameters
# # - kind_trans
# # - retrieval_type
# # - keys in spectrophotometric_data dictionary (ex: wfc3)
# # - keys in photometric_data dictionary (ex: spitzer)
# # - kind_temp
# # - "disso" (if dissociation is True)
# # - "chemEq" (if chemical equilibrium is True)
# # - the sbatch job ID if available
# # So for this example, the run_name will be: emission_HRR_modif_disso
# run_name: null

# # Path where the walker steps will be saved
# # If null, the path will be set to <base_dir>/DataAnalysis/walker_steps/<pl_name>
# walker_path: null

# # Name of the file where the walker steps will be saved
# # If null, the file will be named walker_steps_<run_name>.h5
# walker_file_out: null

# # Walker file used to init walkers
# # If set to null, the retrieval will use random initialisation from the prior
# # Possible values: null, 'file_name.h5'
# walker_file_in: null

# # How to init walkers
# # Options: "burnin", "continue" (not used if walker_file_in is null)
# # If "burnin", the walkers will be initialised from the last step of the burnin chain in walker_file_in
# # If "continue", the walkers will be initialised from the last step of the chain in walker_file_in.
# #    Note that this option allows to keep the walker_file_in as the output file, and append the new walkers to it.
# init_mode: burnin

# # Path where this yaml file will be copied
# # If null, the path will be set to base_dir/DataAnalysis/retrieval_params/<pl_name>
# params_path: null

# # Name of the file where this yaml file will be copied
# # If null, the file will be named params_<run_name>.yaml
# params_file_out: null

# # Types of models : in emission or transmission
# kind_trans: emission

# # --- Number of cpu to use ---
# # If null, the code will use the number of cpus from
# # the slurm environment variable named SLURM_CPUS_PER_TASK
# # (null is recommended if using sbatch)
# n_cpu: null

# # --- Number of walkers ---
# # If null, the number of walkers will be set to n_wlkr_per_cpu * n_cpu
# # (we recommend to set it to null and using n_wlkr_per_cpu instead)
# n_walkers: null

# # --- Number of walkers per cpu ---
# n_walkers_per_cpu: 18

# # Downsampling factor for the high resolution model
# # It will be used to downsample the high resolution model
# # to increase the speed of the retrieval.
# # Ex: 1 will use the full resolution of 1 000 000
# #     4 will use a resolution of 1e6/4 = 250 000
# opacity_sampling: 4

# # List of Orders to use (null will use all orders)
# orders: null

# # -----------------------------------------------------------

# # -----------------------------------------------------------
# # --- Data parameters ---
# # -----------------------------------------------------------

# # Dictionary of planet parameters that will replace the ones in ExoFile
# # --------------------------------------------------------------------------------------------------
# # units must be inputed in astropy string format (see table here https://docs.astropy.org/en/stable/units/ref_api.html#module-astropy.units.si)
# # (Set to null or comment if you want to use the default values from ExoFile)
# pl_params:
#   M_star: # Star mass
#     value: 1.561
#     unit: M_sun

#   R_star: # Star radius
#     value: 1.5093043  # Computed from planet.ap / 3.69
#     unit: R_sun
  
#   M_pl: # Planet mass
#     value: null
#     unit: M_jup
  
#   R_pl: # Planet radius
#     value: 1.6787561  # Computed from 0.1143 * planet.R_star
#     unit: R_jup
  
#   RV_sys: # Systemic radial velocity
#     value: null
#     unit: km/s
  
#   mid_tr: # Mid-transit time
#     value: null
#     unit: d
  
#   t_peri: # Time of periastron passage
#     value: null
#     unit: d
  
#   trandur: # Transit duration
#     value: null
#     unit: h
  
#   period: # Orbital period
#     value: null
#     unit: d
  
#   excent: # Eccentricity
#     value: 0.0
#     unit: null
  
#   incl: # Inclination
#     value: 86.63
#     unit: deg
  
#   Teff: # Star effective temperature
#     value: 7300
#     unit: K
  
#   Tp: # Planet temperature
#     value: null 
#     unit: K
#   w: # Longitude of periastron?
#     value: -90
#     unit: deg
#   ap: # Semi-major axis
#     value: 0.0259
#     unit: au

# # -----------------------------------------------------------
# # --- High resolution data ---
# # -----------------------------------------------------------

# # - Name of the instrument (used to get the resolution and wavelength range)
# # This should be a list, matching the number of high resolution files
# # If a single element is given, it will be used for all observation files.
# instrum: 
#   - spirou

# # -----------------------------------------------------------
# # --- Input for model intialisation ---
# # -----------------------------------------------------------

# # - Name of the species that contribute to the line opacities
# # ** THIS IS NOT THE LINE LIST NAME **
# line_opacities: [CO, H2O]

# # - Species that contribute to the continuum opacities.
# # By default, H2-H2 and H2-He are included.
# continuum_opacities: [H-]

# # - Other species to add to the model (ex: e- and H needed for H- opacity):
# other_species: [e-, H]

# # Which species lists will be include in the fit?
# # This will automatically add the species in the params_prior dictionnary,
# # with a log_uniform prior between 1e-12 and 1e-0.5.
# # You can leave the list empty if you want to set the priors manually.
# # For chemical equilibrium, all the species will be added to the fixed_params dictionnary.
# # The possible elements of the list are:
# # - one of the species list (like line_opacities, continuum_opacities, other_species)
# # - or species name. This is useful if you want to fit only some of the species in a list.
# species_in_prior: [line_opacities, continuum_opacities, e-]

# # TODO: Not well implemented yet. ****
# # - Dictionnary of linelists to use for each species in line_opacities
# # The default linelists are defined in the retrieval_utils.linelist_dict
# linelist_names: 
#   # Here is an example of a specific linelist defined by the user.
#   # high:  # line-by-line linelist name
#   #   CO: user_defined_linelist_name_high
#   # low:  # correlated-k linelist name
#   #   CO: user_defined_linelist_name_low

# # -----------------------------------------------------------
# # --- Parameters and Priors ---
# # -----------------------------------------------------------

# # Note: All the parameters used for modeling will be put in a dictionnary
# # named theta_dict, which is a combination of the keys in fixed_params and params_prior.
# # The parameters in params_prior will overwrite the fixed_params if they have the same key.
# # For multi-region models, there will be a list of dictionnaries
# # named theta_regions, which includes a theta_dict for each region.
# # If there is only one region, theta_regions will be a list with one element.

# # - Default parameters needed for the retrieval. If you want to fit one of them,
# # you need to add it to the params_prior dictionnary. No need to remove it from the fixed parameters.
# # Note that you can have unused parameters (so you don't need to erase them if you don't use them)
# fixed_params:

#   # Reference pressure (for transmission spectra)
#   P0: 0.001

#   ###### Fudge factors ######
#   # Scaling factor for WFC3
#   log_f: 0.0
#   # Scaling factor for the Fp/Fs spectrum
#   spec_scale: 1.0

#   ###### Abundances ######
#   # Electron abundance
#   e-: 1e-06
#   # H abundance
#   H: 1e-99  # Some values are computed later in the code, so this is just a placeholder.

#   # Abundance ratios for chemical equilibrium
#   # C/O
#   C/O: 0.54  # Solar value is 0.54
#   # Fe/H
#   Fe/H: 0.0  # Solar value is 0.0

#   # Cloud parameters (set to null if not used)
#   p_cloud: null  # Cloud pressure (in bar)
#   scat_factor: null  # Scattering factor

#   ###### TP profile parameters ######
#   # Equilibrium temperature (for all TP profiles)
#   T_eq: 2500.0

#   # Guillot and Guillot_modif parameters
#   # gamma parameter (for Guillot and Guillot_modif)
#   tp_gamma: 10.0
#   # Internal temperature (for Guillot and Guillot_modif)
#   T_int: 500.0
  
#   # Guillot only parameters
#   # IR opacity (for Guillot)
#   kappa_IR: 0.01
#   # Gravity (for Guillot)
#   gravity: 23.0

#   # Guillot_modif only parameters
#   # Delta parameter (for Guillot_modif)
#   tp_delta: 1e-07
#   # Transition pressure (for Guillot_modif)
#   ptrans: 1e-03
#   # Alpha parameter (for Guillot_modif)
#   tp_alpha: 0.3

#   # Akima Spline TP profile parameters
#   # These lists will appear as P1, P2, ..., Pn and T1, T2, ..., Tn in the theta_dict.
#   # List of pressure nodes
#   akima_P: [1e-10, 1e-05, 1e-02, 1e+01]
#   # List of temperature nodes
#   akima_T: [1000.0, 1500.0, 2000.0, 2500.0]

#   ###### Orbital parameters ######
#   # Semi amplitude of the planet's radial velocity (in km/s)
#   kp: 150.0
#   # Radial velocity of the system (in km/s)
#   rv: 0.0

#   ###### Rotation kernel parameters ######
#   # Wind speed (in km/s)  (Old parameter, not used anymore)
#   wind: null  # Set to null if not used
#   # Phase of the first longitude in Citrus kernel
#   phase1: 0.5
#   # Phase of the second longitude in Citrus kernel
#   phase2: 0.75
#   # Rotation factor in Citrus kernel
#   rot_factor: 1.0


# ##### Priors #####

# # Here, we define the priors for the parameters.
# # The priors are defined as a list with the following format:
# # [prior_type, prior_arg_1, prior_arg_2, ...]
# # The prior_type is a string that defines the type of prior.
# # It can be one of the default priors found in retrieval_utils.default_prior_func.
# # or a custom prior function (see explanation below).
# # The prior_args are the arguments needed for the prior function.

# # The default priors are:
# # - uniform or log_uniform: Uniform prior between prior_arg_1 and prior_arg_2
# # - gaussian: Gaussian prior with mean prior_arg_1 and standard deviation prior_arg_2
# # 
# # NOTE: The difference between uniform and log_uniform is that the values are
# # converted to log scale (exponentiated) when creating the theta_dict.
# # Ex:
# # >>> params_prior['CO'] = ['log_uniform', -12.0, -0.5]
# # will give in the logl function:
# # >>> theta_dict['CO'] = 10 ** theta_dict['CO']

# # - Note about Abundances priors -
# # In the current example, we don't use chemical equilibrium, so the priors are set for each molecule.
# # Note that if you want to fit only a subset of the molecules in line_opacities, continuum_opacities and other_species,
# # you need to assign a value to the species that are not fitted in fixed_params.
# # Here is an example of a possible set of priors:
# params_prior:
#   # You don't need to add the species since they can be automatically added
#   # based on the species_in_prior list.
#   # But if you want to manually add the species in the prior, you can add it here.
#   # In this example, I want to fit the e- in other_species, but not H, so I add it here.
#   # CO: [log_uniform, -12.0, -0.5]

#   # Orbital parameters
#   rv: [uniform, -40, 40]  # For RV
#   kp: [uniform, 100, 250]  # For RV

#   # TP-profile parameters
#   T_eq: [uniform, 100, 4500]  # For iso, guillot and modif
#   tp_gamma: [log_uniform, -2, 6]  # For guillot and modif
#   # kappa_IR: [log_uniform, -4, 0]  # For guillot
#   gravity: [uniform, 1000, 10000]  # For guillot
#   tp_delta: [log_uniform, -8, -3]  # For modif
#   ptrans: [log_uniform, -8, 3]  # For modif
#   tp_alpha: [uniform, -1.0, 1.0]  # For modif

#   # Other parameters related to the atmosphere
#   # params_prior['p_cloud'] = ['log_uniform', -5.0, 2]
#   # params_prior['R_pl'] = ['uniform', 1.0, 1.6]

#   # Scaling factor for WFC3 uncertainties
#   # params_prior['log_f'] = ['uniform', -2, 3]  # yes, it is a uniform prior, not a log_uniform


# # If the planet is splitted in multiple regions,
# # specify which parameters are specific to each region.
# # Each region is identified with an ID starting at 1 (not 0).
# region_id: [1] 

# # List of parameters that are specific to each region (some parameters are shared by all regions)
# # It won't do anything if there is only one region.
# # Put empty list if not used.
# reg_params: []


# # Dictionary of fixed parameters for each region.
# # It may be useful to fix some parameters in specific regions.
# # In this example, we force a low water abundance in region 2.
# reg_fixed_params: 
#   # H2O_2: 1e-12  # Force water abundance to ~zero in region 2


# #### Note about theta_dict ####

# # Apart from all the parameters in the fixed_params and params_prior dictionnaries,
# # the theta_dict will also contain the following parameters:
# # - pressure: The pressure grid used for the model
# # - temperatures: The temperature grid used for the model, computed based on the TP profile parameters
# # - M_pl: The planet mass in Jupiter mass
# # - R_pl: The planet radius in Jupiter radius
# # - R_star: The star radius in Solar radius

# # The custom functions (like the rotation kernel function or custom priors)
# # are expected to receive the full theta_dict as an argument.
# # This way, they can access all the parameters needed for the model.


# #####################################################
# # --- Custom prior functions ---
# #####################################################
# # The default prior functions are defined in the retrieval_utils module.
# # If you want to add your own prior function, you can put the function in a file
# # which will be imported in the script.
# # The prior function should take `theta_dict`, `param` and `prior_args` as arguments
# # Where `param` is the key of the parameter in `theta_dict`
# # and `prior_args` is a list regrouping all arguments defined
# # in `params_prior`, after the prior type.
# # NOTE: Some prior functions may use other parameters (fitted or not). That's why
# # the function is taking the full `theta_dict` as an argument.
# # Here is an example of the uniform prior function in retrieval_utils:
# # 
# # def uniform_prior(theta_dict, key, prior_inputs):
# #     """Uniform prior"""
# #     x = theta_dict[key]
# #     low, high = prior_inputs
# #     if x < low:
# #         out = -np.inf
# #     elif x > high:
# #         out = -np.inf
# #     else:
# #         out = 0.
# # 
# #     return out
# # 
# #####################################################
# # --- Custom prior initialisation functions ---
# #####################################################
# # Each prior type should have an initialisation function, which is used to
# # initialise the walkers (to start the burnin phase for example).
# # If a custom prior function is defined, it should also have an initialisation function in the same file.
# # The initialisation function should take `prior_inputs` and `n_wlkr` (the number of walker) as arguments.
# # It should return a sample of the prior distribution with the shape (n_wlkr, 1).
# # (The empty dimension is important to combine the samples with the other parameters in the theta_dict.)
# # Here is an example of the uniform prior initialisation function in retrieval_utils:
# #
# # def init_uniform_prior(prior_inputs, n_wlkr):
# #     """Initialize from uniform prior"""
# #     low, high = prior_inputs
# #     sample = np.random.uniform(low, high, size=(n_wlkr, 1))

# #     return sample
# #
# # Now, in the file with the custom prior function, you also need 2 dictionaries named
# # prior_init_func and prior_func, that assign a name (a string) to the initialisation
# # function and the prior function.
# # Here is an example of the dictionary for the uniform prior:
# # prior_init_func = {'custom_uniform': init_uniform_prior}
# # prior_func = {'custom_uniform': uniform_prior}

# # Put here the full adress to the file containing the custom prior function
# # Set to null if not used
# custom_prior_file: null

# # --- Special treatment for the initialisation of some parameters ---
# # Some parameters may need a special treatment for the initialisation from the prior.
# # For instance, I recommend to start initialise the retrieval with a kp value
# # and a rv value close to the expected values, since we have a good idea of what they should be.
# # This will improve the convergence of the retrieval.
# special_init:
#   kp: ['uniform', 200, 250]
#   rv: ['uniform', -30, 30]
#   CO: ['log_uniform', -5.0, -2.0]
#   Fe: ['log_uniform', -6.0, -2.0]
#   T_eq: ['uniform', 2500, 3500]

# # --- Rotation kernel function ---
# # It is possible to use a custom function to generate the rotation kernel.
# # The function should take as input `theta_regions` and `tr_i`.
# # `theta_regions` is a list of `theta_dict` for each region, where `theta_dict` is a dictionnary
# # with all the parameters needed for the model of region.
# # `tr_i` is the index of the high resolution data sequence.
# # The function should return a list of rotation kernels (a 1D array) at the same sampling
# # and resolution as the data (so uses instrum['resol'] and lbl_res).
# # The function should be named `get_ker`.
# # Here is an example of a function that generates a Citrus rotation kernel:
# #
# # def get_ker(theta_regions, tr_i=0):
# #     all_phases = (data_trs[tr_i]['t_start'] - planet.mid_tr.value) / planet.period.to('d').value % 1
# #     mean_phase = np.mean(all_phases[data_trs[tr_i]['i_pl_signal']])  # Only  where there is planet signal
# #     theta_dict = theta_regions[0]
# #     longitudes = [theta_dict['phase1'], 0.99]
# #     R_pl = theta_dict['R_pl'] * 0.01  # cgs to SI  (cm to m)
# #     angular_freq = 2 * np.pi / planet.period[0].to('s').value
# #     rot_factor = theta_dict['rot_factor']
# #
# #     ker_obj = CitrusRotationKernel(longitudes, R_pl, rot_factor * angular_freq, instrum['resol'])
# #
# #     rot_ker = ker_obj.resample(lbl_res, phase=mean_phase, n_os=1000)
# #
# #     return rot_ker
# #
# # The full or relative path to the file is the input here.
# # If set to null, only the instrumental profile will be used to convolve the model.
# get_ker_file: null


# -----------------------------------------------------------
# ##################################################################
# ############# New version of the retrieval that uses the yaml file
# ##################################################################
# -----------------------------------------------------------

# Do imports when needed

import sys
sys.path.append('/home/fgenest/')
sys.path.append('/home/fgenest/starships/')

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from pathlib import Path
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
from starships.analysis import bands, resamp_model
import starships.planet_obs as pl_obs
from starships.planet_obs import Observations, Planet
import starships.petitradtrans_utils as prt
from starships.homemade import unpack_kwargs_from_command_line, pop_kwargs_with_message
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
    from petitRADTRANS.nat_cst import guillot_global, guillot_modif
    
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
        if input_params[key] is None:
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
    
                
    return input_params


# # Here I just define all the parameters so they are recognized by the code
# pl_name = 'WASP-33 b'
# base_dir = Path('/scratch/adb')
# high_res_path = Path('~/DataAnalysis/SPIRou/Reductions')
# reduc_name = 'WASP-33b_v07232'
# high_res_file_stem_list = ['retrieval_input_4-pc_mask_wings97_day2']
# spectrophotometric_data = {'wfc3': {'file_path': '/home/adb/projects/def-dlafre/adb/Observations/HST/WFC3', 'file_name': 'WASP-33_WFC3_full_res.ecsv'}}
# photometric_data = {}
# retrieval_type = 'HRR'
# white_light = False
# chemical_equilibrium = False
# dissociation = True
# add_spitzer = False
# kind_temp = 'modif'
# n_steps_burnin = 300
# n_steps_sampling = 3000
# run_name = 'emission_HRR_wfc3_modif_disso_30303539'
# walker_path = Path('/scratch/adb/DataAnalysis/walker_steps/WASP-33_b')
# walker_file_out = Path('walker_steps_emission_HRR_wfc3_modif_disso_30303539.h5')
# walker_file_in = None
# init_mode = 'from_burnin'
# slurm_array_behaviour = 'burnin'
# params_path = Path('/scratch/adb/DataAnalysis/retrieval_params/WASP-33_b')
# params_file_out = Path('params_emission_HRR_wfc3_modif_disso_30303539.yaml')
# kind_trans = 'emission'
# n_cpu = 2
# n_walkers = 36
# n_walkers_per_cpu = 18
# opacity_sampling = 4
# orders = None
# pl_params = {'M_star': 1.561 *u.solMass, 'R_star': 1.5093043 *u.solRad, 'R_pl': 1.6787561* u.jupiterRad, 'excent': 0.0,
#               'incl': 86.63 *u.deg, 'Teff': 7300. *u.K, 'w': -90.* u.deg, 'ap': 0.0259 *u.AU}
# instrum = ['spirou']
# line_opacities = ['CO', 'H2O']
# continuum_opacities = ['H-']
# other_species = ['e-', 'H']
# species_in_prior = ['CO', 'H2O', 'H-', 'e-']
# linelist_names = {}
# fixed_params = {'P0': 0.001, 'log_f': 0.0, 'spec_scale': 1.0, 'e-': '1e-06', 'H': '1e-99', 'C/O': 0.54, 'Fe/H': 0.0, 'p_cloud': None, 'scat_factor': None, 'T_eq': 2500.0, 'tp_gamma': 10.0, 'T_int': 500.0, 'kappa_IR': 0.01, 'gravity': 23.0, 'tp_delta': '1e-07', 'ptrans': '1e-03', 'tp_alpha': 0.3, 'akima_P': ['1e-10', '1e-05', '1e-02', '1e+01'], 'akima_T': [1000.0, 1500.0, 2000.0, 2500.0], 'kp': 150.0, 'rv': 0.0, 'wind': None, 'phase1': 0.5, 'phase2': 0.75, 'rot_factor': 1.0}
# params_prior = {'rv': ['uniform', -40, 40], 'kp': ['uniform', 100, 250], 'T_eq': ['uniform', 100, 4500], 'tp_gamma': ['log_uniform', -2, 6], 'gravity': ['uniform', 1000, 10000], 'tp_delta': ['log_uniform', -8, -3], 'ptrans': ['log_uniform', -8, 3], 'tp_alpha': ['uniform', -1.0, 1.0]}
# region_id = [1]
# reg_params = []
# reg_fixed_params = {}
# custom_prior_file = None
# special_init = {'kp': ['uniform', 200, 250], 'rv': ['uniform', -30, 30], 'CO': ['log_uniform', -5.0, -2.0], 'Fe': ['log_uniform', -6.0, -2.0], 'T_eq': ['uniform', 2500, 3500]}
# get_ker_file = None

# limP = [-10, 2]    # pressure log range, standard
# n_pts = 50  
# star_spectrum = '/home/adb/projects/def-dlafre/adb/Observations/SPIRou/Reductions/WASP-33b_v07232/WASP-33b_v07232_star_spectrum.npz'

# The functions above should be in a separate file (maybe retrieval_utils.py)

# -----------------------------------------------------------

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
    # Get wv_range for each high_res instrument
    instrum_param_list = [load_instrum(instrum_name) for instrum_name in instrum]
    wv_range_high = [instrum_param['high_res_wv_lim'] for instrum_param in instrum_param_list]
    wv_range_high = get_wv_range(wv_range_high)
    log.info(f'wavelength range for model at high res: {wv_range_high}')

    # Define the low resolution wavelength range based on low resolution data
    # NOTE: The low-res models are taking less memory, so we model the full range,
    #       even the regions in between where there is no data.
    load_low_res_data()
    load_photometry()
    wv_range_all_low = [infos['wv_range'] for infos in spectrophotometric_data.values()]
    wv_range_all_low += [infos['wv_range'] for infos in photometric_data.values()]

    if retrieval_type != 'LRR':
        # In JR or HRR, model at least the full range of the high-res data.
        # It won't necessarily be used, but it is useful for analysis later on.
        # So add it to the list of wv_range
        wv_range_all_low.append([np.min(wv_range_high), np.max(wv_range_high)])
    
    # The low resolution data will model the full range
    wv_range_low = [[np.min(wv_range_all_low), np.max(wv_range_all_low)]]
    log.info(f'wavelength range for model at low res: {wv_range_low}')
    
    # Assign (to each low res dataset) which kind of model (high or low)
    # will be used to compare with the data (only use in JR mode)
    if retrieval_type == 'JR':
        assign_model_type(wv_range_high)
        
    # --- Define the default resolution for high-res models ---
    # The models at high resolution will be downgraded
    # to the highest instruments resolution when generated.
    # Then the model can be downgraded again to match other high-res instrument's
    global res_instru
    res_instru = max([p_list['resol'] for p_list in instrum_param_list])

    # --- Initialize model objects to None ---
    # Initialize atmo objects based on the wavelength ranges (put None for now)
    for mode in ['high', 'low']:
        wv_rng = globals()[f'wv_range_{mode}']
        for i_range, _ in enumerate(wv_rng):
            globals()[f'atmo_{mode}_{i_range}'] = None

    # Same for the stellar spectra
    for mode in ['high', 'low']:
        globals()[f'fct_star_{mode}'] = None

    # --- Resolution of the planet model ---
    global prt_res
    prt_res = {'high': int(1e6 / opacity_sampling), 'low': 1000}

    # --- Additional variables ---
    global inj_alpha, nolog, do_tr
    inj_alpha = 'ones'
    nolog = True
    do_tr = [1]

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
    # TODO: Import correctly using the file in yaml file
    global get_ker
    get_ker = lambda theta_regions, tr_i: [None for _ in theta_regions]

    return input_params

# ONce the setup_retrieval function is run, all the parameters will be accessible in the global variables.

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
    The function reads the input data and prepare it."""

    global data_info, data_trs

    data_info = {'trall_alpha_frac': [], 'trall_icorr': [], 'trall_N': [], 'bad_indexs': []}
    data_trs = []

    for high_res_file_stem in high_res_file_stem_list:
        log.debug(f'Hires files stem: {high_res_path / high_res_file_stem}')
        log.info('Loading Hires files.')
        data_info_i, data_trs_i = pl_obs.load_sequences(high_res_file_stem, do_tr, path=high_res_path)
        # Add index of the exposures where we expect to see the planet signal (to be used in kernel function)
        # trall_alpha_frac is the fraction of the total planet signal received during the exposure.
        data_trs_i['0']['i_pl_signal'] = data_info_i['trall_alpha_frac'] > 0.5
        for data_tr in data_trs_i.values():
            data_trs.append(data_tr)
        # Patch for now data_info. Need to modify how the logl is computed to make it more clean.
        # Would not work with different instruments
        for key in ['trall_alpha_frac', 'trall_N', 'bad_indexs']:
            data_info[key].append(data_info_i[key])
        try:
            data_info['trall_icorr'].append(data_info_i['trall_icorr'] + data_info['trall_icorr'][-1][-1] + 1)
        except IndexError:
            data_info['trall_icorr'].append(data_info_i['trall_icorr'])
    for key in data_info.keys():
        data_info[key] = np.concatenate(data_info[key], axis=0)
    
    data_info['bad_indexs'] = None  # Leave it to None. Not implemented yet.

    return data_info, data_trs


def load_low_res_data(pad_n_res_elem=5):
    """Load the low resolution data.

    This function loads the low resolution data for each instrument
    specified in the `spectrophotometric_data` dictionary, which needs
    to exist in the global variables.
    The function reads the data file (found in `spectrophotometric_data`),
    extracts the wavelengths, data, uncertainties, instrument resolution, and wavelength range.

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
            
        # Read the data file (astropy table)
        low_res_path = Path(infos['file_path'])
        low_res_file = Path(infos['file_name'])
        data_table = Table.read(low_res_path / low_res_file)
        
        # Get the wavelenghts
        default_name = 'wave'
        col_name = infos.get('wv_col_name', default_name)
        # infos['wave'] = data_table[col_name].to('um').value
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
    
def assign_model_type(wv_rng_list_high):
    """Assign the kind of model (high res or low res) that will be used
    to create synthetic data. The input is the list of wavelength ranges
    that are covered by the high res models. If one of these ranges covers
    entirely the data of a specific instrument, then the high res model is used.
    The low res model is used otherwise."""
    
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
               

def init_stellar_spectrum(mode, wl_range=None):

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
        resamp_star = resamp_model(star_wv[is_in_range], star_flux[is_in_range], star_res, Raf=Raf)
        resamp_star = np.ma.masked_invalid(resamp_star)
        fct_star = interp1d(star_wv[is_in_range], resamp_star)
        
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
        for key in ['wind', 'p_cloud', 'gamma_scat', 'scat_factor', 'C/O', 'Fe/H', 'cloud_fraction']:
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
        if theta_dict['cloud_fraction'] == None or theta_dict['cloud_fraction'] == 1:
            wv_out, model_out = prt.retrieval_model_plain(atmo_obj, species, planet, *args, **kwargs)
        else:
            cloud_f = theta_dict['cloud_fraction']
            clear_f = 1 - cloud_f
            wv_out, model_out_cloudy = prt.retrieval_model_plain(atmo_obj, species, planet, *args, **kwargs)
            theta_dict['p_cloud_clear'] = None
            args_clear = [theta_dict[key] for key in ['pressures', 'temperatures', 'gravity', 'P0', 'p_cloud_clear', 'R_pl', 'R_star']]
            wv_out, model_out_clear = prt.retrieval_model_plain(atmo_obj, species, planet, *args_clear, **kwargs)
            model_out = (cloud_f * model_out_cloudy) + (clear_f * model_out_clear)

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
                wv_out, model_out = prt.prepare_model(wv_out, model_out, prt_res[mode], Raf=Raf,
                                                    rot_ker=rot_ker, **rot_kwargs)

        wv_all.append(wv_out)
        model_all.append(model_out)

    wv_all = np.concatenate(wv_all)
    model_all = np.concatenate(model_all)

    return wv_all, model_all

def prepare_model_multi_reg(theta_regions, mode, rot_ker_list=None, atmo_obj=None, tr_i=0, Raf=None):
    
    # Get the list of rotation kernels
    rot_ker_list = get_ker(theta_regions, tr_i=tr_i)
    
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


def prepare_photometry(wv_mod, spec_mod, model_res, data_info, mod_sampling=None, integrate_fct='simpson'):
    
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
    kwargs = dict(Raf=instru_res, Rbf=model_res, sample=wv_mod_sub)
    _, resamp_prt = spectrum.resampling(wv_mod_sub, spec_mod_sub, **kwargs)
    
    # Apply the response function to the spectrum
    mod_out = list()
    for fct_i in fct_band:
        response_i = fct_i(wv_mod_sub)
        norm = integrate_fct(response_i, x=wv_mod_sub)
        mod_i = integrate_fct(response_i * resamp_prt, x=wv_mod_sub) / norm
        mod_out.append(mod_i)
    mod_out = np.array(mod_out)
    

    return wv_band, mod_out


def prepare_spectrophotometry(wv_mod, spec_mod, model_res, data_info, mod_sampling=None):
    
    if mod_sampling is None:
        mod_sampling = model_res
    
    # Get the values needed from the data_info dictionary
    wv_rng, instru_res, wv_grid = (data_info[key] for key in ['wv_range', 'res', 'wave'])
    
    # TODO: Add the possibility to use unequal spectral bins
    # The binning function spectrum.box_binning needs to be replaced
    # because it assumes evenly spaced grid for now.
    # The function that reads the spectrophotometry should also
    # be changed to be able to read bin limits.
    
    # Downgrade to instrument resolution
    cond = (wv_mod >= wv_rng[0]) & (wv_mod <= wv_rng[-1])
    kwargs = dict(Raf=instru_res, Rbf=model_res, sample=wv_mod[cond])
    _, resamp_prt = spectrum.resampling(wv_mod[cond], spec_mod[cond], **kwargs)
    
    # Bin the spectrum and interpolate
    # TODO: replace the binning function, which is just a box convolution for now.
    binned_prt = spectrum.box_binning(resamp_prt, mod_sampling / instru_res)
    fct_prt = interp1d(wv_mod[cond], binned_prt)
    # Project into instrument wv grid
    mod = fct_prt(wv_grid)

    return wv_grid, mod


def lnprob(theta, ):
    
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

        logl_i = []
        # --- Computing the logL for all sequences
        for tr_i, data_tr_i in enumerate(data_trs):
            
            # NOTE: Not optimal to re-compute the model for each sequence.
            # Could be done once for all regions and then the rotation kernel
            # could be applied to the model for each region depending on the phase.
            wv_high, model_high = prepare_model_multi_reg(theta_regions,
                                                          'high',
                                                          tr_i=tr_i,
                                                          Raf=res_instru)
            if not np.isfinite(model_high[100:-100]).all():
                log.warning("NaN in high res model spectrum encountered")
                return -np.inf
            
            vrp_orb = rv_theo_t(theta_dict['kp'],
                                data_tr_i['t_start'] * u.d, planet.mid_tr,
                                planet.period, plnt=True).value

            args = (theta_dict['rv'], data_tr_i, planet, wv_high, model_high)
            kwargs = dict(vrp_orb=vrp_orb, vr_orb=-vrp_orb * Kp_scale, nolog=nolog,
                          alpha=np.ones_like(data_tr_i['t_start']), kind_trans=kind_trans)
            logl_tr = corr.calc_log_likelihood_grid_retrieval(*args, **kwargs)

            if not np.isfinite(logl_tr).all():
                return -np.inf

            logl_i.append(logl_tr)

        logl_all_visits = np.concatenate(logl_i, axis=0)
        log.debug(f'Shape of individual logl for all exposures (all visits combined): {logl_all_visits.shape}')
        total += corr.sum_logl(logl_all_visits, data_info['trall_icorr'], orders,
                               data_info['trall_N'], axis=0, del_idx=data_info['bad_indexs'], nolog=True,
                               alpha=data_info['trall_alpha_frac'])

    ###################
    # --- LOW RES --- #
    ###################
    if (retrieval_type == 'JR') or (retrieval_type == 'LRR') or white_light:
        
        # If at least one instrument need the low-res model, then compute it
        model_type = [infos.get('model_type', 'low') for infos
                      in list(spectrophotometric_data.values()) + list(photometric_data.values())]
        if 'low' in model_type:
            wv_low, model_low = prepare_model_high_or_low(theta_dict, 'low')
            
            if np.sum(np.isnan(model_low)) > 0:
                log.info("NaN in low res model spectrum encountered")
                return -np.inf

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
                    args = (wv_high, model_high, res_instru, infos, prt_res['high'])
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

        del wv_low, model_low

    if retrieval_type != 'LRR':
        del wv_high, model_high

    gc.collect()

    log.debug(f'logL = {total}')

    return total


def find_max_lnprob(param_init):
    """Find the maximum likelihood value using scipy.optimize.minimize
    param_init is the initial position. It has a shape (n_try, n_params)
    """
    
    
    


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


# NOTE: theses checks could be replaced by a schema validation
def check_cmd_line_args(cmd_line_kw):

    # Make sure some variables are converted to integers
    int_keys = ['n_steps_burnin', 'n_steps_sampling', 'n_walkers', 'n_cpu',
                'n_walkers_per_cpu']

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
