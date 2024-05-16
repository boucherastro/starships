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

def set_save_location(obs_dir, pl_name, reduction, instrument):

    ''' Note: Better to use the scratch space to save the reductions.
    You have infinte space, but the files are deleted if untouched for 2 months. It allows to save as 
    many reductions as desired. Once the correct reduction parameters are set, you can either move them 
    into your home directory. '''

    # finding WASP-127b files to be reduced
    # os.system("!ls /home/opereira/projects/def-dlafre/fgenest/nirps/WASP-127b/")

    # pl_name = 'WASP-127 b'
    # reduction = 'genest'    # naming of reduction, YAML input
    # instrument = input on YAML file

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
    path_fig = Path('Figures/Reductions')

def load_planet(pl_name, obs_dir, pl_kwargs, instrument):
    # --- Where to find the observations?
    # obs_dir = Path.home() / Path(f'projects/def-dlafre/fgenest/nirps/WASP-127b/')

    # All the observations must be listed in files.
    # We need the e2ds, the telluric corrected and the reconstructed spectra.
    list_filenames = {'list_e2ds': 'list_e2ds',
                    'list_tcorr': 'list_tellu_corrected',
                    'list_recon': 'list_tellu_recon'}

    # Available instruments
    # print(list(instruments_drs.keys()))

    # Some parameter of the planet system can be specified here.
    # If not, the default parameter from exofile are taken. Make sure they are ok (see cell below).
    pl_kwargs = {
    #     'M_star': 1.89 *u.M_sun,
    #     'R_star': 1.60 *u.R_sun,
    #     'M_pl' : 3.38 * u.M_jup,
    #     'R_pl' : 1.83 * u.R_jup,
                }

    # check if any planet attributes were manually specified
    if bool(pl_kwargs): 
        obs = Observations(name=pl_name, instrument=instrument, pl_kwargs=pl_kwargs)
    else: 
        obs = Observations(name=pl_name, instrument=instrument)

    p = obs.planet

    # Print some parameters (to check if they are satisfying)
    # print((f"M_pl: {p.M_pl.to('Mjup')}\n"
    #     f"M_star: {p.M_star.to('Msun')}\n"
    #     f"R_star: {p.R_star.to('Rsun')}\n"
    #     f"R_pl: {p.R_pl.to('Rjup')}\n"
    #     f"RV_sys: {p.RV_sys}"))

    # Get the data
    obs.fetch_data(obs_dir, **list_filenames)

    # new_mask = obs.count.mask | (obs.count < 400.)
    # obs.flux = np.ma.array(obs.flux, mask=new_mask)
    return p, obs

'''**********************************************************************************************'''
'''                                  Build Transmission Spectrum                                 '''

def build_trans_spec(visit_name, mask_tellu, mask_wings, n_pc, coeffs, ld_model, kind_trans, iout_all, 
                        clip_ratio, clip_ts, unberv_it, obs, planet):

    # visit_name = 'tr1'  # Choose a visit name for the saved file names

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
    # mask_tellu = 0.2  # Identify the deep tellurics that will be masked (in fraction of abosption)
    # mask_wings = 0.9  # Mask wings of these deep tellurics (in fraction of absorption)
    # n_pc = 3
    params_all=[[mask_tellu, mask_wings, 51, 41, 5, n_pc, 5.0, 5.0, 5.0, 5.0]]


    # --- For emission spectra
    # coeffs = [0.532]
    # ld_model = 'linear'
    # kind_trans='emission'
    # --- For transmission spectra
    # coeffs = [ 0.02703969,  1.10037972, -0.96372403,  0.28750393]  # For transits
    # ld_model = 'nonlinear'
    # kind_trans='transmission'

    # cbp=True  # Correct bad pixels

    RVsys = [planet.RV_sys.value[0]]
    # iout_all = ['all']  # Which exposures are used to build the star's reference spectrum
    transit_tags = [None]

    kwargs_gen_tr = {
    'coeffs' : coeffs,
    'ld_model' : ld_model,
    'do_tr' : [1],
    'kind_trans' : kind_trans,
    'polynome' : [False],
    'cbp': True
    }

    kwargs_build_ts = {
    'clip_ratio' : clip_ratio,
    'clip_ts' : clip_ts,
    'unberv_it' : unberv_it,
    }

    # If you want to test many different parameters (like n_pc, mask_tellu, mask_wings),
    # You could start a for loop here, changing the corresponding value in `params_all`.

    # Extract the planetary signal
    list_tr = pl_obs.generate_all_transits(obs, transit_tags, RVsys, params_all, iout_all,
                                        **kwargs_gen_tr, **kwargs_build_ts)

    return list_tr


def save_pl_sig(list_tr, out_dir, n_pc, mask_wings, visit_name, do_tr = [1]):
    # Save sequence with all reduction steps
    # For this example, we will comment this part to save disk space on your computer
    # out_filename = f'sequence_{n_pc}-pc_mask_wings{mask_wings*100:n}_{visit_name}'
    # pl_obs.save_single_sequences(out_filename, list_tr['1'], path=out_dir, save_all=True)

    # Save sequence with only the info needed for a retrieval (to compute log likelihood).
    out_filename = f'retrieval_input_{n_pc}-pc_mask_wings{mask_wings*100:n}_{visit_name}'
    pl_obs.save_sequences(out_filename, list_tr, do_tr, path=out_dir)

    # print('Kp =', list_tr['1'].Kp)

'''**********************************************************************************************'''
'''                                      Output Result Plots                                     '''

# visit_list = [list_tr['1']]  # You could put multiple visits in the same figure
# pf.plot_airmass(visit_list)

# sequence_obj = list_tr['1']

# idx_ord = 40
# pf.plot_steps(sequence_obj, idx_ord)
