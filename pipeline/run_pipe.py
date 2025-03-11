import numpy as np
import yaml
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import os
import traceback

from multiprocessing import Pool
from itertools import product
import astropy.units as u

import pipeline.reduction as red
import pipeline.make_model as mod
import pipeline.correlations as corr
import pipeline.split_nights as split
import pipeline.injection_recovery as inj_rec

from importlib import reload

'''--------------------------------------Helper Functions----------------------------------------'''
def main_loop(mask_tellu, mask_wings, n_pc, mol, config_dict, planet, obs, dirs_dict, 
              visit_name, wave_mod, mod_spec):
    ''' Main loop for the pipeline. '''
    naming_params = {'molecule': mol, 'n_pc': n_pc, 'mask_tellu': mask_tellu, 'mask_wings': mask_wings}
    print('CURRENT PARAMETERS:', naming_params)

    # reducing the data
    reduc = red.reduce_data(config_dict, planet, obs, dirs_dict['scratch_dir'], 
                            dirs_dict['red_steps_dir'], n_pc, mask_tellu, mask_wings, visit_name)

    # perfoming classic ccf (translates model)
    try:
        nametag = f'_{visit_name}_{mol}_maskwings{mask_wings*100:n}_masktellu{mask_tellu*100:n}_pc{n_pc}'
        corr.classic_ccf(config_dict, reduc, wave_mod, mod_spec, dirs_dict['classic_ccf_dir'], nametag) 
    except TypeError:
        print('Classic CCF could not be performed. Skipping...')
        traceback.print_exc()

    # performing injected ccf and saving the files
    corr.perform_ccf(config_dict, reduc, mol, wave_mod, mod_spec, n_pc, mask_tellu, mask_wings, 
                     dirs_dict['scratch_dir'], dirs_dict['injected_ccf_dir'], visit_name)

    # combining visits
    if (len(config_dict['visit_name']) > 1) and (visit_name == config_dict['visit_name'][-1]):
        if ([mask_tellu, mask_wings, n_pc] == config_dict['night_params']):
            try:
                comb_dir_dict = red.set_save_location(config_dict['pl_name'], 'combined', 
                                                    config_dict['reduction'], config_dict['instrument'])
                corr.combined_visits_ccf(planet, mol, wave_mod, mod_spec, comb_dir_dict, config_dict)
            except:
                print('Could not combine visits. Skipping...')
                traceback.print_exc()

def main_loop_wrapper(*args):
    ''' Wrapper function to pass multiple arguments to the main loop. '''
    *iterable_args, kwargs = args
    return main_loop(*iterable_args, **kwargs)

def multi_param_plots(config_dict, planet, mol, dirs_dict, visit_name):
    ''' Function to create plots for all parameters. '''
    # plots for each pc
    for mask_tellu in config_dict['mask_tellu']:
        for mask_wings in config_dict['mask_wings']:
            corr.plot_all_ccf(config_dict, mol, mask_tellu, mask_wings, dirs_dict['scratch_dir'], 
                            visit_name, planet, id_pc0=None, order_indices=np.arange(75), 
                            path_fig = dirs_dict) # give dict as input, will select correct path based on that

    # plots for each mask_wings 
    for mask_tellu in config_dict['mask_tellu']:
        for n_pc in config_dict['n_pc']:
            # load all ccf map and reductions for (mask_tellu, n_pc) at each mask_wings
            corr.plot_all_maskwings(config_dict, planet, mol, mask_tellu, n_pc, dirs_dict['scratch_dir'], 
                                    visit_name, id_pc0=None, order_indices=np.arange(75), 
                                    path_fig = str(dirs_dict['param_dir']) + '/')

    # plots for each mask_tellu
    for n_pc in config_dict['n_pc']:
        for mask_wings in config_dict['mask_wings']:
            corr.plot_all_masktellu(config_dict, planet, mol, mask_wings, n_pc, dirs_dict['scratch_dir'], 
                                    visit_name, id_pc0=None, order_indices=np.arange(75), 
                                    path_fig = str(dirs_dict['param_dir']) + '/')

def pool_processing(config_dict, planet, obs, dirs_dict, visit_name, wave_mod, mod_spec, mol):
    ''' Function to run the main loop in parallel using multiprocessing '''
    iterables = product(config_dict['mask_tellu'], config_dict['mask_wings'], config_dict['n_pc'])

    # Input n_processes to see how many iterables to run in parallel
    n_processes = 4
    with Pool(n_processes) as pool:
        args = {
            'mol': mol,
            'config_dict': config_dict,
            'planet': planet,
            'obs': obs,
            'dirs_dict': dirs_dict,
            'visit_name': visit_name,
            'wave_mod': wave_mod,
            'mod_spec': mod_spec
        }

        # Create a list of all arguments including those from iterables
        all_args = [(arg_tuple + (args,)) for arg_tuple in iterables]

        # Use pool.starmap to pass multiple arguments
        pool.starmap(main_loop_wrapper, all_args)
           
def injection_recovery(config_dict, planet, obs, mol, dirs_dict, visit_name, wave_mod, mod_spec):
    # only doing for one set of parameters, so no need for pool processing or multi param plots

    # change directories so everything saved in injected-recovery results folder
    dirs_dict_new = dirs_dict.copy()
    dirs_dict_new['scratch_dir'] = dirs_dict['scratch_dir'] / 'injected_fits'
    dirs_dict_new['scratch_dir'].mkdir(parents=True, exist_ok=True)

    inj_dir = dirs_dict['out_dir'] / Path('Results') / Path('Injected')
    inj_dir.mkdir(parents=True, exist_ok=True)

    dirs_dict_new['red_steps_dir'] = inj_dir
    dirs_dict_new['classic_ccf_dir'] = inj_dir
    dirs_dict_new['injected_ccf_dir'] = inj_dir
    dirs_dict_new['ttest_dir'] = inj_dir

    # inject the model into the data and save the new fits files
    inj_rec.main(config_dict, planet, obs, visit_name, wave_mod, mod_spec, 
                    scratch_dir = dirs_dict_new['scratch_dir'], path_fig = str(inj_dir))

    mask_tellu = config_dict['inj_params'][0]
    mask_wings = config_dict['inj_params'][1]
    n_pc = config_dict['inj_params'][2]

    # Make a copy of the config dict and change the observations file so it grabs the new
    config_dict_new = config_dict.copy()
    config_dict_new['obs_dir'] = dirs_dict_new['scratch_dir']
    config_dict_new['n_pc'] = [n_pc]

    # make a new planet and obs object using the injected filed
    planet_new, obs_new = red.load_planet(config_dict_new, visit_name)

    main_loop(mask_tellu, mask_wings, n_pc, mol, config_dict_new, planet_new, obs_new, dirs_dict_new, 
            visit_name, wave_mod, mod_spec)

    # plot the injected ccf and ttest results
    # corr.plot_all_ccf(config_dict_new, mol, mask_tellu, mask_wings, dirs_dict_new['scratch_dir'], 
    #                         visit_name, planet, id_pc0=n_pc, order_indices=np.arange(75), 
    #                         path_fig = dirs_dict_new, param = None)
    
    corr.plot_all_ccf(config_dict_new, mol, mask_tellu, mask_wings, dirs_dict_new['scratch_dir'], 
                            visit_name, planet, id_pc0=None, order_indices=np.arange(75), 
                            path_fig = dirs_dict_new) # give dict as input, will select correct path based on that


    
'''-------------------------------------Actual pipeline------------------------------------------'''
def run_pipe(config_filepath, run_name):
    # unpack input parameters into config dictionary
    with open(config_filepath, 'r') as file:
        config_dict = yaml.safe_load(file)

    config_dict['reduction'] = run_name

    config_dict['obs_dir'] = Path.home() / Path(config_dict['obs_dir'])

    if config_dict['visit_name'] == []:
        config_dict['visit_name'] = split.split_night(config_dict['obs_dir'], str(config_dict['obs_dir']))
    print('VISIT NAMES: ', config_dict['visit_name'])
    
    # loop over all visits
    for visit_name in config_dict['visit_name']:
        print('Running night:' + visit_name)
        # creating the planet and observation objects
        planet, obs = red.load_planet(config_dict, visit_name)

        # if multiple mid_tr have been given, use the one for that visit
        # note: this is working, just commented out because yaml files have not been updated yet
        # if config_dict['mid_tr'] != []:
        #     config_dict['pl_params']['mid_tr'] = {
        #         'value' : config_dict['mid_tr'][config_dict['visit_name'].index(visit_name)],
        #         'unit' : 'd'
        #     }

        dirs_dict = red.set_save_location(config_dict['pl_name'], visit_name, 
                                          config_dict['reduction'], config_dict['instrument'])


        # Case 1: a list of existing models was given to test: 
        if isinstance(config_dict['input_model_file'], list):
            for file in config_dict['input_model_file']:
                model = np.load(file)
                wave_mod = model['wave_mod']
                mod_spec = model['mod_spec']
                mol = model['mol']

                # quick plot of the model
                plt.figure()
                plt.plot(wave_mod, mod_spec)
                plt.title(f'{mol} model')
                plt.xlabel('Wavelength (um)')
                plt.ylabel('Flux')
                plt.savefig(str(dirs_dict['out_dir']) + f'/{mol}_model.pdf')
                
                # wrapping in try-except loop
                try:
                    pool_processing(config_dict, planet, obs, dirs_dict, visit_name, wave_mod, mod_spec, mol)
                    multi_param_plots(config_dict, planet, mol, dirs_dict, visit_name)
                except:
                    #print the error
                    print(f'Error in processing {visit_name} night. Skipping...')
                    traceback.print_exc()
                    continue

                if config_dict['do_injection']: 
                    mod_spec = mod_spec * 1e-6
                    injection_recovery(config_dict, planet, obs, mol, dirs_dict, visit_name, wave_mod, mod_spec)


        # Case 2: if a single model file was given to test
        elif config_dict['input_model_file'].endswith('.npz'):
            model = np.load(config_dict['input_model_file'])
            wave_mod = model['wave_mod']
            mod_spec = model['mod_spec']
            mol = model['mol']

            plt.figure()
            plt.plot(wave_mod, mod_spec)
            plt.title(f'{mol} model')
            plt.xlabel('Wavelength (um)')
            plt.ylabel('Flux')
            plt.savefig(str(dirs_dict['out_dir']) + f'/{mol}_model.pdf')

            try:
                pool_processing(config_dict, planet, obs, dirs_dict, visit_name, wave_mod, mod_spec, mol)
                multi_param_plots(config_dict, planet, mol, dirs_dict, visit_name)
            except:
                print(f'Error in processing {visit_name} night. Skipping...')
                traceback.print_exc()
                continue

            if config_dict['do_injection']: 
                # THIS ASSUMES THE MODEL IS GIVEN IN PPM AND NEEDS TO BE CONVERTED (SCARLET MODELS)
                mod_spec = mod_spec * 1e-6
                injection_recovery(config_dict, planet, obs, mol, dirs_dict, visit_name, wave_mod, mod_spec)


        # Case 3: else, we make our own model based on the input file
        else: 
            reload(mod)
            
            with open(config_dict['input_model_file'], 'r') as file:
                config_model = yaml.safe_load(file)

            # initialize all molecule abundances
            if config_model['species_vmr'] == {}:
                print('Generating dummy VMRs before making model...')
                for mol in config_model['line_opacities']:
                    config_model['species_vmr'][mol] = -99.0

            # Start with model with all molecules
            wave_mod, mod_spec, abundances, MMW, VMR = mod.make_model(config_model, planet, 
                                                                        dirs_dict['out_dir'], config_dict)

            if len(config_model['line_opacities']) == 1: mol = config_model['line_opacities'][0]
            else: mol = 'all'

            if visit_name == config_dict['visit_name'][0]:
                mod.plot_model_components(config_model, planet, path_fig = str(dirs_dict['out_dir']))
        
            try:
                pool_processing(config_dict, planet, obs, dirs_dict, visit_name, wave_mod, mod_spec, mol)
                multi_param_plots(config_dict, planet, mol, dirs_dict, visit_name)
            except:
                print(f'Error in processing {visit_name} night. Skipping...')
                traceback.print_exc()
                continue

            if config_dict['do_injection']: 
                injection_recovery(config_dict, planet, obs, mol, dirs_dict, visit_name, wave_mod, mod_spec)
            
            # iterate over individual molecules if there are more than 1
            if len(config_model['line_opacities']) > 1:
                for single_mol in config_model['line_opacities']:
                    # Copy input dict
                    theta_dict_single = dict(config_model)
                    abundances_subtracted = dict(abundances)

                    # Need to be a list
                    if not isinstance(single_mol, list): single_mol = [single_mol]

                    # set all other molecules to zero
                    for mol in config_model['line_opacities']:
                        if mol != single_mol[0]:
                            linelist = config_model['linelist_names'][config_model['mode']][mol]
                            abundances_subtracted[linelist] = abundances_subtracted[linelist]*0 + 1e-99

                    theta_dict_single['chemical_equilibrium'] = False
                    wave_mod, mod_spec, _, _, _ = mod.make_model(theta_dict_single, planet, out_dir = None, 
                                                    abundances = abundances_subtracted,
                                                    MMW = MMW, config_dict=config_dict)

                    try: 
                        pool_processing(config_dict, planet, obs, dirs_dict, visit_name, wave_mod, mod_spec, single_mol[0])
                        multi_param_plots(config_dict, planet, single_mol[0], dirs_dict, visit_name)
                    except:
                        print(f'Error in processing {single_mol} for {visit_name}. Skipping...')
                        traceback.print_exc()
                        continue


'''----------------------------------------------------------------------------------------------'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the pipeline with the given config files.')
    parser.add_argument('config_filepath', type=Path, help='Path to the config.yaml file.')
    parser.add_argument('run_name', type=str, help='Name of reduction.')

    args = parser.parse_args()

    # Call the run_pipe function with the parsed arguments
    run_pipe(args.config_filepath, args.run_name)