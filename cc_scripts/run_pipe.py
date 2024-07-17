import numpy as np
import yaml
from sys import path
from pathlib import Path
import argparse

from multiprocessing import Pool
from itertools import product
import functools

path.append("opereira/starships/cc_scripts/")

import cc_scripts.reduction as red
import cc_scripts.make_model as mod
import cc_scripts.correlations as corr
import cc_scripts.split_nights as split

def main_loop(mask_tellu, mask_wings, n_pc, mol, config_dict, planet, obs, scratch_dir, out_dir, path_fig, visit_name, wave_mod, mod_spec):
    
    naming_params = {'molecule': mol, 'n_pc': n_pc, 'mask_tellu': mask_tellu, 'mask_wings': mask_wings}
    print('CURRENT PARAMETERS:', naming_params)

    # reducing the data
    reduc = red.reduce_data(config_dict, planet, obs, scratch_dir, out_dir, path_fig, n_pc, mask_tellu, mask_wings, visit_name)

    # performing correlations
    corr.perform_ccf(config_dict, reduc, mol, wave_mod, mod_spec, n_pc, mask_tellu, mask_wings, scratch_dir, path_fig, visit_name)

def main_loop_wrapper(*args):
    *iterable_args, kwargs = args
    return main_loop(*iterable_args, **kwargs)

## running the pipeline
def run_pipe(config_filepath, run_name):

    # unpack input parameters into config dictionary
    with open(config_filepath, 'r') as file:
        config_dict = yaml.safe_load(file)

    config_dict['reduction'] = run_name

    # unpack the model input parameters
    # eventually change this so that it can also open model files OR yaml files
    with open(config_dict['input_model_file'], 'r') as file:
        config_model = yaml.safe_load(file)

    # initialize all molecule abundances
    if config_model['species_vmr'] == {}:
        for mol in config_model['line_opacities']:
            config_model['species_vmr'][mol] = -99.0

    config_dict['obs_dir'] = Path.home() / Path(config_dict['obs_dir'])

    if config_dict['visit_name'] == []:
        config_dict['visit_name'] = split.split_night(config_dict['obs_dir'], str(config_dict['obs_dir']))
    print('VISIT NAMES: ', config_dict['visit_name'])
    
    # loop over all visits
    for visit_name in config_dict['visit_name']:

        scratch_dir, out_dir, path_fig = red.set_save_location(config_dict['pl_name'], visit_name, config_dict['reduction'], config_dict['instrument'])

        # creating the planet and observation objects
        if visit_name != 'combined':
            planet, obs = red.load_planet(config_dict, visit_name)

        # Start with model with all molecules
        wave_mod, mod_spec, abundances, MMW, VMR = mod.make_model(config_model, planet, out_dir, config_dict)

        if len(config_model['line_opacities']) == 1:
            mol = config_model['line_opacities'][0]
        else: mol = 'all'

        if visit_name == config_dict['visit_name'][0]:
            mod.plot_model_components(config_model, planet, out_dir.parent, path_fig)

# --------------------------------------------------------------------------------------------------------------------------------
        iterables = product(config_dict['mask_tellu'], config_dict['mask_wings'], config_dict['n_pc'])

        # Input n_processes to see how many iterables to run in parallel
        n_processes = 4
        with Pool(n_processes) as pool:
            args = {
                'mol': mol,
                'config_dict': config_dict,
                'planet': planet,
                'obs': obs,
                'scratch_dir': scratch_dir,
                'out_dir': out_dir,
                'path_fig': path_fig,
                'visit_name': visit_name,
                'wave_mod': wave_mod,
                'mod_spec': mod_spec
            }

            # Create a list of all arguments including those from iterables
            all_args = [(arg_tuple + (args,)) for arg_tuple in iterables]


            # Use pool.starmap to pass multiple arguments
            pool.starmap(main_loop_wrapper, all_args)
        
        # include injection step here, need to change the dictionaries everything is saved to

# --------------------------------------------------------------------------------------------------------------------------------

        # plots for each pc
        for mask_tellu in config_dict['mask_tellu']:
            for mask_wings in config_dict['mask_wings']:
                corr.plot_all_ccf(config_dict, mol, mask_tellu, mask_wings, scratch_dir, 
                                visit_name, planet, id_pc0=None, order_indices=np.arange(75), 
                                path_fig = path_fig)

        # plots for each mask_wings 
        for mask_tellu in config_dict['mask_tellu']:
            for n_pc in config_dict['n_pc']:
                # load all ccf map and reductions for (mask_tellu, n_pc) at each mask_wings
                corr.plot_all_maskwings(config_dict, planet, mol, mask_tellu, n_pc, 
                                                            scratch_dir, visit_name, id_pc0=None, 
                                                            order_indices=np.arange(75), path_fig = path_fig)

        # plots for each mask_tellu
        for n_pc in config_dict['n_pc']:
            for mask_wings in config_dict['mask_wings']:
                corr.plot_all_masktellu(config_dict, planet, mol, mask_wings, n_pc, 
                                                            scratch_dir, visit_name, id_pc0=None, 
                                                            order_indices=np.arange(75), path_fig = path_fig)
        

        if ([mask_tellu, mask_wings, n_pc] == config_dict['night_params']) and (visit_name == config_dict['visit_name'][-1]):
            comb_scratch_dir, comb_out_dir, comb_path_fig = red.set_save_location(config_dict['pl_name'], 'combined', config_dict['reduction'], config_dict['instrument'])
            corr.combined_visits_ccf(planet, mol, wave_mod, mod_spec, comb_scratch_dir, comb_path_fig, comb_out_dir, config_dict)

        # iterate over individual molecules if there are more than 1
        if len(config_model['line_opacities']) > 1:
            for single_mol in config_model['line_opacities']:
                
                # Copy input dict
                theta_dict_single = dict(config_model)

                abundances_subtracted = dict(abundances)

                # Need to be a list
                if not isinstance(single_mol, list):
                    single_mol = [single_mol]

                # set all other molecules to zero
                for mol in config_model['line_opacities']:
                    if mol != single_mol[0]:
                        linelist = config_model['linelist_names'][config_model['mode']][mol]
                        abundances_subtracted[linelist] = abundances_subtracted[linelist] * 0 + 1e-99

                theta_dict_single['chemical_equilibrium'] = False
                wave_mod, mod_spec, _, _, _ = mod.make_model(theta_dict_single, planet, out_dir = None, 
                                                abundances = abundances_subtracted,
                                                MMW = MMW, config_dict=config_dict)

# --------------------------------------------------------------------------------------------------------------------------------
                iterables = product(config_dict['mask_tellu'], config_dict['mask_wings'], config_dict['n_pc'])

                with Pool() as pool:
                    args = {
                        'mol': single_mol[0],
                        'config_dict': config_dict,
                        'planet': planet,
                        'obs': obs,
                        'scratch_dir': scratch_dir,
                        'out_dir': out_dir,
                        'path_fig': path_fig,
                        'visit_name': visit_name,
                        'wave_mod': wave_mod,
                        'mod_spec': mod_spec
                    }

                    # Create a list of all arguments including those from iterables
                    all_args = [(arg_tuple + (args,)) for arg_tuple in iterables]

                    # Use pool.starmap to pass multiple arguments
                    pool.starmap(main_loop_wrapper, all_args)
# --------------------------------------------------------------------------------------------------------------------------------

                # plots for each pc
                for mask_tellu in config_dict['mask_tellu']:
                    for mask_wings in config_dict['mask_wings']:
                        corr.plot_all_ccf(config_dict, mol, mask_tellu, mask_wings, scratch_dir, 
                                        visit_name, planet, id_pc0=None, order_indices=np.arange(75), 
                                        path_fig = path_fig)

                # plots for each mask_wings 
                for mask_tellu in config_dict['mask_tellu']:
                    for n_pc in config_dict['n_pc']:
                        # load all ccf map and reductions for (mask_tellu, n_pc) at each mask_wings
                        corr.plot_all_maskwings(config_dict, planet, single_mol, mask_tellu, n_pc, 
                                                                    scratch_dir, visit_name, id_pc0=None, 
                                                                    order_indices=np.arange(75), path_fig = path_fig)

                # plots for each mask_tellu
                for n_pc in config_dict['n_pc']:
                    for mask_wings in config_dict['mask_wings']:
                        corr.plot_all_masktellu(config_dict, planet, single_mol, mask_wings, n_pc, 
                                                                    scratch_dir, visit_name, id_pc0=None, 
                                                                    order_indices=np.arange(75), path_fig = path_fig)
                         
                if ([mask_tellu, mask_wings, n_pc] == config_dict['night_params']) and (visit_name == config_dict['visit_name'][-1]):
                    comb_scratch_dir, comb_out_dir, comb_path_fig = red.set_save_location(config_dict['pl_name'], 'combined', config_dict['reduction'], config_dict['instrument'])
                    corr.combined_visits_ccf(planet, single_mol, wave_mod, mod_spec, comb_scratch_dir, comb_path_fig, comb_out_dir, config_dict)                            
                                            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the pipeline with the given config files.')
    parser.add_argument('config_filepath', type=Path, help='Path to the config.yaml file.')
    parser.add_argument('run_name', type=str, help='Name of reduction.')

    args = parser.parse_args()

    # Call the run_pipe function with the parsed arguments
    run_pipe(args.config_filepath, args.run_name)