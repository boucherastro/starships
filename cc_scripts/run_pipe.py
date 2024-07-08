import numpy as np
import yaml
from sys import path
from pathlib import Path
import argparse

path.append("opereira/starships/cc_scripts/")

import cc_scripts.reduction as red
import cc_scripts.make_model as mod
import cc_scripts.correlations as corr
import cc_scripts.split_nights as split

## running the pipeline
def run_pipe(config_filepath, model_filepath, run_name):

    # unpack input parameters into config dictionary
    with open(config_filepath, 'r') as file:
        config_dict = yaml.safe_load(file)

    config_dict['reduction'] = run_name

    # unpack the model input parameters
    with open(model_filepath, 'r') as file:
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
        
        if visit_name == config_dict['visit_name'][0]:
            print( "Output directory:", out_dir.parent)
            out_dir_gen = out_dir.parent

        # creating the planet and observation objects
        if visit_name != 'combined':
            planet, obs = red.load_planet(config_dict, visit_name)

        # Start with model with all molecules
        wave_mod, mod_spec, abundances, MMW, VMR = mod.make_model(config_model, planet, out_dir, config_dict)

        if len(config_model['line_opacities']) == 1:
            mol = config_model['line_opacities'][0]
        else: mol = 'all'

        
        # creating the dictionaries to store all reductions
        all_reductions = {}
        all_ccf_map = {}
        all_logl_map = {}

        # performing the reduction for each n_pc, mask_tellu, mask_wings
        for mask_tellu in config_dict['mask_tellu']:
            for mask_wings in config_dict['mask_wings']:
                for n_pc in config_dict['n_pc']:

                    naming_kwargs = {'molecule': mol, 'n_pc': n_pc, 'mask_tellu': mask_tellu, 'mask_wings': mask_wings}
                    print('CURRENT PARAMETERS:', naming_kwargs)

                    # reducing the data
                    reduc = red.reduce_data(config_dict, planet, obs, scratch_dir, out_dir, path_fig, n_pc, mask_tellu, mask_wings, visit_name)
                    
                    # check if reduction was already performed
                    # if reduc != None:
                    #     all_reductions[(n_pc, mask_tellu, mask_wings)] = reduc

                    # performing correlations
                    ccf_map, logl_map = corr.perform_ccf(config_dict, reduc, mol, wave_mod, mod_spec, n_pc, mask_tellu, mask_wings, scratch_dir, path_fig, visit_name)

                    # include injection step here, need to change the dictionaries everything is saved to

                # plot all ccf maps for each n_pc
                # load all ccf maps and reductions for (mask_tellu, mask_wings) at each n_pc

                ccf_obj, logl_obj = corr.plot_all_ccf(config_dict, mol, mask_tellu, mask_wings, scratch_dir, 
                                                      visit_name, planet, id_pc0=None, order_indices=np.arange(75), 
                                                      path_fig = path_fig)
                
            # plots for each mask_wings 
            for n_pc in config_dict['n_pc']:
                # load all ccf map and reductions for (mask_tellu, n_pc) at each mask_wings
                ccf_obj, logl_obj = corr.plot_all_maskwings(config_dict, planet, mol, mask_tellu, n_pc, 
                                                            scratch_dir, visit_name, id_pc0=None, 
                                                            order_indices=np.arange(75), path_fig = path_fig)

        # plots for each mask_tellu
        for n_pc in config_dict['n_pc']:
            for mask_wings in config_dict['mask_wings']:
                ccf_obj, logl_obj = corr.plot_all_masktellu(config_dict, planet, mol, mask_wings, n_pc, 
                                                            scratch_dir, visit_name, id_pc0=None, 
                                                            order_indices=np.arange(75), path_fig = path_fig)
        
        if [mask_tellu, mask_wings, n_pc] == config_dict['night_params']:
            corr.combined_visits_ccf(planet, mol, wave_mod, mod_spec, scratch_dir, path_fig, out_dir, config_dict)
        
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
                
                # creating the dictionaries to store all reductions
                all_reductions = {}
                all_ccf_map = {}
                all_logl_map = {}

                # performing the reduction for each n_pc, mask_tellu, mask_wings
                for mask_tellu in config_dict['mask_tellu']:
                    for mask_wings in config_dict['mask_wings']:
                        for n_pc in config_dict['n_pc']:

                            naming_kwargs = {'molecule': single_mol, 'n_pc': n_pc, 'mask_tellu': mask_tellu, 'mask_wings': mask_wings}
                            print('CURRENT PARAMETERS:', naming_kwargs)

                            # reducing the data
                            reduc = red.reduce_data(config_dict, planet, obs, scratch_dir, out_dir, path_fig, n_pc, mask_tellu, mask_wings, visit_name)
                            
                            # check if reduction was already performed
                            if reduc != None:
                                all_reductions[(n_pc, mask_tellu, mask_wings)] = reduc

                            # performing correlations
                            ccf_map, logl_map = corr.perform_ccf(config_dict, all_reductions[(n_pc, mask_tellu, mask_wings)], single_mol, wave_mod, mod_spec, n_pc, mask_tellu, mask_wings, scratch_dir, path_fig, visit_name)
                            all_ccf_map[(n_pc, mask_tellu, mask_wings)] = ccf_map
                            all_logl_map[(n_pc, mask_tellu, mask_wings)] = logl_map

                        # # plot all ccf maps for each n_pc
                        ccf_obj, logl_obj = corr.plot_all_ccf(all_ccf_map, all_logl_map, all_reductions, 
                                                            config_dict, single_mol, mask_tellu, mask_wings, 
                                                            id_pc0=None, order_indices=np.arange(75), 
                                                            path_fig = path_fig)
                        
                if [mask_tellu, mask_wings, n_pc] == config_dict['night_params']:
                    corr.combined_visits_ccf(planet, single_mol, wave_mod, mod_spec, scratch_dir, path_fig, out_dir, config_dict)                                
                                            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the pipeline with the given config files.')
    parser.add_argument('config_filepath', type=Path, help='Path to the config.yaml file.')
    parser.add_argument('model_filepath', type=Path, help='Path to the model_config.yaml file.')
    parser.add_argument('run_name', type=str, help='Name of reduction.')

    args = parser.parse_args()

    # Call the run_pipe function with the parsed arguments
    run_pipe(args.config_filepath, args.model_filepath, args.run_name)