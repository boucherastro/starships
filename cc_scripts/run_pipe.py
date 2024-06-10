import numpy as np
import yaml
from sys import path
from pathlib import Path

path.append("opereira/starships/cc_scripts/")

import cc_scripts.reduction as red
import cc_scripts.make_model as mod
import cc_scripts.correlations as corr
import cc_scripts.split_nights as split


## running the pipeline
def run_pipe(config_filepath, model_filepath):
    # unpack input parameters into config dictionary
    with open(config_filepath, 'r') as file:
        config_dict = yaml.safe_load(file)

    # unpack the model input parameters
    with open(model_filepath, 'r') as file:
        config_model = yaml.safe_load(file)

    config_dict['obs_dir'] = Path.home() / Path(config_dict['obs_dir'])
    out_dir, path_fig = red.set_save_location(config_dict['pl_name'], config_dict['reduction'], config_dict['instrument'])

    if config_dict['visit_name'] == []:
        visit_names = split.split_night(config_dict['obs_dir'], path_fig)

    else: visit_names = config_dict['visit_name']

    # loop over all visits
    for visit_name in visit_names:

        # creating the planet and observation objects
        planet, obs = red.load_planet(config_dict, visit_name)

        # iterate over models
        for single_mol in config_model['line_opacities']:
            # Copy input dict
            theta_dict_single = dict(config_model)

            # Set all other linelists to zero (10^-99 to avoid division by zero)
            for mol in config_model['line_opacities']:
                if mol != single_mol:
                    theta_dict_single[mol] = 1e-99

            wave_mod, mod_spec = mod.make_model(theta_dict_single, planet, out_dir, path_fig, config_dict)

            # creating the dictionaries to store all reductions
            all_reductions = {}
            all_ccf_map = {}
            all_logl_map = {}

            # performing the reduction for each n_pc, mask_tellu, mask_wings
            for mask_tellu in config_dict['mask_tellu']:
                for mask_wings in config_dict['mask_wings']:
                    for n_pc in config_dict['n_pc']:

                        naming_kwargs = {'visit_name': visit_name, 'molecule': mol, 'n_pc': n_pc, 'mask_tellu': mask_tellu, 'mask_wings': mask_wings}
                        print(naming_kwargs)
                        # reducing the data
                        planet, obs = red.load_planet(config_dict, visit_name)
                        reduc = red.reduce_data(config_dict, planet, obs, out_dir, path_fig, visit_name, mol, n_pc, mask_tellu, mask_wings)
                        all_reductions[(n_pc, mask_tellu, mask_wings)] = reduc

                        # performing correlations
                        ccf_map, logl_map = corr.perform_ccf(config_dict, reduc['1'], wave_mod, mod_spec, [n_pc], mask_tellu, mask_wings, out_dir)
                        all_ccf_map[(n_pc, mask_tellu, mask_wings)] = ccf_map
                        all_logl_map[(n_pc, mask_tellu, mask_wings)] = logl_map

                    # plot all ccf maps for each n_pc
                    ccf_obj, logl_obj = corr.plot_all_ccf(all_ccf_map, all_logl_map, all_reductions, 
                                                        config_dict, mask_tellu, mask_wings, id_pc0=0, 
                                                        order_indices=np.arange(75), path_fig = path_fig)

                    # plotting ttest maps for each n_pc
                    # for n_pc in config_dict['n_pc']:

                    #     obs = all_reductions[(n_pc, mask_tellu, mask_wings)]['1']

                    #     ccf_obj.ttest_map(obs, kind='logl', vrp=np.zeros_like(obs.vrp), 
                    #                     orders=np.arange(75), kp0=0, RV_limit=config_dict['RV_lim'], 
                    #                     kp_step=config_dict['Kp_step'], rv_step=2, RV=None, 
                    #                     speed_limit=3, icorr=obs.iIn, equal_var=False, path_fig = path_fig)


    #         # repeating for model with all molecules
    #         wave_mod, mod_spec = mod.make_model(config_model, planet, out_dir, path_fig, config_dict)

    #         # creating the dictionaries to store all reductions


    #         # performing the reduction for each n_pc, mask_tellu, mask_wings
    #         for mask_tellu in config_dict['mask_tellu']:
    #             for mask_wings in config_dict['mask_wings']:

    #                 all_reductions = {}
    #                 all_ccf_map = {}
    #                 all_logl_map = {}

    #                 for n_pc in config_dict['n_pc']:

    #                     # reducing the data
    #                     reduc = red.reduce_data(config_dict, n_pc, mask_tellu, mask_wings, planet, obs, out_dir, path_fig)
    #                     all_reductions[n_pc] = reduc
    #                     # performing correlations
    #                     ccf_map, logl_map = corr.perform_ccf(config_dict, reduc['1'], wave_mod, mod_spec, n_pc, mask_tellu, mask_wings, out_dir)
    #                     all_ccf_map[n_pc] = ccf_map
    #                     all_logl_map[n_pc] = logl_map

    #                     # include injection step here, need to change the dictionaries everything is saved to
    #                     # reduc.final
    #                     # performing correlations
    #                     # ccf_map, logl_map = corr.perform_ccf(config_dict, reduc, wave_mod, mod_spec, n_pc, mask_tellu, mask_wings, out_dir)
    #                     # all_ccf_map[n_pc] = ccf_map
    #                     # all_logl_map[n_pc] = logl_map

    #                 # plot all ccf maps for each n_pc
    #                 ccf_obj, logl_obj = corr.plot_all_ccf(all_ccf_map, all_logl_map, all_reductions, 
    #                                                     config_dict, mask_tellu, mask_wings, id_pc0=0, 
    #                                                     order_indices=np.arange(75), path_fig = path_fig)

    #                 # plotting ttest maps for each n_pc
    #                 for n_pc in config_dict['n_pc']:

    #                     obs = all_reductions[n_pc]

    #                     ccf_obj.ttest_map(obs, kind='logl', vrp=np.zeros_like(obs.vrp), 
    #                                     orders=np.arange(75), kp0=0, RV_limit=config_dict['RV_lim'], 
    #                                     kp_step=config_dict['Kp_step'], rv_step=2, RV=None, 
    #                                     speed_limit=3, icorr=obs.iIn, equal_var=False, path_fig = path_fig)

    #     # combine all visits

        