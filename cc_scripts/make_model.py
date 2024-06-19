import matplotlib.pyplot as plt
import numpy as np

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Set the log level of 'fonttools' to WARNING
logging.getLogger('fontTools').setLevel(logging.WARNING)

logging.basicConfig()

from starships.mask_tools import interp1d_masked
interp1d_masked.iprint = False
from starships.analysis import resamp_model
import starships.petitradtrans_utils as prt
from starships.instruments import load_instrum

import astropy.units as u
import astropy.constants as const

from scipy.interpolate import interp1d
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)


# from petitRADTRANS import nat_cst as nc
try:
    from petitRADTRANS.physics import guillot_global, guillot_modif
except ModuleNotFoundError:
    from petitRADTRANS.nat_cst import guillot_global, guillot_modif

atmo_high = None
atmo_low = None

def init_model_retrieval(config_model, mol_species=None, kind_res='high', lbl_opacity_sampling=None,
                         wl_range=None, continuum_species=None, pressures=None, **kwargs):
    """
    Initialize some objects needed for modelization: atmo, species, star_fct, pressures
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
        mol_species = config_model['line_opacities']

    if lbl_opacity_sampling is None:
        lbl_opacity_sampling = config_model['opacity_sampling']

    if continuum_species is None:
        continuum_species = config_model['continuum_opacities']

    if pressures is None:
        limP = config_model['limP']
        n_pts = config_model['n_pts']
        pressures = np.logspace(*limP, n_pts)

    species = prt.select_mol_list(mol_species, kind_res=kind_res, **kwargs)
    species_2_lnlst = {mol: lnlst for mol, lnlst in zip(mol_species, species)}


    Raf, wl_range = config_model['Raf'], config_model['wl_range']
    # need to make this independent of instrument
    if kind_res == 'high':
        mode = 'lbl'

        # check if Raf specified, else take from instrument
        if Raf is None:
            Raf = load_instrum(config_model['instrument'])['resol']
        pix_per_elem = 2
        if wl_range is None:
            wl_range = load_instrum(config_model['instrument'])['high_res_wv_lim']

    elif kind_res == 'low':
        mode = 'c-k'
        Raf = 1000
        pix_per_elem = 1
        if wl_range is None:
            wl_range = config_model['intrument']['low_res_wv_lim']
    else:
        raise ValueError(f'`kind_res` = {kind_res} not valid. Choose between high or low')


    atmo, _ = prt.gen_atm_all([*species.keys()], pressures, mode=mode,
                                      lbl_opacity_sampling=lbl_opacity_sampling, wl_range=wl_range,
                                      continuum_opacities=continuum_species)

    # --- downgrading the star spectrum to the wanted resolution
    if config_model['kind_trans'] == 'emission' and config_model['star_wv'] is not None:
        resamp_star = np.ma.masked_invalid(
            resamp_model(config_model['star_wv'][(config_model['star_wv'] >= wl_range[0] - 0.1) & (config_model['star_wv'] <= wl_range[1] + 0.1)],
                         config_model['star_flux'][(config_model['star_wv'] >= wl_range[0] - 0.1) & (config_model['star_wv'] <= wl_range[1] + 0.1)], 500000, Raf=Raf,
                         pix_per_elem=pix_per_elem))
        fct_star = interp1d(config_model['star_wv'][(config_model['star_wv'] >= wl_range[0] - 0.1) & (config_model['star_wv'] <= wl_range[1] + 0.1)],
                                     resamp_star)
    else:
        fct_star = None

    return atmo, species_2_lnlst, fct_star


def prepare_abundances(config_model, mode=None, ref_linelists=None):
    """Use the correct linelist name associated to the species."""

    if ref_linelists is None:
        if mode is None:
            ref_linelists = config_model['line_opacities'].copy()
        else:
            ref_linelists = [config_model['linelist_names'][mode][mol] for mol in config_model['line_opacities']]

    theta_dict = {}
    if config_model['chemical_equilibrium']:
        for mol in config_model['line_opacities']:
            theta_dict[mol] = 10 ** (-99.0) # doing this will change the model depending on whether you use a standard linelist or input your own!
    else:
        for mol in config_model['line_opacities']:
            theta_dict[mol] = 10 ** config_model['species_vmr'][mol]

    # add option where if not chemical equilibrium, takes the inputted abundances from the YAML file
    # --- Prepare the abundances (with the correct linelist name for species)
    species = {lnlst: theta_dict[mol] for lnlst, mol
               in zip(ref_linelists, config_model['line_opacities'])}
    
    # --- Adding continuum opacities
    for mol in config_model['continuum_opacities']:
        species[mol] = theta_dict[mol]
        
    # --- Adding other species
    for mol in config_model['other_species']:
        species[mol] = theta_dict[mol]

    return species

def create_internal_dict(config_dict, planet):
    ''' for internally created variables to be used in other functions'''
    int_dict = {}

    limP = config_dict['limP']
    n_pts = config_dict['n_pts']
    int_dict['pressures'] = np.logspace(*limP, n_pts)

    # need to 
    int_dict['temperatures'] = config_dict['T_eq']* np.ones_like(int_dict['pressures'])

    int_dict['P0'] = config_dict['P0']
    int_dict['p_cloud'] = config_dict['p_cloud']
    int_dict['R_pl'] = planet.R_pl[0].to(u.R_jup).cgs.value
    int_dict['R_star'] = planet.R_star.to(u.R_sun).cgs.value
    int_dict['gravity'] = (const.G * planet.M_pl / (planet.R_pl) ** 2).cgs.value

    return int_dict

def prepare_model_high_or_low(config_model, int_dict, planet, atmo_obj=None, fct_star=None,
                              species_dict=None, Raf=None, rot_ker=None, out_dir = None,
                              abundances = None, MMW = None):

    mode = config_model['mode']
    # if Raf is None:
    #     Raf = load_instrum(config_model['instrument'])['resol']
    
    if atmo_obj is None:
        # Use atmo object in globals parameters if it exists
        atmo_obj = globals()['atmo_high'] if mode == 'high' else globals()['atmo_low']
        # Initiate if not done yet
        if atmo_obj is None:
            log.info(f'Model not initialized for mode = {mode}. Starting initialization...')
            output = init_model_retrieval(config_model, kind_res=mode)
            log.info('Saving values in `linelist_names`.')
            atmo_obj, lnlst_names, config_model['fct_star_global'][mode] = output
            # Update the values of the global variables
            # Need to use globals() otherwise an error is raised.
            if mode == 'high':
                globals()['atmo_high'] = atmo_obj
            else:
                globals()['atmo_low'] = atmo_obj
                
            # Update the line list names
            if config_model['linelist_names'][mode] is None:
                config_model['linelist_names'][mode] = lnlst_names
            else:
                # Keep the predefined values and complete with the new ones
                config_model['linelist_names'][mode] = {**lnlst_names, **config_model['linelist_names'][mode]}

    if fct_star is None:
        fct_star = config_model['fct_star_global'][mode]

    # --- Prepare the abundances (with the correct name for species)
    # Note that if species is None (not specified), `linelist_names[mode]` will be used inside `prepare_abundances`.
    species = prepare_abundances(config_model, mode, species_dict)

    # --- Generating the model
    args = [int_dict[key] for key in ['pressures', 'temperatures', 'gravity', 'P0', 'p_cloud', 'R_pl', 'R_star']]
    kwargs = dict(gamma_scat=config_model['gamma_scat'],
                  kappa_factor=config_model['scat_factor'],
                  C_to_O=config_model['C/O'],
                    Fe_to_H=config_model['Fe/H'],
                    specie_2_lnlst=config_model['linelist_names'][mode],
                    kind_trans=config_model['kind_trans'],
                    dissociation=config_model['dissociation'],
                    fct_star=fct_star)
    
    if config_model['chemical_equilibrium'] == True:
        wv_out, model_out, abundances, MMW, VMR = prt.retrieval_model_plain(atmo_obj, species, planet, save_abundances = True, *args, **kwargs)

    else: wv_out, model_out = prt.retrieval_model_plain(atmo_obj, species, planet, abundances = abundances, MMW = MMW, *args, **kwargs)

    # saving wv_out, model_out
    if out_dir != None:
        # Generate the filename
        species_keys = '_'.join(species.keys())
        filename = str(out_dir) + f'/model_native_{species_keys}.npz'

        # Save wv_out and model_out into the npz file
        np.savez(filename, wave_mod=wv_out, mod_spec=model_out)

    # move this function into the cross correlation step, so the model native resol -> instrument resol
    if config_model['instrument'] != None and mode == 'high':
        # --- Downgrading and broadening the model (if broadening is included)
        if np.isfinite(model_out[100:-100]).all():
            # Get wind broadening parameters
            if config_model['wind'] is not None:
                rot_kwargs = {'rot_params': [config_model['R_pl'] * const.R_jup,
                                             config_model['M_pl'],
                                             config_model['T_eq'] * u.K,
                                             [config_model['wind']]],
                                'gauss': True, 'x0': 0,
                                'fwhm': config_model['wind'] * 1e3, }
            else:
                rot_kwargs = {'rot_params': None}
            
            lbl_res = 1e6 / config_model['opacity_sampling']

            if Raf is None:
                Raf = load_instrum(config_model['instrument'])['resol']

            # Downgrade the model
            wv_out, model_out = prt.prepare_model(wv_out, model_out, lbl_res, Raf=Raf,
                                                  rot_ker=rot_ker, **rot_kwargs)
            
            # saving new downgraded model
            if out_dir != None:
                # Generate the filename
                species_keys = '_'.join(species.keys())
                filename = str(out_dir) + f"/model_{config_model['instrument']}_{species_keys}.npz"

                # Save wv_out and model_out into the npz file
                np.savez(filename, wave_mod=wv_out, mod_spec=model_out)

    # plotting the model
    # if path_fig is not None:
    #     plt.plot(wv_out, model_out)
    #     plt.xlabel('Wavelength')
    #     plt.ylabel('Flux')
    #     keys_string = ', '.join(species.keys())
    #     plt.title(keys_string)
    #     # Generate the filename
    #     species_keys = '_'.join(species.keys())
    #     filename = str(path_fig) + f'model_{species_keys}.pdf'

    #     # Save the figure
    #     plt.savefig(filename)

    if config_model['chemical_equilibrium'] == True:
        return wv_out, model_out, abundances, MMW, VMR
    
    return wv_out, model_out

def add_instrum_model(wv_out, model_out, config_dict, species, Raf = None, rot_ker=None):
   
    if Raf == None:
        Raf = load_instrum(config_dict['instrument'])['resol']
        
    # --- Downgrading and broadening the model (if broadening is included)
    if np.isfinite(model_out[100:-100]).all():
        # Get wind broadening parameters
        if config_dict['wind'] is not None:
            rot_kwargs = {'rot_params': [config_dict['R_pl'] * const.R_jup,
                                            config_dict['M_pl'],
                                            config_dict['T_eq'] * u.K,
                                            [config_dict['wind']]],
                            'gauss': True, 'x0': 0,
                            'fwhm': config_dict['wind'] * 1e3, }
        else:
            rot_kwargs = {'rot_params': None}
        
        lbl_res = 1e6 / config_dict['opacity_sampling']

        # Downgrade the model
        wave_mod, mod_spec = prt.prepare_model(wv_out, model_out, lbl_res, Raf=Raf,
                                                rot_ker=rot_ker, **rot_kwargs)

    # np.savez(f'model_{config_dict['instrument']}_{'_'.join(species.keys())}.npz', wave_mod=wave_mod, mod_spec=mod_spec)
    return wave_mod, mod_spec 

def make_model(config_model, planet, out_dir, config_dict = {}, abundances = None, MMW = None):
    # computing extra parameters needed for model making
    int_dict = create_internal_dict(config_model, planet)

    # create the model, plot and save it 
    if config_model['chemical_equilibrium']:
        wave_mod, mod_spec, abundances, MMW, VMR = prepare_model_high_or_low(config_model, int_dict, planet, out_dir=out_dir)
        if config_model['instrument'] == None:
            wave_mod, mod_spec = add_instrum_model(config_dict, wave_mod, mod_spec)
        return wave_mod, mod_spec, abundances, MMW, VMR

    else: 
        wave_mod, mod_spec = prepare_model_high_or_low(config_model, int_dict, planet, out_dir=out_dir, abundances=abundances, MMW = MMW)
        if config_model['instrument'] == None:
            wave_mod, mod_spec = add_instrum_model(config_dict, wave_mod, mod_spec)
        return wave_mod, mod_spec, None, None, None
    
def plot_model_components(config_model, planet, path_fig = None, config_dict = {}, abundances = None, MMW = None):
    """ Get the contribution by keeping only the molecules in  `list_of_molecules`"""
    
    # Don't modify the input object
    # theta_dict = copy.deepcopy(config_model)
    theta_dict = dict(config_model)

    # Init outputs
    mol_contrib = dict()
    out_spec = dict()

    # Spec with all molecules
    wv, out_spec['All'], abundances, MMW, VMR = make_model(theta_dict, planet, out_dir = None, config_dict=config_dict)

    if config_model['species_vmr'] == {}:
            for mol in config_model['line_opacities']:
                config_model['species_vmr'][mol] = -99.0

    # Spec without any molecule
    # for mol in config_model['line_opacities']:
    #     del theta_dict['line_opacities'][mol] 

    # theta_dict['line_opacities']= []
    # theta_dict['linelist_names']['high'] = None

    # _, out_spec['No Mol'] = mod.make_model(theta_dict, planet, out_dir = None, path_fig = None, config_dict=config_dict)


    # get cloud contribution
    theta_dict = dict(config_model)
    theta_dict['p_cloud'] = 1e99
    wv, spec_no_cloud, _, _, _ = make_model(theta_dict, planet, out_dir = None)
    spec_abs = (out_spec['All'] - spec_no_cloud) 
    out_spec[f'Without clouds'] = spec_no_cloud

    # get haze contribution
    # theta_dict = dict(config_model)
    # theta_dict['gamma_scat'] = 0
    # wv, spec_no_haze, _, _, _ = make_model(theta_dict, planet, out_dir = None, path_fig = None)
    # spec_abs = (out_spec['All'] - spec_no_haze) 
    # out_spec[f'Haze'] = spec_abs

    # Iterate over input molecules, need to add clouds, haze and H2
    for mol_name in config_model['line_opacities']: 
        
        abundances_subtracted = dict(abundances)

        # Need to be a list
        if not isinstance(mol_name, list):
            mol_name = [mol_name]

        # Spec without the targetted molecules contribution
        theta_dict = dict(config_model)

        for mol in config_model['line_opacities']:
            if mol == mol_name[0]:
                linelist = config_model['linelist_names'][config_model['mode']][mol]
                abundances_subtracted[linelist] = abundances_subtracted[linelist] * 0 + 1e-99
                
    #             print(f'Abundances with {mol_name} removed: {abundances_subtracted}')

        theta_dict['chemical_equilibrium'] = False
        wv, spec_no_mol, _, _, _ = make_model(theta_dict, planet, out_dir = None, 
                                        abundances = abundances_subtracted,
                                        MMW = MMW, config_dict=config_dict)

        # Absolute contribution of the molecule in the spectrum
        spec_mol_abs = (out_spec['All'] - spec_no_mol) 

        # Save it in output
        out_spec[f'Without {mol_name[0]}'] = spec_no_mol
        mol_contrib[mol_name[0]] = spec_mol_abs

    
    # plot contributions
    fig = plt.figure(figsize=(7, 10), dpi=200)

    # Determine the number of subplots to create
    n_subplots = max(len(out_spec.items()), len(mol_contrib.items()) + 2)

    # Create the first subplot separately without sharing y-axis
    ax = [fig.add_subplot(n_subplots, 1, 1)]
    ax[0].plot(wv, out_spec['All'], label='All', lw=0.3)
    ax[0].set_title('Full Model')
    ax[0].set_ylabel('Fp/Fstar')
    ax[0].legend()

    # Create the rest of the subplots with sharing y-axis
    for i in range(1, n_subplots):
        ax.append(fig.add_subplot(n_subplots, 1, i+1, sharex=ax[0]))

    # Find the y-limits for the shared y-axis
    y_min = min(spec.min() for spec in mol_contrib.values())
    y_max = max(spec.max() for spec in mol_contrib.values())

    # plot cloud contribution
    ax[1].plot(wv, out_spec['Without clouds'], label = 'Without Clouds', lw = 0.3, alpha = 0.8)
    ax[1].plot(wv, out_spec['All'], label = 'With Clouds', lw = 0.3, alpha = 0.8)
    ax[1].legend()

    # plot haze contribution
    # ax[2].plot(wv, out_spec['Haze'], label = 'Haze', lw = 0.3)
    # ax[2].legend()

    for i, (key, spec) in enumerate(mol_contrib.items(), start=2):
        ax[i].plot(wv, spec, label=key, lw=0.3)
        ax[i].legend(loc='upper left')
        ax[i].set_ylim(y_min, y_max)  # Set the y-limits to be the same for all subplots
        
    ax[1].set_title('Contributions')

    plt.tight_layout()
    if path_fig is not None:
        filename = str(path_fig) + f'model_summary.pdf'
        plt.savefig(filename)