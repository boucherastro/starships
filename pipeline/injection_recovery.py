import numpy as np
import matplotlib.pyplot as plt
import os, yaml
import pipeline.reduction as red
from pathlib import Path
from scipy.interpolate import interp1d
from astropy.io import fits


def plot_scaled_model(wave_mod, mod_spec, mod_spec_scaled, path_fig=None):
    """
    Plot the original model spectrum and the scaled version used for the injection.

    Parameters
    ----------
    wave_mod : array-like
        Wavelength grid (in microns) output by petitRADTRANS for the model.
    mod_spec : array-like
        Original petitRADTRANS model spectrum. Values are transit depths
        in units of (R_pl / R_star)**2. The function converts these
        depths to ppm by multiplying by 1e6.
    mod_spec_scaled : array-like
        Scaled differential model spectrum, computed as
        (mod_spec − (R_pl / R_star)**2) × the scaling factor.
    path_fig : str or Path-like, optional
        Directory in which to save the figure as `scaled_model.pdf`.  
        If None, the figure is not saved.

    Notes
    -----
    The function generates a two-panel figure:
      • Top panel: original petitRADTRANS transit depth spectrum in ppm.  
      • Bottom panel: scaled spectrum plotted as `1 - mod_spec_scaled`.
    """
    
    fig, axs = plt.subplots(2, 1, figsize=(9, 5), dpi=200, sharex=True)
    
    # Top panel: original model
    axs[0].plot(wave_mod, mod_spec * 1e6, label='Original model', linewidth=0.5, color="dodgerblue")
    axs[0].set_ylabel('Transit depth [ppm]', fontsize=12)
    axs[0].legend()

    # Bottom panel: scaled model
    axs[1].plot(wave_mod, 1 - mod_spec_scaled, label='Scaled model', linewidth=0.5, color="dodgerblue")
    axs[1].set_xlabel('Wavelength [μm]', fontsize=12)
    axs[1].legend()

    plt.tight_layout()
    
    if path_fig is not None:
        fig.savefig(str(path_fig) + '/scaled_model.pdf', bbox_inches='tight')
    
    plt.show()


def main(config_dict, p, obs, visit_name, wave_mod, mod_spec, scratch_dir=None, path_fig=None, debug=False):
    """
    Inject a planetary transmission model into time-series spectra for a given visit.

    This function Doppler-shifts and injects a scaled model transmission spectrum into
    each exposure of a night, accounting for the planet radial velocity, barycentric
    motion, and transit light curve. The injected spectra are written to new FITS
    files in a dedicated output directory, preserving the original file structure.

    Parameters
    ----------
    config_dict : dict
        Configuration dictionary containing at least:
        - 'obs_dir' : str or Path
            Directory containing the original reduced spectra.
        - 'instrument' : str
            Instrument name (used for output path construction).
        - 'RV_inj' : float
            Radial velocity offset (km/s) at which to inject the model.
        - 'scaling_factor' : float
            Multiplicative factor applied to the injected model.
    p : object
        Planet object containing planetary and orbital parameters.
    obs : Observations
        Observations object corresponding to the visit to inject.
    visit_name : str
        Identifier of the observing night.
    wave_mod : array-like
        Wavelength grid of the input model spectrum.
    mod_spec : array-like
        Model transmission spectrum expressed as transit depth
        (R_pl / R_star)**2.
    scratch_dir : str or Path, optional
        Directory where injected FITS files are written. If None, a default
        directory in SCRATCH is used.
    path_fig : str or Path, optional
        Directory where diagnostic figures (e.g., scaled model) are saved.
        If None, a 'Figures' subdirectory is created in `scratch_dir`.
    debug : bool, optional
        If True, generate diagnostic plots showing the injection at the
        exposure and order level.

    Notes
    -----
    - Injection is performed in the observer rest frame for each exposure.
    - The model is scaled by the normalized transit light curve to ensure
      injection only during transit.
    - Outside the model wavelength range, no signal is injected.
    """
    
    # Where to save the injected data files
    if scratch_dir == None:
        scratch_dir = Path(os.environ["SCRATCH"])
        scratch_dir /= Path(f'{config_dict["instrument"]}/Reductions/injected_fits/{"".join(p.name.split())}')
        scratch_dir.mkdir(parents=True, exist_ok=True)
    
    if path_fig == None:
        path_fig = scratch_dir / Path('Figures/')
        path_fig.mkdir(parents=True, exist_ok=True)
    
    # Getting the exposures in and out of transit
    obs.calc_sequence(plot=False)
    in_transit = obs.iIn
    out_transit = obs.iOut

    # Getting the lightcurve
    exp_times = obs.t
    lightcurve = obs.alpha

    # Create the window function to scale the model 
    Wc = lightcurve / np.max(lightcurve)

    # Getting velocities for correction after
    obs.norv_sequence(RV=obs.planet.RV_sys.value[0])  # offset the RVs so that they are 0 at mid-transit
    
    vshift = -obs.berv0 + obs.RV_sys + obs.vrp + obs.mid_vrp + config_dict['RV_inj']

    # Scale the inputted model
    mod_spec_scaled = (mod_spec - (p.R_pl / p.R_star)**2) * config_dict['scaling_factor']

    # Save a plot of the scaled model
    plot_scaled_model(wave_mod, mod_spec, mod_spec_scaled, path_fig)

    # Get list of all exposures
    with open(str(config_dict['obs_dir']) + '/' + f'list_tcorr_{visit_name}') as f:
        exp_list = f.readlines()

    # Initialize lists to store data
    wavelengths = []
    counts = []

    for i, exp in enumerate(exp_list):

        # Shift the model wavelength into the observer rest frame for this exposure
        wave_mod_shifted = wave_mod * (1 + vshift[i] / 299792.458)

        # Setup a function to interpolate over the model
        interp_wavelength = interp1d(wave_mod_shifted, mod_spec_scaled, kind='cubic', bounds_error=False, fill_value=0.)
        
        if debug:
            # Plot each exposure
            plt.figure(figsize=(8,3), dpi=200)
        
        # load the exposure
        with fits.open(str(config_dict['obs_dir']) + '/' + exp.strip(), memmap=False) as hdul:

            count = hdul[1].data  # copy of the flux data
            wv = hdul[2].data / 1000  # convert in microns (could also use pl_obs.fits2wavenew to get wv)
            
            new_count = count.copy()  # create a copy to modify

            # Iterate over the orders
            for w in range(len(wv)):

                if debug:
                    # Plot original 
                    if w == 0: plt.plot(wv[w], count[w], label='Original', zorder=1)
                    else: plt.plot(wv[w], count[w], zorder=1)

                # Interpolate the model to the wavelength, and scale by the lightcurve
                mod_interp = 1 - interp_wavelength(wv[w]) * Wc[i]

                # Multiply the count by the model in that range
                new_count[w] = count[w] * mod_interp

                if debug:
                    # Plot new
                    if w == 0: plt.plot(wv[w], new_count[w], label='Injected', color = 'blue', zorder=0)
                    else: plt.plot(wv[w], new_count[w], color = 'blue', zorder=0)
            
            # Assign the modified data back to the HDU
            hdul[1].data = new_count

            # Save to new fits file with same name, in new folder
            hdul.writeto(str(scratch_dir / exp.strip()), overwrite=True)

            # Append wavelengths and counts for this exposure to the lists
            wavelengths.append(wv)
            counts.append(hdul[1].data)

            if debug:
                plt.title(f'Exposure {i + 1}')
                plt.legend()
                # plt.show()

    # Copy e2ds files into scratch so we can use it as a new obs_dir
    # Get list of all exposures
    with open(str(config_dict['obs_dir']) + '/' + f'list_e2ds_{visit_name}') as f:
        e2ds_list = f.readlines()

    for i, e2ds in enumerate(e2ds_list):
        os.system(f'cp {config_dict["obs_dir"]}/{e2ds.strip()} {scratch_dir}')

    # Copy the e2ds and tcorr lists into the scratch
    os.system(f'cp {config_dict["obs_dir"]}/list_tcorr_{visit_name} {scratch_dir}')
    os.system(f'cp {config_dict["obs_dir"]}/list_e2ds_{visit_name} {scratch_dir}')

