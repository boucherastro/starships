"""
phoenix_models.py — Download and interpolate PHOENIX HiRes stellar spectra.

Provides:
- Utilities to download PHOENIX HiRes FITS files from the Göttingen FTP server.
- PhoenixInterpGrid: a class that builds a RegularGridInterpolator over
  (Teff, log g, metallicity, alpha) at a user-specified resolving power and
  wavelength range. Once built, the interpolator is fast to evaluate.

Usage example::

    from starships.phoenix_models import PhoenixInterpGrid

    grid = PhoenixInterpGrid(
        teff=[7000, 7800],
        logg=[4.0, 4.5],
        metal=0.0,
        alpha=0.0,
        wv_range=[0.95, 2.5],
        resolution=70000,
    )

    # Evaluate at a specific point and wavelength array
    flux = grid(wv_array, teff=7400, logg=4.3, metal=0.0)

Originally developed in SpectraUtils; integrated into starships for packaging.
"""

import os
import shutil
import urllib.request
from contextlib import closing
from itertools import product
from pathlib import Path
from typing import Dict, Union
import logging

import numpy as np
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator, interp1d

from starships.convolution import resample_constant_res, gauss_convolve, get_wv_constant_res

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.basicConfig()

# ---------------------------------------------------------------------------
# Default directory for locally cached PHOENIX files
# ---------------------------------------------------------------------------

try:
    _base_dir = os.environ['SCRATCH']
except KeyError:
    _base_dir = Path.home()

DEFAULT_MODEL_DIR = Path(_base_dir) / "Models/PHOENIX_HiRes/"
MODEL_SUB_DIR = Path('PHOENIX-ACES-AGSS-COND-2011/')
URL_ROOT = "ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/"
WAVE_FILENAME = 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'

# ---------------------------------------------------------------------------
# PHOENIX grid parameter axes
# ---------------------------------------------------------------------------

PHOENIX_GRID = {
    'metal': np.array([-4, -3, -2, -1.5, -1, -0.5, 0.0, 0.5, 1]),
    'alpha': np.arange(-0.2, 1.21, 0.2),
    'teff':  np.append(np.arange(2300, 7000, 100), np.arange(7000, 12001, 200)),
    'logg':  np.arange(0, 6, 0.5),
}

# Resolving power of the native PHOENIX HiRes spectra, by wavelength range (µm)
PHOENIX_RESOLUTION = {
    (0.05, 0.3):  5_000,
    (0.3,  2.5):  500_000,
    (2.5,  5.5):  200_000,
}

# FITS filename templates
FILE_FRAME = 'lte{:05g}-{:1.2f}{:+1.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
FILE_FRAME_ALPHA = 'lte{:05g}-{:1.2f}{:+1.1f}.Alpha={:+1.2f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
METAL_DIR_FRAME = 'Z{:+1.1f}'
METAL_DIR_FRAME_ALPHA = 'Z{:+1.1f}.Alpha={:+1.2f}'


# ===========================================================================
# File download helpers
# ===========================================================================

def _download_ftp_file(url, save_path):
    """Download a file from an FTP URL and save it locally."""
    with closing(urllib.request.urlopen(url)) as r:
        with open(save_path, 'wb') as f:
            shutil.copyfileobj(r, f)


def get_phoenix_filepath(teff=6000, logg=4.5, metal=0.0, alpha=0.0):
    """Return the relative FITS file path for a given set of PHOENIX parameters.

    Parameters
    ----------
    teff : float
        Effective temperature in K.
    logg : float
        Log surface gravity (cgs).
    metal : float
        Metallicity [Fe/H].
    alpha : float
        Alpha-element abundance [α/Fe].

    Returns
    -------
    Path
        Relative path within MODEL_SUB_DIR (use DEFAULT_MODEL_DIR to make absolute).
    """
    # Force a minus sign before metallicity = 0 to match the PHOENIX filename convention
    if metal == 0.0:
        metal = -1.0e-8

    if alpha == 0.0:
        filename = Path(FILE_FRAME.format(teff, logg, metal))
        file_dir = METAL_DIR_FRAME.format(metal)
    elif -3.0 <= metal <= 0.0:
        # Alpha-enhanced models are only available for -3 ≤ [Fe/H] ≤ 0
        filename = Path(FILE_FRAME_ALPHA.format(teff, logg, metal, alpha))
        file_dir = Path(METAL_DIR_FRAME_ALPHA.format(metal, alpha))
    else:
        raise ValueError(
            "Alpha-element enhanced models ([α/Fe] ≠ 0) are only available "
            "for −3.0 ≤ [Fe/H] ≤ 0.0."
        )

    return MODEL_SUB_DIR / file_dir / filename


def _get_local_or_download(relative_path, query=True):
    """Return the local path of a PHOENIX file, downloading it if necessary.

    Parameters
    ----------
    relative_path : Path
        Path relative to DEFAULT_MODEL_DIR.
    query : bool
        If True and the file is missing locally, download it from the FTP server.

    Returns
    -------
    Path
        Absolute local path to the file.
    """
    local_path = DEFAULT_MODEL_DIR / relative_path
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.is_file():
        log.debug(f'Using cached file: {local_path}')
    elif query:
        url = URL_ROOT + str(relative_path)
        log.info(f'Downloading: {url}')
        _download_ftp_file(url, local_path)
        log.info(f'Saved to: {local_path}')
    else:
        raise FileNotFoundError(
            f'File not found: {local_path}\n'
            f'Set query=True to download it automatically.'
        )

    return local_path


def get_phoenix_wv_grid(query=True):
    """Return the local path of the PHOENIX wavelength grid FITS file."""
    return _get_local_or_download(Path(WAVE_FILENAME), query=query)


# ===========================================================================
# Grid parameter selection
# ===========================================================================

def get_phoenix_grid_in_range(teff=6000, logg=4.5, metal=0.0, alpha=0.0,
                               extrapolate=False):
    """Find the PHOENIX grid points that bracket the requested parameter ranges.

    Each parameter can be a scalar (fixed value) or a [min, max] pair. The
    function returns the smallest subset of the PHOENIX grid that covers the
    requested range.

    Parameters
    ----------
    teff, logg, metal, alpha : float or array-like
        Requested parameter value(s). Scalars are treated as fixed; arrays
        specify a range [min, max].
    extrapolate : bool
        If True, clip to the grid edges when the request is outside the grid.
        If False (default), raise ValueError for out-of-range requests.

    Returns
    -------
    dict
        {param_name: np.ndarray of grid values} for each parameter.
    """
    input_dict = dict(teff=teff, logg=logg, metal=metal, alpha=alpha)
    grid_in_range = {}

    for key, value in input_dict.items():
        max_val = np.max(value)
        min_val = np.min(value)

        max_grid = PHOENIX_GRID[key].max()
        min_grid = PHOENIX_GRID[key].min()

        if max_val > max_grid:
            if not extrapolate:
                raise ValueError(
                    f'{key} max ({max_val}) exceeds the PHOENIX grid max ({max_grid}).'
                )
            max_val = max_grid

        if min_val < min_grid:
            if not extrapolate:
                raise ValueError(
                    f'{key} min ({min_val}) is below the PHOENIX grid min ({min_grid}).'
                )
            min_val = min_grid

        max_idx = np.searchsorted(PHOENIX_GRID[key], max_val, side='left')
        min_idx = np.searchsorted(PHOENIX_GRID[key], min_val, side='right') - 1

        grid_in_range[key] = PHOENIX_GRID[key][min_idx : max_idx + 1]

    return grid_in_range


def _get_all_phoenix_files(param_grid: Dict[str, list], query=True):
    """Ensure all PHOENIX FITS files for a parameter grid are available locally.

    Parameters
    ----------
    param_grid : dict
        {param_name: array of values}, as returned by get_phoenix_grid_in_range.
    query : bool
        Whether to download missing files.

    Returns
    -------
    dict
        {tuple_of_params: local_path} mapping.
    """
    param_keys = list(param_grid.keys())
    filepath_dict = {}

    for params in product(*param_grid.values()):
        kwargs = dict(zip(param_keys, params))
        rel_path = get_phoenix_filepath(**kwargs)
        key = tuple(kwargs.items())
        filepath_dict[key] = _get_local_or_download(rel_path, query=query)

    return filepath_dict


# ===========================================================================
# Spectrum conversion to target resolution
# ===========================================================================

def convert_phoenix_at_resolution(wave, flux, resolution, wv_range,
                                   oversampling=2, n_fwhm=7, output_wv_grid=None):
    """Convolve and resample a PHOENIX spectrum to a target resolving power.

    PHOENIX HiRes spectra have different native resolutions in different
    wavelength regions (see PHOENIX_RESOLUTION). This function handles each
    region separately, then assembles the output on a single grid.

    Parameters
    ----------
    wave : np.ndarray
        Native PHOENIX wavelength grid in µm.
    flux : np.ndarray
        Native PHOENIX flux values (erg/s/cm²/cm).
    resolution : float
        Target resolving power of the output spectrum.
    wv_range : list of float
        [wv_min, wv_max] in µm of the desired output.
    oversampling : int
        Oversampling factor for the output grid (output grid step =
        resolution / oversampling). Higher values give smoother interpolation.
    n_fwhm : int
        Number of FWHMs on each side of the Gaussian kernel (for edge padding).
    output_wv_grid : np.ndarray, optional
        If provided, the output is evaluated on this grid instead of being
        built from resolution and oversampling.

    Returns
    -------
    output_wv_grid : np.ndarray
        Output wavelength grid in µm.
    output_flux : np.ndarray
        Convolved and resampled flux.
    """
    # Build the output grid at the target resolution × oversampling
    if output_wv_grid is None:
        grid_sampling = oversampling * resolution
        output_wv_grid = get_wv_constant_res(wv_range=wv_range, resolution=grid_sampling)

    output_flux = np.zeros_like(output_wv_grid)

    # Process each native-resolution region independently to handle the
    # different PHOENIX_RESOLUTION values in each wavelength range
    for phnx_rng, phnx_res in PHOENIX_RESOLUTION.items():

        # Skip regions that don't overlap with the requested wv_range
        if phnx_rng[0] >= wv_range[-1] or phnx_rng[-1] <= wv_range[0]:
            continue

        wv_min = max(phnx_rng[0], wv_range[0])
        wv_max = min(phnx_rng[-1], wv_range[-1])

        # Pad the sub-range to avoid edge effects in the Gaussian convolution
        pad = wv_min / resolution * n_fwhm
        fkwargs = dict(wv_range=[wv_min - pad, wv_max + pad], resolution=phnx_res)
        wv_resampled, flux_resampled = resample_constant_res(wave, flux, **fkwargs)

        wv_conv, flux_conv = gauss_convolve(wv_resampled, flux_resampled, resolution,
                                            n_fwhm=n_fwhm)

        fct_flux = interp1d(wv_conv, flux_conv, kind='cubic')

        # Fill points that fall within both the PHOENIX resolution region and
        # the actual convolved grid. The two ranges can differ by ~1e-5 µm at
        # the edges due to the 'valid' mode trimming of gauss_convolve, so we
        # intersect both conditions to avoid any out-of-bounds interpolation.
        in_range = ((phnx_rng[0] <= output_wv_grid) &
                    (output_wv_grid < phnx_rng[-1]) &
                    (wv_conv[0] <= output_wv_grid) &
                    (output_wv_grid <= wv_conv[-1]))
        output_flux[in_range] = fct_flux(output_wv_grid[in_range])

    return output_wv_grid, output_flux


# ===========================================================================
# Main interpolation function and class
# ===========================================================================

def interp_phoenix_grid(teff=7500, logg=4.5, metal=0.0, alpha=0.0,
                        wv_range=(1, 2.5), resolution=70000, oversampling=2,
                        n_fwhm=7, output_wv_grid=None, method='linear', query=True):
    """Build a RegularGridInterpolator over the PHOENIX parameter space.

    Downloads any missing PHOENIX files, convolves them to `resolution`, and
    assembles a multi-dimensional interpolation grid. Fixed parameters (scalars)
    are not included as interpolation axes.

    Parameters
    ----------
    teff : float or list of float
        Effective temperature in K, or [min, max] range.
    logg : float or list of float
        Log surface gravity, or [min, max] range.
    metal : float or list of float
        Metallicity [Fe/H], or [min, max] range.
    alpha : float or list of float
        Alpha-element abundance, or [min, max] range.
    wv_range : tuple of float
        (wv_min, wv_max) in µm.
    resolution : float
        Target resolving power of the output spectra.
    oversampling : int
        Oversampling factor for the internal wavelength grid.
    n_fwhm : int
        Gaussian kernel half-width in FWHMs.
    output_wv_grid : np.ndarray, optional
        Fixed output wavelength grid. If None, built from resolution and oversampling.
    method : str
        Interpolation method for RegularGridInterpolator ('linear' or 'nearest').
        'linear' is recommended — higher-order methods are much slower.
    query : bool
        Whether to download missing PHOENIX files automatically.

    Returns
    -------
    phoenix_interp : RegularGridInterpolator
        Interpolator. Call as phoenix_interp(points) where points has shape
        (..., n_free_params + 1) with wavelength as the last column.
    free_param_keys : list of str
        Names of the parameters that are interpolation axes (non-fixed ones).
    """
    param_grids = get_phoenix_grid_in_range(teff=teff, logg=logg, metal=metal,
                                             alpha=alpha)

    if not query:
        _get_all_phoenix_files(param_grids, query=False)

    key_list = list(param_grids.keys())
    grids_length = [len(param_grids[key]) for key in key_list]
    param_grid_idx = [np.arange(n) for n in grids_length]

    # Only parameters with more than one grid point become interpolation axes
    free_idx = [i for i, n in enumerate(grids_length) if n > 1]
    free_param_keys = [key_list[i] for i in free_idx]
    out_dims = [grids_length[i] for i in free_idx]

    output_flux_grid = None
    native_wv_grid = None

    for p_idx in product(*param_grid_idx):
        param = {key: param_grids[key][idx] for key, idx in zip(key_list, p_idx)}

        # Load wavelength grid once (same for all PHOENIX models)
        if native_wv_grid is None:
            wave_file = get_phoenix_wv_grid(query=query)
            hdu = fits.open(wave_file)
            native_wv_grid = hdu[0].data / 1e4  # Angstrom → µm
            hdu.close()

        # Load the flux for this parameter combination.
        # param contains scalar values (one grid point), so we resolve the
        # file path directly instead of going through _get_all_phoenix_files,
        # which expects list/array values.
        rel_path = get_phoenix_filepath(**param)
        flux_path = _get_local_or_download(rel_path, query=query)
        hdu = fits.open(flux_path)
        native_flux = hdu[0].data
        hdu.close()

        args = (native_wv_grid, native_flux, resolution, wv_range)
        kwargs = dict(oversampling=oversampling, n_fwhm=n_fwhm,
                      output_wv_grid=output_wv_grid)
        output_wv_grid, flux_conv = convert_phoenix_at_resolution(*args, **kwargs)

        # Initialise the output array on the first iteration
        if output_flux_grid is None:
            out_dims.append(len(output_wv_grid))  # wavelength axis is last
            output_flux_grid = np.zeros(out_dims)

        idx = tuple(p_idx[i] for i in free_idx)
        output_flux_grid[idx] = flux_conv

    # Build the interpolator: axes are the free parameter grids + wavelength
    interp_axes = [param_grids[key] for key in free_param_keys] + [output_wv_grid]
    # bounds_error=False: return NaN for out-of-bounds queries instead of
    # raising ValueError. This can happen e.g. when the optimiser proposes
    # a parameter value that passes the prior check due to floating-point
    # rounding but sits just outside the interpolation grid. NaN propagates
    # to -inf in the log-likelihood, so the optimiser moves away from that
    # region without crashing.
    phoenix_interp = RegularGridInterpolator(interp_axes, output_flux_grid,
                                             method=method, bounds_error=False,
                                             fill_value=np.nan)

    return phoenix_interp, free_param_keys


class PhoenixInterpGrid:
    """Interpolated PHOENIX stellar spectrum grid.

    Builds a multi-dimensional interpolation grid over (Teff, log g,
    metallicity, alpha) at a fixed resolving power and wavelength range.
    Parameters that are given as scalars are treated as fixed (not
    interpolated). Once initialised, the grid can be evaluated very quickly
    at any point within the covered parameter space.

    Parameters
    ----------
    teff : float or list of float
        Effective temperature in K, or [min, max] to cover a range.
    logg : float or list of float
        Log surface gravity (cgs).
    metal : float or list of float
        Metallicity [Fe/H].
    alpha : float or list of float
        Alpha-element abundance [α/Fe].
    wv_range : tuple of float
        (wv_min, wv_max) in µm for the output spectra.
    resolution : float
        Target resolving power R.
    oversampling : int
        Internal grid oversampling factor.
    n_fwhm : int
        Gaussian convolution kernel half-width (in FWHMs).
    output_wv_grid : np.ndarray, optional
        Fixed output wavelength grid. If None, built automatically.
    method : str
        Interpolation method ('linear' recommended).
    query : bool
        If True, download missing PHOENIX files automatically.

    Examples
    --------
    >>> grid = PhoenixInterpGrid(
    ...     teff=[7000, 8000], logg=[4.0, 4.5], metal=0.0, alpha=0.0,
    ...     wv_range=[0.95, 2.5], resolution=70000,
    ... )
    >>> flux = grid(wv_array, teff=7400, logg=4.3, metal=0.0)
    """

    def __init__(self, teff=7500, logg=4.5, metal=0.0, alpha=0.0,
                 wv_range=(1, 2.5), query=True, resolution=70000,
                 oversampling=2, n_fwhm=7, output_wv_grid=None, method='linear'):

        self.fct_interp, self.parameters = interp_phoenix_grid(
            teff=teff, logg=logg, metal=metal, alpha=alpha,
            query=query, wv_range=wv_range, resolution=resolution,
            oversampling=oversampling, n_fwhm=n_fwhm,
            output_wv_grid=output_wv_grid, method=method,
        )
        self.n_parameters = len(self.parameters)

    def __call__(self, wv, **params):
        """Evaluate the interpolated spectrum at a given wavelength array and parameters.

        Parameters
        ----------
        wv : np.ndarray
            Wavelength points at which to evaluate the spectrum (µm).
            Must lie within the wv_range used at construction.
        **params
            Values for each free parameter (those not fixed at construction).
            Example: grid(wv, teff=7400, logg=4.3)

        Returns
        -------
        np.ndarray
            Interpolated flux at each wavelength point.
        """
        required_keys = self.parameters
        try:
            param_values = [params.pop(key) for key in required_keys]
        except KeyError:
            raise ValueError(
                f"All interpolation parameters {required_keys} must be provided as "
                "keyword arguments. Parameters that were fixed at construction time "
                "(single grid value) are not required and are silently ignored."
            )
        # Parameters that were fixed at construction time (single grid value) are
        # not interpolation axes, so they are not in self.parameters. Silently
        # ignore any remaining kwargs rather than raising an error, so the caller
        # can always pass the full theta_dict without worrying about which
        # parameters happen to be fixed.

        # Build the input array: each row is (param_1, ..., param_n, wv_i)
        n_wv = len(wv)
        param_repeat = np.broadcast_to(param_values, (n_wv, self.n_parameters))
        fct_input = np.append(param_repeat, wv[:, None], axis=1)

        return self.fct_interp(fct_input)
