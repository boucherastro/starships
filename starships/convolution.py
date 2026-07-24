"""
convolution.py — Spectral resampling and Gaussian convolution utilities.

Provides helpers to:
- Build wavelength grids at constant resolving power (Δλ/λ = const).
- Resample a spectrum onto such a grid.
- Convolve a spectrum with a Gaussian kernel matching a target resolution.
- Extend a wavelength range by a Doppler velocity pad (for edge-effect avoidance).

Originally developed in SpectraUtils; integrated into starships for packaging.
"""

import numpy as np
from scipy.interpolate import interp1d

from starships.homemade import calc_shift

SIGMA_TO_FWHM = 2 * np.sqrt(2 * np.log(2))


def gaussian(x, x0=0, sig=1, amp=None):
    """Evaluate a Gaussian function.

    Parameters
    ----------
    x : array-like
        Points at which to evaluate the Gaussian.
    x0 : float
        Centre of the Gaussian.
    sig : float
        Standard deviation.
    amp : float, optional
        Amplitude. If None, the Gaussian is normalised to unit integral.

    Returns
    -------
    np.ndarray
    """
    if amp is None:
        amp = 1 / np.sqrt(2 * np.pi * sig ** 2)
    return amp * np.exp(-0.5 * ((x - x0) / sig) ** 2)


def add_dv_pad_to_wv_range(dv_pad, wv_range):
    """Extend a wavelength range by a Doppler velocity pad.

    Useful to avoid edge effects when the spectrum will be shifted by up to
    dv_pad km/s, or when a convolution kernel extends beyond the nominal range.

    Parameters
    ----------
    dv_pad : float
        Velocity pad in km/s (one-sided, applied symmetrically).
    wv_range : list or tuple of float
        [wv_min, wv_max] in µm (or any consistent unit).

    Returns
    -------
    list of float
        Extended [wv_min, wv_max].
    """
    wv_min, wv_max = wv_range
    # calc_shift(v) returns the relativistic Doppler factor for velocity v in km/s.
    # Dividing wv_min by the factor extends the blue end;
    # multiplying wv_max extends the red end.
    wv_shift = calc_shift(dv_pad)
    wv_min /= wv_shift
    wv_max *= wv_shift
    return [wv_min, wv_max]


class SamplingError(ValueError):
    """Raised when a wavelength grid is not sampled at constant resolution."""
    pass


def get_res_from_grid(grid, res_rtol=1e-6):
    """Estimate the resolving power of a constant-resolution wavelength grid.

    Parameters
    ----------
    grid : np.ndarray
        Wavelength grid assumed to be sampled at constant Δλ/λ.
    res_rtol : float
        Relative tolerance used to verify that the sampling is truly constant.

    Returns
    -------
    float
        Median resolving power R = λ/Δλ.

    Raises
    ------
    SamplingError
        If the sampling resolution is not constant within res_rtol.
    """
    res_grid = grid[:-1] / np.diff(grid)
    res_in = np.median(res_grid)

    if (np.abs(res_grid - res_in) / res_in > res_rtol).any():
        raise SamplingError('Wavelength sampling resolution is not constant.')

    return res_in


def gauss_convolve(wv, spec, resolution, n_fwhm=7, res_rtol=1e-6, mode='valid'):
    """Convolve a spectrum with a Gaussian kernel to match a target resolution.

    The input spectrum must be sampled at constant resolving power (Δλ/λ = const),
    as produced by get_wv_constant_res() or resample_constant_res().

    Parameters
    ----------
    wv : np.ndarray
        Wavelength grid (constant Δλ/λ).
    spec : np.ndarray
        Flux values at each wavelength point.
    resolution : float
        Target resolving power R = λ/Δλ of the output spectrum.
    n_fwhm : int
        Half-width of the convolution kernel in units of the FWHM. 7 is a
        safe default that keeps truncation errors well below 1 ppm.
    res_rtol : float
        Tolerance passed to get_res_from_grid() to verify constant sampling.
    mode : str
        Convolution mode: 'valid' (output shorter, no edge effects) or
        'same' (output same length, with edge effects at the boundaries).

    Returns
    -------
    wv_out : np.ndarray
        Output wavelength grid (trimmed if mode='valid').
    spec_out : np.ndarray
        Convolved spectrum.
    """
    # Verify and retrieve the input sampling resolution
    res_in = get_res_from_grid(wv, res_rtol=res_rtol)

    # Build the Gaussian kernel on a grid with the same step as the input
    res_elem_grid = 1.0 / res_in        # wavelength step (in fractional units)
    fwhm = 1.0 / resolution             # FWHM of the target resolution element
    half_width = n_fwhm * fwhm / 2.0
    kernel_grid = np.arange(0, half_width + 0.1 * res_elem_grid, res_elem_grid)
    kernel_grid = np.concatenate([-kernel_grid[-1:0:-1], kernel_grid])

    sigma = fwhm / SIGMA_TO_FWHM
    gauss_kernel = gaussian(kernel_grid, sig=sigma)
    gauss_kernel /= gauss_kernel.sum()  # normalise to conserve flux

    spec_out = np.convolve(spec, gauss_kernel, mode=mode)

    # Trim the wavelength grid to match the (potentially shorter) output
    if mode == 'same':
        wv_out = wv.copy()
    elif mode == 'valid':
        ker_h_len = (gauss_kernel.size - 1) // 2
        wv_out = wv[ker_h_len:-ker_h_len].copy()
    else:
        raise ValueError(f"Convolution mode '{mode}' is not supported. Use 'valid' or 'same'.")

    return wv_out, spec_out


def get_wv_constant_res(wv=None, wv_range=None, resolution=None):
    """Build a wavelength grid sampled at constant resolving power (Δλ/λ = const).

    The grid is constructed in log-wavelength space, which gives equal
    velocity spacing (Δv = c/R) at every wavelength.

    Parameters
    ----------
    wv : np.ndarray, optional
        Reference wavelength grid. Used to infer wv_range (if not given)
        and/or resolution (if not given).
    wv_range : tuple of float, optional
        (wv_min, wv_max) of the desired output grid. If None, inferred from wv.
    resolution : float, optional
        Target resolving power R. If None, inferred from the input grid wv.

    Returns
    -------
    np.ndarray
        Wavelength grid with constant Δλ/λ = 1/R.
    """
    if wv_range is None:
        if wv is None:
            raise ValueError('`wv` must be specified if `wv_range` is not.')
        wv_range = (np.min(wv), np.max(wv))

    if resolution is None:
        if wv is None:
            raise ValueError('Either `wv` or `resolution` must be specified.')
        dlog_wv = np.min(np.diff(np.log(wv)))
    else:
        dlog_wv = np.log(1 + 1.0 / resolution)

    log_range = np.log(wv_range)
    log_wv = np.arange(log_range[0], log_range[-1] + dlog_wv, dlog_wv)

    return np.exp(log_wv)


def resample_constant_res(wv, flux, wv_range=None, resolution=None, kind='cubic', **kwargs):
    """Resample a spectrum onto a constant-resolution wavelength grid.

    Parameters
    ----------
    wv : np.ndarray
        Input wavelength grid.
    flux : np.ndarray
        Input flux values.
    wv_range : tuple of float, optional
        (wv_min, wv_max) of the output grid. Defaults to the input range.
    resolution : float, optional
        Target resolving power of the output grid. If None, inferred from wv.
    kind : str
        Interpolation kind passed to scipy.interpolate.interp1d.
    **kwargs
        Additional keyword arguments forwarded to interp1d.

    Returns
    -------
    wv_resampled : np.ndarray
    flux_resampled : np.ndarray
    """
    flux_spl = interp1d(wv, flux, kind=kind, **kwargs)
    wv_resampled = get_wv_constant_res(wv, wv_range=wv_range, resolution=resolution)
    flux_resampled = flux_spl(wv_resampled)
    return wv_resampled, flux_resampled
