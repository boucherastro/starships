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


def required_margin(wv_min: float, resolution: float, n_fwhm: int = 7) -> float:
    """Wavelength margin (same units as `wv_min`) `degrade_and_resample` needs
    beyond its `sample` range, to have room for a full convolution kernel at
    `resolution` -- the exact formula `degrade_and_resample` uses internally for
    its own padding (see its `Notes`), factored out here so a caller that needs
    to know *in advance* how much margin a future `degrade_and_resample` call
    will require (e.g. how wide to generate a model, or how much padding a
    loaded data file's own wavelength range needs) uses the same number, not an
    independently maintained guess that can silently drift out of sync with it
    (found as a real bug, Chantier A, 2026-09-16: a photometric instrument's
    `wv_range` was padded with an unrelated fixed constant, `pad_n_res_elem`,
    that happened to leave zero margin left over for `prepare_photometry`'s own
    `degrade_and_resample` call downstream -- silent NaN in the synthetic data).

    Parameters
    ----------
    wv_min : float
        Shortest wavelength in the region that will need a margin -- the margin
        scales with wavelength, since resolving power is wavelength / d(wavelength).
    resolution : float
        Target resolving power the eventual `degrade_and_resample` call will use.
    n_fwhm : int, default 7
        Same meaning as `degrade_and_resample`'s own `n_fwhm`.

    Returns
    -------
    float
    """
    return n_fwhm * wv_min / resolution


def degrade_and_resample(wv: np.ndarray, flux: np.ndarray, resolution: float,
                          input_resolution: float, sample: np.ndarray,
                          n_fwhm: int = 7, kind: str = 'cubic') -> np.ndarray:
    """Degrade a spectrum to a target resolving power and evaluate it on a given grid.

    This chains resample_constant_res() and gauss_convolve() -- the same two-step
    pattern already used by phoenix_models.convert_phoenix_at_resolution() to degrade
    PHOENIX stellar spectra -- and adds the interpolation back onto an arbitrary
    output grid needed by the retrieval pipeline's model-preparation functions
    (init_stellar_spectrum, prepare_photometry, prepare_spectrophotometry,
    petitradtrans_utils.prepare_model, retrieval_utils.downgrade_mod).

    The key difference with the older spectrum.py::resampling()/analysis.py::resamp_model()
    engine: gauss_convolve() measures the *actual* sampling resolution of the
    intermediate grid from the array itself (get_res_from_grid()), instead of
    trusting a caller-supplied number as if it were the true input sampling density.
    That conflation (documented in spectrum.py::resampling()'s own docstring) is what
    produced ~30% kernel-width errors whenever the target and input resolutions were
    of the same order of magnitude (e.g. the final degradation to instrument
    resolution) -- this function is the fix.

    Note for future work (Chantier A Phase 2/3, not implemented here): this is the
    natural place to plug in an extra kernel -- a simple (region-less) rotation
    profile, or a box kernel representing the RV smearing accumulated over an
    exposure's integration time -- by convolving `flux_resamp` with it between the
    resample_constant_res() and gauss_convolve() calls below.

    Parameters
    ----------
    wv : np.ndarray
        Input wavelength grid. Does not need to be sampled at constant resolution,
        and does not need to be cropped to `sample`'s range -- a wavelength pad is
        added internally (see Notes) so it is best to pass the widest array
        available, to leave room for that pad.
    flux : np.ndarray
        Input flux values, same shape as `wv`.
    resolution : float
        Target (output) resolving power.
    input_resolution : float
        Native/physical resolving power of the input spectrum. Used to build the
        intermediate constant-resolution grid before convolving -- this is what
        `spectrum.py::resampling()` calls `Rbf`.
    sample : np.ndarray
        Wavelength grid to evaluate the degraded spectrum on.
    n_fwhm : int
        Passed to gauss_convolve(); also used to size the wavelength pad added
        around `sample`'s range before resampling.
    kind : str
        Interpolation kind used both for the initial resampling and for the final
        evaluation on `sample`.

    Returns
    -------
    np.ndarray
        Degraded flux evaluated at `sample`. Points too close to the edge of the
        available input data (not enough margin for a full convolution kernel) come
        back as NaN -- consistent with how the rest of the codebase already handles
        invalid values (np.ma.masked_invalid), rather than the old engine's silent
        zero-padded edge approximation.

    Notes
    -----
    A wavelength pad of `n_fwhm` resolution elements (at `resolution`) is added
    around `sample`'s range before resampling, so that the edge trimming done by
    gauss_convolve()'s default 'valid' mode does not clip any of the requested
    `sample` points -- see `required_margin`, which computes this same pad for
    callers that need to know it in advance (e.g. to size how wide a model or a
    loaded data file's wavelength range needs to be for this function not to run
    out of margin and return NaN near the edges). The pad is clipped to whatever
    is actually available in `wv`.
    """
    wv_min, wv_max = np.min(sample), np.max(sample)
    pad = required_margin(wv_min, resolution, n_fwhm=n_fwhm)
    wv_range = (max(wv_min - pad, np.min(wv)), min(wv_max + pad, np.max(wv)))

    cond = (wv >= wv_range[0]) & (wv <= wv_range[1])
    # get_wv_constant_res() (used internally by resample_constant_res()) can return
    # a grid extending up to one resolution element beyond `wv_range` by
    # construction -- pass bounds_error=False here so that harmless overshoot
    # (trimmed away below by gauss_convolve()'s 'valid' mode) does not raise.
    wv_resamp, flux_resamp = resample_constant_res(wv[cond], flux[cond], wv_range=wv_range,
                                                    resolution=input_resolution, kind=kind,
                                                    bounds_error=False, fill_value=np.nan)
    wv_conv, flux_conv = gauss_convolve(wv_resamp, flux_resamp, resolution, n_fwhm=n_fwhm)

    # A single leftover NaN (e.g. the get_wv_constant_res() overshoot above, if not
    # fully trimmed away by gauss_convolve()'s 'valid' mode) would otherwise corrupt
    # the *entire* cubic spline built below, since interp1d(kind='cubic') fits one
    # global spline rather than a local window -- drop non-finite points first.
    finite = np.isfinite(flux_conv)
    fct = interp1d(wv_conv[finite], flux_conv[finite], kind=kind, bounds_error=False, fill_value=np.nan)
    return fct(sample)
