"""
stellar_fit.py — Fit a stellar spectrum to high-resolution SPIRou data.

This module fits a PHOENIX model stellar spectrum to a master out-of-transit
spectrum computed by starships. The fit solves for global stellar parameters
(Teff, logg, metallicity, vsini, limb-darkening coefficient, systemic RV)
while handling the per-order continuum correction analytically.

The key idea is the **profile log-likelihood**: for a given set of global
parameters, the polynomial continuum correction for each spectral order can
be solved analytically (linear least squares in log flux), without including
the polynomial coefficients as free parameters in the sampler. This keeps
the problem in a manageable low-dimensional space.

Mathematical formulation (one order):
    log(data) = log(model) + Φ c + ε
    where Φ is the Chebyshev basis matrix and c are the polynomial coefficients.
    For fixed model, c* = (Φᵀ W Φ)⁻¹ Φᵀ W r  with  r = log(data) - log(model).
    The profile chi² at the optimal c* is:  χ²_profile = rᵀWr - bᵀc*
    where  b = Φᵀ W r.

Workflow:
    1. Edit the YAML config file (see stellar_fit_inputs_example.yaml).
    2. Run the fit:
       - Point estimate:  stellar_fit.main("my_config.yaml")
       - Or interactively in Python:
             import starships.stellar_fit as sf
             sf.setup_stellar_fit("my_config.yaml")
             result, theta_best, theta_dict = sf.run_minimize()
             dynesty_results = sf.run_dynesty()
    3. Import in a notebook for post-processing:
             import starships.stellar_fit as sf
             sf.setup_stellar_fit("my_config.yaml")
             models = sf.make_model_with_best_poly(theta_dict)
    4. Save the best-fit stellar spectrum for use as `star_spectrum` in a
       starships.retrieval config:
             sf.save_stellar_spectrum(theta_dict, "star_spectrum.npz")

Dependencies:
    - starships.phoenix_models.PhoenixInterpGrid
    - PyAstronomy.pyasl                 (pyasl.fastRotBroad)
    - dynesty                           (optional, for posterior sampling)

Important note on order of convolutions:
    This code applies rotational broadening on the PHOENIX spectrum that has
    already been convolved to `phoenix_resolution`. This is an approximation
    that is accurate when vsini >> instrument_FWHM (i.e., vsini >> c/R ≈ 4 km/s
    for SPIRou). For slowly-rotating stars (vsini < ~10 km/s), set
    phoenix_resolution much higher than SPIRou's resolution (e.g., 300 000)
    to avoid systematic errors.
"""

import os
import logging
import warnings
import pickle
from pathlib import Path

import numpy as np
import yaml
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize

from starships.homemade import calc_shift
from starships.convolution import add_dv_pad_to_wv_range, get_wv_constant_res
from starships import retrieval_utils as ru

# ---------------------------------------------------------------------------
# Optional imports — PyAstronomy must be installed; phoenix_models is bundled
# ---------------------------------------------------------------------------

try:
    from starships.phoenix_models import PhoenixInterpGrid
    _PHOENIX_AVAILABLE = True
except ImportError:
    _PHOENIX_AVAILABLE = False

try:
    from PyAstronomy import pyasl
    _PYASL_AVAILABLE = True
except ImportError:
    _PYASL_AVAILABLE = False

try:
    import dynesty
    _DYNESTY_AVAILABLE = True
except ImportError:
    _DYNESTY_AVAILABLE = False


log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

# ===========================================================================
# Global variables
# ---------------------------------------------------------------------------
# All globals are set by setup_stellar_fit(). Declaring them here documents
# what they are and makes them accessible from notebooks after import.
# ===========================================================================

# Data (loaded from the reduction .npz file)
global ref_wave        # (n_orders, n_pixels) wavelength grid in µm
global ref_spectrum    # (n_orders, n_pixels) master out-of-transit flux
global ref_mask        # (n_orders, n_pixels) True = bad pixel
global ref_uncert      # (n_orders, n_pixels) flux uncertainty

# Model infrastructure
global phoenix_interp    # PhoenixInterpGrid object for PHOENIX interpolation
global _wave_rot_grids   # list of per-order wavelength grids for pyasl.fastRotBroad
global _cheb_bases       # list of pre-computed Chebyshev basis matrices (one per order)

# Config parameters (populated from the YAML)
global star_name
global n_poly            # number of Chebyshev polynomial terms per order
global rot_broad_samp    # resolving power used for the rotation broadening grid
global phoenix_resolution, phoenix_oversampling, phoenix_n_fwhm, phoenix_method
global dv_pad            # maximum Doppler padding in km/s (vsini_max + v_shift_max)
global params_prior      # dict: {param_name: [prior_type, arg1, arg2]}
global fixed_params      # dict: non-fitted parameters (e.g., alpha)
global special_init      # dict: optional starting point overrides for minimisation
global prior_func_dict   # dict: prior_type → log-prior function
global prior_init_func_dict  # dict: prior_type → walker-init function

# Output paths
global run_name, base_dir
global walker_path, walker_file_out
global n_live_points

# Controls dynesty's progress bar. Set to True/False to override auto-detection.
# When None (default), the bar is shown only if stdout is a terminal (isatty).
# In notebooks, set:  sf.verbose_dynesty = True  before calling run_dynesty().
verbose_dynesty = None

# Precomputed inverse-CDF tables for combined_split_gaussian priors
# (populated by _build_prior_icdf_tables during setup)
global _prior_icdf_tables


# ===========================================================================
# Prior utility helpers
# ---------------------------------------------------------------------------
# These functions know about all supported prior types and return the
# effective range / mode / inverse CDF used in several places (bounds,
# starting-point sampling, PHOENIX grid setup, and dynesty prior_transform).
# ===========================================================================

def _get_prior_effective_range(prior_info):
    """Return the (lo, hi) parameter range implied by a prior specification.

    For bounded priors (uniform), this is the literal range. For Gaussian-type
    priors, we use ±5σ around the mode as a practical bound that contains
    virtually all of the probability mass.

    Parameters
    ----------
    prior_info : list
        Prior specification as stored in params_prior: [type, arg1, arg2, ...].

    Returns
    -------
    lo, hi : float or None
        Effective lower and upper bounds, or (None, None) for unknown types.
    """
    prior_type = prior_info[0]
    args = prior_info[1:]

    if prior_type in ('uniform', 'log_uniform'):
        return float(args[0]), float(args[1])

    elif prior_type == 'gaussian':
        mu, sigma = float(args[0]), float(args[1])
        return mu - 5.0 * sigma, mu + 5.0 * sigma

    elif prior_type == 'split_gaussian':
        # Format: [mu, sigma_lo, sigma_hi]
        mu, sigma_lo, sigma_hi = float(args[0]), float(args[1]), float(args[2])
        return mu - 5.0 * sigma_lo, mu + 5.0 * sigma_hi

    elif prior_type == 'combined_split_gaussian':
        # Format: [mu1, s_lo1, s_hi1, mu2, s_lo2, s_hi2, ...]
        n_refs = len(args) // 3
        lo = min(float(args[3*i]) - 5.0 * float(args[3*i+1]) for i in range(n_refs))
        hi = max(float(args[3*i]) + 5.0 * float(args[3*i+2]) for i in range(n_refs))
        return lo, hi

    return None, None


def _get_prior_mode(prior_info):
    """Return the mode (most probable value) of a prior distribution.

    For uniform priors the midpoint is returned as a neutral default; for all
    Gaussian-type priors the mode is the mean/mode parameter mu.

    Parameters
    ----------
    prior_info : list
        Prior specification as stored in params_prior.

    Returns
    -------
    float
    """
    prior_type = prior_info[0]
    args = prior_info[1:]

    if prior_type in ('uniform', 'log_uniform'):
        lo, hi = float(args[0]), float(args[1])
        return (lo + hi) / 2.0

    elif prior_type in ('gaussian', 'split_gaussian'):
        # The mode is the first argument (mu) for both of these
        return float(args[0])

    elif prior_type == 'combined_split_gaussian':
        # The mode of a product of split-Gaussians is found numerically
        return _find_combined_split_gaussian_mode(args)

    # Fallback: first numeric arg (covers unknown types gracefully)
    return float(args[0])


def _find_combined_split_gaussian_mode(args):
    """Find the mode of a product of split-Gaussian PDFs numerically.

    The combined distribution is piecewise-quadratic in log space, so the
    mode is found efficiently with a bounded 1-D minimisation.

    Parameters
    ----------
    args : tuple
        Flat triplets (mu1, s_lo1, s_hi1, mu2, s_lo2, s_hi2, ...).

    Returns
    -------
    float
    """
    from scipy.optimize import minimize_scalar

    n_refs = len(args) // 3
    lo = min(float(args[3*i]) - 5.0 * float(args[3*i+1]) for i in range(n_refs))
    hi = max(float(args[3*i]) + 5.0 * float(args[3*i+2]) for i in range(n_refs))

    def neg_log_pdf(x):
        total = 0.0
        for i in range(n_refs):
            mu = float(args[3*i])
            sigma_lo = float(args[3*i+1])
            sigma_hi = float(args[3*i+2])
            sigma = sigma_lo if x < mu else sigma_hi
            total += 0.5 * ((x - mu) / sigma) ** 2
        return total

    result = minimize_scalar(neg_log_pdf, bounds=(lo, hi), method='bounded')
    return float(result.x)


def _build_prior_icdf_tables(n_grid=2000):
    """Pre-compute inverse-CDF tables for combined_split_gaussian priors.

    For uniform / gaussian / split_gaussian priors, the inverse CDF is
    analytically invertible and evaluated directly in prior_transform. For
    combined_split_gaussian, the product of multiple split-Gaussians does not
    have a closed-form inverse CDF, so we compute it numerically once here
    and store a fast interpolator.

    This function is called once in setup_stellar_fit() after params_prior is
    set. The resulting tables are stored in the global _prior_icdf_tables dict
    and used inside prior_transform() during dynesty sampling.

    Parameters
    ----------
    n_grid : int
        Number of grid points for the numerical CDF computation. 2000 is
        accurate to better than 0.1σ for typical Gaussian-shaped distributions.
    """
    global _prior_icdf_tables
    from scipy.integrate import cumulative_trapezoid
    from scipy.interpolate import interp1d

    _prior_icdf_tables = {}

    for key, prior_info in params_prior.items():
        if prior_info[0] != 'combined_split_gaussian':
            continue

        args = prior_info[1:]
        lo, hi = _get_prior_effective_range(prior_info)

        # Evaluate the log-PDF on a fine grid (working in log space first
        # avoids numerical underflow when individual Gaussians are narrow)
        x_grid = np.linspace(lo, hi, n_grid)
        n_refs = len(args) // 3
        log_pdf = np.zeros(n_grid)
        for i_ref in range(n_refs):
            mu = float(args[3*i_ref])
            sigma_lo = float(args[3*i_ref+1])
            sigma_hi = float(args[3*i_ref+2])
            sigmas = np.where(x_grid < mu, sigma_lo, sigma_hi)
            log_pdf += -0.5 * ((x_grid - mu) / sigmas) ** 2

        # Normalise to avoid overflow in exp, then compute the CDF
        pdf = np.exp(log_pdf - log_pdf.max())
        cdf = cumulative_trapezoid(pdf, x_grid, initial=0)
        cdf /= cdf[-1]   # rescale to [0, 1]

        # Inverse CDF: u → x (used by dynesty's prior_transform)
        _prior_icdf_tables[key] = interp1d(
            cdf, x_grid,
            kind='linear',
            bounds_error=False,
            fill_value=(x_grid[0], x_grid[-1]),
        )
        log.info(
            f"Built inverse-CDF table for '{key}' "
            f"(combined_split_gaussian over [{lo:.1f}, {hi:.1f}], "
            f"mode ≈ {_get_prior_mode(prior_info):.1f})"
        )


# ===========================================================================
# Config loading and global setup
# ===========================================================================

def setup_stellar_fit(input_parameters, **kwargs):
    """Load the YAML config and set up the module-level global state.

    This function must be called before any fitting. It:
      - Reads the YAML config file (or dict) and propagates all settings to
        the global namespace of this module.
      - Loads the observed spectrum from the reduction .npz file.
      - Initialises the PHOENIX interpolation grid.
      - Pre-computes the per-order wavelength and basis matrices (cached for
        speed, since they are constant across likelihood evaluations).

    Calling this function from a notebook after importing the module is the
    recommended way to reproduce a previous fit and regenerate model spectra.

    Parameters
    ----------
    input_parameters : str, Path, or dict
        Path to the YAML config file, or a dict with the same structure.
    **kwargs
        Keyword overrides for any key present in the config dict.
        Example: setup_stellar_fit("config.yaml", n_live_points=1000)
    """
    # --- Read config --------------------------------------------------------
    if isinstance(input_parameters, dict):
        cfg = dict(input_parameters)
    else:
        with open(input_parameters, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Apply any runtime overrides (e.g. passed from the command line)
    for key, val in kwargs.items():
        if key in cfg:
            cfg[key] = val
            log.info(f"Config override: {key} = {val}")
        else:
            raise KeyError(
                f"Key '{key}' not found in the config. "
                f"Available keys: {list(cfg.keys())}"
            )

    # Propagate all config entries to the global namespace of this module
    for key, val in cfg.items():
        globals()[key] = val
        log.info(f"Config: {key} = {val}")

    # --- Resolve output paths -----------------------------------------------
    global base_dir, walker_path, walker_file_out, run_name
    pl_fname = star_name.replace(' ', '_')

    # Expand '~' for all keys that represent file system paths.
    # This must be done before any Path() logic below, so that paths like
    # '~/scratch/...' read from the YAML work correctly.
    path_keys = ['data_file', 'base_dir', 'walker_path',
                 'walker_file_out', 'custom_prior_file']
    for key in path_keys:
        val = cfg.get(key)
        if val is not None:
            globals()[key] = Path(val).expanduser()

    if base_dir is None:
        base_dir = Path(os.environ.get('SCRATCH', Path.home()))
    base_dir = Path(base_dir).expanduser()

    if walker_path is None:
        walker_path = base_dir / f"DataAnalysis/StellarFit/{pl_fname}"
    walker_path = Path(walker_path).expanduser()

    if run_name is None:
        run_name = f"stellar_fit_{pl_fname}"

    if walker_file_out is None:
        walker_file_out = f"results_{run_name}"

    # --- Set up prior functions (reuse from retrieval_utils) ----------------
    global prior_func_dict, prior_init_func_dict
    prior_func_dict = dict(ru.default_prior_func)
    prior_init_func_dict = dict(ru.default_prior_init_func)

    if cfg.get('custom_prior_file') is not None:
        custom_prior, custom_init = ru.load_custom_prior(cfg['custom_prior_file'])
        prior_func_dict.update(custom_prior)
        prior_init_func_dict.update(custom_init)

    # Pre-compute numerical inverse-CDF tables for combined_split_gaussian priors.
    # Must be called after params_prior is set (above) and before run_dynesty().
    _build_prior_icdf_tables()

    # --- Load data ----------------------------------------------------------
    load_reduced_data(
        cfg['data_file'],
        bad_pix_frac=cfg.get('bad_pix_frac', 0.2),
        min_valid_pixels=cfg.get('min_valid_pixels', 800),
    )

    # --- Initialise model infrastructure ------------------------------------
    _check_dependencies()
    _init_phoenix_grid()
    validate_phoenix_grid()   # warn immediately if any grid point has NaN
    _precompute_rotation_grids()
    _precompute_chebyshev_bases()

    log.info("setup_stellar_fit() complete — ready to fit.")


def load_reduced_data(data_file, bad_pix_frac=0.2, min_valid_pixels=800):
    """Load the master out-of-transit spectrum from a starships reduction file.

    Populates the module globals ref_wave, ref_spectrum, ref_mask, ref_uncert.

    Orders are discarded if they fail either of two criteria:
    - Too many bad pixels **in the interior** (between the first and last valid
      pixel): more than bad_pix_frac of interior pixels are masked. Edge
      masking (contiguous masked pixels at the order boundaries) is excluded
      from this count because it is expected and does not affect the usable
      spectral range.
    - Too few valid pixels overall: fewer than min_valid_pixels valid pixels
      across the whole order.

    Parameters
    ----------
    data_file : str or Path
        Path to the .npz file produced by a starships reduction.
    bad_pix_frac : float
        Maximum fraction of masked pixels allowed **in the interior** of an
        order (between the first and last valid pixel). Default 0.2.
    min_valid_pixels : int
        Minimum number of valid (unmasked) pixels an order must have to be
        kept. Default 800. Orders below this threshold are discarded even if
        their interior bad-pixel fraction is acceptable.
    """
    global ref_wave, ref_spectrum, ref_mask, ref_uncert

    reduction = np.load(Path(data_file).expanduser())

    # The master out-of-transit spectrum and its pixel mask
    ref_spectrum = reduction['reference_spec'].copy()
    ref_mask = reduction['mask_reference_spec'].astype(bool).copy()

    # Wavelength grid — take the first exposure (all are identical)
    ref_wave = reduction['wave'][0].copy()

    # Per-exposure relative noise -> combined absolute uncertainty
    # The noise array has shape (n_exposures, n_orders, n_pixels).
    # We combine all exposures in quadrature, then convert to flux units.
    all_uncert = reduction['noise']
    n_exp = all_uncert.shape[0]
    ref_uncert = (np.sqrt(np.nansum(all_uncert ** 2, axis=0)) / n_exp) * ref_spectrum

    # Remove the barycentric / systemic RV already applied during reduction
    # so that ref_wave is as close as possible to the stellar rest frame
    ref_wave = ref_wave / calc_shift(reduction['RV_const'])

    # Extend the mask to all non-physical values
    ref_mask |= ~np.isfinite(ref_spectrum)
    ref_mask |= ~np.isfinite(ref_uncert)
    ref_mask |= (ref_uncert <= 0)   # zero uncertainty → infinite weight → skip
    ref_mask |= (ref_spectrum <= 0) # log(flux) undefined for non-positive values

    # Replace masked values with NaN for safety
    ref_spectrum = np.where(ref_mask, np.nan, ref_spectrum)
    ref_uncert = np.where(ref_mask, np.nan, ref_uncert)

    # --- Decide which orders to keep -----------------------------------------
    # We apply two criteria, evaluated independently:
    #   1. Minimum valid pixel count  (catches completely dead orders)
    #   2. Interior bad-pixel fraction  (catches orders with many gaps,
    #      while ignoring the expected edge masking at order boundaries)
    keep = np.array([
        _order_is_usable(ref_mask[i], bad_pix_frac, min_valid_pixels)
        for i in range(ref_mask.shape[0])
    ])

    ref_wave = ref_wave[keep]
    ref_spectrum = ref_spectrum[keep]
    ref_mask = ref_mask[keep]
    ref_uncert = ref_uncert[keep]

    log.info(
        f"Loaded {ref_wave.shape[0]} orders ({np.sum(~keep)} discarded) "
        f"— {ref_wave.shape[1]} pixels per order."
    )


def _order_is_usable(mask_ord, bad_pix_frac, min_valid_pixels):
    """Return True if a spectral order has enough usable pixels.

    Two conditions must both be satisfied:
    - At least min_valid_pixels unmasked pixels in the order.
    - The fraction of masked pixels **inside** the valid span (between the
      first and last unmasked pixel) must not exceed bad_pix_frac.
      Contiguous masked pixels at the edges are excluded because those are
      normal order-boundary effects, not data quality issues.

    Parameters
    ----------
    mask_ord : np.ndarray of bool, shape (n_pixels,)
        Pixel mask for one order (True = bad pixel).
    bad_pix_frac : float
        Maximum allowed fraction of interior bad pixels.
    min_valid_pixels : int
        Minimum number of unmasked pixels.

    Returns
    -------
    bool
    """
    valid_idx = np.where(~mask_ord)[0]
    n_valid = len(valid_idx)

    # Criterion 1: minimum absolute count of valid pixels
    if n_valid < min_valid_pixels:
        return False

    # Criterion 2: interior bad-pixel fraction
    # The "interior" spans from the first to the last valid pixel.
    # This excludes the contiguous masked edges that are expected at the
    # boundaries of every echelle order.
    first_valid = valid_idx[0]
    last_valid = valid_idx[-1]
    interior = mask_ord[first_valid : last_valid + 1]
    interior_bad_frac = interior.sum() / len(interior)

    if interior_bad_frac > bad_pix_frac:
        return False

    return True


# ===========================================================================
# Stellar model: PHOENIX + rotational broadening
# ===========================================================================

def _check_dependencies():
    """Raise a descriptive ImportError if mandatory packages are missing."""
    if not _PHOENIX_AVAILABLE:
        raise ImportError(
            "The 'phoenix_models' package is required but not found. "
            "This package is only available on the Narval cluster."
        )
    if not _PYASL_AVAILABLE:
        raise ImportError(
            "PyAstronomy is required for rotational broadening. "
            "Install it with: pip install PyAstronomy"
        )



def _init_phoenix_grid():
    """Initialise the PHOENIX interpolation grid over the data wavelength range.

    The grid is pre-built once per setup call and stored as the global
    `phoenix_interp`. Subsequent calls to `_generate_stellar_model_ord()`
    evaluate it at different (Teff, logg, metallicity) points without
    rebuilding the grid.

    The `dv_pad` (velocity padding) is computed from the prior bounds on
    v_shift and vsini, so the grid covers all physically accessible wavelengths.
    """
    global phoenix_interp, dv_pad

    # Compute the velocity pad from the prior edges
    v_prior = params_prior.get('v_shift', ['uniform', -100, 100])
    vsini_prior = params_prior.get('vsini', ['uniform', 0, 140])
    dv_pad = float(np.max(np.abs(v_prior[1:]))) + float(np.max(vsini_prior[1:]))

    wv_range = [float(np.nanmin(ref_wave)), float(np.nanmax(ref_wave))]
    wv_range = add_dv_pad_to_wv_range(dv_pad, wv_range)

    # Build parameter ranges for the PHOENIX grid.
    # Free parameters (in params_prior) are passed as [low, high] so that
    # PhoenixInterpGrid builds an interpolation axis covering the full prior.
    # Fixed parameters (in fixed_params) are passed as scalars so that they
    # are not interpolation axes (saves memory and build time).
    # Parameters absent from both dicts keep their PHOENIX default values.
    phoenix_params = {}
    for param in ('teff', 'logg', 'metal', 'alpha'):
        if param in fixed_params:
            phoenix_params[param] = fixed_params[param]
        elif param in params_prior:
            prior_info = params_prior[param]
            # _get_prior_effective_range handles all prior types, including
            # split_gaussian and combined_split_gaussian (uses ±5σ bounds).
            lo, hi = _get_prior_effective_range(prior_info)
            phoenix_params[param] = [lo, hi]

    phoenix_interp = PhoenixInterpGrid(
        wv_range=wv_range,
        resolution=phoenix_resolution,
        oversampling=phoenix_oversampling,
        n_fwhm=phoenix_n_fwhm,
        method=phoenix_method,
        query=False,
        **phoenix_params,
    )
    log.info(
        f"PHOENIX grid initialised: "
        f"{wv_range[0]:.3f}–{wv_range[1]:.3f} µm at R={phoenix_resolution}"
    )


def validate_phoenix_grid(n_interior_edge=10):
    """Check the PHOENIX interpolation grid for coverage and NaN values.

    After setup_stellar_fit(), this function performs two checks:

    1. **Coverage check**: verifies that the grid's parameter ranges cover
       the full prior bounds. If teff (or any other free parameter) in the
       prior goes up to 11000 K but the grid only covers [7400, 7500], all
       likelihood evaluations above 7500 K will return NaN because the
       RegularGridInterpolator returns fill_value=nan for out-of-bounds
       queries.

    2. **NaN check**: looks for NaN values stored in the grid itself. NaN
       can appear at the wavelength edges of each grid point due to the
       'valid' mode convolution trimming. With linear interpolation, a NaN
       at any corner of the interpolation hypercube propagates to the query
       result.

    Parameters
    ----------
    n_interior_edge : int
        Number of edge wavelength pixels to ignore when checking for NaN
        (these are expected due to 'valid' convolution trimming). Set to 0
        to check the full wavelength range including edges.

    Returns
    -------
    ok : bool
        True if both checks pass (correct coverage and no unexpected NaN).
    """
    interp = phoenix_interp.fct_interp
    values = interp.values   # shape: (n_param1, n_param2, ..., n_wv)
    grid_axes = interp.grid  # tuple: (param1_grid, ..., wv_grid)
    param_axes = grid_axes[:-1]
    wv_axis = grid_axes[-1]
    param_names_grid = phoenix_interp.parameters  # free param names in the grid

    ok = True

    # ------------------------------------------------------------------
    # Check 1: parameter coverage vs. prior bounds
    # The grid must span at least [prior_low, prior_high] for each free
    # parameter, otherwise queries within the prior will return NaN.
    # ------------------------------------------------------------------
    log.info("PHOENIX grid parameter coverage:")
    for name, ax in zip(param_names_grid, param_axes):
        grid_lo, grid_hi = float(ax[0]), float(ax[-1])
        n_pts = len(ax)
        log.info(f"  {name}: [{grid_lo}, {grid_hi}]  ({n_pts} points)")

        # Compare against prior bounds when available
        if name in params_prior:
            prior_info = params_prior[name]
            prior_lo, prior_hi = _get_prior_effective_range(prior_info)
            if prior_lo is None:
                continue
            if grid_lo > prior_lo + 1e-6:
                log.warning(
                    f"  *** {name}: grid lower bound {grid_lo} is ABOVE prior "
                    f"lower bound {prior_lo}. Queries near {prior_lo} will return NaN."
                )
                ok = False
            if grid_hi < prior_hi - 1e-6:
                log.warning(
                    f"  *** {name}: grid upper bound {grid_hi} is BELOW prior "
                    f"upper bound {prior_hi}. Queries near {prior_hi} will return NaN. "
                    f"You may need to download more PHOENIX files."
                )
                ok = False

    # ------------------------------------------------------------------
    # Check 2: NaN values stored in the grid
    # ------------------------------------------------------------------
    lo = n_interior_edge
    hi = values.shape[-1] - n_interior_edge
    interior_values = values[..., lo:hi]
    n_nan = int(np.sum(np.isnan(interior_values)))
    n_total = interior_values.size

    if n_nan == 0:
        log.info(
            f"PHOENIX grid OK: no NaN in interior wavelength range "
            f"[{wv_axis[lo]:.4f}, {wv_axis[hi-1]:.4f}] µm "
            f"({n_total} values checked)"
        )
    else:
        ok = False
        log.warning(
            f"PHOENIX grid has NaN values: {n_nan} / {n_total} interior pixels "
            f"({100 * n_nan / n_total:.1f}%) are NaN. "
            "These will cause -inf in the likelihood during the fit."
        )
        nan_mask_collapsed = np.any(np.isnan(interior_values), axis=-1)
        bad_indices = list(zip(*np.where(nan_mask_collapsed)))
        if bad_indices:
            log.warning("  Affected grid points:")
            for idx_tuple in bad_indices[:20]:
                param_vals = {name: float(ax[i])
                              for name, ax, i in zip(param_names_grid, param_axes, idx_tuple)}
                n_nan_here = int(np.sum(np.isnan(interior_values[idx_tuple])))
                n_here = interior_values[idx_tuple].size
                log.warning(f"    {param_vals}  →  {n_nan_here}/{n_here} NaN pixels")
            if len(bad_indices) > 20:
                log.warning(f"    ... and {len(bad_indices) - 20} more grid points.")

    return ok


def _precompute_rotation_grids():
    """Pre-compute the per-order uniform wavelength grids for pyasl.fastRotBroad.

    PyAstronomy's fastRotBroad requires a uniformly-sampled wavelength grid
    (constant Δλ, not constant Δv). We build one grid per order, extended
    by dv_pad on each side to avoid edge effects in the broadening convolution.
    The grids are stored in the global list `_wave_rot_grids`.
    """
    global _wave_rot_grids

    _wave_rot_grids = []
    for wv_ord in ref_wave:
        wv_min = float(np.nanmin(wv_ord))
        wv_max = float(np.nanmax(wv_ord))
        wv_min, wv_max = add_dv_pad_to_wv_range(dv_pad, [wv_min, wv_max])
        # Step size: Δλ = λ / rot_broad_samp  (constant resolving power grid)
        delta_wv = wv_min / rot_broad_samp
        _wave_rot_grids.append(np.arange(wv_min, wv_max, delta_wv))


def _generate_stellar_model_ord(idx_ord, teff, logg, metal, alpha,
                                vsini, epsilon, v_shift, **_ignored):
    """Generate a rotationally broadened, RV-shifted PHOENIX spectrum for one order.

    Parameters
    ----------
    idx_ord : int
        Index into ref_wave / ref_mask (0-indexed).
    teff : float
        Effective temperature of the star in K.
    logg : float
        Log surface gravity log g in cgs units.
    metal : float
        Metallicity [Fe/H].
    alpha : float
        Alpha-element abundance [α/Fe].
    vsini : float
        Projected rotational velocity in km/s.
    epsilon : float
        Linear limb-darkening coefficient for the rotation profile (0 to 1).
    v_shift : float
        Systemic radial velocity in km/s (positive = receding from observer).
    **_ignored
        Extra keys from theta_dict (e.g. from fixed_params) are silently ignored.

    Returns
    -------
    model_norm : np.ndarray, shape (n_pixels,)
        Model flux normalised by its median. Masked pixels are set to NaN.
    """
    # The rotation grid for this order (uniform Δλ, needed by pyasl)
    wv_rot = _wave_rot_grids[idx_ord]

    # 1. Evaluate the PHOENIX spectrum on the rotation grid
    flux_rot = phoenix_interp(wv_rot, teff=teff, logg=logg, metal=metal, alpha=alpha)

    # Handle NaN at the edges of the rotation grid. This can happen when
    # wv_rot extends slightly beyond the PHOENIX wavelength grid (e.g. due
    # to the Doppler padding added by dv_pad). NaN values must be replaced
    # before convolution because pyasl.fastRotBroad uses mode='same', so
    # a single NaN at the edge propagates through the rotation kernel
    # (width ~2*vsini/c × n_pixels ≈ 55 px for vsini=120 km/s) and can
    # corrupt the entire output spectrum.
    # We fill edge NaN with the nearest valid flux value — these pixels are
    # outside the observed wavelength range so their exact value does not
    # affect the likelihood.
    nan_mask = ~np.isfinite(flux_rot)
    if nan_mask.any():
        valid_idx = np.where(~nan_mask)[0]
        if len(valid_idx) == 0:
            # Entire order is outside the PHOENIX grid — skip it
            return np.full(ref_wave[idx_ord].shape, np.nan)
        # Nearest-neighbour fill: propagate edge values inward
        flux_rot[:valid_idx[0]] = flux_rot[valid_idx[0]]
        flux_rot[valid_idx[-1] + 1:] = flux_rot[valid_idx[-1]]

    # 2. Apply rotational broadening.
    #    pyasl.fastRotBroad expects wavelength in Angstroms → multiply µm by 1000.
    flux_rot = pyasl.fastRotBroad(wv_rot * 1000.0, flux_rot, epsilon, vsini)

    # 3. Build a spline interpolant and resample onto the observed wavelength grid.
    #    Dividing the observed wavelengths by the Doppler shift factor is equivalent
    #    to shifting the model: if the star recedes at v_shift km/s, observed lines
    #    appear at λ_obs = λ_rest × shift, so we query the rest-frame model at
    #    λ_rest = λ_obs / shift.
    flux_spline = UnivariateSpline(wv_rot, flux_rot, k=3, s=0)
    wv_shift = calc_shift(v_shift)

    mask = ref_mask[idx_ord]
    wv_obs = ref_wave[idx_ord]
    model_norm = np.full(wv_obs.shape, np.nan)
    model_norm[~mask] = flux_spline(wv_obs[~mask] / wv_shift)

    # 4. Normalise by the median to remove the absolute flux scale.
    #    The polynomial correction handles any remaining order-level shape.
    med = np.nanmedian(model_norm)
    if med > 0:
        model_norm /= med

    return model_norm


# ===========================================================================
# Profile log-likelihood
# ===========================================================================

def _precompute_chebyshev_bases():
    """Pre-compute and cache Chebyshev basis matrices for all orders.

    The basis matrix Φ for order i has shape (n_valid_pixels, n_poly), where
    each column is a Chebyshev polynomial evaluated on a [-1, 1] grid.
    Because the mask (and therefore n_valid_pixels) is constant across MCMC
    iterations, we build these matrices once and reuse them in every likelihood
    call. This avoids a significant bottleneck.
    """
    global _cheb_bases
    _cheb_bases = []
    n_pixels = ref_wave.shape[-1]   # full order size (same for all orders)
    for _ in range(ref_wave.shape[0]):
        # Build the basis over ALL pixels so that each pixel's x-coordinate
        # is its position in the full order (x = -1 at pixel 0, +1 at the last).
        # The valid rows are extracted at fit time with [~mask], keeping the
        # x-mapping consistent between fitting and evaluation.
        # Building on n_valid instead would map the first valid pixel to x=-1
        # and the last to x=+1, which differs from the x used when plotting
        # the correction on the full order — an inconsistency that degrades the
        # continuum correction near masked edges.
        _cheb_bases.append(_build_chebyshev_basis(n_pixels, n_poly))


def _build_chebyshev_basis(n_pixels, n_coeffs):
    """Build a Chebyshev polynomial basis matrix.

    Parameters
    ----------
    n_pixels : int
        Number of pixels (rows of the output matrix).
    n_coeffs : int
        Number of Chebyshev terms (columns). Degree = n_coeffs - 1.

    Returns
    -------
    basis : np.ndarray, shape (n_pixels, n_coeffs)
        Column k is the k-th Chebyshev polynomial T_k(x) evaluated on
        n_pixels equally spaced points on [-1, 1].
    """
    # Map pixel indices to [-1, 1] for numerical stability
    x = np.linspace(-1.0, 1.0, n_pixels)
    # Build the Vandermonde-like matrix for Chebyshev polynomials
    basis = np.column_stack([
        np.polynomial.chebyshev.chebval(x, np.eye(n_coeffs)[k])
        for k in range(n_coeffs)
    ])
    return basis


def _profile_logl_one_order(idx_ord, model_norm, log_noise_s=0.0):
    """Profile log-likelihood for a single spectral order.

    For a given normalised stellar model, this function analytically finds
    the Chebyshev polynomial correction (in log flux) that minimises χ²,
    then returns the log-likelihood evaluated at that optimum.

    This is the core of the "profile likelihood" approach: instead of
    including polynomial coefficients as free parameters in the sampler,
    we solve for them exactly at each likelihood evaluation.

    A global noise scaling factor s = exp(log_noise_s) inflates the formal
    uncertainties by s. The optimal polynomial coefficients are unchanged by
    this scaling (the normal equations cancel s²), so only the final log-
    likelihood needs to be adjusted:

        logl(s) = −χ²_profile / (2s²)  +  log_norm  −  n_valid × log(s)

    Parameters
    ----------
    idx_ord : int
        Spectral order index.
    model_norm : np.ndarray, shape (n_pixels,)
        Normalised model flux for this order (NaN at masked pixels).
    log_noise_s : float
        Natural log of the noise scaling factor (default 0 → s = 1, no scaling).
        Fit this as a free parameter when formal uncertainties may be under-
        or over-estimated.

    Returns
    -------
    logl : float
        Profile log-likelihood for this order (−∞ if computation fails).
    """
    mask = ref_mask[idx_ord]
    data = ref_spectrum[idx_ord, ~mask]
    sigma = ref_uncert[idx_ord, ~mask]
    model = model_norm[~mask]

    # All values must be strictly positive because we work in log space
    if np.any(data <= 0) or np.any(model <= 0) or np.any(sigma <= 0):
        return -np.inf

    # --- Log-space residual -------------------------------------------------
    # The model in log space: log(data) ≈ log(model) + Φ c
    # The polynomial Φ c corrects for continuum shape errors (blaze, etc.)
    log_residual = np.log(data) - np.log(model)

    # Propagate uncertainties to log space via the delta method: σ_log ≈ σ/f
    sigma_log = sigma / data
    W = 1.0 / sigma_log ** 2   # diagonal weight matrix elements

    # --- Solve the normal equations for the optimal polynomial coefficients --
    # Minimise: Σ_λ W(λ) [r(λ) - Φ(λ)·c]²  over c
    # Solution: (ΦᵀWΦ) c* = Φᵀ W r
    # Extract only the valid rows so we fit on observed pixels, while keeping
    # the x-coordinates consistent with the full-order basis (built at setup).
    Phi = _cheb_bases[idx_ord][~mask]  # shape (n_valid_pixels, n_poly)
    WPhi = W[:, None] * Phi     # shape (n_valid_pixels, n_poly)
    A = Phi.T @ WPhi            # shape (n_poly, n_poly)  — normal matrix
    b = WPhi.T @ log_residual   # shape (n_poly,)         — right-hand side

    # Use lstsq for numerical robustness (handles nearly singular A)
    c_opt, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # --- Profile chi² -------------------------------------------------------
    # χ²_profile = rᵀWr - bᵀc*
    # The second term is the variance explained by the optimal polynomial.
    chi2_profile = np.dot(W, log_residual ** 2) - np.dot(b, c_opt)

    # --- Full Gaussian log-likelihood with optional noise scaling -----------
    # log_norm uses the original (unscaled) uncertainties — the s dependence
    # is captured by the two correction terms below.
    log_norm = -0.5 * np.sum(np.log(2.0 * np.pi * sigma_log ** 2))

    if log_noise_s == 0.0:
        # Fast path: no scaling, avoids two exp/log evaluations per order
        logl = -0.5 * chi2_profile + log_norm
    else:
        # chi2_profile / s² = chi2_profile * exp(−2 * log_noise_s)
        # n_valid * log(s) = n_valid * log_noise_s
        n_valid = np.sum(~mask)
        logl = (-0.5 * chi2_profile * np.exp(-2.0 * log_noise_s)
                + log_norm
                - n_valid * log_noise_s)

    return logl


def profile_log_likelihood(theta_dict):
    """Profile log-likelihood summed over all spectral orders.

    Generates the stellar model for each order and evaluates the profile
    log-likelihood (with analytically optimal polynomial per order).

    Parameters
    ----------
    theta_dict : dict
        Stellar parameters. Must contain at least the keys expected by
        _generate_stellar_model_ord (teff, logg, metal, alpha, vsini,
        epsilon, v_shift). Additional keys (from fixed_params) are ignored.

    Returns
    -------
    logl : float
        Total profile log-likelihood, or −∞ if any order fails.
    """
    # Extract noise scaling (optional parameter; defaults to s=1 if absent)
    log_noise_s = theta_dict.get('log_noise_s', 0.0)

    total_logl = 0.0

    for idx_ord in range(ref_wave.shape[0]):
        model_norm = _generate_stellar_model_ord(idx_ord, **theta_dict)

        # Sanity check: model must contain at least some finite values
        if not np.any(np.isfinite(model_norm)):
            return -np.inf

        logl_ord = _profile_logl_one_order(idx_ord, model_norm, log_noise_s)

        if not np.isfinite(logl_ord):
            return -np.inf

        total_logl += logl_ord

    return total_logl


# ===========================================================================
# Prior and full log-probability
# ===========================================================================

def unpack_theta(theta):
    """Convert a parameter vector to a dictionary.

    The dictionary contains both the fitted parameters (from params_prior,
    in order) and the fixed parameters (from fixed_params). Fixed parameters
    are overwritten by fitted ones if they share the same key.

    Parameters
    ----------
    theta : array-like, shape (n_free_params,)
        Parameter values in the order defined by params_prior.

    Returns
    -------
    theta_dict : dict
        Dictionary with all parameters needed by the model.
    """
    theta_dict = dict(fixed_params)
    for key, val in zip(params_prior.keys(), theta):
        theta_dict[key] = val
    return theta_dict


def pack_theta(theta_dict):
    """Convert a parameter dictionary to a vector (inverse of unpack_theta).

    Parameters
    ----------
    theta_dict : dict
        Parameter dictionary.

    Returns
    -------
    theta : np.ndarray, shape (n_free_params,)
        Values in the order defined by params_prior.
    """
    return np.array([theta_dict[key] for key in params_prior])


def lnprob(theta, debug=False):
    """Full log-posterior: log prior + profile log-likelihood.

    Parameters
    ----------
    theta : array-like
        Parameter vector in the order defined by params_prior.
    debug : bool
        If True, print a detailed breakdown per parameter of which prior
        produces a non-finite value, and which spectral order(s) produce
        a non-finite log-likelihood. Useful when the fit returns -inf.

    Returns
    -------
    float
        Log-posterior, or −∞ if the prior is violated.
    """
    # --- Prior check ---------------------------------------------------------
    # Evaluate each parameter's prior independently. In debug mode, ALL
    # parameters are evaluated and printed, even after one fails, so the
    # user can see every individual prior value and immediately identify
    # which parameter is out of bounds or causing a non-finite result.
    theta_dict_free = {key: val for key, val in zip(params_prior.keys(), theta)}
    lp = 0.0
    for key, prior_info in params_prior.items():
        prior_name = prior_info[0]
        prior_args = prior_info[1:]
        prior_func = prior_func_dict[prior_name]
        val_prior = prior_func(theta_dict_free, key, prior_args)
        if debug:
            log.info(
                f"  [prior] {key}={theta_dict_free[key]:.6g}  "
                f"{prior_name}({list(prior_args)}) → {val_prior}"
            )
        lp += val_prior
        if not np.isfinite(lp) and not debug:
            # Fast path: exit early when prior is violated (no debug needed)
            return -np.inf

    if not np.isfinite(lp):
        return -np.inf

    # --- Likelihood ----------------------------------------------------------
    theta_dict = unpack_theta(theta)

    if debug:
        log.info(f"[debug] Prior OK (lp={lp:.4f}). Evaluating likelihood at {theta_dict}")
        total = 0.0
        log_noise_s_debug = theta_dict.get('log_noise_s', 0.0)
        for idx_ord in range(ref_wave.shape[0]):
            model_norm = _generate_stellar_model_ord(idx_ord, **theta_dict)
            logl_ord = _profile_logl_one_order(idx_ord, model_norm, log_noise_s_debug)
            finite = np.isfinite(logl_ord)
            log.info(f"  order {idx_ord:2d}: logl={logl_ord:.2f}  {'OK' if finite else '<<< -inf'}")
            if np.isfinite(logl_ord):
                total += logl_ord
        log.info(f"  TOTAL logl = {total:.2f}")
        return lp + total

    logl = profile_log_likelihood(theta_dict)
    return lp + logl


def prior_transform(u):
    """Map the unit hypercube [0, 1]^n to physical parameter space.

    This is the function required by dynesty (nested sampling). Each
    parameter is transformed independently based on its prior type:

      - uniform / log_uniform [low, high]
            x = low + u * (high - low)
      - gaussian [mu, sigma]
            x = scipy.stats.norm.ppf(u, mu, sigma)
      - split_gaussian [mu, sigma_lo, sigma_hi]
            Analytic inverse CDF of the split-normal distribution.
      - combined_split_gaussian [mu1, s_lo1, s_hi1, ...]
            Numerical inverse CDF (pre-computed in _build_prior_icdf_tables).

    Parameters
    ----------
    u : np.ndarray, shape (n_free_params,)
        Point in the unit hypercube (one value per free parameter).

    Returns
    -------
    theta : np.ndarray, shape (n_free_params,)
        Physical parameter values.
    """
    from scipy.stats import norm as scipy_norm

    theta = np.empty_like(u)
    for i, (key, prior_info) in enumerate(params_prior.items()):
        prior_type = prior_info[0]
        args = prior_info[1:]

        if prior_type in ('uniform', 'log_uniform'):
            # Linear mapping from [0, 1] to [low, high]
            low, high = float(args[0]), float(args[1])
            theta[i] = low + u[i] * (high - low)

        elif prior_type == 'gaussian':
            # Standard percent-point function (inverse CDF of the Gaussian)
            mu, sigma = float(args[0]), float(args[1])
            theta[i] = scipy_norm.ppf(u[i], loc=mu, scale=sigma)

        elif prior_type == 'split_gaussian':
            # Analytic inverse CDF of the split-normal distribution.
            # The split-normal assigns fraction  p_lo = sigma_lo/(sigma_lo+sigma_hi)
            # of its probability mass to the left side (x < mu).
            # Derivation of the inverse CDF:
            #   CDF(x < mu) = 2*p_lo * Phi((x-mu)/sigma_lo)
            #   CDF(x >= mu) = p_lo + (1-p_lo) * (2*Phi((x-mu)/sigma_hi) - 1)
            mu = float(args[0])
            sigma_lo = float(args[1])
            sigma_hi = float(args[2])
            p_lo = sigma_lo / (sigma_lo + sigma_hi)
            if u[i] <= p_lo:
                # Invert the left-half CDF: u/(2*p_lo) goes from 0 to 0.5
                theta[i] = mu + sigma_lo * scipy_norm.ppf(u[i] / (2.0 * p_lo))
            else:
                # Invert the right-half CDF
                theta[i] = mu + sigma_hi * scipy_norm.ppf(
                    0.5 * (1.0 + (u[i] - p_lo) / (1.0 - p_lo))
                )

        elif prior_type == 'combined_split_gaussian':
            # Product of multiple split-Gaussians: no closed-form inverse CDF.
            # Use the precomputed interpolation table built in setup_stellar_fit.
            theta[i] = float(_prior_icdf_tables[key](u[i]))

        else:
            raise ValueError(
                f"Prior type '{prior_type}' for parameter '{key}' is not "
                "supported in prior_transform. Supported types: "
                "'uniform', 'log_uniform', 'gaussian', "
                "'split_gaussian', 'combined_split_gaussian'."
            )
    return theta


def _logl_for_dynesty(theta):
    """Log-likelihood wrapper for dynesty (receives physical params, not the unit cube)."""
    # Apply the prior explicitly to reject out-of-bound proposals from dynesty
    lp = ru.log_prior(theta, params_prior, prior_func_dict=prior_func_dict)
    if not np.isfinite(lp):
        return -1e300  # dynesty requires a finite float

    theta_dict = unpack_theta(theta)
    logl = profile_log_likelihood(theta_dict)

    return logl if np.isfinite(logl) else -1e300


# ===========================================================================
# Point estimate: scipy.optimize
# ===========================================================================

def run_minimize(n_restarts=5, method='Nelder-Mead', verbose=False):
    """Find best-fit parameters by minimising −log L (the chi² equivalent).

    Runs scipy.optimize.minimize from multiple starting points drawn from
    the prior (or from special_init if provided) and returns the best result.
    This is much faster than MCMC and gives a good point estimate as well
    as a starting point for nested sampling.

    Parameters
    ----------
    n_restarts : int
        Number of independent minimisation attempts. More restarts increase
        the chance of finding the global minimum at the cost of more time.
    method : str
        Optimisation method passed to scipy.optimize.minimize.
        'Nelder-Mead' is a good default (gradient-free, handles noise well).
    verbose : bool
        If True, print the current parameter values and log-likelihood at
        every iteration. Useful for debugging (e.g. when the fit goes to
        infinity or does not converge).

    Returns
    -------
    best_result : OptimizeResult
        The best result (lowest −log L) among all attempts.
    theta_best : np.ndarray
        Best-fit parameter vector.
    theta_dict_best : dict
        Best-fit parameters as a dictionary (including fixed_params).
    """
    # Build parameter bounds from the prior for bounded optimisers
    bounds = _get_bounds_from_prior()
    lower = np.array([b[0] if b[0] is not None else -np.inf for b in bounds])
    upper = np.array([b[1] if b[1] is not None else  np.inf for b in bounds])
    param_names = list(params_prior.keys())

    # Keep track of every evaluation for verbose mode
    _call_counter = [0]

    def neg_lnprob(theta):
        theta_clipped = np.clip(theta, lower, upper)
        val = -lnprob(theta_clipped)
        if verbose:
            _call_counter[0] += 1
            param_str = '  '.join(f'{k}={v:.4g}' for k, v in zip(param_names, theta_clipped))
            log.info(f"  [{_call_counter[0]:4d}]  −logL={val:.4f}   {param_str}")
        return val

    def _callback(theta):
        # Called once per Nelder-Mead iteration (not every function evaluation).
        # Only used when verbose=False to give a progress line every 50 iterations.
        if not verbose:
            _call_counter[0] += 1
            if _call_counter[0] % 50 == 0:
                val = neg_lnprob(theta)
                param_str = '  '.join(f'{k}={v:.4g}' for k, v in zip(param_names, theta))
                log.info(f"  iter {_call_counter[0]:4d}:  −logL={val:.2f}   {param_str}")

    best_result = None
    best_fun = np.inf

    for i_try in range(n_restarts):
        theta0 = _sample_starting_point(bounds)
        log.info(f"Minimisation attempt {i_try + 1}/{n_restarts}")

        _call_counter[0] = 0  # reset counter for each attempt

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = minimize(
                neg_lnprob, theta0, method=method, bounds=bounds,
                callback=None if verbose else _callback,
                options={
                    # Nelder-Mead converges in a few hundred steps for ~6
                    # parameters; 5000 is a generous ceiling.
                    'maxiter': 5000,
                    # We only need a good starting point for dynesty, not the
                    # exact minimum. xatol=1 stops when parameters change by
                    # less than ~1 unit per step (1 K for Teff, 1 km/s for
                    # vsini…) which is already far more precise than the
                    # posterior width for these stellar parameters.
                    'xatol': 1.0,
                    # The total log-likelihood is typically ~-10^5 to -10^6,
                    # so stopping at 10 log-likelihood units of improvement
                    # is a sensible convergence criterion.
                    'fatol': 10.0,
                },
            )

        log.info(f"  −log L = {result.fun:.2f}  (success={result.success})")

        if result.fun < best_fun:
            best_fun = result.fun
            best_result = result

    theta_best = best_result.x
    theta_dict_best = unpack_theta(theta_best)
    log.info(f"Best-fit: {theta_dict_best}")

    return best_result, theta_best, theta_dict_best


def _get_bounds_from_prior():
    """Build a list of (low, high) bounds from params_prior.

    For uniform priors these are the literal bounds; for Gaussian-type priors
    we use the ±5σ range returned by _get_prior_effective_range.
    """
    return [_get_prior_effective_range(prior_info)
            for prior_info in params_prior.values()]


def _sample_starting_point(bounds):
    """Sample a starting point for the minimiser.

    Priority order for each parameter:
    1. special_init (if the key is listed there) — the user knows best.
    2. Prior mode  (for gaussian / split_gaussian / combined_split_gaussian) —
       when informative literature constraints exist, starting at the most
       likely value avoids wasting iterations exploring improbable regions.
    3. Random draw from the prior bounds (uniform priors) — allows multiple
       restarts to explore different regions of parameter space.

    Parameters
    ----------
    bounds : list of (float, float)
        Prior bounds for each parameter, in the order of params_prior.

    Returns
    -------
    theta0 : np.ndarray
    """
    theta0 = np.empty(len(params_prior))
    si = globals().get('special_init') or {}

    for i, (key, prior_info) in enumerate(params_prior.items()):
        prior_type = prior_info[0]

        if key in si and si[key] is not None:
            # Explicit starting point always takes priority
            theta0[i] = float(si[key])

        elif prior_type in ('gaussian', 'split_gaussian', 'combined_split_gaussian'):
            # For informative priors, start at the prior mode (mu). This is the
            # literature estimate and already a well-motivated starting point.
            theta0[i] = _get_prior_mode(prior_info)

        else:
            # Uniform prior: draw randomly so that multiple restarts explore
            # different parts of the prior volume (improves global optimisation).
            low, high = bounds[i]
            if low is not None and high is not None:
                theta0[i] = np.random.uniform(low, high)
            else:
                theta0[i] = float(prior_info[1])  # fallback

    return theta0


# ===========================================================================
# Full posterior: dynesty nested sampling
# ===========================================================================

def run_dynesty(n_live=500, save_file=None, n_workers=1,
                bound='multi', sample='rwalk', **dynesty_kwargs):
    """Sample the posterior with dynesty dynamic nested sampling.

    Uses DynamicNestedSampler, which adaptively allocates live points to the
    regions of parameter space that contribute most to the posterior. This is
    more efficient than the static NestedSampler for parameter estimation
    (as opposed to pure Bayesian evidence computation).

    Parameters
    ----------
    n_live : int
        Number of live points for both the initial and dynamic batch phases
        (nlive_init = nlive_batch = n_live). Increase for more accurate
        posteriors. 400 is a good default for ~6 parameters; 500–1000 if you
        need tight constraints on teff or vsini.
    save_file : str or Path, optional
        File name (relative to walker_path) to save the results as a pickle.
        If None, results are not saved to disk.
    n_workers : int
        Number of parallel worker processes for likelihood evaluation.
        Uses dynesty's native pool (``dynesty.pool.Pool``) when available,
        which is more robust than a raw multiprocessing pool because it
        serialises loglike and prior_transform to the workers explicitly
        (avoids pitfalls with global state on some systems).

        To use parallelism on Narval, add to your sbatch script::

            #SBATCH --cpus-per-task=8

        and set ``n_workers: 8`` in your YAML config.
        Default is 1 (single-threaded).
    bound : str
        Bounding method for dynesty. ``'multi'`` (multiple ellipsoids, default)
        is well suited to the elongated teff–vsini degeneracy in stellar
        spectroscopy. ``'balls'`` (overlapping spheres) can be better for
        highly multimodal posteriors with irregular geometry.
    sample : str
        Sampling method within each bound. ``'rwalk'`` (random walk, default)
        works well for 5–7 dimensional problems. ``'rslice'`` (random slice)
        can improve efficiency when the posterior has strong correlations.
    **dynesty_kwargs
        Additional keyword arguments forwarded to
        ``dynesty.DynamicNestedSampler``.

    Returns
    -------
    results : dynesty.results.Results
        Dynesty results object. Key attributes:

          - results.samples     : posterior samples (shape n_samples × n_dim)
          - results.logwt       : log importance weights
          - results.logz[-1]    : log evidence estimate
          - results.logzerr[-1] : uncertainty on log evidence
    """
    if not _DYNESTY_AVAILABLE:
        raise ImportError(
            "dynesty is not installed. Install it with: pip install dynesty"
        )

    n_dim = len(params_prior)
    log.info(
        f"Starting dynesty: DynamicNestedSampler  "
        f"n_live={n_live}  n_dim={n_dim}  "
        f"bound='{bound}'  sample='{sample}'  n_workers={n_workers}"
    )

    # run_nested kwargs shared by all code paths below.
    # pfrac=1.0 tells dynesty to allocate all effort to posterior estimation
    # rather than splitting between posterior and evidence (the default).
    #
    # print_progress is enabled only when stdout is connected to a real terminal
    # (interactive session or notebook). When stdout is redirected — as in an
    # sbatch job — dynesty's \r-based progress bar produces garbled output, so
    # we suppress it. The "Starting dynesty..." and "Dynesty done." log lines
    # are always emitted and serve as progress markers in HPC logs.
    import sys
    if verbose_dynesty is None:
        interactive = sys.stdout.isatty()
    else:
        interactive = bool(verbose_dynesty)
    run_kwargs = dict(
        nlive_init=n_live,
        nlive_batch=n_live,
        wt_kwargs={'pfrac': 1.0},
        stop_kwargs={'pfrac': 1.0},
        print_progress=interactive,
    )

    if n_workers > 1:
        # Prefer dynesty's native pool, which explicitly serialises loglike and
        # prior_transform to workers — more robust than multiprocessing.Pool on
        # systems that use 'spawn' (e.g., macOS). On Narval (Linux, 'fork' is
        # the default), both approaches work, but dypool is cleaner.
        try:
            from dynesty import pool as dypool
            log.info(f"Using dynesty.pool.Pool with {n_workers} workers.")
            with dypool.Pool(n_workers,
                             loglike=_logl_for_dynesty,
                             prior_transform=prior_transform) as pool:
                sampler = dynesty.DynamicNestedSampler(
                    pool.loglike, pool.prior_transform, n_dim,
                    bound=bound, sample=sample, pool=pool,
                    **dynesty_kwargs,
                )
                sampler.run_nested(**run_kwargs)

        except (ImportError, AttributeError):
            # Fall back to multiprocessing.Pool if dynesty.pool is not available
            log.warning("dynesty.pool not available; falling back to multiprocessing.Pool.")
            import multiprocessing
            with multiprocessing.Pool(n_workers) as pool:
                sampler = dynesty.DynamicNestedSampler(
                    _logl_for_dynesty, prior_transform, n_dim,
                    bound=bound, sample=sample, pool=pool,
                    queue_size=n_workers, **dynesty_kwargs,
                )
                sampler.run_nested(**run_kwargs)
    else:
        sampler = dynesty.DynamicNestedSampler(
            _logl_for_dynesty, prior_transform, n_dim,
            bound=bound, sample=sample, **dynesty_kwargs,
        )
        sampler.run_nested(**run_kwargs)

    results = sampler.results
    log.info(
        f"Dynesty done. "
        f"log Z = {results.logz[-1]:.2f} ± {results.logzerr[-1]:.2f}"
    )

    if save_file is not None:
        save_path = Path(walker_path) / save_file
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as fh:
            pickle.dump(results, fh)
        log.info(f"Results saved to {save_path}")

    return results


# ===========================================================================
# Visualization helpers (intended for notebook use)
# ===========================================================================

def make_model_with_best_poly(theta_dict, return_poly=False):
    """Generate model spectra with the optimal polynomial correction applied.

    For each order, this function:
      1. Generates the PHOENIX + rotation model.
      2. Solves for the optimal log-polynomial correction (same as in the
         profile likelihood).
      3. Returns model × exp(poly), which is directly comparable to the data.

    This is the recommended function to call in an analysis notebook to
    visually inspect the quality of a fit.

    Parameters
    ----------
    theta_dict : dict
        Stellar parameters (e.g. output of unpack_theta).
    return_poly : bool
        If True, also return the multiplicative polynomial correction arrays.

    Returns
    -------
    models_corrected : np.ndarray, shape (n_orders, n_pixels)
        Model flux after polynomial correction. Masked pixels are NaN.
    poly_corrections : np.ndarray, shape (n_orders, n_pixels)
        Only returned if return_poly=True.
        The multiplicative polynomial correction exp(Φ c*) applied per order.
    """
    n_orders, n_pixels = ref_wave.shape
    models_corrected = np.full((n_orders, n_pixels), np.nan)
    poly_corrections = np.ones((n_orders, n_pixels))

    for idx_ord in range(n_orders):
        model_norm = _generate_stellar_model_ord(idx_ord, **theta_dict)

        mask = ref_mask[idx_ord]
        data = ref_spectrum[idx_ord, ~mask]
        sigma = ref_uncert[idx_ord, ~mask]
        model = model_norm[~mask]

        if np.any(data <= 0) or np.any(model <= 0):
            continue

        # Solve for optimal polynomial (same computation as in the likelihood).
        # _cheb_bases[idx_ord] is built on the full pixel grid, so x-coordinates
        # are consistent between fitting and evaluation.
        log_residual = np.log(data) - np.log(model)
        sigma_log = sigma / data
        W = 1.0 / sigma_log ** 2

        Phi_full = _cheb_bases[idx_ord]          # (n_pixels, n_poly)
        Phi = Phi_full[~mask]                    # (n_valid, n_poly) — fit on valid pixels
        WPhi = W[:, None] * Phi
        A = Phi.T @ WPhi
        b = WPhi.T @ log_residual
        c_opt, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        # Polynomial correction on the full pixel grid: same x-coordinates as fitting
        log_poly_full = Phi_full @ c_opt
        poly_full = np.exp(log_poly_full)

        # Corrected model: model_norm × poly ≈ data
        corrected = np.full(n_pixels, np.nan)
        corrected[~mask] = model * np.exp((Phi_full @ c_opt)[~mask])
        models_corrected[idx_ord] = corrected
        poly_corrections[idx_ord] = poly_full

    if return_poly:
        return models_corrected, poly_corrections
    return models_corrected


def save_stellar_spectrum(theta_dict, save_dir=None, save_name=None, wv_range=None, apply_rotation=True):
    """Build the best-fit PHOENIX stellar spectrum and save it for use in starships.retrieval.

    This is the function to call after a fit to produce the ``star_spectrum``
    .npz file consumed by ``starships.retrieval`` (the ``star_spectrum`` entry
    of the retrieval YAML config). The spectrum is saved in the star's rest
    frame: ``v_shift`` (systemic RV) is *not* applied, since
    ``starships.retrieval`` evaluates the stellar spectrum directly on the
    rest-frame wavelength grid of the atmosphere model and applies
    systemic/orbital Doppler shifts separately later in the pipeline (see
    ``petitradtrans_utils.retrieval_model_plain``). Rotational broadening
    (``vsini``, ``epsilon``) *is* intrinsic to the star and is applied here.

    The .npz stores the physical resolving power (``resolution``) and the
    wavelength-grid oversampling factor (``oversampling``) as two separate
    fields, rather than a single combined "sampling resolution" -- this
    matches the vocabulary already used by
    ``phoenix_models.convert_phoenix_at_resolution`` and is meant to be
    forward-compatible with a planned rework of how
    ``starships.retrieval`` handles the stellar spectrum's resolution
    (currently it only reads a single ``sampling_res`` key, so that key is
    also written here -- as ``resolution * oversampling`` -- for backward
    compatibility with the current code).

    Parameters
    ----------
    theta_dict : dict
        Stellar parameters, as produced by `unpack_theta()` or built by hand.
        Must contain ``teff``, ``logg``, ``metal``, ``alpha``, ``vsini``,
        ``epsilon``. ``v_shift`` is accepted but ignored (see note above).
        Extra keys (e.g. from `fixed_params`) are ignored.
    save_dir : str or Path, optional
        Directory where the .npz file will be saved. If None (default), uses the current directory.
    save_name : str, optional
        Name for the saved .npz file. If None (default), a default name is generated based on the stellar parameters.
    wv_range : list of float, optional
        [wv_min, wv_max] in µm for the saved spectrum. If None (default),
        reuses the wavelength range already covered by the fit's PHOENIX
        grid (the global `phoenix_interp`, built by `setup_stellar_fit()`).
        Give an explicit range to cover a broader band than what was used
        for the fit (e.g. the full instrument range for the planet retrieval).
    apply_rotation : bool
        If True (default), apply rotational broadening (`vsini`, `epsilon`)
        via `pyasl.fastRotBroad` before saving. Set to False to save the
        non-rotating PHOENIX spectrum (see the note on convolution order at
        the top of this module: rotation should only be skipped/reconsidered
        for slowly-rotating stars where vsini is not >> instrument FWHM).

    Returns
    -------
    wave : np.ndarray
        Wavelength grid (µm), sampled at constant resolving power
        `phoenix_resolution * phoenix_oversampling`.
    flux : np.ndarray
        Stellar flux (PHOENIX native units, erg/s/cm^2/cm).
    """
    _check_dependencies()
    if apply_rotation and theta_dict.get('vsini', 0) > 0 and not _PYASL_AVAILABLE:
        raise ImportError(
            "PyAstronomy is required for rotational broadening. "
            "Install it with: pip install PyAstronomy"
        )
        
    if save_dir is None:
        save_dir = Path.cwd()
        log.info(f"No save_dir specified; using current directory: {save_dir}")
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    params = dict(teff=theta_dict['teff'], logg=theta_dict['logg'],
                  metal=theta_dict['metal'], alpha=theta_dict.get('alpha', 0.0))
    
    
    if save_name is None:
        save_name = (f"starspec_teff{params['teff']:.0f}"
                     f"_logg{params['logg']:.2f}"
                     f"_metal{params['metal']:.2f}"
                     f"_alpha{params['alpha']:.2f}"
                     f"_vsini{theta_dict.get('vsini', 0):.0f}"
                     f"_eps{theta_dict.get('epsilon', 0):.2f}.npz")
        log.info(f"No save_name specified; using default: {save_name}")
    
    log.info(f"Generating stellar spectrum for: ")
    for key, val in params.items():
        log.info(f"  {key} = {val}")

    if wv_range is None:
        # Reuse the grid already built for the fit (last axis of the interpolator)
        grid = phoenix_interp
        wave = phoenix_interp.fct_interp.grid[-1]
    else:
        # Rebuild a grid over the requested range. Parameters are passed as
        # scalars (not [min, max]), so no interpolation axes are built -- this
        # is a single-point evaluation, not a grid meant to be reused for a fit.
        grid = PhoenixInterpGrid(
            wv_range=wv_range, resolution=phoenix_resolution,
            oversampling=phoenix_oversampling, n_fwhm=phoenix_n_fwhm,
            method=phoenix_method, query=False, **params,
        )
        wave = get_wv_constant_res(wv_range=wv_range,
                                   resolution=phoenix_resolution * phoenix_oversampling)

    if apply_rotation and theta_dict.get('vsini', 0) > 0:
        # Same approach as _generate_stellar_model_ord(): evaluate the PHOENIX
        # spectrum on a uniform-Δλ grid (required by pyasl.fastRotBroad),
        # convolve with the rotation profile, then spline back onto the
        # constant-R output grid. Pad the uniform grid so the rotation kernel
        # (width ~ vsini) does not pull in edge artefacts near wave's bounds.
        log.info(f"Applying rotational broadening: vsini={theta_dict['vsini']:.2f} km/s, "
                 f"epsilon={theta_dict['epsilon']:.2f}")
        vsini = theta_dict['vsini']
        wv_min, wv_max = add_dv_pad_to_wv_range(3 * vsini, [float(wave[0]), float(wave[-1])])
        delta_wv = wv_min / rot_broad_samp
        wv_uniform = np.arange(wv_min, wv_max, delta_wv)

        flux_uniform = grid(wv_uniform, **params)
        nan_mask = ~np.isfinite(flux_uniform)
        if nan_mask.any():
            valid_idx = np.where(~nan_mask)[0]
            flux_uniform[:valid_idx[0]] = flux_uniform[valid_idx[0]]
            flux_uniform[valid_idx[-1] + 1:] = flux_uniform[valid_idx[-1]]

        flux_uniform = pyasl.fastRotBroad(wv_uniform * 1000.0, flux_uniform,
                                          theta_dict['epsilon'], vsini)
        flux = UnivariateSpline(wv_uniform, flux_uniform, k=3, s=0)(wave)
    else:
        vsini = False
        flux = grid(wave, **params)

    # Save the spectrum and metadata to a .npz file
    save_path = save_dir / save_name
    np.savez(
        save_path.as_posix(),
        wave=wave,
        flux=flux,
        resolution=phoenix_resolution,
        oversampling=phoenix_oversampling,
        **params,  # Also save the stellar parameters in the .npz for reference (teff, logg, metal, alpha)
        vsini=vsini,  # =False if rotation was skipped, else the vsini value used
        # Backward-compat with the current starships.retrieval, which only
        # reads a single combined 'sampling_res' key.
        sampling_res=phoenix_resolution * phoenix_oversampling,
        
    )
    log.info(
        f"Stellar spectrum saved to {save_path} "
        f"(R={phoenix_resolution}, oversampling={phoenix_oversampling}, "
        f"{wave[0]:.3f}-{wave[-1]:.3f} µm)"
    )

    return wave, flux


# ===========================================================================
# PHOENIX file download (must be run on a login node, not a compute node)
# ===========================================================================

def download_phoenix_files(input_parameters):
    """Download the PHOENIX model files required by a given fit configuration.

    This function must be run **before** the fit, from a node with internet
    access (e.g. a Narval login node). Compute nodes cannot reach the
    internet, so running the fit directly without this step will fail if the
    required files are not already cached locally.

    Typical workflow on Narval::

        # On the login node (internet access):
        python -m starships.stellar_fit download my_config.yaml

        # Then submit the fit to the scheduler:
        sbatch run_stellar_fit.sh

    The files are saved in DEFAULT_MODEL_DIR, which defaults to
    ``$SCRATCH/Models/PHOENIX_HiRes/``.

    Parameters
    ----------
    input_parameters : str, Path, or dict
        Path to the YAML config file, or a dict with the same structure.
        The PHOENIX parameter ranges are inferred from ``params_prior``
        (prior bounds define the range) and ``fixed_params`` (fixed value).
        For Gaussian priors, the range ±5σ around the mean is used.
    """
    from starships.phoenix_models import (
        get_phoenix_grid_in_range,
        _get_all_phoenix_files,
        get_phoenix_wv_grid,
        DEFAULT_MODEL_DIR,
    )

    # Read the config without initialising the model (files may not exist yet)
    if isinstance(input_parameters, dict):
        cfg = dict(input_parameters)
    else:
        with open(input_parameters, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

    params_prior_cfg = cfg.get('params_prior', {})
    fixed_params_cfg = cfg.get('fixed_params', {})

    # Determine the range of each PHOENIX grid parameter from the config.
    # Parameters that are fixed use a scalar; parameters in the prior use
    # their full range so that all needed grid points are downloaded.
    phoenix_param_names = ['teff', 'logg', 'metal', 'alpha']
    phoenix_params = {}

    for param in phoenix_param_names:
        if param in params_prior_cfg:
            prior_info = params_prior_cfg[param]
            # _get_prior_effective_range handles all supported prior types.
            # For Gaussian-type priors it returns ±5σ, which covers all grid
            # points the sampler is likely to visit.
            lo, hi = _get_prior_effective_range(prior_info)
            if lo is None:
                raise ValueError(
                    f"Cannot determine download range for PHOENIX parameter "
                    f"'{param}': unsupported prior type '{prior_info[0]}'. "
                    f"Supported types: uniform, log_uniform, gaussian, "
                    f"split_gaussian, combined_split_gaussian."
                )
            phoenix_params[param] = [lo, hi]
        elif param in fixed_params_cfg and fixed_params_cfg[param] is not None:
            phoenix_params[param] = float(fixed_params_cfg[param])
        else:
            phoenix_params[param] = 0.0  # default: solar alpha abundance

    # Find the grid points that bracket the requested parameter ranges
    grid = get_phoenix_grid_in_range(**phoenix_params, extrapolate=True)
    n_files = int(np.prod([len(v) for v in grid.values()]))

    log.info(f"PHOENIX files will be saved to: {DEFAULT_MODEL_DIR}")
    log.info("Parameter grid to download:")
    for k, v in grid.items():
        log.info(f"  {k}: {v}")
    log.info(f"Total files to download: {n_files} model file(s) + 1 wavelength grid")

    # Download the shared wavelength grid first
    log.info("Downloading wavelength grid...")
    get_phoenix_wv_grid(query=True)

    # Download all model FITS files
    log.info("Downloading model files (this may take a while)...")
    _get_all_phoenix_files(grid, query=True)

    log.info("Download complete. You can now run the fit on a compute node.")


# ===========================================================================
# Entry point helpers — SLURM ID and config archiving
# ===========================================================================

def get_slurm_id():
    """Return the SLURM job ID as a string, or None if not running under SLURM.

    For array jobs (``sbatch --array``), combines the array job ID and task
    ID with an underscore — e.g. ``'12345678_3'``. For regular jobs, returns
    just the job ID — e.g. ``'12345678'``. Returns None when called outside
    of a SLURM environment (login node, local machine, notebook).
    """
    if 'SLURM_ARRAY_JOB_ID' in os.environ:
        keys = ['SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID']
    else:
        keys = ['SLURM_JOB_ID']
    try:
        return '_'.join(os.environ[k] for k in keys)
    except KeyError:
        log.info("SLURM job ID not found — running outside of SLURM.")
        return None


def _save_run_config(input_parameters, slurm_id=None):
    """Archive the input configuration alongside the results.

    Saves a copy of the YAML config to ``walker_path`` with the SLURM job ID
    embedded in the filename. This makes it trivial to find the inputs that
    produced a given results file:

        inputs_stellar_fit_KELT-20_12345678.yaml   ←→   results_stellar_fit_KELT-20_12345678.pkl

    The saved YAML is a snapshot of the actual parameters used, enriched with
    a few extra fields (slurm_id, results_file) for traceability.

    This function is called automatically by ``main()`` and should not need to
    be called manually from notebooks (which manage their own files).

    Parameters
    ----------
    input_parameters : str, Path, or dict
        Original input config (file path or dict). File paths are read and
        re-written so that extra metadata fields can be added.
    slurm_id : str or None
        SLURM job ID string (from ``get_slurm_id()``), embedded in the filename.
        If None (local run), the file is saved without a job-ID suffix.
    """
    import shutil

    # Build output filename:  inputs_<run_name>[_<slurm_id>].yaml
    stem = f"inputs_{run_name}"
    if slurm_id is not None:
        stem = f"{stem}_{slurm_id}"
    out_path = Path(walker_path) / f"{stem}.yaml"
    Path(walker_path).mkdir(parents=True, exist_ok=True)

    # Load the config (file or dict) and add traceability fields
    if isinstance(input_parameters, dict):
        cfg = {k: (str(v) if isinstance(v, Path) else v)
               for k, v in input_parameters.items()}
    else:
        with open(Path(input_parameters).expanduser(), 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Add metadata that makes it easy to match inputs ↔ results
    cfg['slurm_id'] = slurm_id
    cfg['results_file'] = str(Path(walker_path) / f"{walker_file_out}.pkl")

    with open(out_path, 'w') as f:
        yaml.dump(cfg, f, sort_keys=False)

    log.info(f"Input config archived to: {out_path}")
    return out_path


# ===========================================================================
# Entry point
# ===========================================================================

def main(yaml_file):
    """Run the full stellar fit pipeline from a YAML config file.

    Intended for use in an sbatch script on Narval:
        python -m starships.stellar_fit my_config.yaml

    Steps:
      1. Load config and data.
      2. Append the SLURM job ID to output filenames (prevents overwriting).
      3. Archive the input YAML alongside the results for reproducibility.
      4. Quick minimisation to find the best-fit point estimate.
      5. Full posterior with dynesty.

    Parameters
    ----------
    yaml_file : str or Path
        Path to the YAML config file.

    Returns
    -------
    dynesty_results : dynesty.results.Results
    """
    global run_name, walker_file_out

    setup_stellar_fit(yaml_file)

    # Append the SLURM job ID to all output filenames so that re-submitting
    # the same YAML never overwrites a previous run's results.
    # Without this, results_stellar_fit_KELT-20.pkl would be silently replaced
    # every time the script is submitted, making it impossible to compare runs.
    slurm_id = get_slurm_id()
    if slurm_id is not None:
        run_name = f"{run_name}_{slurm_id}"
        walker_file_out = f"{walker_file_out}_{slurm_id}"
        log.info(f"SLURM job {slurm_id}: outputs will include _{slurm_id} suffix.")

    # Archive the input config BEFORE starting the run, so the inputs are
    # always traceable even if the job is killed or crashes partway through.
    _save_run_config(yaml_file, slurm_id=slurm_id)

    log.info("=== Step 1: Point estimate (minimisation) ===")
    run_minimize()

    log.info("=== Step 2: Full posterior (dynesty) ===")
    results = run_dynesty(
        n_live=n_live_points,
        save_file=f"{walker_file_out}.pkl",
        n_workers=globals().get('n_workers', 1),
        bound=globals().get('dynesty_bound', 'multi'),
        sample=globals().get('dynesty_sample', 'rwalk'),
    )
    return results


if __name__ == '__main__':
    import sys

    usage = (
        "Usage:\n"
        "  python -m starships.stellar_fit <config.yaml>           # run the fit\n"
        "  python -m starships.stellar_fit download <config.yaml>  # download PHOENIX files first\n"
    )

    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)

    if sys.argv[1] == 'download':
        if len(sys.argv) < 3:
            print(usage)
            sys.exit(1)
        download_phoenix_files(sys.argv[2])
    else:
        main(sys.argv[1])
